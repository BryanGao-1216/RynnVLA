"""RynnVLA's WET-OP LeRobot v3 vectors and GenManip's Lift2 wire format."""

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from rynnvla.constants import RobotType, VIEW_ROLE_TO_ID


ROBOT_TYPE = RobotType.LIFT2
VECTOR_SIZE = 16
CAMERA_ROLES = {
    "video.top_camera_view": "head",
    "video.left_camera_view": "left_wrist",
    "video.right_camera_view": "right_wrist",
    "video.overlook_camera_view": "front_third",
}
CAMERA_SLOTS = {f"view{VIEW_ROLE_TO_ID[role]}": VIEW_ROLE_TO_ID[role] for role in CAMERA_ROLES.values()}


def state_vector(value, size, field):
    values = np.asarray(value, dtype=np.float64).squeeze()
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"{field} must contain {size} finite values, received shape {values.shape}")
    return values


def observations(native):
    """Pack state as L xyz/rotvec/fingers, then R, in the training camera slots."""
    text = native.get("instruction")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("WET-OP observation requires a nonempty instruction")

    grippers = state_vector(native.get("state.gripper"), 4, "state.gripper").reshape(2, 2)
    poses = native.get("state.ee_pose")
    if not isinstance(poses, (tuple, list, np.ndarray)) or len(poses) != 2:
        raise ValueError("state.ee_pose must contain left and right [XYZ, WXYZ] pairs")
    state = []
    for arm, (pose, fingers) in enumerate(zip(poses, grippers)):
        if not isinstance(pose, (tuple, list, np.ndarray)) or len(pose) != 2:
            raise ValueError(f"state.ee_pose[{arm}] must contain [XYZ, WXYZ]")
        xyz = state_vector(pose[0], 3, f"state.ee_pose[{arm}].xyz")
        quat = state_vector(pose[1], 4, f"state.ee_pose[{arm}].wxyz")
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3, rtol=0):
            raise ValueError(f"state.ee_pose[{arm}] must contain a unit WXYZ quaternion")
        state.extend((xyz, Rotation.from_quat(quat[[1, 2, 3, 0]]).as_rotvec(), fingers))

    images = {}
    for field, role in CAMERA_ROLES.items():
        frame = native.get(field)
        if isinstance(frame, Image.Image):
            frame = frame.convert("RGB")
        image = np.asarray(frame)
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
                or min(image.shape[:2]) == 0):
            raise ValueError(f"{field} ({role}) must be a nonempty RGB uint8 HWC image")
        images[f"view{VIEW_ROLE_TO_ID[role]}"] = image
    return text.strip()[:512], images, np.concatenate(state).astype(np.float32)


def action_chunk(actions, worker_id):
    """Encode SE(3) deltas and absolute finger targets at the native step rate."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or not len(actions) or actions.shape[1] != VECTOR_SIZE or not np.isfinite(actions).all():
        raise ValueError(f"Expected a nonempty, finite (T, {VECTOR_SIZE}) action chunk, got {actions.shape}")

    arms = actions.reshape(-1, 2, 8)
    quats = Rotation.from_rotvec(arms[:, :, 3:6].reshape(-1, 3)).as_quat()
    quats = quats[:, [3, 0, 1, 2]].reshape(-1, 2, 4)
    chunk = []
    for step, rotations in zip(arms, quats):
        chunk.append({worker_id: {
            "action": [
                [arm[:3].tolist(), quat.tolist(), arm[6:].tolist()]
                for arm, quat in zip(step, rotations)
            ],
            "control_type": "ee_pose",
            # GenManip left-composes each delta with the last commanded pose.
            # The demonstration recorder computes this same delta between consecutive targets.
            "is_rel": True,
            "base_motion": [0.0, 0.0, 0.0],
            "base_is_rel": True,
        }})
    return chunk


class WetopPolicy:
    def __init__(self, checkpoint, *, device, dtype, denoising_steps, chunk_size):
        import torch

        from rynnvla.inference_wrappers.rynn_brain_vla import RynnBrainVLAInferenceWrapper

        self.torch = torch
        self.denoising_steps = denoising_steps
        self.wrapper = RynnBrainVLAInferenceWrapper(
            model_path=str(checkpoint), device=device, dtype=getattr(torch, dtype), attn_implementation="sdpa",
        )
        processor = self.wrapper.processor
        schema = processor.schema or {}
        for section in ("state", "action"):
            fields = schema.get(section, {}).get(ROBOT_TYPE.value, {})
            leaf = fields.get("native_vector", {})
            if (set(fields) - {"type"} != {"native_vector"} or leaf.get("dim") != VECTOR_SIZE
                    or leaf.get("is_relative") is not False or leaf.get("allow_relative") is not False):
                raise ValueError(f"Checkpoint requires the Lift2 {VECTOR_SIZE}D LeRobot v3 native {section} schema")

        config = self.wrapper.model.config
        self.action_dim = config.action_dim
        self.horizon = config.action_chunk_size
        if config.use_latent_actions or self.action_dim < VECTOR_SIZE or self.horizon < 1 or not processor.use_state:
            raise ValueError("Use a state-conditioned RynnVLA Stage 2 checkpoint trained on WET-OP")
        self.chunk_size = chunk_size if chunk_size is not None else self.horizon
        if not 1 <= self.chunk_size <= self.horizon:
            raise ValueError(f"--chunk-size must be between 1 and checkpoint horizon {self.horizon}")

    def __call__(self, native):
        from rynnvla.utils.robot import Position, RobotState

        torch = self.torch
        text, images, values = observations(native)
        state = RobotState(native_vector=Position(torch.from_numpy(values), allow_relative=False))
        with torch.no_grad():
            inputs = self.wrapper.process(text, images, state.to_dict(), ROBOT_TYPE.value, CAMERA_SLOTS)
            inputs = {
                key: value.to(self.wrapper.device) if isinstance(value, torch.Tensor) else value
                for key, value in self.wrapper.collate([inputs]).items()
            }
            cache = self.wrapper.prefill(inputs)
            actions = self.wrapper.decode(inputs, cache, self.denoising_steps, robot_type=ROBOT_TYPE.value)
            if actions.shape != (1, self.horizon, self.action_dim):
                raise ValueError(f"Unexpected RynnVLA action tensor shape {tuple(actions.shape)}")
            action = self.wrapper.processor.post_process(
                actions[0, :self.chunk_size].detach().cpu().float(), robot_type=ROBOT_TYPE, state=None,
            )
        return action.native_vector.data.numpy()
