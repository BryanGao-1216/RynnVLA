from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmark.wetop_online import main, run_online
from benchmark.wetop_policy import CAMERA_SLOTS, ROBOT_TYPE, WetopPolicy, action_chunk, observations


def native_observation():
    return {
        "instruction": "Move the container",
        "state.ee_pose": [
            [[1, 2, 3], [2**-0.5, 2**-0.5, 0, 0]],
            [[4, 5, 6], [2**-0.5, 0, -2**-0.5, 0]],
        ],
        "state.gripper": [[0.01, 0.02, 0.03, 0.04]],
        "video.top_camera_view": np.full((3, 4, 3), 11, dtype=np.uint8),
        "video.left_camera_view": np.full((3, 4, 3), 22, dtype=np.uint8),
        "video.right_camera_view": np.full((3, 4, 3), 33, dtype=np.uint8),
        "video.overlook_camera_view": np.full((3, 4, 3), 44, dtype=np.uint8),
    }


def test_lift2_observation_and_action_use_training_layout():
    text, images, vector = observations(native_observation())
    assert text == "Move the container"
    assert CAMERA_SLOTS == {"view0": 0, "view1": 1, "view2": 2, "view3": 3}
    assert [images[name][0, 0, 0] for name in sorted(images)] == [11, 22, 33, 44]
    np.testing.assert_allclose(vector, [
        1, 2, 3, np.pi / 2, 0, 0, 0.01, 0.02,
        4, 5, 6, 0, -np.pi / 2, 0, 0.03, 0.04,
    ])

    step = action_chunk(vector[None, :], "2")[0]["2"]
    assert step["control_type"] == "ee_pose"
    assert step["is_rel"] and step["base_is_rel"]
    assert step["base_motion"] == [0.0, 0.0, 0.0]
    for arm, (position, quat, fingers) in enumerate(step["action"]):
        row = vector[arm * 8:arm * 8 + 8]
        assert position == row[:3].tolist()
        assert fingers == row[6:].tolist()
        np.testing.assert_allclose(Rotation.from_quat(np.array(quat)[[1, 2, 3, 0]]).as_rotvec(), row[3:6])


def test_checkpoint_normalization_preserves_all_native_dimensions(monkeypatch):
    import torch

    from rynnvla.inference_wrappers.rynn_brain_vla import RynnBrainVLAInferenceWrapper
    from rynnvla.models.rynn_brain_vla import RynnBrainVLAProcessor

    leaf = {"type": "Position", "dim": 16, "is_relative": False, "allow_relative": False,
            "min": list(range(16)), "max": list(range(2, 18))}
    processor = object.__new__(RynnBrainVLAProcessor)
    processor.schema = {
        "action": {"lift2": {"type": "RobotAction", "native_vector": leaf}},
        "state": {"lift2": {"type": "RobotState", "native_vector": deepcopy(leaf)}},
    }
    processor.action_norm_type = "min_max_sym"
    processor.use_state = True
    config = SimpleNamespace(action_dim=81, action_chunk_size=3, use_latent_actions=False)
    monkeypatch.setattr(RynnBrainVLAInferenceWrapper, "load_model", lambda self: SimpleNamespace(config=config))
    monkeypatch.setattr(RynnBrainVLAInferenceWrapper, "load_processor", lambda self: processor)

    def process(wrapper, text, images, state, robot_type, camera_slot_map):
        assert robot_type == ROBOT_TYPE.value and camera_slot_map == CAMERA_SLOTS
        assert state["native_vector"]["data"][0][6:8] == pytest.approx([0.01, 0.02])
        return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    monkeypatch.setattr(RynnBrainVLAInferenceWrapper, "process", process)
    monkeypatch.setattr(RynnBrainVLAInferenceWrapper, "collate", lambda self, batch: batch[0])
    monkeypatch.setattr(RynnBrainVLAInferenceWrapper, "prefill", lambda self, inputs: {})
    monkeypatch.setattr(RynnBrainVLAInferenceWrapper, "decode", lambda *args, **kwargs: torch.zeros(1, 3, 81))

    policy = WetopPolicy("unused", device="cpu", dtype="float32", denoising_steps=3, chunk_size=2)
    np.testing.assert_array_equal(policy(native_observation()), [np.arange(1, 17)] * 2)


def test_stage1_checkpoint_is_rejected_before_loading_weights(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.json").write_text('{"use_latent_actions": true}', encoding="utf-8")
    (tmp_path / "processor_config.json").write_text('{}', encoding="utf-8")
    monkeypatch.setenv("EBENCH_SUBMIT_TOKEN", "test-token")

    with pytest.raises(SystemExit) as exc:
        main(["run", "--checkpoint", str(tmp_path), "--endpoint", "https://example.test", "--run-id", "test"])

    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "Stage 1 latent-action checkpoint" in message
    assert "exported WET-OP Stage 2 model directory" in message


@pytest.mark.parametrize("max_episodes", [0, 1])
def test_episode_limit_counts_scored_episodes_and_never_uses_next_reset_frame(max_episodes):
    decisions = []

    def policy(native):
        decisions.append(native["episode_id"])
        return np.zeros((2, 16))

    def response(episode, *, result=None, lock_lost=False):
        native = {"episode_id": episode, "reset": True} if episode else None
        return {"3": {"obs": native, "episode_result": result, "metric": {}, "lock_lost": lock_lost}}

    next_responses = iter([
        (response("retry", lock_lost=True), False),
        (response("next", result={"episode_id": "retry", "score": 0.0, "sr": 0.0}), False),
        (response(None, result={"episode_id": "next", "score": 1.0, "sr": 1.0}), True),
    ])

    def step(chunk):
        assert len(chunk) == 2 and all(set(step) == {"3"} for step in chunk)
        return next(next_responses)

    client = SimpleNamespace(reset=lambda: response("original"), step=step)
    assert run_online(client, policy, "3", max_episodes) == (1 if max_episodes else 2)
    assert decisions == (["original", "retry"] if max_episodes else ["original", "retry", "next"])
