"""Run a local RynnVLA Stage 2 policy against the WET-OP online worker."""

import argparse
import json
from numbers import Real
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = ROOT / "configs" / "config.json"


def run_online(client, policy, worker_id, max_episodes):
    from .wetop_policy import action_chunk

    response = client.reset()
    done = False
    active_episode = None
    completed = set()
    while True:
        worker = response.get(worker_id) if isinstance(response, dict) else None
        if not isinstance(worker, dict) or "obs" not in worker:
            raise ValueError(f"Platform response is missing the observation for worker {worker_id}")
        if worker.get("reset_pending"):
            raise RuntimeError("GenManip client returned an unresolved pending reset")

        result = worker.get("episode_result")
        if result is not None:
            if (not isinstance(result, dict) or active_episode is None
                    or result.get("episode_id") != active_episode):
                raise ValueError(f"Platform episode result does not match active episode {active_episode!r}")
            if any(not isinstance(result.get(key), Real) or not 0 <= result[key] <= 1
                   for key in ("score", "sr")):
                raise ValueError(f"Invalid score or success rate for episode {active_episode!r}")
            completed.add(active_episode)
            print(f"{active_episode}  score={result['score']}  sr={result['sr']}", flush=True)
            active_episode = None

        if worker.get("lock_lost"):
            print(f"Worker {worker_id} lost its lock on {active_episode}; continuing with the returned observation",
                  file=sys.stderr, flush=True)
            active_episode = None
        native = worker["obs"]
        if native is None:
            if active_episode is not None or worker.get("metric") is None:
                raise RuntimeError("Platform ended the worker without an episode result or final metrics")
            return len(completed)
        if done:
            raise RuntimeError("GenManip reported a finished job with a live observation")
        if max_episodes and len(completed) >= max_episodes:
            return len(completed)
        if not isinstance(native, dict):
            raise ValueError("Platform observation must be a dictionary")
        episode = native.get("episode_id")
        if not isinstance(episode, str) or not episode or episode in completed:
            raise ValueError(f"Unexpected platform episode ID {episode!r}")
        if active_episode is not None and (episode != active_episode or native.get("reset")):
            raise RuntimeError(f"Platform changed episode without returning a result for {active_episode}")
        active_episode = episode
        response, done = client.step(action_chunk(policy(native), worker_id))


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def configured_checkpoint():
    from rynnvla.utils.lerobot_v3 import resolve_path

    config = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
    return resolve_path(TRAINING_CONFIG.parent, config["rynnvla"]["training_output_dir"]) / "stage2"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    run = parser.add_subparsers(dest="command", required=True).add_parser("run", help="evaluate one worker")
    run.add_argument("--checkpoint", type=Path, default=os.getenv("RYNNVLA_CHECKPOINT") or None,
                     help="exported Stage 2 HF model (default: training_output_dir/stage2 from configs/config.json)")
    run.add_argument("--endpoint", default=os.getenv("EBENCH_ONLINE_ENDPOINT"))
    run.add_argument("--run-id", default=os.getenv("EBENCH_TASK_ID"))
    run.add_argument("--worker-id", type=int, default=0)
    run.add_argument("--max-episodes", type=int, default=1, help="score this many episodes; 0 runs the full job")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                     help="default: bfloat16 on CUDA, float32 elsewhere")
    run.add_argument("--denoising-steps", type=positive_int, default=10)
    run.add_argument("--chunk-size", type=positive_int, help="execute this many predictions before replanning")
    args = parser.parse_args(argv)

    token = os.getenv("EBENCH_SUBMIT_TOKEN", "").strip()
    if not token or not args.run_id or not args.endpoint:
        run.error("set EBENCH_SUBMIT_TOKEN, EBENCH_TASK_ID, and EBENCH_ONLINE_ENDPOINT from the submitted job")
    endpoint = urlsplit(args.endpoint)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
            or endpoint.username is not None or endpoint.password is not None or endpoint.query or endpoint.fragment):
        run.error("--endpoint must be an HTTP(S) URL with no credentials, query or fragment")
    if args.worker_id < 0 or args.max_episodes < 0:
        run.error("--worker-id and --max-episodes must be nonnegative")
    try:
        checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else configured_checkpoint()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        run.error(f"Unable to resolve a Stage 2 model from {TRAINING_CONFIG}: {exc}. Pass --checkpoint.")
    if not all((checkpoint / name).is_file() for name in ("config.json", "processor_config.json")):
        run.error(f"No exported Stage 2 model at {checkpoint}; set RYNNVLA_CHECKPOINT or pass --checkpoint")
    model_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if model_config.get("use_latent_actions") is True:
        run.error(
            f"{checkpoint} is a Stage 1 latent-action checkpoint. "
            "Set RYNNVLA_CHECKPOINT in scripts/eval.sh to the exported WET-OP Stage 2 model directory."
        )

    from genmanip_client import EvalClient
    from .wetop_policy import WetopPolicy

    dtype = args.dtype or ("bfloat16" if args.device.startswith("cuda") else "float32")
    print(f"Loading RynnVLA from {checkpoint} on {args.device}", flush=True)
    policy = WetopPolicy(checkpoint, device=args.device, dtype=dtype,
                         denoising_steps=args.denoising_steps, chunk_size=args.chunk_size)
    worker_id = str(args.worker_id)
    os.environ.setdefault("GENMANIP_RESULT_DIR", str(ROOT / "runs" / "wetop" / "client_results"))
    client = EvalClient(
        base_url=args.endpoint, run_id=args.run_id, token=token, worker_ids=[worker_id],
        save_result=False, save_process=False, verbose=False,
    )
    try:
        count = run_online(client, policy, worker_id, args.max_episodes)
    finally:
        client.close()
    print(f"Worker {worker_id} scored {count} episode(s) in this session.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        sys.exit(f"[wetop] {exc}")
