# RynnVLA on Wet-OP Bench

Run the launchers from the repository root with `gmp` and the RynnVLA inference environment available. Use an exported **Stage 2** WET-OP checkpoint. [Platform Quick Start](https://internrobotics.shlab.org.cn/eval/quick-start)

Set the platform token in `get_eval_job.sh` and submit the job.

```bash
./get_eval_job.sh
```

The script submits a private `wet_op_bench_test` job and prints `task_id` and `endpoint` when it is ready. Fill in those values at the top of `eval.sh` and provide `EBENCH_SUBMIT_TOKEN` in the evaluation environment.

```bash
./eval.sh
```

Set `RYNNVLA_CHECKPOINT` at the top of `eval.sh` to the exported Stage 2 model directory. The local model consumes the platform instruction, the four training camera views and the measured Lift2 state. It uses the checkpoint's normalization and predicts relative end-effector transforms with absolute finger positions. Each prediction is one native simulator action. Direct RynnVLA inference needs no ARK API key.

The launcher runs worker 0 and counts each scored episode toward `--max-episodes`, including episodes with score zero. Set `--max-episodes 0` in `eval.sh` for the full job. `--chunk-size N` executes the first `N` actions of each predicted chunk before obtaining a new observation. The default executes the full checkpoint horizon.
