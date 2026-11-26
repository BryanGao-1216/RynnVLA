# RynnVLA on Wet-OP Bench

Run the launchers from the repository root with `gmp` and the RynnVLA inference environment available. Use an exported **Stage 2** WET-OP checkpoint. [Platform Quick Start](https://internrobotics.shlab.org.cn/eval/quick-start)

Set the platform token in `scripts/get_eval_job.sh` and submit the job.

```bash
bash scripts/get_eval_job.sh
```

The script submits a private `wet_op_bench_test` job and prints `task_id` and `endpoint` when it is ready. Fill in those values and the platform token at the top of `scripts/eval.sh`.

```bash
bash scripts/eval.sh
```

Set `RYNNVLA_CHECKPOINT` at the top of `scripts/eval.sh` to the exported WET-OP Stage 2 directory, saved under `rynnvla.training_output_dir/stage2` by the training launcher. `RynnVLA-Latent-4B` is the Stage 1 initialization and predicts latent actions. The WET-OP Stage 2 model consumes the platform instruction, four training camera views and measured Lift2 state, and predicts relative end-effector transforms with absolute finger positions. It uses the checkpoint's normalization and produces one native simulator action per prediction.

The launcher runs worker 0 and counts each scored episode toward `--max-episodes`, including episodes with score zero. Set `--max-episodes 0` in `scripts/eval.sh` for the full job. `--chunk-size N` executes the first `N` actions of each predicted chunk before obtaining a new observation. The default executes the full checkpoint horizon.
