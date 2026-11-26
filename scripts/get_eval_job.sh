#!/usr/bin/env bash

export EBENCH_SUBMIT_TOKEN='sk-pRlYogu2W9Cu0ImFeQL9R98V0egQbeKewytF4rDeGTBWD7BG'

gmp online submit \
  --base_url https://internrobotics.shlab.org.cn/eval \
  --token "$EBENCH_SUBMIT_TOKEN" \
  --benchmark_set wet_op_bench_test \
  --model_name RynnVLA \
  --model_type VLA \
  --submitter_name smoke \
  --submitter_homepage smoke \
  --is_public 0
