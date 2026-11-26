#!/usr/bin/env bash

export EBENCH_TASK_ID='EBENCH_TASK_ID'
export EBENCH_ONLINE_ENDPOINT='EBENCH_ONLINE_ENDPOINT'
export EBENCH_SUBMIT_TOKEN='sk-pRlYogu2W9Cu0ImFeQL9R98V0egQbeKewytF4rDeGTBWD7BG'

python -m benchmark.wetop_online run \
  --checkpoint "/gaowenbin/models/base_models/RynnVLA-Latent-4B" \
  --endpoint "$EBENCH_ONLINE_ENDPOINT" \
  --run-id "$EBENCH_TASK_ID" \
  --worker-id 0 \
  --max-episodes 10
