#!/usr/bin/env bash

export EBENCH_TASK_ID='EBENCH_TASK_ID'
export EBENCH_ONLINE_ENDPOINT='EBENCH_ONLINE_ENDPOINT'

python -m benchmark.wetop_online run \
  --checkpoint "/gaowenbin/models/trained_models/RynnVLA-S1/stage2" \
  --endpoint "$EBENCH_ONLINE_ENDPOINT" \
  --run-id "$EBENCH_TASK_ID" \
  --worker-id 0 \
  --max-episodes 10
