#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

pids=()

for i in {7..7}; do
    config_path="$script_dir/../configs/config-${i}.json"

    "${PYTHON_BIN:-python}" "$script_dir/label_lerobot_v3.py" \
        --config "$config_path" "$@" \
        > "label-${i}.log" 2>&1 &

    pids+=("$!")
done

status=0

for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

exit "$status"