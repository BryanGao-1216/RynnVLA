#!/usr/bin/env bash
set -euo pipefail

# Set the default config here. Relative paths are relative to the repository root.
config_path="configs/config.json"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$config_path" != /* ]]; then
    config_path="$script_dir/../$config_path"
fi

exec "${PYTHON_BIN:-python}" "$script_dir/label_lerobot_v3.py" --config "$config_path" "$@"
