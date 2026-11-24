#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON_BIN:-python}" "$script_dir/label_lerobot_v3.py" "$@"
