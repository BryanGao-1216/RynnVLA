#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export SWANLAB_MODE=online
export SWANLAB_PROJ_NAME="RynnVLA-WETOP"
export SWANLAB_EXP_NAME="wetop-stage1"
export SWANLAB_LOGDIR="./runs/swanlab"

# 在仓库根目录运行。第二阶段把 stage1 改成 stage2。。
# 只准备数据时加 --prepare-only，续训时加 --resume。
python scripts/train_lerobot_v3.py stage1 \
    --nproc-per-node 1 \
    --config configs/config.json
