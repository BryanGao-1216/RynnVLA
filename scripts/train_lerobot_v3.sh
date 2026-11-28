#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# SwanLab 首次安装：python -m pip install '.[swanlab]'
# local 写本地，online 上传到 SwanLab，disabled 关闭记录。
export SWANLAB_MODE=online
export SWANLAB_PROJ_NAME="RynnVLA-WETOP"
export SWANLAB_EXP_NAME="wetop-stage1"
export SWANLAB_LOGDIR="./runs/swanlab"
# online 模式先运行 swanlab login，或在启动前设置 SWANLAB_API_KEY。
# export SWANLAB_WORKSPACE="your-workspace"

# 在仓库根目录运行。第二阶段把 stage1 改成 stage2。
# --nproc-per-node 填上面选择的 GPU 数量。
# 只准备数据时加 --prepare-only，续训时加 --resume。
python scripts/train_lerobot_v3.py stage1 \
    --nproc-per-node 1 \
    --config configs/config.json
