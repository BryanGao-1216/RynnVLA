#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# 在仓库根目录运行。第二阶段把 stage1 改成 stage2。
# --nproc-per-node 填上面选择的 GPU 数量。
# 只准备数据时加 --prepare-only，续训时加 --resume。
python scripts/train_lerobot_v3.py stage1 \
    --nproc-per-node 1 \
    --config configs/config.json
    
