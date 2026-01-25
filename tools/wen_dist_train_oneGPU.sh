# 使用方法： ./tools/wen_dist_train_oneGPU.sh ./projects/configs/MonoOcc/MonoOcc-S.py
#!/usr/bin/env bash

CONFIG=$1
GPUS=1     # 強制設為 1
PORT=${PORT:-21417}
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/wen_train.py $CONFIG --launcher pytorch ${@:3} --deterministic --gpus $GPUS --seed 42 --no-validate