#!/bin/bash
# srun -p mllm --gres gpu:8 bash scripts/share4video/eval/8b/vbench.sh

CKPT=${1-"llava-v1.6-llama3-8b"}
CKPT_DIR=${2-"checkpoints"}
MODEL_BASE=pretrained/llava-v1.6-llama3-8b
NUM_GRID=16

python -m llava.eval.video.eval_vbench \
            --model-path ${CKPT_DIR}/${CKPT} \
            --num_frames ${NUM_GRID} \
            --conv-mode eval_vbench_llama3 \
            --save_path ./playground/results/vbench/${CKPT}