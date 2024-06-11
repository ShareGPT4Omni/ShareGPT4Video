#!/bin/bash
# bash scripts/share4video/eval/8b/mvbench.sh


CKPT=${1-"llava-v1.6-8b_llama3-8b_siglip_video-sft_ft-mlp-llm-lora_lr-mlp-2e-5-llm-2e-4"}
CKPT_DIR=${2-"checkpoints"}
MODEL_BASE=pretrained/llava-v1.6-llama3-8b
NUM_GRID=16

python -m llava.eval.video.eval_mvbench \
            --model-path ${CKPT_DIR}/${CKPT} \
            --num_frames ${NUM_GRID} \
            --conv-mode eval_mvbench_llama3 \
            --save_path ./playground/results/mvbench/${CKPT}