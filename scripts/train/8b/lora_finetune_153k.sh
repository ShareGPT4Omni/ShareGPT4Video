#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=2
export GRADIENT_ACCU_STEPS=2
export BATCH_SIZE=4
export MASTER_PORT=29508
export CPUS_PER_TASK=24
export QUOTA=reserved

export DATA_PATH=data/image_grid/sharegpt4video_sft-mix153k.json
export SAVE_PATH=llava-v1.6-8b_lama3-8b_video-sft-mix153k_ft-mlp-llm-lora_lr-mlp-2e-5-llm-2e-4_run2
export BASE_LR=2e-4
export MLP_LR=2e-5

SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p mllm \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA} \
    ${SRUN_ARGS} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr ${MLP_LR} \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path pretrained/llava-v1.6-llama3-8b \
    --version llava_llama_3 \
    --data_path ${DATA_PATH} \
    --image_folder data \
    --vision_tower pretrained/vision_encoder/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --unfreeze_mm_vision_tower False \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SAVE_PATH}'