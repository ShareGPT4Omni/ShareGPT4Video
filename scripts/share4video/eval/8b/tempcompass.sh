!/bin/bash
# bash scripts/share4video/eval/8b/tempcompass.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT=${1-"yes_no"}
CKPT=${2-""}
CKPT_DIR="checkpoints"
MAX_NEW_TOKENS=128

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_tempcompass \
        --model-path ${CKPT_DIR}/${CKPT} \
        --num-grid 16 \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --image-folder playground/data/videos/TempCompass/video \
        --chat_conversation_output_folder playground/results/tempcompass/predictions/$SPLIT/${CKPT}/${CHUNKS}_${IDX}.json \
        --Eval_QA_root playground/data/videos/TempCompass/$SPLIT.json \
        --task_type $SPLIT \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llava_llama_3 &
done
wait

cd playground/results/tempcompass
python combine_result.py --output_file predictions/$SPLIT/${CKPT}/$SPLIT.json --num_chunks $CHUNKS
python eval_$SPLIT.py --video_llm ${CKPT}
