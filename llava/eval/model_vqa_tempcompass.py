import argparse
import json
import math
import os

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from tqdm import tqdm

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def create_frame_grid(img_array, interval_width):
    n, h, w, c = img_array.shape
    grid_size = int(np.ceil(np.sqrt(n)))

    horizontal_band = np.ones((h, interval_width, c),
                              dtype=img_array.dtype) * 255
    vertical_band = np.ones((interval_width, w + (grid_size - 1)
                            * (w + interval_width), c), dtype=img_array.dtype) * 255

    rows = []
    for i in range(grid_size):
        row_frames = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n:
                frame = img_array[idx]
            else:
                frame = np.ones_like(img_array[0]) * 255
            if j > 0:
                row_frames.append(horizontal_band)
            row_frames.append(frame)
        combined_row = np.concatenate(row_frames, axis=1)
        if i > 0:
            rows.append(vertical_band)
        rows.append(combined_row)

    final_grid = np.concatenate(rows, axis=0)
    return final_grid


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def load_video(vis_path, num_frm=8):
    vr = VideoReader(vis_path)
    total_frame_num = len(vr)
    frame_idx = get_seq_frames(total_frame_num, num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)
    img_grid = create_frame_grid(img_array, 50)
    img_grid = Image.fromarray(img_grid).convert("RGB")

    return [img_grid]


def process_data(video_id, qs, model_config, image_folder, tokenizer, processor, num_grid=-1):
    if num_grid != -1:
        qs = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: " + qs
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = os.path.join(image_folder, video_id)
    image_grid = load_video(image, num_grid)
    image_size = image_grid[0].size
    image_tensor = process_images(
        image_grid, processor, model_config)[0]

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    return input_ids, image_tensor, image_size


def eval_dataset(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, lora_alpha=args.lora_alpha)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    qa_json = args.Eval_QA_root
    image_folder = args.image_folder

    with open(qa_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    key_list = list(data.keys())
    key_list.sort()
    keys = get_chunk(key_list, args.num_chunks, args.chunk_idx)

    answer_prompt = {
        # "multi-choice": "\nBest Option:",     # The old version
        "multi-choice": "\nPlease directly give the best option:",
        "yes_no": "\nPlease answer yes or no:",
        # "caption_matching": "\nBest Option:",     #The old version
        "caption_matching": "\nPlease directly give the best option:",
        "captioning": ""    # The answer "Generated Caption:" is already contained in the question
    }

    eval_dict = {}
    for v_id in tqdm(keys):
        items = data[v_id]
        for dim in items:
            for item in items[dim]:
                question = item['question'] + answer_prompt[args.task_type]
                # =================================You need to change this code =========================
                # ......
                input_ids, image_tensor, image_size = process_data(
                    v_id+'.mp4', question, model.config, image_folder, tokenizer, image_processor, args.num_grid)
                input_ids = input_ids.unsqueeze(0).to(
                    device='cuda', non_blocking=True)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).to(
                            dtype=torch.float16, device='cuda', non_blocking=True),
                        image_sizes=[image_size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True)

                    outputs = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True)[0].strip()

                # ......
                # =======================================================================================

                if v_id not in eval_dict:
                    eval_dict[v_id] = {}
                if dim not in eval_dict[v_id]:
                    eval_dict[v_id][dim] = []

                pred = {
                    'question': item['question'],
                    'answer': item['answer'],
                    'prediction': outputs
                }
                eval_dict[v_id][dim].append(pred)

    eval_dataset_json = args.chat_conversation_output_folder
    os.makedirs(os.path.dirname(eval_dataset_json), exist_ok=True)
    with open(eval_dataset_json, 'w', encoding='utf-8') as f:
        json.dump(eval_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-grid", type=int, default=16)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--task_type", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--lora-alpha", default=None,
                        type=int, help="lora alpha for scaling weight")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dataset_name", type=str,
                        default=None, help="The type of LLM")
    parser.add_argument("--Eval_QA_root", type=str,
                        default='./', help="folder containing QA JSON files")
    parser.add_argument("--Eval_Video_root", type=str,
                        default='./', help="folder containing video data")
    parser.add_argument("--chat_conversation_output_folder",
                        type=str, default='./Chat_results', help="")
    args = parser.parse_args()

    eval_dataset(args)
