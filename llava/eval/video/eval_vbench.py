
import functools
import itertools
import logging
import multiprocessing as mp
import os
import pdb
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import torch
import transformers
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from llava.eval.video.general_utils import (conv_templates, create_frame_grid,
                                            resize_image_grid, video_answer)
from llava.eval.video.vbench_utils import (VBenchDataset, check_ans,
                                           load_results, save_results)
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model_and_dataset(rank, world_size, args):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device_map='cpu')
    logger.info('done loading llava')

    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = VBenchDataset(num_segments=args.num_frames)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, tokenizer, processor, dataset


def infer_mvbench(
    model,
    processor,
    tokenizer,
    data_sample,
    conv_mode,
    pre_query_prompt=None,  # add in the head of question
    post_query_prompt=None,  # add in the end of question
    answer_prompt=None,  # add in the begining of answer
    return_prompt=None,  # add in the begining of return message
    print_res=False,
):
    video_list = data_sample["video_pils"]
    conv = conv_templates[conv_mode].copy()
    conv.user_query(data_sample['question'],
                    pre_query_prompt, post_query_prompt, is_mm=True)
    if answer_prompt is not None:
        conv.assistant_response(answer_prompt)

    llm_message, conv = video_answer(
        conv=conv,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        img_grid=video_list,
        max_new_tokens=32,
        do_sample=False,
        print_res=print_res
    )

    # if answer_prompt is not None:
    #     llm_message = ''.join(llm_message.split(answer_prompt)[1:])
    if return_prompt is not None:
        llm_message = return_prompt + llm_message

    return llm_message


def single_test(model, processor, tokenizer, vid_path,  num_frames=4, conv_mode="plain"):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(video_path, num_segments=8, return_msg=False, num_frames=4):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments)
        img_array = vr.get_batch(frame_indices).asnumpy()
        img_grid = create_frame_grid(img_array, 50)
        img_grid = Image.fromarray(img_grid).convert("RGB")
        img_grid = resize_image_grid(img_grid)
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return img_grid, msg
        else:
            return img_grid
    if num_frames != 0:
        vid, msg = load_video(
            vid_path, num_segments=num_frames, return_msg=True)
    else:
        vid, msg = None, 'num_frames is 0, not inputing image'
    img_grid = vid
    conv = conv_templates[conv_mode].copy()
    conv.user_query("Describe the video in details.", is_mm=True)
    llm_response, conv = video_answer(conv=conv, model=model, processor=processor, tokenizer=tokenizer,
                                      do_sample=False, img_grid=img_grid, max_new_tokens=256, print_res=True)


def run(rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    print_res = True
    conv_mode = args.conv_mode

    pre_query_prompt = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames."
    post_query_prompt = "\nOnly give the best option."

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, tokenizer, processor, dataset = load_model_and_dataset(
        rank, world_size, args)
    logger.info('done model and dataset...')
    logger.info('constructing dataset...')
    logger.info('single test...')

    vid_path = "images/104554.webm"
    if rank == 0:
        single_test(model,
                    processor,
                    tokenizer,
                    vid_path,
                    num_frames=args.num_frames,
                    conv_mode=args.conv_mode)
        logger.info('single test done...')
        tbar = tqdm(total=len(dataset))

    correct = 0
    total = 0
    result_list = []
    acc_dict = {}
    done_count = 0

    for example in dataset:
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0]  # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = infer_mvbench(
            model,
            processor,
            tokenizer,
            example,
            conv_mode=conv_mode,
            pre_query_prompt=pre_query_prompt,
            post_query_prompt=post_query_prompt,
            answer_prompt="Best option:(",
            return_prompt='(',
            print_res=print_res,
        )
        gt = example['answer']
        result_list.append({
            'pred': pred,
            'gt': gt,
            'task_type': task_type,
            'task_split': example['task_split'],
            'video_path': example['video_path'],
            'question': example['question'],

        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        if rank == 0:
            tbar.update(len(result_list) - done_count, )
            tbar.set_description_str(
                f"One Chunk--Task Type: {task_type}, Chunk Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%;"
                f" Chunk Total Acc: {correct / total * 100 :.2f}%"
            )
            done_count = len(result_list)
    return result_list


def main():
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    args = parse_args()
    save_path = args.save_path
    json_data = load_results(save_path)
    if json_data is None:
        if multiprocess:
            logger.info(f'started benchmarking, saving to: {save_path}')
            n_gpus = torch.cuda.device_count()
            # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
            world_size = n_gpus
            with Pool(world_size) as pool:
                func = functools.partial(run, args=args, world_size=world_size)
                result_lists = pool.map(func, range(world_size))

            logger.info('finished running')
            result_list = [res for res in itertools.chain(*result_lists)]
        else:
            result_list = run(0, world_size=1, args=args)  # debug

    else:
        logger.info(f'loaded results from {save_path}')
        result_list = json_data
    save_results(result_list, save_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path",
                        type=str,
                        default='checkpoints/llava-v1.6-7b_vicuna-1.5-7b_clip-large-336_video-sft-mix294k_ft-mlp-llm-lora_lr-mlp-2e-5-llm-2e-4')
    parser.add_argument("--model-base",
                        type=str,
                        default=None)
    parser.add_argument("--save_path",
                        type=str,
                        default='./playground/results/vbench')
    parser.add_argument("--num_frames",
                        type=int,
                        default=16)
    parser.add_argument("--conv-mode",
                        type=str,
                        default='eval_vbench')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
