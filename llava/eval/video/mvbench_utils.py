import copy
import dataclasses
import json
import os
from enum import Enum, auto
from typing import Any, List

import numpy as np
import torch
from PIL import Image
from transformers import StoppingCriteria

from llava.constants import IMAGE_TOKEN_INDEX
from llava.eval.video.general_utils import EvalDataset, EasyDict, load_json, dump_json
from llava.mm_utils import process_images, tokenizer_image_token
from llava.eval.video.general_utils import create_frame_grid, resize_image_grid


def load_results(save_path):
    all_results = load_json(save_path, 'all_results.json')
    if all_results is not None:
        result_list = all_results['result_list']
    else:
        result_list = None
    # json_data = load_json(save_path, 'all_results.json')['result_list']
    return result_list


def save_results(result_list, save_path):

    final_res, acc_dict = {}, {}
    correct, total = 0, 0
    for res in result_list:
        task_type = res['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0]  # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = res['pred']
        gt = res['gt']
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]
    final_res['Avg'] = correct / total * 100

    all_results = {
        "acc_dict": acc_dict,
        "result_list": result_list
    }
    dump_json(all_results, save_path, 'all_results.json')
    dump_json(final_res, save_path, 'upload_leaderboard.json')


def check_ans(pred, gt):
    flag = False

    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]

    if not any([c in pred_option for c in 'abcdefgABCDEFG']):
        print(f"model doesn't follow instructions: {pred}")
    elif pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag


class MVBenchDataset(EvalDataset):
    data_list_info = {
        # "task_type (sub task name)": ("json file name", "image/video prefix", "data_type", "bound")
        # has start & end
        "Action Sequence": ("action_sequence.json", "playground/data/mvbench/star/Charades_v1_480/", "video", True),
        # has start & end
        "Action Prediction": ("action_prediction.json", "playground/data/mvbench/star/Charades_v1_480/", "video", True),
        "Action Antonym": ("action_antonym.json", "playground/data/mvbench/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", "playground/data/mvbench/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", "playground/data/mvbench/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", "playground/data/mvbench/clevrer/video_validation/", "video", False),
        # has start & end
        "Object Interaction": ("object_interaction.json", "playground/data/mvbench/star/Charades_v1_480/", "video", True),
        "Object Shuffle": ("object_shuffle.json", "playground/data/mvbench/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", "playground/data/mvbench/clevrer/video_validation/", "video", False),
        # has start & end
        "Action Localization": ("action_localization.json", "playground/data/mvbench/sta/sta_video/", "video", True),
        "Scene Transition": ("scene_transition.json", "playground/data/mvbench/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", "playground/data/mvbench/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", "playground/data/mvbench/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", "playground/data/mvbench/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", "playground/data/mvbench/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", "playground/data/mvbench/nturgbd/", "video", False),
        "Character Order": ("character_order.json", "playground/data/mvbench/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", "playground/data/mvbench/vlnqa/", "video", False),
        # has start & end, read frame
        "Episodic Reasoning": ("episodic_reasoning.json", "playground/data/mvbench/tvqa/frames_fps3_hq/", "frame", True),
        "Counterfactual Inference": ("counterfactual_inference.json", "playground/data/mvbench/clevrer/video_validation/", "video", False),
    }
    data_dir = "playground/data/mvbench/json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_list_info = self.data_list_info
        data_dir = self.data_dir

        self.data_list = []
        for k, v in data_list_info.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        # self.data_list = self.data_list[:100] # for debug
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

    def __getitem__(self, idx):
        question, answer = self.qa_template(self.data_list[idx]['data'])
        task_type = self.data_list[idx]['task_type']
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(
            self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])

        try:  # might be problem with decord
            images_group = decord_method(video_path, bound)
            img_group = np.stack(np.array([np.asarray(image)
                                 for image in images_group]), axis=0)
            img_grid = create_frame_grid(img_group)
            img_grid = [resize_image_grid(
                Image.fromarray(img_grid).convert("RGB"))]
        except Exception as e:
            print(f'Error! {e}')
            print(f'error decoding {video_path}')
            task_type = 'error_reading_video'
            img_grid = None

        return {
            'video_path': video_path,
            'video_pils': img_grid,
            'question': question,
            'answer': answer,
            'task_type': task_type,
        }

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer


# conversation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True
            )
            flag = True
            for output in outputs:
                for keyword in self.keywords:
                    if keyword not in output:
                        flag = False
                        return False
            return flag
