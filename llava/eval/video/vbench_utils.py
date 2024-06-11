import json
import os

import numpy as np
from PIL import Image

from llava.eval.video.general_utils import (EvalDataset, create_frame_grid,
                                            dump_json, load_json,
                                            resize_image_grid)


def load_results(save_path):
    all_results = load_json(save_path, 'all_results.json')
    if all_results is not None:
        result_list = all_results['result_list']
    else:
        result_list = None
    return result_list


def save_results(result_list, save_path):
    final_res, acc_dict = {}, {}
    correct, total = 0, 0
    for res in result_list:
        task_split = res['task_split']
        if task_split not in acc_dict:
            acc_dict[task_split] = [0, 0]  # correct, total
        acc_dict[task_split][1] += 1
        total += 1
        pred = res['pred']
        gt = res['gt']
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_split][0] += 1
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
    gt_option = gt_list[0]

    if not any([c in pred_option for c in 'abcdefghABCDEFGH']):
        print(f"model doesn't follow instructions: {pred}")
    elif pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag


class VBenchDataset(EvalDataset):
    data_list_info = {
        # "task_type (sub task name)": ("json file name", "image/video prefix", "data_type", "bound")
        "ActivityNet": ("ActivityNet_QA_new.json",),
        "Driving-decision-making": ("Driving-decision-making_QA_new.json",),
        "Driving-exam": ("Driving-exam_QA_new.json",),
        "MOT": ("MOT_QA_new.json",),
        "MSRVTT": ("MSRVTT_QA_new.json",),
        "MSVD": ("MSVD_QA_new.json",),
        "MV": ("MV_QA_new.json",),
        "NBA": ("NBA_QA_new.json",),
        "SQA3D": ("SQA3D_QA_new.json",),
        "TGIF": ("TGIF_QA_new.json",),
        "TVQA": ("TVQA_QA_new.json",),
        "Ucfcrime": ("Ucfcrime_QA_new.json",),
        "Youcook2": ("Youcook2_QA_new.json",)
    }
    data_dir = "playground/data/vbench/Eval_QA"
    video_dir = "playground/data/vbench"

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
                    'data': data
                })
        # self.data_list = self.data_list[:100] # for debug
        self.decord_method = self.read_video

    def __getitem__(self, idx):
        question, answer = self.qa_template(self.data_list[idx]['data'])
        task_type = self.data_list[idx]['task_type']
        task_split = self.data_list[idx]['data']['task_split']
        video_path = os.path.join(
            self.video_dir, self.data_list[idx]['data']['video_path'])

        try:  # might be problem with decord
            images_group = self.decord_method(video_path)
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
            'task_split': task_split,
        }

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += f"Options:\n {data['options']}"
        answer = data['answer']
        question = question.rstrip()
        return question, answer
