import copy
import dataclasses
import itertools
import json
import os
import re
from enum import Enum, auto
from typing import Any, List

import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from PIL import Image
from torch.utils.data import Dataset
from transformers import StoppingCriteria

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token


def load_json(load_dir_path, json_file_name):

    load_path = os.path.join(load_dir_path, json_file_name)
    if not os.path.exists(load_path):
        return None
    with open(load_path, 'r', encoding='utf-8') as f:
        obj_serializable = json.load(f)
    return obj_serializable


def dump_json(obj_serializable, save_dir_path, json_file_name):
    os.makedirs(save_dir_path, exist_ok=True)
    save_path = os.path.join(save_dir_path, json_file_name)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj_serializable, f, indent=4, ensure_ascii=False, )


def create_frame_grid(img_array, interval_width=50):
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


def resize_image_grid(image, max_length=1920):
    width, height = image.size
    if max(width, height) > max_length:
        if width > height:
            scale = max_length / width
        else:
            scale = max_length / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        img_resized = image.resize((new_width, new_height), Image.BILINEAR)
    else:
        img_resized = image
    return img_resized


class EasyDict(dict):
    """
    Get attributes

    >>> d = EasyDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'bar'

    Works recursively

    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1

    Bullet-proof

    >>> EasyDict({})
    {}
    >>> EasyDict(d={})
    {}
    >>> EasyDict(None)
    {}
    >>> d = {'a': 1}
    >>> EasyDict(**d)
    {'a': 1}

    Set attributes

    >>> d = EasyDict()
    >>> d.foo = 3
    >>> d.foo
    3
    >>> d.bar = {'prop': 'value'}
    >>> d.bar.prop
    'value'
    >>> d
    {'foo': 3, 'bar': {'prop': 'value'}}
    >>> d.bar.prop = 'newer'
    >>> d.bar.prop
    'newer'


    Values extraction

    >>> d = EasyDict({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
    >>> isinstance(d.bar, list)
    True
    >>> from operator import attrgetter
    >>> map(attrgetter('x'), d.bar)
    [1, 3]
    >>> map(attrgetter('y'), d.bar)
    [2, 4]
    >>> d = EasyDict()
    >>> d.keys()
    []
    >>> d = EasyDict(foo=3, bar=dict(x=1, y=2))
    >>> d.foo
    3
    >>> d.bar.x
    1

    Still like a dict though

    >>> o = EasyDict({'clean':True})
    >>> o.items()
    [('clean', True)]

    And like a class

    >>> class Flower(EasyDict):
    ...     power = 1
    ...
    >>> f = Flower()
    >>> f.power
    1
    >>> f = Flower({'height': 12})
    >>> f.height
    12
    >>> f['power']
    1
    >>> sorted(f.keys())
    ['height', 'power']

    update and pop items
    >>> d = EasyDict(a=1, b='2')
    >>> e = EasyDict(c=3.0, a=9.0)
    >>> d.update(e)
    >>> d.c
    3.0
    >>> d['c']
    3.0
    >>> d.get('c')
    3.0
    >>> d.update(a=4, b=4)
    >>> d.b
    4
    >>> d.pop('a')
    4
    >>> d.a
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'a'
    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(
                x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, d)


class EvalDataset(Dataset):

    def __init__(self, num_segments, test_ratio=None):
        super().__init__()
        self.num_segments = num_segments
        self.test_ratio = test_ratio
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_clip_gif,
            'frame': self.read_frame,
        }

    def __getitem__(self, index) -> Any:
        raise NotImplementedError('')

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])

        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        return images_group

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
                if len(images_group) == len(frame_indices):
                    break

        # might be some really short videos in the gif datasets
        if len(images_group) < self.num_segments:
            multiplier = int(self.num_segments/len(images_group)) + 1
            images_group = [image for _ in range(
                multiplier) for image in images_group][:self.num_segments]
            assert len(images_group) == self.num_segments

        return images_group

    def read_clip_gif(self, video_path, bound=None, fps=25):
        gif = VideoFileClip(video_path)
        frames = gif.iter_frames()
        max_frame = gif.reader.nframes - 1
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for index, frame in enumerate(frames):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)

        # might be some really short videos in the gif datasets
        if len(images_group) < self.num_segments:
            multiplier = int(self.num_segments/len(images_group)) + 1
            images_group = [image for _ in range(
                multiplier) for image in images_group][:self.num_segments]
            assert len(images_group) == self.num_segments

        return images_group

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(
            bound, fps, max_frame, first_idx=1)  # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(
                video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        return images_group

    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # self.data_list = self.data_list[::200] # debug
        if self.test_ratio is None:
            self.data_list = self.data_list[rank::world_size]
        else:
            np.random.RandomState(42).shuffle(self.data_list)
            if isinstance(self.test_ratio, (float, int)):
                num_samples = int(len(self.data_list) * self.test_ratio)
            else:
                num_samples = int(self.test_ratio)
            self.data_list = self.data_list[rank:num_samples:world_size]


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


class MultiModalConvStyle(Enum):
    """Different separator style."""
    MM_ALONE = 'mm_alone'
    MM_INTERLEAF = 'mm_inferleaf'


@dataclasses.dataclass
class Conversation(EasyDict):
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    sep: List[str]
    mm_token: str

    mm_style: MultiModalConvStyle = MultiModalConvStyle.MM_INTERLEAF
    pre_query_prompt: str = None
    post_query_prompt: str = None
    answer_prompt: str = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.sep, str):
            self.sep = [self.sep for _ in self.roles]

    def get_prompt(self):
        # if only one sep given, then both sep are the sames
        sep = [self.sep for _ in self.roles] if isinstance(
            self.sep, str) else self.sep
        sep = dict(zip(self.roles, sep))
        ret = self.system + sep[self.roles[0]] if self.system != "" else ""
        for i, (role, message) in enumerate(self.messages):
            # if is last msg(the prompt for assistant), if answer prompt exists, no sep added
            if i+1 == len(self.messages):
                if role != self.roles[-1]:  # last role is not the model
                    ret += role + message + sep[role] + self.roles[-1]
                else:
                    ret += role + message
            else:
                ret += role + message + sep[role]
        return ret

    def user_query(self, query=None, pre_query_prompt=None, post_query_prompt=None, is_mm=False, num_mm_token=1):
        if post_query_prompt is not None:
            query = f"{query} {post_query_prompt}"

        if pre_query_prompt is not None:
            query = f"{pre_query_prompt} {query}"
        role = self.roles[0]
        # TODO: remove the num_mm_token and hack the self.mm_token outside
        if is_mm:
            mm_str = num_mm_token*self.mm_token[:-1] + self.mm_token[-1]
            if self.mm_style == MultiModalConvStyle.MM_ALONE:
                self._append_message(role, mm_str)
            elif self.mm_style == MultiModalConvStyle.MM_INTERLEAF:
                if self.mm_token not in query:
                    query = f'{mm_str} {query}'
        self._append_message(role, query)

    def assistant_response(self, response, pre_query_prompt=None, post_query_prompt=None):
        if post_query_prompt is not None:
            response = f"{response} {post_query_prompt}"

        if pre_query_prompt is not None:
            response = f"{post_query_prompt} {response}"

        role = self.roles[1]
        self._append_message(role, response)

    def _append_message(self, role, message):
        message = '' if message is None else message
        self.messages.append([role, message])

    def copy(self):
        return copy.deepcopy(self)


def video_answer(conv: Conversation, model, processor, tokenizer, img_grid, do_sample=True, max_new_tokens=200, num_beams=1, top_p=0.9,
                 temperature=1.0, print_res=False, **kwargs):
    prompt = conv.get_prompt()
    if not isinstance(img_grid, (list, tuple)):
        img_grid = [img_grid]
    image_size = img_grid[0].size
    image_tensor = process_images(img_grid, processor, model.config)[0]
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(
        device=model.device, non_blocking=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(
                dtype=torch.float16, device=model.device, non_blocking=True),
            image_sizes=[image_size],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            use_cache=True,
            **kwargs)
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
    if print_res:  # debug usage
        print('### PROMPTING LM WITH: ', prompt)
        print('### LM OUTPUT TEXT:  ', outputs)

    conv.messages[-1][1] = outputs
    return outputs, conv


conv_plain_v1 = Conversation(
    system="",
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token='<image>'
)

SYSTEM_MVBENCH = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
conv_eval_mvbench = Conversation(
    system=SYSTEM_MVBENCH,
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ", "</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
)
conv_eval_mvbench_llama3 = Conversation(
    system=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_MVBENCH}""",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n",
           "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    messages=[],
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
)

SYSTEM_VBENCH = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
conv_eval_vbench = Conversation(
    system=SYSTEM_VBENCH,
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    sep=[" ", "</s>"],
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
)
conv_eval_vbench_llama3 = Conversation(
    system=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_VBENCH}""",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n",
           "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    messages=[],
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
    mm_token='<image>\n',
    mm_style=MultiModalConvStyle.MM_INTERLEAF,
)


conv_templates = {
    "plain": conv_plain_v1,
    "eval_mvbench": conv_eval_mvbench,
    "eval_mvbench_llama3": conv_eval_mvbench_llama3,
    "eval_cvrrbench": conv_eval_cvrrbench,
    "eval_vbench": conv_eval_vbench,
    "eval_vbench_llama3": conv_eval_vbench_llama3
}
