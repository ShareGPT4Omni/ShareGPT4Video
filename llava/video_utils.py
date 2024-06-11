import argparse
import io
import json
import os
import random
import tempfile
from multiprocessing import Manager, Pool, cpu_count

import cv2
import imageio
import numpy as np
from decord import VideoReader
from PIL import Image


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(
            start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(
                    range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        # gap between frames, this is also the clip length each frame represents
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def get_index(num_frames, bound, fps, max_frame, first_idx=0):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_frames
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_frames)
    ])
    return frame_indices


def read_frames_gif(
    video_path, num_frames, sample='rand', fix_start=None,
    max_num_frames=-1, client=None, clip=None,
):
    if video_path.startswith('s3') or video_path.startswith('p2'):
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        max_num_frames=max_num_frames
    )
    frames = []
    reference_size = None
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            if reference_size is None:
                reference_size = (frame.shape[1], frame.shape[0])
            frame = cv2.resize(frame, reference_size,
                               interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
    frames = np.stack(frames, axis=0)  # .float() / 255

    return frames


def read_frames_decord(
    video_path, num_frames, sample='rand', fix_start=None,
    max_num_frames=-1, client=None, clip=None
):
    if video_path.startswith('s3') or video_path.startswith('p2') or video_path.startswith('p_hdd') or video_path.startswith('cluster1'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if clip:
        vlen = int(duration * fps)
        frame_indices = get_index(num_frames, clip, fps, vlen)
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps, max_num_frames=max_num_frames
        )
    # if clip:
    #     frame_indices = [f + start_index for f in frame_indices]

    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
    return frames


def read_diff_frames_decord(
    video_path, clip, client=None
):
    if video_path.startswith('s3') or video_path.startswith('p2') or video_path.startswith('p_hdd') or video_path.startswith('cluster1') or video_path.startswith('s_hdd'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()

    start_idx = round(clip[0]*fps)
    end_idx = min(round(clip[1]*fps), vlen)
    frame_indices = [start_idx, end_idx]

    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
    return frames
