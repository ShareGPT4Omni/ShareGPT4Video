import os
from torch.nn import functional as F
import cv2
import math
from transformers import CLIPFeatureExtractor,CLIPVisionModel
import numpy as np


model_path = 'openai/clip-vit-large-patch14-336'
feature_extractor = CLIPFeatureExtractor.from_pretrained(model_path)
vision_tower = CLIPVisionModel.from_pretrained(model_path).cuda()
vision_tower.requires_grad_(False)


def get_resized_wh(width, height, max_size):
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
    else:
        new_width = width
        new_height = height
    return new_width, new_height

def check_pure(mtx):
    unique_elements = np.unique(mtx)
    return len(unique_elements) == 1

def extract_second(image_filename):
    return image_filename.split('/')[-1].replace('.png', '').split('_')[-1]

def calculate_clip_feature_sim_2(image_1, image_2):
    input_1 = feature_extractor(images=image_1, return_tensors="pt", padding=True)
    input_2 = feature_extractor(images=image_2, return_tensors="pt", padding=True)
    image_feature_1 = vision_tower(**input_1.to(device=vision_tower.device), output_hidden_states=True).hidden_states[-1][:, 0]
    image_feature_2 = vision_tower(**input_2.to(device=vision_tower.device), output_hidden_states=True).hidden_states[-1][:, 0]
    similarity = F.cosine_similarity(image_feature_1.to(device='cpu'), image_feature_2.to(device='cpu'), dim=1)
    print(f'Sim: {similarity}')
    return similarity

def frame_interval_file(video_path, keyframe_interval, shortest_duration, longest_duration, window_threshold, output_dir):    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = math.ceil(video_fps * keyframe_interval)
    frame_list = []
    cnt_tmp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cnt_tmp / video_fps > longest_duration:
            break
        if cnt_tmp == 0 and check_pure(frame) == True:
            pure_cnt = 1
            while pure_cnt < frame_interval:
                ret, frame = cap.read()
                if check_pure(frame) != True:
                    break
                pure_cnt += 1
        frame_list.append(frame)
        cnt_tmp += 1
    if len(frame_list) > math.ceil(video_fps * shortest_duration):
        start_frame_idx = 0
        selected_frame_list = [0]
        if len(frame_list) > frame_interval:
            for i in range(1, len(frame_list)):
                if i % frame_interval == 0:
                    dynamic_sim = calculate_clip_feature_sim_2(frame_list[start_frame_idx], frame_list[i])
                    if dynamic_sim < window_threshold:
                        selected_frame_list.append(i)
                        start_frame_idx = i
        if len(selected_frame_list) == 1:
            selected_frame_list.append(len(frame_list)-1)
        elif (len(frame_list)-selected_frame_list[-1]) >= frame_interval:
            selected_frame_list.append(len(frame_list)-1)
        for fc in selected_frame_list:
            current_time = fc / video_fps
            time_str = f"{current_time:04.2f}"
            frame_filename = f"frame_{time_str}.png"
            frame_filename = os.path.join(output_dir, frame_filename)
            os.makedirs(output_dir, exist_ok=True)
            new_width, new_height = get_resized_wh(width, height, 1024)
            if new_width == width and new_height == height:
                pass
            else:
                frame_list[fc] = cv2.resize(frame_list[fc], (new_width, new_height), interpolation=cv2.INTER_AREA)
            suc = cv2.imwrite(frame_filename, frame_list[fc])
            if not suc:
                print(f"Failed to save frame {time_str} to {frame_filename}.")
    cap.release()


if __name__=="__main__":
    video_path = 'example.mp4'
    keyframe_interval = 2
    shortest_duration = 10
    longest_duration = 120
    window_threshold = 0.93
    output_dir = 'example/'
    # save keyframes to 'output_dir'
    frame_interval_file(video_path, keyframe_interval, shortest_duration, longest_duration, window_threshold, output_dir)
