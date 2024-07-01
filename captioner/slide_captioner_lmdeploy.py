import argparse
import json
import os

from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.model.utils import rewrite_ctx
from contextlib import contextmanager
import torch

def get_image_list(video_path):
    frames = sorted(os.listdir(video_path))
    img_path_list = []
    for frame in frames:
        img_path = os.path.join(video_path, frame)
        # NOTE lazy load image
        # img = load_image(img_path)
        img_path_list.append(img_path)
    return img_path_list

def _forward_4khd_7b(self, images):
    """internlm-xcomposer2-4khd-7b vit forward."""
    outputs = [x.convert('RGB') for x in images]
    outputs = [self.HD_transform(x, hd_num=9) for x in outputs]
    outputs = [
        self.model.vis_processor(x).unsqueeze(0).to(dtype=torch.half)
        for x in outputs
    ]
    embeds, split = self.model.vit(outputs, self.model.plora_glb_GN,
                                    self.model.plora_sub_GN)
    embeds = self.model.vision_proj(embeds)
    embeds = torch.split(embeds, split, dim=1)
    embeds = [x.squeeze() for x in embeds]
    return embeds

@contextmanager
def custom_forward():
    origin_func_path = [
        'lmdeploy.vl.model.xcomposer2.Xcomposer2VisionModel._forward_4khd_7b',
    ]
    rewrite_func = [
        _forward_4khd_7b
    ]
    with rewrite_ctx(origin_func_path, rewrite_func):
        yield

class VideoData():
    def __init__(self, video_path):
        self.video_path = video_path
        self.img_path_list = get_image_list(self.video_path)
        self.frame_ptr = 0
        self.caption_list = [""]	    # hack for unified code, remember to remove the first item

    @property
    def is_finished(self):
        return self.frame_ptr == len(self.img_path_list)
    
    def get_prepared_data(self,):
        curr_img = load_image(self.img_path_list[self.frame_ptr])
        if self.frame_ptr == 0:
            query = 'This is the first frame of a video, describe it in detail.'
        else:
            query = "Here are the Video frame {} at {}.00 Second(s) and Video frame {} at {}.00 Second(s) of a video, describe what happend between them. What happend before is: {}".format(
                self.frame_ptr, int(self.frame_ptr * 2), self.frame_ptr + 1, int((self.frame_ptr + 1) * 2), self.caption_list[-1])
        self.frame_ptr += 1
        return (query, curr_img)
    
    def get_finish_data(self):
        prompt = ""
        for frame_idx, caption in enumerate(self.caption_list[1:]):
            prompt += 'Video frame {} at {}.00 Second(s) description: {}\n'.format(
                frame_idx+1, frame_idx*2, caption)
        return dict(
            video_path=self.video_path,
            frame_num=len(self.img_path_list),
            summary_prompt=prompt
        )
    
    def record_caption(self, caption):
        self.caption_list.append(caption)


class VideoPool():
    def __init__(self, pool_size=6, video_path=None):
        self.size = pool_size
        self.video_path = json.load(open(video_path, 'r'))
        self.video_ptr = 0
        self.video_pool = set()
        self._init_pool()

    def _load(self):
        # load ptr video
        video = VideoData(self.video_path[self.video_ptr])
        self.video_ptr += 1
        return video

    def get_batch_data(self):
        batch_data = []
        for video in data_pool.video_pool:
            data = video.get_prepared_data()
            batch_data.append(data)
        return batch_data

    def _init_pool(self):
        while len(self.video_pool) < self.size and \
                self.video_ptr < len(self.video_path):
            print("Load Video")
            self.video_pool.add(self._load())

    def record_caption(self, caption_list):
        # put the model generation back to the video list
        for caption, video in zip(caption_list, self.video_pool):
            video.record_caption(caption)

    def check_finished_video(self,):
        remove_list = []
        finish_list = []
        for video in self.video_pool:
            if video.is_finished:
                final_data = video.get_finish_data()
                finish_list.append(final_data)
                remove_list.append(video)
        for remove_item in remove_list:
            self.video_pool.remove(remove_item)
        self._init_pool()
        return finish_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-name", type=str,
                        default="/mnt/petrelfs/chenlin/MLLM/hw_home/hf_ckpts/ShareGPT4Video/sharegpt4video/sharecaptioner_v1")
    parser.add_argument("--videos-file", type=str, default="describe.json",
                        help="a list, each element is a string for image path")
    parser.add_argument("--save-path", type=str, default="outputs/")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = pipeline(args.model_name, chat_template_config=ChatTemplateConfig(model_name='internlm-xcomposer2-4khd'))
    data_pool = VideoPool(pool_size=args.batch_size, video_path=args.videos_file)
    cnt = 0
    while True:
        batch_data = data_pool.get_batch_data()
        if len(batch_data) == 0:
            break
        with custom_forward():
            responses = model(batch_data)
        responses = [resp.text for resp in responses]
        data_pool.record_caption(responses)
        finish_list = data_pool.check_finished_video()
        cnt += 1
        if len(finish_list) == 0:
            continue
        batch_infer_list = [(data['summary_prompt']) for data in finish_list]
        final_responses = model(batch_infer_list)
        for finish_data, finish_response in zip(finish_list, final_responses):
            finish_data['summary_prompt'] = finish_response.text
            filename = finish_data['video_path'].split('/')[-1]
            with open(os.path.join(args.save_path, filename+'.json'), 'w') as f:
                f.write(json.dumps(finish_data, indent=2, ensure_ascii=False))