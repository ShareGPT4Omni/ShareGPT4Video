import argparse
import json
import os

from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.model.utils import rewrite_ctx
from contextlib import contextmanager
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def _forward_4khd_7b(self, images):
    """internlm-xcomposer2-4khd-7b vit forward."""
    outputs = [x.convert('RGB') for x in images]
    outputs = [self.HD_transform(x, hd_num=16) for x in outputs]
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

class ImageGridDataset(Dataset):
    def __init__(self, video_list, img_grid_w, img_grid_h, img_h, img_w):
        self.video_dir_list = json.load(open(video_list))
        self.img_grid_w = img_grid_w
        self.img_grid_h = img_grid_h
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.video_dir_list)

    def __getitem__(self, idx):
        video_img_list = self.video_dir_list[idx]
        try:
            img_path_list = os.listdir(video_img_list)
            img_path_list.sort()
            images = [Image.open(os.path.join(video_img_list, img_path)).convert('RGB') for img_path in img_path_list]
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

        img_width, img_height = images[0].size
        grid_image = Image.new('RGB', (self.img_grid_w * img_width, self.img_grid_h * img_height))
        
        for index, image in enumerate(images):
            x = (index % self.img_grid_w) * img_width
            y = (index // self.img_grid_w) * img_height
            grid_image.paste(image, (x, y))
        
        grid_image = grid_image.resize((self.img_w, self.img_h), Image.ANTIALIAS)
        prompt = "Here are a few key frames of a video, discribe this video in detail."
        
        return (prompt, grid_image), video_img_list

def custom_collate_fn(batch):
    input_list = []
    filename_list = []
    for item in batch:
        if item:
            input_list.append(item[0])
            filename_list.append(item[1])
    return input_list, filename_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--model-name", type=str,
                        default="Lin-Chen/ShareCaptioner-Video",)
    parser.add_argument("--img-path", type=str, default="describe.json",
            help="a list, each element is a string for video folder path."
            "make sure that each image path contains exactly 30 images per video")
    parser.add_argument("--save-path", type=str, default="outputs/")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = pipeline(args.model_name, chat_template_config=ChatTemplateConfig(model_name='internlm-xcomposer2-4khd'))
    img_grid_w, img_grid_h = 5, 6  # Grid dimensions, defaultly 30 images per video
    img_h, img_w = 600, 800  # Desired height and width

    # Create the dataset
    dataset = ImageGridDataset(args.img_path, img_grid_w, img_grid_h, img_h, img_w)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

    # Using the DataLoader in a training loop or inference
    for batch_data, filenames in dataloader:
        # grid_image is now a batch of images ready for inference
        with custom_forward():
            final_responses = model(batch_data)
        for filename, response in zip(filenames, final_responses):
            finish_data = {
                "filename": filename,
                "response": response.text
            }
            filename = filename.split('/')[-1]
            with open(os.path.join(args.save_path, filename+'.json'), 'w') as f:
                f.write(json.dumps(finish_data, indent=2, ensure_ascii=False))
