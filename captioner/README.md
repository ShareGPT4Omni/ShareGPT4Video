## Batch Inference with ShareGPT4Video

We support two types of sampling strategy for video inference: (1) fixed sampling interval, Slide Caption (2) fixed sampling frames, and Fast Caption. For fixed sampling interval, we iteratively infer the frame based on the previous generated caption, and finally summarize the results. This strategy provides detailed captions, but it runs a little bit slow. For fixed sampling frames, we directly infer the video by concatenating 16 frames into one image and inferring it at one time. It runs much faster but the quality may be worse than the former one.

### Usage

1. We use lmdeploy to speed up the inference.
```
pip install lmdeploy
```
2. List your video path at `videos_to_describe.json`.
3. run the inference code.
```bash
# fast caption (fixed sampling frames)
python fast_captioner_lmdeploy.py --batch-size 2
# slide caption (fixed sampling interval)
python slide_captioner_lmdeploy.py --batch-size 2
```
