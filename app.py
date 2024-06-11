import os
import shutil
import tempfile

import gradio as gr
import torch

from llava.conversation import Conversation, conv_templates
from llava.serve.gradio_utils import (Chat, block_css, learn_more_markdown,
                                      title_markdown)


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(
        tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename


def generate(video, textbox_in, first_run, state, state_):
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    video = video if video else "none"

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()

    first_run = False if len(state.messages) > 0 else True

    text_en_out, state_ = handler.generate(
        video, textbox_in, first_run=first_run, state=state_)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    textbox_out = text_en_out

    if flag:
        state.append_message(state.roles[0], textbox_in)
    state.append_message(state.roles[1], textbox_out)
    torch.cuda.empty_cache()
    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), gr.update(value=video if os.path.exists(video) else None, interactive=True))


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True),
            True, state, state_, state.to_gradio_chatbot())


conv_mode = "llava_llama_3"
model_path = 'Lin-Chen/sharegpt4video-8b'
device = 'cuda'
load_8bit = False
load_4bit = False
dtype = torch.float16
handler = Chat(model_path, conv_mode=conv_mode,
               load_8bit=load_8bit, load_4bit=load_8bit, device=device)

textbox = gr.Textbox(
    show_label=False, placeholder="Enter text and press ENTER", container=False
)
with gr.Blocks(title='ShareGPT4Video-8B🚀', theme=gr.themes.Default(), css=block_css) as demo:
    gr.Markdown(title_markdown)
    state = gr.State()
    state_ = gr.State()
    first_run = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            video = gr.Video(label="Input Video")

            cur_dir = os.path.dirname(os.path.abspath(__file__))

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="ShareGPT4Video-8B",
                                 bubble_full_width=True)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=True
                    )
            with gr.Row(elem_id="buttons") as button_row:
                regenerate_btn = gr.Button(
                    value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(
                    value="🗑️  Clear history", interactive=True)

    with gr.Row():
        gr.Examples(
            examples=[
                [
                    f"{cur_dir}/examples/sample_demo_1.mp4",
                    "Why is this video funny?",
                ],
                [
                    f"{cur_dir}/examples/C_1_0.mp4",
                    "Write a poem for this video.",
                ],
                [
                    f"{cur_dir}/examples/yoga.mp4",
                    "What is happening in this video?",
                ]
            ],
            inputs=[video, textbox],
        )
    gr.Markdown(learn_more_markdown)

    submit_btn.click(generate, [video, textbox, first_run, state, state_],
                     [state, state_, chatbot, first_run, textbox, video])
    clear_btn.click(clear_history, [state, state_],
                    [video, textbox, first_run, state, state_, chatbot])

demo.launch()
