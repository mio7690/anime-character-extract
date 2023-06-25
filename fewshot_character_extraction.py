import os
import tqdm
import numpy as np
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess
import gradio as gr
import imageio
from face_detector import has_face
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_video_files(folder_path):
    allowed_patterns = [
        '*.mp4', '*.mkv', '.wav'
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns
        for path in Path(folder_path).glob(pattern)
    ]
    return image_path_list

# 生成 PIL Image 的生成器


def image_generator(video_path, frame_interval):
    # 判断是视频文件还是视频文件夹
    if os.path.isdir(video_path):
        # 获取目录下所有视频文件
        video_files = get_video_files(video_path)
    else:
        video_files = [video_path]
    #  import ipdb;ipdb.set_trace()
    for video_file in video_files:
        reader = imageio.get_reader(video_file, 'ffmpeg')
        frame_num = reader.count_frames()

        print(f"{frame_num} frames in total,will sample one frame every {frame_interval} frames, totally {frame_num//frame_interval} frames")
        for i in range(0, frame_num, frame_interval):
            pil_image = Image.fromarray(reader.get_data(i))
            yield pil_image


def get_similar_images(video_path, text_prompts, image_prompt, ratio=0.5, sim_topk=30, frame_interval=24, output_dir="output_images"):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        "clip_feature_extractor", model_type="ViT-B-16", is_eval=True, device=device)
    # 输出参数配置

    # 检查output目录是否存在,若不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查是否提供了文本或图像提示
    if text_prompts is None and image_prompt is None:
        assert False, "Please provide text prompts or image prompts"

    # 获取文本提示的特征
    if text_prompts is not None:
        text_prompts = model.extract_features({"text_input": text_prompts})

    # 获取图像提示的特征
    if image_prompt is not None:
        image_prompt = Image.fromarray(np.uint8(image_prompt * 255))
        image_prompt = vis_processors["eval"](
            image_prompt).unsqueeze(0).to(device)
        image_prompt = model.extract_features({'image': image_prompt})

    # 如果提供了文本和图像提示，则使用它们的加权平均作为提示特征
    if text_prompts is not None and image_prompt is not None:
        text_and_image_prompt = ratio * \
            text_prompts + (1 - ratio) * image_prompt
    else:
        text_and_image_prompt = text_prompts if text_prompts is not None else image_prompt

    # 生成器，逐帧读取视频并计算相似度,并保存前sim_topk的图片
    image_iter = image_generator(video_path, frame_interval=frame_interval)
    topk_similar_images = []
    topk_sim = []
    for i, pil_image in tqdm.tqdm(enumerate(image_iter)):
        #  if i%20 == 0 :
        #  import ipdb;ipdb.set_trace()
        if not has_face(pil_image):
            print("图片无角色,跳过该帧")
            continue
        image = vis_processors["eval"](pil_image).unsqueeze(0).to(device)
        feature = model.extract_features({'image': image})
        with torch.no_grad():
            sim = torch.cosine_similarity(
                text_and_image_prompt, feature, dim=-1).cpu().detach().float()

        # 若属于前sim_topk相似的图片,则保存至topk_similar_images
        if len(topk_similar_images) < sim_topk:
            topk_similar_images.append(pil_image)
            topk_sim.append(sim)
        else:
            min_sim = min(topk_sim)
            if sim > min_sim:
                min_sim_index = topk_sim.index(min_sim)
                topk_similar_images[min_sim_index] = pil_image
                topk_sim[min_sim_index] = sim

    print(
        f"The max similarity is {float(max(topk_sim))}, min similarity is {float(min(topk_sim))}")
    # 保存topk相似的图片
    similar_images = []
    for i, pil_image in enumerate(topk_similar_images):
        output_path = os.path.join(output_dir, f"similar_frame_{i}.jpg")
        # save pil_image to output_path
        pil_image.save(output_path)
        similar_images.append(output_path)
    return similar_images


def predict(video_file, frame_interval, text_prompts, image_prompt, text_image_ratio, sim_topk, output_dir):
    return get_similar_images(video_path=video_file,
                              text_prompts=text_prompts,
                              image_prompt=image_prompt,
                              ratio=text_image_ratio,
                              sim_topk=sim_topk,
                              output_dir=output_dir,
                              frame_interval=frame_interval)


text_prompts = 'one anime girl with short yellow hair and green eyes'
video_path = r'/mnt/e/wsl/data/Tamako_Love_Story.mkv'
output_path = r'/mnt/e/wsl/data/Tamako_Love_Story_output'

# 定义 Gradio 界面
video_upload = gr.Textbox(
    label="Video File Path or Video Files Folder", value=video_path)
frame_interval = gr.Slider(minimum=1, maximum=100,
                           step=1,  label="Frame Sampling")
text_input = gr.Textbox(label="Text Prompts (optional)", value=text_prompts)
image_upload = gr.Image(label="Image Prompt (optional)")
text_image_ratio = gr.Slider(
    minimum=0, maximum=1, step=0.1, label="Ratio (text:image)")
sim_threshold = gr.Slider(minimum=0, maximum=1,
                          step=0.05, label="Threshold (optional)")
topk = gr.Slider(minimum=0, maximum=300, step=1,
                 label="Top K Similar Images(optional)")
output_image_path = gr.Textbox(
    label="Output Images Directory", value=output_path)

preview = gr.outputs.Image(type="numpy", label="Preview")

gradio_interface = gr.Interface(
    fn=predict,
    inputs=[video_upload, frame_interval, text_input,
            image_upload, text_image_ratio, topk, output_image_path,],
    outputs=None,
    title="Find Similar Frames in a Video",
    description="Input a video file and text/image prompts to retrieve frames with similar features",
)

gradio_interface.launch()
