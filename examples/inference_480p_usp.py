import torch
from diffsynth import ModelManager, save_video, VideoData, WanAnimateX_simple_VideoPipeline
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image
import os
import pickle
from PIL import Image
import numpy as np
import random
import pickle
import torch
from io import BytesIO
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import  torch.nn  as nn
import cv2
import sys  
import torch.distributed as dist
from tqdm import tqdm
sys.path.append("../")  

# define hight and width
height = 832
width = 480
seed = 0
max_frames = 81
use_teacache = False

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument('--cfg', type=float, default=5, help='Path to config file')
    parser.add_argument('--model_path', type=str, default="checkpoints/animate-x++_simple.ckpt", help='Path to model file')
    parser.add_argument('--save_dir', type=str, default="./outputs", help='Directory to save results')
    args = parser.parse_args()
    return args



# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 examples/inference_480p_usp.py 
# or
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 examples/inference_480p_usp.py 

def test(args):


    test_list_path= [
    [2, "data/images/4.jpeg", "data/saved_pose/balei", "a person is dancing."], 
    [2, "data/images/7.jpeg", "/data/saved_pose/normal_video", "a person is dancing，背景中的火焰正熊熊燃烧，明亮的火光在空气中跳跃舞动。火焰的形态灵动多变，橙红色的火舌时而高高跃起，时而收拢低伏，仿佛有生命一般。明暗交错的火光映照出温暖的色彩，燃烧时迸发出微小的火星和缕缕轻烟。火焰轮廓分明细腻，边缘泛着金黄和蓝紫色，整体看上去既有力量，又富有动态美感。栩栩如生的火焰让整个背景充满了热烈、鲜活的氛围，带来温暖且富有活力的视觉感受。"], 
    [2, "data/images/8.jpeg", "data/saved_pose/case-36-1", "a person is dancing."],
    [1, "data/images/28.jpeg", "data/saved_pose/dance_3", "a person is dancing."],
    [2, "data/images/30.jpeg", "data/saved_pose/dance_1", "a person is dancing."],
    [2, "data/images/35.jpeg", "data/saved_pose/normal_video", "a person is dancing，背景中的光圈在空间中闪闪发光，边缘流转着柔和的光晕。每个光圈都仿佛拥有生命一般，缓缓旋转、变幻大小，光芒时隐时现，像波纹般轻盈地在画面里滑动。光圈散发出五彩斑斓的色彩，不断交叠、穿梭，营造出梦幻而富有动感的视觉效果。它们的运动轨迹让整个背景充满了流动的节奏感与未来的科技氛围，仿佛置身于一个光影交错、能量环绕的奇幻空间之中。"],
    [2, "data/images/40.png", "data/saved_pose/case-55", "a person is dancing."],
    [2, "data/images/41.png", "data/saved_pose/case-28", "a person is dancing."], 
    ]

    misc_size = [height,width]

    # Download models
    # snapshot_download("Wan-AI/Wan2.1-I2V-14B-720P", local_dir="./Wan2.1-I2V-14B-720P")

    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        ["./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32, # Image Encoder is loaded with float32
    )
    model_manager.load_models(
        [
            [
                
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",

            ],
            "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
            "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )


    model_manager.load_lora_v2(args.model_path, lora_alpha=1.0)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    from xfuser.core.distributed import (initialize_model_parallel,
                                        init_distributed_environment)
    init_distributed_environment(
        rank=dist.get_rank(), world_size=dist.get_world_size())

    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=1,
        ulysses_degree=dist.get_world_size(),
    )
    torch.cuda.set_device(dist.get_rank())

    pipe = WanAnimateX_simple_VideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=f"cuda:{dist.get_rank()}", 
                                            use_usp=True if dist.get_world_size() > 1 else False)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
    # pipe.enable_vram_management(num_persistent_param_in_dit=0) 

    def resize(image):
        
        image = torchvision.transforms.functional.resize(
            image,
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return torch.from_numpy(np.array(image))



    for path_dir_per in tqdm(test_list_path):
        sample_fps = path_dir_per[0] # frame interval for sampling
        pose_file_path = path_dir_per[2]
        ref_image_path = path_dir_per[1]
        caption = path_dir_per[3]

        dwpose_all = {}
        frames_all = {}
        
        save_path = args.save_dir+"/{}_{}_{}.mp4".format(ref_image_path.split('/')[-1], pose_file_path.split('/')[-1], caption[:30])
        if os.path.exists(save_path):
            continue

        for ii_index in sorted(os.listdir(pose_file_path)):
            if ii_index != "pose.jpeg":
                dwpose_all[ii_index] = Image.open(os.path.join(pose_file_path, ii_index))
                frames_all[ii_index] = Image.fromarray(cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB))
        
        try:
            frames_pose_ref = Image.open(os.path.join(pose_file_path, "pose.jpeg"))
        except:
            frames_pose_ref = Image.open(os.path.join(pose_file_path, sorted(os.listdir(pose_file_path))[0]))

        stride = sample_fps
        _total_frame_num = len(frames_all)
        cover_frame_num = (stride * max_frames)
        
        if _total_frame_num < cover_frame_num + 1:
            start_frame = 0
            end_frame = _total_frame_num-1
            stride = max((_total_frame_num//max_frames),1)
            end_frame = min(stride*max_frames, _total_frame_num)
        else:
            start_frame = 0 # random.randint(0, _total_frame_num-cover_frame_num-1)
            end_frame = start_frame + cover_frame_num
        frame_list = []
        dwpose_list = []

        random_ref_frame = frames_all[list(frames_all.keys())[0]]
        if random_ref_frame.mode != 'RGB':
            random_ref_frame = random_ref_frame.convert('RGB')
        random_ref_dwpose = frames_pose_ref
        if random_ref_dwpose.mode != 'RGB':
            random_ref_dwpose = random_ref_dwpose.convert('RGB')
        
        # sample pose sequence
        for i_index in range(start_frame, end_frame, stride):
            if i_index < len(frames_all):  # Check index within bounds
                i_key = list(frames_all.keys())[i_index]
                i_frame = frames_all[i_key]
                if i_frame.mode != 'RGB':
                    i_frame = i_frame.convert('RGB')
                
                i_dwpose = dwpose_all[i_key]
                if i_dwpose.mode != 'RGB':
                    i_dwpose = i_dwpose.convert('RGB')
                frame_list.append(i_frame)
                dwpose_list.append(i_dwpose)
        
        if (end_frame-start_frame) < max_frames:
            for _ in range(max_frames-(end_frame-start_frame)):
                i_key = list(frames_all.keys())[end_frame-1]
                
                i_frame = frames_all[i_key]
                if i_frame.mode != 'RGB':
                    i_frame = i_frame.convert('RGB')
                i_dwpose = dwpose_all[i_key]
                
        
                frame_list.append(i_frame)
                dwpose_list.append(i_dwpose)

        have_frames = len(frame_list)>0
        middle_indix = 0

        if have_frames:

            l_hight = random_ref_frame.size[1]
            l_width = random_ref_frame.size[0]
            
            ref_frame = random_ref_frame 
            
            random_ref_frame_tmp = torchvision.transforms.functional.resize(
                random_ref_frame,
                (height, width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            random_ref_dwpose_tmp = resize(random_ref_dwpose) 
            
            dwpose_data_tmp = torch.stack([resize(ss).permute(2,0,1) for ss in dwpose_list], dim=0)

        
        dwpose_data = torch.zeros(max_frames, 3, misc_size[0], misc_size[1])

        if have_frames:
            
            dwpose_data[:len(frame_list), ...] = dwpose_data_tmp
            
        
        dwpose_data = dwpose_data.permute(1,0,2,3)

        def image_compose_width(imag, imag_1):
            # read the size of image1
            rom_image = imag
            width, height = imag.size
            # read the size of image2
            rom_image_1 = imag_1
            
            width1 = rom_image_1.size[0]
            # create a new image
            to_image = Image.new('RGB', (width+width1, height))
            # paste old images
            to_image.paste(rom_image, (0, 0))
            to_image.paste(rom_image_1, (width, 0))
            return to_image

        # caption = "a person is dancing"
        video_out_condition = []
        for ii in range(dwpose_data_tmp.shape[0]):
            ss = dwpose_list[ii]
            video_out_condition.append(image_compose_width(random_ref_frame_tmp, torchvision.transforms.functional.resize(
                ss,
                (height, width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )))

        # Image-to-video
        video = pipe(
            prompt=caption,
            negative_prompt="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=ref_frame,
            num_inference_steps=50,
            cfg_scale=args.cfg,
            seed=seed, tiled=True,
            dwpose_data=dwpose_data,
            random_ref_dwpose=random_ref_dwpose_tmp,
            height=height,
            width=width,
            tea_cache_l1_thresh=0.3 if use_teacache else None,
            tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,

        )

        video_out = []
        for ii in range(len(video)):
            ss = video[ii]
            video_out.append(image_compose_width(video_out_condition[ii], ss))
        if dist.get_rank() == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_video(video_out, args.save_dir+"/{}_{}_{}.mp4".format(ref_image_path.split('/')[-1], pose_file_path.split('/')[-1], caption[:30]), fps=10, quality=5)


if __name__ == "__main__":
    args = get_args()
    print("cfg:", args.cfg)
    print("model_path:", args.model_path)
    print("save_dir:", args.save_dir)
    test(args)
