import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched
# import torch_fidelity
import cv2
import numpy as np
import os
import copy
import random
import time
import json
from tqdm import tqdm
import imageio
from models.tools import encode_prompts
import numpy.typing as npt
from decord import VideoReader, cpu
from torch.nn import functional as F
from torchvision.transforms import Lambda, Compose
from models.tools import ToTensorVideo, CenterCropResizeVideo
import torchvision.transforms as transforms
from PIL import Image



def load_and_transform_image(image_path, height, width):
    # 打开图像
    image = Image.open(image_path).convert('RGB')
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # 调整图像大小为给定尺寸
        transforms.ToTensor(),  # 将图像转换为张量，并且将像素值归一化到[0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
    ])
    # 应用转换
    image_tensor = transform(image)
    # 增加批量维度
    image_tensor = image_tensor.unsqueeze(0)  # 从 [C, H, W] -> [B, C, H, W]，其中 B=1
    return image_tensor


def array_to_video(image_array: npt.NDArray, fps: float = 30.0, output_file: str = 'output_video.mp4') -> None:
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))
    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)
    video_writer.release()

def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(0, 2, 3, 1).float().numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return


def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    s = 0
    e = sample_frames_len
    print(f'sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}', video_path, total_frames)

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


def preprocess(video_data: torch.Tensor, height: int = 128, width: int = 128) -> torch.Tensor:
    transform = Compose(
        [
            ToTensorVideo(),
            CenterCropResizeVideo((height, width)),
            Lambda(lambda x: 2. * x - 1.)
        ]
    )
    video_outputs = transform(video_data)
    video_outputs = torch.unsqueeze(video_outputs, 0)
    return video_outputs


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)



def train_one_epoch(text_tokenizer, text_model, model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        if args.file_type=='image':
            samples = samples.unsqueeze(1)   #  [B, C, H, W] ->  [B, T, C, H, W]
        samples = samples.to(device, non_blocking=True).permute(0,2,1,3,4)       #  [B, C, T, H, W]
        # ------- Cosmos VAE ------- #
        with torch.no_grad():
            (x,) = vae.encode(samples)       # [B, C, T, H, W]
            x = x * 0.5
        x = x.permute(0, 2, 1, 3, 4).to(torch.bfloat16)       # [B, T, C, H, W]

        # ------- Text Encoder ------- #
        with torch.no_grad():
            (context_mask, labels) = encode_prompts(       # [B, 300]   [B, 300, 1536]
                labels,
                text_model,
                text_tokenizer,
                text_tokenizer_max_length=300,
                use_llm_system_prompt=True,
            )
            labels = labels.to(torch.bfloat16)     # ([B, 300, 1536])
        video_len = args.num_frames // args.vae_tempotal_stride + 1
        if args.file_type=='image':
            ini_frame = 0
            context_mask = context_mask.unsqueeze(1).repeat(1, x.shape[-1] * x.shape[-2] * (ini_frame+1), 1)   # [B, 1440, 300]
        else:
            ini_frame = random.randint(1, video_len-1)
        # context_mask = context_mask.unsqueeze(1).repeat(1, x.shape[-1] * x.shape[-2] * (ini_frame+1), 1)   # [B, 1440, 300]

        # ------- Training ------- #
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(x, labels, ini_frame, context_mask)
        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        update_ema(ema_params, model_params, rate=args.ema_rate)
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and int(os.environ.get('RANK'))==0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def evaluate(text_tokenizer, text_model, model_without_ddp, vae, ema_params, args, batch_size=16, cfg=3.0, use_ema=True, epoch=0):
    model_without_ddp.eval()
    num_steps = 1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ------- switch to ema params ------- #
    if use_ema:
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    # ------- load image/video ------- #
    if args.i2v:
        image_path = '/ossfs/workspace/yuhu.yh/code/VideoMAR/data/a book on fire with flames coming out of it.jpg' 
        prompt = ["a book on fire with flames coming out of it"]
        image_path = '/ossfs/workspace/yuhu.yh/code/VideoMAR/data/Small fireworks explode and unfold in the air.jpg' 
        prompt = ["Small fireworks explode and unfold in the air"]
        image_path = '/ossfs/workspace/yuhu.yh/code/VideoMAR/data/a sailboat is drifting on the ocean.jpg' 
        prompt = ["a sailboat is drifting on the ocean"]
        Img_cond = load_and_transform_image(image_path, args.img_size_h, args.img_size_w).cuda().to(torch.bfloat16).unsqueeze(2)    # [B, C, T, H, W]
    elif args.v2v:
        prompt = ["Waves gently wash over a sandy beach, creating intricate patterns in the wet sand. The water is clear and slightly foamy as it moves in and out. The horizon is visible in the distance with a clear blue sky above. The beach is empty, and the scene is calm and serene. The overall style is realistic with high clarity and no noticeable blur."]
        video_path = "/ossfs/workspace/yuhu.yh/code/VideoMAR/data/waves.mp4"
        Img_cond = preprocess(read_video(video_path, 25, 1), args.img_size_h, args.img_size_w).cuda().to(torch.bfloat16)    # [B, C, T, H, W]
    
    # ------- VAE latent ------- #
    print(Img_cond.shape)
    (Img_cond_latents,) = vae.encode(Img_cond)
    Img_cond_latents = Img_cond_latents * 0.5
    Img_cond_latents = Img_cond_latents.permute(0, 2, 1, 3, 4).to(torch.bfloat16)       # [B, T, C, H, W]



    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))
        
        with torch.no_grad():
            (context_mask, labels_gen) = encode_prompts(
                prompt,
                text_model,
                text_tokenizer,
                text_tokenizer_max_length=300,
                use_llm_system_prompt=True,
            )
            labels_gen = labels_gen.to(torch.bfloat16)      # ([B, 300, 1536])
        labels_gen = labels_gen.repeat(batch_size, 1, 1)

        torch.cuda.synchronize()
        device = torch.device("cuda")
        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                sampled_video = model_without_ddp.sample_tokens(args.cond_frame, vae, Img_cond_latents, bsz=batch_size, context_mask=context_mask, num_iter=args.num_iter, cfg=cfg,
                                                                 cfg_schedule=args.cfg_schedule, labels=labels_gen, device=device,
                                                                 temperature=args.temperature, output_dir=args.output_dir)

        # torch.distributed.barrier()
        sampled_video = ((sampled_video[0] + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        imageio.mimsave(os.path.join(args.output_dir, f"epoch{epoch}_{args.cond_frame}.mp4"), sampled_video, fps=12)
        print(f"Saved video to {args.output_dir}")

