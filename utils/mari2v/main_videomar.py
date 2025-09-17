import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.data import AntDiffData, AntDiffData_clap
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from models.vae import AutoencoderKL
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer
import models.videomar_rope as videomar
from engine_videomar import train_one_epoch, evaluate
import copy
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True



def get_args_parser():
    parser = argparse.ArgumentParser('VideoMAR training with Diffusion Loss', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='videomar', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--mask_ratio_min', type=float, default=0.7, help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1, help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=0)

    # Text Encoder
    parser.add_argument('--text_model_path', default='', help='path to text model')

    # VAE parameters
    parser.add_argument('--img_size_h', default=256, type=int, help='images input size')
    parser.add_argument('--img_size_w', default=256, type=int, help='images input size')
    parser.add_argument('--num_frames', default=33, type=int)
    parser.add_argument('--vae_embed_dim', default=16, type=int, help='vae output embedding dimension')
    parser.add_argument('--vae_spatial_stride', default=16, type=int, help='tokenizer stride')
    parser.add_argument('--vae_tempotal_stride', default=8, type=int, help='tokenizer stride')
    parser.add_argument('--patch_size', default=1, type=int, help='number of tokens to group as a patch.')
    parser.add_argument('--Cosmos_VAE', action='store_true', dest='Cosmos_VAE', help='VAE of Cosmos')
    parser.add_argument('--MAR_VAE', action='store_true', dest='MAR_VAE', help='VAE of MAR')
    parser.add_argument('--vae_path', default='', type=str, help='path to vae')

    # Generation parameters
    parser.add_argument('--file_type', type=str, default="video")
    parser.add_argument('--i2v', action='store_true')
    parser.add_argument('--v2v', action='store_true')
    parser.add_argument('--cond_frame', default=1, type=int, help='number of conditioning frames for video generation')
    parser.add_argument('--num_iter', default=256, type=int, help='number of autoregressive iterations to generate an image')
    parser.add_argument('--cfg', default=3.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=2, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=24, help='generation batch size')

    # Optimizer parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay (default: 0.02)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant', help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=200, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_rate', default=0.999, type=float)

    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=4)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

######################### VAE #########################
    if args.MAR_VAE:
        vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    elif args.Cosmos_VAE:
        # vae = ImageTokenizer(checkpoint_enc=f'{args.vae_path}/encoder.jit', checkpoint_dec=f'{args.vae_path}/decoder.jit').to(local_rank).eval()       # Image tokenizer
        vae = CausalVideoTokenizer(checkpoint_enc=f'{args.vae_path}/encoder.jit', checkpoint_dec=f'{args.vae_path}/decoder.jit').cuda().eval()         # Video tokenizer
    for param in vae.parameters():
        param.requires_grad = False

######################### Text Encoder #########################
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).cuda().eval()
    for param in text_model.parameters():
        param.requires_grad = False
    
######################### VideoMAR And Optimizer #########################
    model = videomar.__dict__[args.model](
        img_size_h=args.img_size_h,
        img_size_w=args.img_size_w,
        num_frames=args.num_frames,
        vae_spatial_stride=args.vae_spatial_stride,
        vae_tempotal_stride=args.vae_tempotal_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
    )

    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    model.to(device)
    model_without_ddp = model
    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # no weight decay on bias, norm layers, and diffloss MLP
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

######################### Load checkpoint for VideoMAR #########################
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del ema_state_dict
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

######################### Judge If Evaluate First #########################
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(text_tokenizer, text_model, model_without_ddp, vae, ema_params, args, batch_size=args.eval_bsz, cfg=args.cfg, use_ema=True)
        return

######################### Load Dataset from OSS #########################
    dataset_train = AntDiffData(oss_config=args.data_path, img_size=[args.img_size_h, args.img_size_w], num_frames=args.num_frames, file_type=args.file_type)
    sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

######################### Start Training #########################
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # -------per epoch-------
        train_one_epoch(
            text_tokenizer, text_model,
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # -------save checkpoint-------
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params)

        # -------online evaluation-------
        # if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
        #     torch.cuda.empty_cache()
        #     evaluate(text_tokenizer, text_model, model_without_ddp, vae, ema_params, args, batch_size=args.eval_bsz, cfg=args.cfg, use_ema=True, epoch=epoch)
        #     torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)