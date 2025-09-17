from functools import partial
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from timm.models.vision_transformer import Block
from diffusers.models.attention import BasicTransformerBlock
from stepvideo.modules.blocks import StepVideoTransformerBlock
from models.diffloss import DiffLoss
# from models.diffloss_rf import DiffLoss
import torch.nn.functional as F
import torchvision.transforms as T
import random
import os
import cv2
import numpy.typing as npt
from torchvision.utils import make_grid
from typing import Optional
from PIL import Image


def videomar(**kwargs):
    model = MAR(
        encoder_embed_dim=1536, encoder_depth=36, encoder_num_heads=24,
        decoder_embed_dim=1536, decoder_depth=36, decoder_num_heads=24,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size_h=256, img_size_w=256, num_frames=33, vae_spatial_stride=16, vae_tempotal_stride=8, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=300,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim
        decoder_heads_dim = int(decoder_embed_dim / decoder_num_heads)
        self.decoder_embed_dim =decoder_embed_dim
        self.patch_size = patch_size

        self.seq_h = img_size_h // vae_spatial_stride // patch_size
        self.seq_w = img_size_w // vae_spatial_stride // patch_size
        self.video_len = num_frames // vae_tempotal_stride + 1
        self.spatial_len = self.seq_h * self.seq_w
        self.seq_len = self.seq_h * self.seq_w * self.video_len
        self.token_embed_dim = vae_embed_dim * patch_size**2

        # --------------------------------------------------------------------------
        # Class Embedding
        self.context_embed = nn.Linear(1536, decoder_embed_dim, bias=True)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, 300, decoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, decoder_embed_dim, bias=True)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            StepVideoTransformerBlock(
                dim=decoder_embed_dim,
                attention_head_dim=decoder_heads_dim,
                attention_type="torch",
                spatial_len=self.spatial_len,
            )
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.self_attn_mask = self.create_attention_mask(self.video_len, self.seq_len)
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
        )
        self.diffusion_batch_mul = diffusion_batch_mul


    def initialize_weights(self):
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, t, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, t, c, h_, p, w_, p)
        x = torch.einsum('ntchpwq->nthwcpq', x)
        x = x.reshape(bsz, t * h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, self.video_len, h_, w_, c, p, p)
        x = torch.einsum('nthwcpq->ntchpwq', x)
        x = x.reshape(bsz, self.video_len, c, h_ * p, w_ * p)
        return x  # [n, t, c, h, w]


    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            for i in range(self.video_len):
                # 计算每段的开始和结束索引
                start_idx = i * self.spatial_len
                end_idx = (i + 1) * self.spatial_len
                
                # 提取段并打乱
                segment = order[start_idx:end_idx]
                np.random.shuffle(segment)
                
                # 将打乱的段插回原数组
                order[start_idx:end_idx] = segment

            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def sample_orders_img(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.spatial_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders


    def mask_loss_weigiht(self, x, orders, num_masked_tokens, num_masked_tokens_frame):
        bsz, seq_len, embed_dim = x.shape
        mask_loss = torch.zeros(bsz, seq_len, device=x.device)
        start = seq_len - num_masked_tokens
        selected_indices = orders[:, start:start + num_masked_tokens_frame]
        num_selected = len(selected_indices[0,:])
        linear_values = torch.linspace(1.0, 1.0, num_selected).repeat(bsz, 1).cuda()
        mask_loss[:, selected_indices] = linear_values
        return mask_loss


    def random_masking(self, x, orders, ini_frame):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(self.spatial_len * mask_rate)) + (self.video_len - ini_frame - 1) * self.spatial_len
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, -num_masked_tokens:], src=torch.ones(bsz, seq_len, device=x.device))
        mask_loss = self.mask_loss_weigiht(x, orders, num_masked_tokens, int(np.ceil(self.spatial_len * mask_rate)))
        return mask, mask_loss

    def mask_by_order(self, mask_len, order, bsz, seq_len, ini_frame):
        masking = torch.zeros(bsz, seq_len).cuda()
        masking = torch.scatter(masking, dim=-1, index=order[:, -mask_len.long():], src=torch.ones(bsz, seq_len).cuda())
        return masking

        
    def mask_by_order_img(self, mask_len, order, bsz):
        masking = torch.zeros(bsz, self.spatial_len).cuda()
        masking = torch.scatter(masking, dim=-1, index=order[:, -mask_len.long():], src=torch.ones(bsz, self.spatial_len).cuda())
        return masking

    def create_attention_mask(self, T, seq_len_q):
        tokens_per_frame = int(seq_len_q/T)
        total_tokens = seq_len_q
        attn_mask = torch.full((total_tokens, total_tokens), False, dtype=torch.bool).cuda()
        # Fill in the allowed attention regions
        for i in range(T):
            start = int(i * tokens_per_frame)
            end = start + tokens_per_frame
            attn_mask[start:end, :end] = True
        return attn_mask



    def forward_mae_cross(self, x, mask, text_embedding, context_mask=None, frame_index=None, causal_attn=True):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(1).unsqueeze(-1).cuda().to(x.dtype)
            text_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * text_embedding

        x = x[(1-mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask.shape[0], mask.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x_after_pad

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(
                x,                     # [1, 3072, 1536]
                text_embedding,       # [1, 300, 1536]
                timestep=None,
                self_attn_mask=self.self_attn_mask if causal_attn else None,
                attn_mask=context_mask,
                rope_positions=[self.video_len, self.seq_h, self.seq_w],
                frame_index=frame_index,
            )
        x = self.decoder_norm(x)
        return x



    def forward_loss(self, z, target, mask, mask_loss=None):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        mask_loss = mask_loss.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask, mask_loss=mask_loss)
        return loss



    def forward(self, imgs, prompt, ini_frame, context_mask=None):
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()

        orders = self.sample_orders(bsz=x.size(0))
        mask, mask_loss = self.random_masking(x, orders, ini_frame)

        text_embedding = self.context_embed(prompt)
        z = self.forward_mae_cross(x, mask, text_embedding, context_mask)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask, mask_loss=mask_loss)

        return loss



######---------------------------- KV Cache ----------------------------######
    def sample_tokens(self, vae, Img_cond_latents, bsz, context_mask=None, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False, device=None, output_dir=None):
        ini_frame = random.randint(1, min(self.video_len-1, 7))
        ini_frame = 2

        Img_cond_latents = self.patchify(Img_cond_latents)
        tokens_video = torch.zeros(bsz, self.seq_len, self.token_embed_dim).to(device).to(torch.bfloat16)
        tokens_video[:,:self.spatial_len*ini_frame,:] = Img_cond_latents[:,:self.spatial_len*ini_frame,:]

        text_embedding = self.context_embed(labels)
        text_embedding = torch.cat([text_embedding, self.fake_latent.to(torch.bfloat16).repeat(bsz, 1, 1)], dim=0)

        import time
        start_time_total = time.time()
        ## pre-filling
        [setattr(blk.attn1, "cache_kv", True) for blk in self.decoder_blocks]
        [setattr(blk.attn1, "cache_kv_more", True) for blk in self.decoder_blocks]
        # for blk in self.decoder_blocks:
        #     print(blk.attn1.cache_kv_more)
        for frame_index in range(ini_frame):
            tokens_image = tokens_video[:,frame_index*self.spatial_len:(frame_index+1)*self.spatial_len,:]
            print('000', tokens_image.shape)
            mask = torch.zeros(bsz, self.spatial_len).to(device).to(torch.bfloat16)
            tokens_image = torch.cat([tokens_image, tokens_image], dim=0).to(torch.bfloat16)
            mask = torch.cat([mask, mask], dim=0)
            z = self.forward_mae_cross(tokens_image, mask, text_embedding, context_mask, [False, frame_index], causal_attn=False)

        ## generation
        for frame_index in range(ini_frame, self.video_len):
            # temperature = 0.95 - 0.05 * (frame_index - ini_frame)/(self.video_len - ini_frame)
            [setattr(blk.attn1, "cache_kv_more", False) for blk in self.decoder_blocks]
            tokens_image = torch.zeros(bsz, self.spatial_len, self.token_embed_dim).to(device).to(torch.bfloat16)
            mask = torch.ones(bsz, self.spatial_len).to(device).to(torch.bfloat16)
            orders = self.sample_orders_img(bsz)
            start_time_frame = time.time()
            for step in list(range(num_iter)):
                cur_tokens = tokens_image.clone()
                tokens_image = torch.cat([tokens_image, tokens_image], dim=0).to(torch.bfloat16)
                mask = torch.cat([mask, mask], dim=0)
                
                z = self.forward_mae_cross(tokens_image, mask, text_embedding, context_mask, [False, frame_index], causal_attn=False)

                mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
                mask_len = torch.Tensor([np.floor(self.spatial_len * mask_ratio)]).to(device)
                mask_len = torch.maximum(torch.Tensor([1]).cuda(), torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

                mask_next = self.mask_by_order_img(mask_len[0], orders, bsz)
                if step >= num_iter - 1:
                    mask_to_pred = mask[:bsz].bool()
                else:
                    mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
                mask = mask_next
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

                # sample token latents for this step
                z = z[mask_to_pred.nonzero(as_tuple=True)]

                cfg_iter = 1 + (cfg - 1) * (self.spatial_len - mask_len[0]) / self.spatial_len

                sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter, device)     # torch.Size([512, 16])
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples.  torch.Size([256, 16])
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                
                if frame_index > ini_frame:
                    cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent.to(torch.bfloat16).clip(-2.0, 2.0)
                else:
                    cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent.to(torch.bfloat16)

                tokens_image = cur_tokens.clone()
            tokens_video[:,frame_index*self.spatial_len:(frame_index+1)*self.spatial_len,:] = tokens_image
            end_time_frame = time.time()
            print(f"inference time for frame {frame_index}: ", end_time_frame - start_time_frame)
            if frame_index < self.video_len-1:
                [setattr(blk.attn1, "cache_kv_more", True) for blk in self.decoder_blocks]
                mask = torch.zeros(bsz, self.spatial_len).to(device).to(torch.bfloat16)
                tokens_image = torch.cat([tokens_image, tokens_image], dim=0).to(torch.bfloat16)
                mask = torch.cat([mask, mask], dim=0)
                z = self.forward_mae_cross(tokens_image, mask, text_embedding, context_mask, [False, frame_index], causal_attn=False)

        end_time_total = time.time()
        print("inference time: ", end_time_total - start_time_total)

        [setattr(blk.attn1, "cache_kv", False) for blk in self.decoder_blocks]
        tokens_video = self.unpatchify(tokens_video)
        tokens_video = vae.decode(tokens_video.permute(0,2,1,3,4)/0.5).permute(0, 2, 1, 3, 4)   # [B, T, C, H, W]
        return tokens_video






# class MAR(nn.Module):
#     """ Masked Autoencoder with VisionTransformer backbone
#     """
#     def __init__(self, img_size_h=256, img_size_w=256, num_frames=33, vae_spatial_stride=16, vae_tempotal_stride=8, patch_size=1,
#                  encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
#                  decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
#                  mlp_ratio=4., norm_layer=nn.LayerNorm,
#                  vae_embed_dim=16,
#                  mask_ratio_min=0.7,
#                  label_drop_prob=0.1,
#                  attn_dropout=0.1,
#                  proj_dropout=0.1,
#                  buffer_size=300,
#                  diffloss_d=3,
#                  diffloss_w=1024,
#                  num_sampling_steps='100',
#                  diffusion_batch_mul=4
#                  ):
#         super().__init__()

#         # --------------------------------------------------------------------------
#         # VAE and patchify specifics
#         self.vae_embed_dim = vae_embed_dim
#         decoder_heads_dim = int(decoder_embed_dim / decoder_num_heads)
#         self.decoder_embed_dim =decoder_embed_dim
#         self.patch_size = patch_size

#         self.seq_h = img_size_h // vae_spatial_stride // patch_size
#         self.seq_w = img_size_w // vae_spatial_stride // patch_size
#         self.video_len = num_frames // vae_tempotal_stride + 1
#         self.spatial_len = self.seq_h * self.seq_w
#         self.seq_len = self.seq_h * self.seq_w * self.video_len
#         self.token_embed_dim = vae_embed_dim * patch_size**2

#         # --------------------------------------------------------------------------
#         # Text Embedding
#         self.context_embed = nn.Linear(1536, decoder_embed_dim, bias=True)
#         self.label_drop_prob = label_drop_prob
#         # Fake text embedding for CFG's unconditional generation
#         self.fake_latent = nn.Parameter(torch.zeros(1, 300, decoder_embed_dim))

#         # --------------------------------------------------------------------------
#         # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
#         self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

#         # --------------------------------------------------------------------------
#         # MAR encoder specifics
#         self.z_proj = nn.Linear(self.token_embed_dim, decoder_embed_dim, bias=True)
#         # --------------------------------------------------------------------------
#         # MAR decoder specifics
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

#         self.decoder_blocks = nn.ModuleList([
#             StepVideoTransformerBlock(
#                 dim=decoder_embed_dim,
#                 attention_head_dim=decoder_heads_dim,
#                 attention_type="torch", 
#                 spatial_len=self.spatial_len,
#             )
#             for _ in range(decoder_depth)])

#         self.decoder_norm = norm_layer(decoder_embed_dim)
#         self.initialize_weights()

#         # --------------------------------------------------------------------------
#         # Diffusion Loss
#         self.diffloss = DiffLoss(
#             target_channels=self.token_embed_dim,
#             z_channels=decoder_embed_dim,
#             width=diffloss_w,
#             depth=diffloss_d,
#             num_sampling_steps=num_sampling_steps,
#         )
#         self.diffusion_batch_mul = diffusion_batch_mul


#     def initialize_weights(self):
#         torch.nn.init.normal_(self.fake_latent, std=.02)
#         torch.nn.init.normal_(self.mask_token, std=.02)
#         self.apply(self._init_weights)


#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#             if m.weight is not None:
#                 nn.init.constant_(m.weight, 1.0)

#     def patchify(self, x):
#         bsz, t, c, h, w = x.shape
#         p = self.patch_size
#         h_, w_ = h // p, w // p
#         x = x.reshape(bsz, t, c, h_, p, w_, p)
#         x = torch.einsum('ntchpwq->nthwcpq', x)
#         x = x.reshape(bsz, t * h_ * w_, c * p ** 2)
#         return x  # [n, l, d]

#     def unpatchify(self, x):
#         bsz = x.shape[0]
#         p = self.patch_size
#         c = self.vae_embed_dim
#         h_, w_ = self.seq_h, self.seq_w
#         x = x.reshape(bsz, self.video_len, h_, w_, c, p, p)
#         x = torch.einsum('nthwcpq->ntchpwq', x)
#         x = x.reshape(bsz, self.video_len, c, h_ * p, w_ * p)
#         return x  # [n, t, c, h, w]


#     def sample_orders(self, bsz, video_len):
#         orders = []
#         for _ in range(bsz):
#             order = np.array(list(range(video_len * self.spatial_len)))
#             start_idx = (video_len-1) * self.spatial_len
#             end_idx = video_len * self.spatial_len
#             # 提取段并打乱
#             segment = order[start_idx:end_idx]
#             np.random.shuffle(segment)
#             # 将打乱的段插回原数组
#             order[start_idx:end_idx] = segment
#             orders.append(order)
#         orders = torch.Tensor(np.array(orders)).cuda().long()
#         return orders


#     def sample_orders_img(self, bsz):
#         orders = []
#         for _ in range(bsz):
#             order = np.array(list(range(self.spatial_len)))
#             np.random.shuffle(order)
#             orders.append(order)
#         orders = torch.Tensor(np.array(orders)).cuda().long()
#         return orders


#     def random_masking(self, x, orders):
#         bsz, seq_len, embed_dim = x.shape
#         mask_rate = self.mask_ratio_generator.rvs(1)[0]
#         num_masked_tokens = int(np.ceil(self.spatial_len * mask_rate))
#         mask = torch.zeros(bsz, seq_len, device=x.device)
#         mask = torch.scatter(mask, dim=-1, index=orders[:, -num_masked_tokens:], src=torch.ones(bsz, seq_len, device=x.device))
#         return mask

#     def mask_by_order_img(self, mask_len, order, bsz):
#         masking = torch.zeros(bsz, self.spatial_len).cuda()
#         masking = torch.scatter(masking, dim=-1, index=order[:, -mask_len.long():], src=torch.ones(bsz, self.spatial_len).cuda())
#         return masking

#     def create_attention_mask(self, T):
#         tokens_per_frame = self.spatial_len
#         total_tokens = T * self.spatial_len
#         attn_mask = torch.full((total_tokens, total_tokens), False, dtype=torch.bool).cuda()
#         for i in range(T):
#             start = int(i * tokens_per_frame)
#             end = start + tokens_per_frame
#             attn_mask[start:end, :end] = True
#         return attn_mask


#     def forward_mae_cross(self, x, mask, text_embedding, video_len, context_mask=None, frame_index=None, causal_attn=True):
#         x = self.z_proj(x)
#         bsz, seq_len, embed_dim = x.shape

#         # random drop class embedding during training
#         if self.training:
#             drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
#             drop_latent_mask = drop_latent_mask.unsqueeze(1).unsqueeze(-1).cuda().to(x.dtype)
#             text_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * text_embedding

#         x = x[(1-mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
#         mask_tokens = self.mask_token.repeat(mask.shape[0], mask.shape[1], 1).to(x.dtype)
#         x_after_pad = mask_tokens.clone()
#         x_after_pad[(1 - mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
#         x = x_after_pad

#         # print(f"x.shape: {x.shape}")
#         # print(f"context_mask.shape: {context_mask.shape}")
#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(
#                 x,                     # [1, 3072, 1536]
#                 text_embedding,       # [1, 300, 1536]
#                 timestep=None,
#                 self_attn_mask=self.create_attention_mask(video_len) if causal_attn else None,
#                 attn_mask=context_mask,
#                 rope_positions=[self.video_len, self.seq_h, self.seq_w],
#                 frame_index=frame_index,
#             )
#         x = self.decoder_norm(x)
#         return x



#     def forward_loss(self, z, target, mask):
#         bsz, seq_len, _ = target.shape
#         target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
#         z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
#         mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
#         loss = self.diffloss(z=z, target=target, mask=mask)
#         return loss



#     def forward(self, imgs, prompt, ini_frame, context_mask=None):
#         imgs = imgs[:, :ini_frame+1, :, :, :]
#         x = self.patchify(imgs)   # [bsz, t, c, h, w] -> [bsz, t*h*w, c]
#         gt_latents = x.clone().detach()

#         orders = self.sample_orders(bsz=x.size(0), video_len=ini_frame+1)   # frame-wise random order
#         mask = self.random_masking(x, orders)                               # frame-wise mask

#         text_embedding = self.context_embed(prompt)                                                                             # Text encoder
#         z = self.forward_mae_cross(x, mask, text_embedding, ini_frame+1, context_mask, [True, ini_frame], causal_attn=True)     # VideoMAR modeling
#         loss = self.forward_loss(z=z, target=gt_latents, mask=mask)                                                             # Diffusion loss

#         return loss



#     def sample_tokens(self, ini_frame, vae, Img_cond_latents, bsz, context_mask=None, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False, device=None, output_dir=None):
#         Img_cond_latents = self.patchify(Img_cond_latents)
#         tokens_video = torch.zeros(bsz, self.seq_len, self.token_embed_dim).to(device).to(torch.bfloat16)
#         tokens_video[:,:self.spatial_len*ini_frame,:] = Img_cond_latents[:,:self.spatial_len*ini_frame,:]

#         text_embedding = self.context_embed(labels)
#         text_embedding = torch.cat([text_embedding, self.fake_latent.to(torch.bfloat16).repeat(bsz, 1, 1)], dim=0)

#         import time
#         start_time_total = time.time()
#         ## pre-filling
#         [setattr(blk.attn1, "cache_kv", True) for blk in self.decoder_blocks]
#         [setattr(blk.attn1, "cache_kv_more", True) for blk in self.decoder_blocks]
#         for frame_index in range(ini_frame):
#             tokens_image = tokens_video[:,frame_index*self.spatial_len:(frame_index+1)*self.spatial_len,:]
#             print('000', tokens_image.shape)
#             mask = torch.zeros(bsz, self.spatial_len).to(device).to(torch.bfloat16)
#             tokens_image = torch.cat([tokens_image, tokens_image], dim=0).to(torch.bfloat16)
#             mask = torch.cat([mask, mask], dim=0)
#             z = self.forward_mae_cross(tokens_image, mask, text_embedding, frame_index+1, context_mask, [False, frame_index], causal_attn=False)

#         ## generation
#         for frame_index in range(ini_frame, self.video_len):
#             [setattr(blk.attn1, "cache_kv_more", False) for blk in self.decoder_blocks]
#             tokens_image = torch.zeros(bsz, self.spatial_len, self.token_embed_dim).to(device).to(torch.bfloat16)
#             mask = torch.ones(bsz, self.spatial_len).to(device).to(torch.bfloat16)
#             orders = self.sample_orders_img(bsz)
#             start_time_frame = time.time()
#             for step in list(range(num_iter)):
#                 cur_tokens = tokens_image.clone()
#                 tokens_image = torch.cat([tokens_image, tokens_image], dim=0).to(torch.bfloat16)
#                 mask = torch.cat([mask, mask], dim=0)
                
#                 z = self.forward_mae_cross(tokens_image, mask, text_embedding, frame_index+1, context_mask, [False, frame_index], causal_attn=False)

#                 mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
#                 mask_len = torch.Tensor([np.floor(self.spatial_len * mask_ratio)]).to(device)
#                 mask_len = torch.maximum(torch.Tensor([1]).cuda(), torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

#                 mask_next = self.mask_by_order_img(mask_len[0], orders, bsz)
#                 if step >= num_iter - 1:
#                     mask_to_pred = mask[:bsz].bool()
#                 else:
#                     mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
#                 mask = mask_next
#                 mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

#                 # sample token latents for this step
#                 z = z[mask_to_pred.nonzero(as_tuple=True)]

#                 cfg_iter = 1 + (cfg - 1) * (self.spatial_len - mask_len[0]) / self.spatial_len

#                 sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter, device)     # torch.Size([512, 16])
#                 sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples.  torch.Size([256, 16])
#                 mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                
#                 if frame_index > ini_frame:
#                     cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent.to(torch.bfloat16).clip(-2.0, 2.0)
#                 else:
#                     cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent.to(torch.bfloat16)

#                 tokens_image = cur_tokens.clone()
#             tokens_video[:,frame_index*self.spatial_len:(frame_index+1)*self.spatial_len,:] = tokens_image
#             end_time_frame = time.time()
#             print(f"inference time for frame {frame_index}: ", end_time_frame - start_time_frame)

#             if frame_index < self.video_len-1:
#                 [setattr(blk.attn1, "cache_kv_more", True) for blk in self.decoder_blocks]
#                 mask = torch.zeros(bsz, self.spatial_len).to(device).to(torch.bfloat16)
#                 tokens_image = torch.cat([tokens_image, tokens_image], dim=0).to(torch.bfloat16)
#                 mask = torch.cat([mask, mask], dim=0)
#                 z = self.forward_mae_cross(tokens_image, mask, text_embedding, frame_index+1, context_mask, [False, frame_index], causal_attn=False)

#         end_time_total = time.time()
#         print("inference time: ", end_time_total - start_time_total)

#         [setattr(blk.attn1, "cache_kv", False) for blk in self.decoder_blocks]
#         tokens_video = self.unpatchify(tokens_video)
#         tokens_video = vae.decode(tokens_video.permute(0,2,1,3,4)/0.5).permute(0, 2, 1, 3, 4)   # [B, T, C, H, W]
#         return tokens_video


