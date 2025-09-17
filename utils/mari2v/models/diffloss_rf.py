import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math


# class RectifiedFlowLoss(nn.Module):
#     """Rectified Flow Loss - 基于确定性ODE路径的生成模型"""
#     def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
#         super(RectifiedFlowLoss, self).__init__()
#         self.in_channels = target_channels


class DiffLoss(nn.Module):
    """Rectified Flow Loss - 基于确定性ODE路径的生成模型"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels

        # 转换num_sampling_steps从字符串为整数(与原DiffLoss保持兼容)
        self.num_sampling_steps = int(num_sampling_steps) if isinstance(num_sampling_steps, str) else num_sampling_steps

        # 保持原有网络结构不变，只更改输出通道数(不需要预测方差)
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # 只需要预测速度场
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

    def forward(self, target, z, mask=None, mask_loss=None):
        """使用与原DiffLoss相同的参数接口计算Rectified Flow损失"""
        batch_size = target.shape[0]
        
        # 随机采样时间点 t ∈ [0, 1]
        t = torch.rand(batch_size, device=target.device)

        # # 生成标准正态分布，然后映射到 [0, 1]
        sigma = 0.5
        beta = 0
        # t = torch.sigmoid(torch.randn(batch_size, device=target.device)*sigma+beta)   

        # 在t处插值得到x_t = (1-t)*x_0 + t*x_1，其中x_1是噪声，x_0是真实数据
        noise = torch.randn_like(target)
        t_view = t.view(-1, *([1] * (target.dim() - 1)))  # 适应任何维度的目标
        x_t = (1 - t_view) * target + t_view * noise

        # 计算真实速度场 v = x_0 - x_1
        true_velocity = target - noise

        # 预测速度场，保持与原始代码相同的接口
        pred_velocity = self.net(x_t, t, z)

        # 计算L2损失
        loss = torch.nn.functional.mse_loss(pred_velocity, true_velocity, reduction='none')
        loss = loss.mean(dim=-1)  # 在特征维度上平均

        if mask_loss is not None:
            loss = (loss * mask * mask_loss).sum() / mask.sum()
        else:
            loss = (loss * mask).sum() / mask.sum()

        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        """保持与原DiffLoss相同的接口进行采样"""
        batch_size = z.shape[0]
        device = z.device
        
        # 初始化为纯噪声
        if cfg != 1.0:
            # 使用分类器引导时
            noise = torch.randn(batch_size // 2, self.in_channels, device=device)
            noise = torch.cat([noise, noise], dim=0) * temperature
            use_cfg = True
        else:
            # 不使用分类器引导
            noise = torch.randn(batch_size, self.in_channels, device=device) * temperature
            use_cfg = False
            
        # 使用欧拉法求解ODE
        x = noise
        steps = self.num_sampling_steps
        # import ipdb;ipdb.set_trace()
        time_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)[:-1]
        step_size = 1.0 / steps
        
        for t in time_steps:
            # 创建batch的时间步
            t_batch = torch.ones(batch_size, device=device) * t
            
            # 预测速度场
            with torch.no_grad():
                if use_cfg:
                    # 分类器引导
                    half = x[: batch_size // 2]
                    combined = torch.cat([half, half], dim=0)
                    v_combined = self.net(combined, t_batch, z)
                    v_cond, v_uncond = torch.split(v_combined, batch_size // 2, dim=0)
                    v = v_uncond + cfg * (v_cond - v_uncond)
                    # 重复以匹配原始形状
                    v = torch.cat([v, v], dim=0)
                else:
                    v = self.net(x, t_batch, z)
                
            # Euler步进：x_{t-dt} = x_t - v_t * dt
            x = x + v * step_size
            
        return x

# 保留原有代码的其他类和函数(SimpleMLPAdaLN, TimestepEmbedder, ResBlock, FinalLayer等)
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # def forward(self, x, t, c):
    #     """
    #     Apply the model to an input batch.
    #     :param x: an [N x C] Tensor of inputs.
    #     :param t: a 1-D batch of timesteps.
    #     :param c: conditioning from AR transformer.
    #     :return: an [N x C] Tensor of outputs.
    #     """
    #     # import ipdb;ipdb.set_trace()
    #     x = self.input_proj(x)
    #     t = self.time_embed(t)
    #     c = self.cond_embed(c)

    #     y = t + c

    #     if self.grad_checkpointing and not torch.jit.is_scripting():
    #         for block in self.res_blocks:
    #             x = checkpoint(block, x, y)
    #     else:
    #         for block in self.res_blocks:
    #             x = block(x, y)

    #     return self.final_layer(x, y)

    # 以下是SimpleMLPAdaLN的forward方法，需要适应时间步t是标量而非整数
    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps (0 to 1).
        :param c: conditioning from AR transformer.
        :return: velocity field prediction
        """
        x = self.input_proj(x)
        # 将t从[0,1]映射到网络期望的范围
        t = t * 1000  # 简单缩放让embedding有意义
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
