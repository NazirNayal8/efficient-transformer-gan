"""
Adapted from: https://github.com/VITA-Group/TransGAN/blob/master/models/TransGAN_8_8_G2_1.py
"""

import torch
import torch.nn as nn
from .layers import Block, pixel_upsample, trunc_normal_

model_type_to_token_dim = {
    'S': 384,
    'M': 512,
    'L': 768,
    'XL': 1024
}

model_type_to_stage_2_size = {
    'S': 2,
    'M': 2,
    'L': 2,
    'XL': 4
}


def get_token_dim(model_type: str) -> int:
    """
    Returns input token dimension based on TransGAN paper setup for CIFAR-10 dataset.
    """
    if model_type not in ['S', 'M', 'L', 'XL']:
        raise ValueError(f'Unregistered Model Type: {model_type}')

    return model_type_to_token_dim[model_type]


def get_stage_2_depth(model_type: str) -> int:

    if model_type not in ['S', 'M', 'L', 'XL']:
        raise ValueError(f'Unregistered Model Type: {model_type}')

    return model_type_to_stage_2_size[model_type]


class Generator(nn.Module):
    """
    Generator Module from TransGAN Paper. This particular version is for CIFAR-10 dataset. This version consists of
    3 stages. It supports the 4 sizes described in the paper:  [S, M, L, XL]

    Params:
    - bottom_width: the height and width of the feature map expected at the beginning of the generator, which will be
        later upsampled to the target height and width of the desired image size.
    - latent_dim: dimension size of input noise vector to generator
    - depth: the number of layers for the first stage of the generator, 5 is the default by the paper.
    - model_type: refers to the model size as described in the paper. It must a value in ['S', 'M', 'L', 'XL']
    - num_heads: number of heads for multi-head attention computation
    - mlp_ratio: used to compute the hidden size of MLP operations inside the transformer encoder blocks. The hidden
        size is equal to mlp_ratio * in_features.
    - qkv_bias: if true, bias term is added to the query-key-value projection operation
    - qk_scale: scale term for for scaled-dot product operation in attention
    - proj_drop: dropout probability for projection step
    - attn_drop: dropout probability after attention operation
    - activation: type of activation function to be used.
    - drop_path_rate: parameter for drop path (stochastic depth) operation
    - norm_layer: normalization layer, default is layer normalization
    """
    def __init__(
            self,
            bottom_width: int,
            latent_dim: int,
            depth: int = 5,
            model_type: str = 'XL',
            num_heads: int = 4,
            mlp_ratio: int = 4.,
            qkv_bias: bool = False,
            qk_scale=None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            activation: str = 'gelu',
            drop_path_rate: float = 0.,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.bottom_width = bottom_width
        self.token_dim = token_dim = get_token_dim(model_type)
        self.latent_dim = latent_dim
        self.depth = depth
        self.model_type = model_type

        self.linear1 = nn.Linear(latent_dim, (self.bottom_width ** 2) * self.token_dim)

        # positional embedding for each of the 3 stages.
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, token_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, token_dim // 4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, token_dim // 16))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3
        ]

        depth_prob = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                token_dim=token_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                proj_drop=proj_drop, attn_drop=attn_drop, drop_path_prob=depth_prob[i], norm_layer=norm_layer,
                activation=activation
            )
            for i in range(depth)
        ])

        stage_2_depth = get_stage_2_depth(model_type)

        self.upsample_blocks = nn.ModuleList([
            nn.ModuleList([
                Block(
                    token_dim=token_dim // 4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, proj_drop=proj_drop, attn_drop=attn_drop, drop_path_prob=0,
                    norm_layer=norm_layer, is_mask=0, activation=activation
                ) for _ in range(stage_2_depth)
            ]
            ),
            nn.ModuleList([
                Block(
                    token_dim=token_dim // 16, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, proj_drop=proj_drop, attn_drop=attn_drop, drop_path_prob=0,
                    norm_layer=norm_layer, is_mask=0, activation=activation
                ),
                Block(
                    token_dim=token_dim // 16, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, proj_drop=proj_drop, attn_drop=attn_drop, drop_path_prob=0,
                    norm_layer=norm_layer, is_mask=(self.bottom_width * 4) ** 2, activation=activation
                )
            ]
            )
        ])
        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        self.deconv = nn.Sequential(
            nn.Conv2d(self.token_dim // 16, 3, 1, 1, 0)
        )

    def set_arch(self, x, cur_stage):
        pass

    def forward(self, z, epoch):
        """
        Expected Input Shape: (B, latent_dim), where:
        - B: batch size
        - latent_dim: size of input noise vector

        Expected Output Shape: (B, 3, H, W), where:
        - H: target image height
        - W: target image width
        - 3: image channel size
        """
        B = z.shape[0]
        x = self.linear1(z)  # -> (B, H * W * token_dim)
        x = x.view(B, self.bottom_width ** 2, self.token_dim)  # -> (B, H * W, token_dim)
        # add positional embedding to stage 1
        x = x + self.pos_embed[0].to(x.get_device())  # -> (B, H * W, token_dim)

        # stage 1
        H, W = self.bottom_width, self.bottom_width
        for index, blk in enumerate(self.blocks):
            x = blk(x, epoch)  # -> (B, H * W, token_dim)

        # stage 2 + 3 ( with upsampling)
        for index, blk in enumerate(self.upsample_blocks):
            x, H, W = pixel_upsample(x, H, W)  # -> (B, 4 * H * W, token_dim / 4)

            x = x + self.pos_embed[index + 1].to(x.get_device())
            for b in blk:
                x = b(x, epoch)  # -> (B, H * W, token_dim_new)

        output = self.deconv(x.permute(0, 2, 1).view(-1, self.token_dim // 16, H, W))
        return output
