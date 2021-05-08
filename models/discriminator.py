import torch
import torch.nn as nn
from .layers import Block, trunc_normal_
from .diff_aug import DiffAugment
from typing import Any


class Discriminator(nn.Module):
    """
    Discriminator module from TransGAN paper. It is essentially an implementation of the Vision Transformer.

    Params:
    - token_dim: the size of the channel dimension of image tokens
    - diff_aug: if not None, differentiable augmentation will be applied to the input.
    - depth: number of transformer encoder layers
    - patch_size: size of image patches for tokenization step
    - img_size: size of input image (height and width)
    - num_classes: number of classification classes. Default is 2 for fake vs real.
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
            token_dim: int,
            diff_aug=None,
            depth: int = 7,
            patch_size: int = 16,
            img_size: int = 32,
            num_classes: int = 2,
            num_heads: int = 4,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_scale=None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            activation: str = 'gelu',
            drop_path_rate: float = 0.,
            norm_layer: Any = nn.LayerNorm
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = token_dim  # num_features for consistency with other models
        self.token_dim = token_dim
        self.depth = depth
        self.img_size = img_size
        self.diff_aug = diff_aug

        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, token_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, token_dim))
        self.pos_drop = nn.Dropout(p=proj_drop)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                token_dim=token_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                proj_drop=proj_drop, attn_drop=attn_drop, drop_path_prob=dpr[i], norm_layer=norm_layer,
                activation=activation
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(token_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.classifier = nn.Linear(token_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.token_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        Expected Input Shape: (B, 3, H, W), where:
        - B: batch size
        - H: image height
        - W: image width

        Expected Output Shape (B, token_dim) (cls token)
        """
        if self.diff_aug is not None:
            x = DiffAugment(x, self.diff_aug, True)  # -> (B, 3, H, W)
        B = x.shape[0]

        x = self.patch_embed(x)  # -> (B, token_dim, patch_size, patch_size)
        x = x.flatten(2)  # -> (B, token_dim, patch_size ** 2)
        x = x.permute(0, 2, 1)  # -> (B, patch_size ** 2, token_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # -> (B, 1, token_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # -> (B, patch_size ** 2 + 1, token_dim)
        x = x + self.pos_embed  # -> (B, patch_size ** 2 + 1, token_dim)
        x = self.pos_drop(x)  # -> (B, patch_size ** 2 + 1, token_dim)

        for blk in self.blocks:
            x = blk(x)  # -> (B, patch_size ** 2 + 1, token_dim)

        x = self.norm(x)  # -> (B, patch_size ** 2 + 1, token_dim)

        # return cls head only
        return x[:, 0]

    def forward(self, x):
        """
        Expected Input Shape: (B, 3, H, W), where:
        - B: batch size
        - H: image height
        - W: image width

        Expected Output Shape (B, num_classes)
        """
        x = self.forward_features(x)  # -> (B, token_dim)
        x = self.classifier(x)  # -> (B, num_classes)
        return x
