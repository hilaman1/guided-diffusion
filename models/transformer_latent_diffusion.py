"""transformer based denoiser"""

import torch
from einops.layers.torch import Rearrange
from torch import nn
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalEmbedding(nn.Module):
    def __init__(self, emb_min_freq=1.0, emb_max_freq=1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        frequencies = torch.exp(
            torch.linspace(np.log(emb_min_freq), np.log(emb_max_freq), embedding_dims // 2)
        )

        self.register_buffer("angular_speeds", 2.0 * torch.pi * frequencies)

    def forward(self, x):
        embeddings = torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1
        )
        return embeddings


class MHAttention(nn.Module):
    def __init__(self, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):
        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, "bs n (h d) -> bs h n d", h=self.n_heads) for x in [q, k, v]]

        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=self.dropout_level if self.training else 0,
        )

        out = rearrange(out, "bs h n d -> bs n (h d)", h=self.n_heads)

        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x, y):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q, k, v)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSepConv(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.mlp = nn.Sequential(
            # this Conv with kernel size 1 is equivalent to the Linear layer in a "regular" transformer MLP
            nn.Conv2d(embed_dim, mlp_multiplier * embed_dim, kernel_size=1, padding="same"),
            nn.Conv2d(
                mlp_multiplier * embed_dim,
                mlp_multiplier * embed_dim,
                kernel_size=3,
                padding="same",
                groups=mlp_multiplier * embed_dim,
            ),  # <- depthwise conv
            nn.GELU(),
            nn.Conv2d(mlp_multiplier * embed_dim, embed_dim, kernel_size=1, padding="same"),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        w = h = int(np.sqrt(x.size(1)))  # only square images for now
        x = rearrange(x, "bs (h w) d -> bs d h w", h=h, w=w)
        x = self.mlp(x)
        x = rearrange(x, "bs d h w -> bs (h w) d")
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout_level: float,
        mlp_class: type[MLP] | type[MLPSepConv],
    ):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, is_causal, dropout_level, n_heads=embed_dim // 64)
        self.cross_attention = CrossAttention(
            embed_dim, is_causal=False, dropout_level=0, n_heads=embed_dim // 64
        )
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(self.norm1(x)) + x
        x = self.cross_attention(self.norm2(x), y) + x
        x = self.mlp(self.norm3(x)) + x
        return x


class DenoiserTransBlock(nn.Module):
    def __init__(
        self,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.img_size / self.patch_size) * (self.img_size / self.patch_size))
        patch_dim = self.n_channels * self.patch_size * self.patch_size

        self.patchify_and_embed = nn.Sequential(
            nn.Conv2d(
                self.n_channels,
                patch_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("bs d h w -> bs (h w) d"),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.rearrange2 = Rearrange(
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=int(self.img_size / self.patch_size),
            p1=self.patch_size,
            p2=self.patch_size,
        )

        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc", torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=self.embed_dim,
                    mlp_multiplier=self.mlp_multiplier,
                    # note that this is a non-causal block since we are
                    # denoising the entire image no need for masking
                    is_causal=False,
                    dropout_level=self.dropout,
                    mlp_class=MLPSepConv,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim), self.rearrange2)

    def forward(self, x, cond):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[: x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)


class Denoiser(nn.Module):
    def __init__(
        self,
        image_size: int,
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        text_emb_size: int = 768,
        mlp_multiplier: int = 4,
        n_channels: int = 4
    ):
        super().__init__()

        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels

        self.fourier_feats = nn.Sequential(
            SinusoidalEmbedding(embedding_dims=noise_embed_dims),
            nn.Linear(noise_embed_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers, mlp_multiplier, n_channels)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Sequential(
            nn.Linear(text_emb_size, 2048),
            nn.GELU(),
            nn.Linear(2048, self.embed_dim)
        )

        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
    #
    #     # Initialize patch_embed for x embedder like nn.Linear (instead of nn.Conv2d):
    #     w = self.x_embedder.proj.weight.data
    #     nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    #     nn.init.constant_(self.x_embedder.proj.bias, 0)
    #
    #     # Initialize for y embedder patch_embed like nn.Linear (instead of nn.Conv2d):
    #     w = self.y_embedder.proj.weight.data
    #     nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    #     nn.init.constant_(self.y_embedder.proj.bias, 0)
    #
    #     # Initialize timestep embedding MLP:
    #     nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
    #     nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
    #
    #     # Zero-out adaLN modulation layers in DiT blocks:
    #     for block in self.blocks:
    #         nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
    #         nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
    #
    #     # Zero-out output layers:
    #     nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
    #     nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
    #     nn.init.constant_(self.final_layer.linear.weight, 0)
    #     nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, noise_level, label):
        noise_level = torch.unsqueeze(noise_level, dim=-1)
        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = label.view(label.size(0), -1)
        label = self.label_proj(label).unsqueeze(1)

        noise_label_emb = torch.cat([noise_level, label], dim=1)  # bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x, noise_label_emb)
        return x


if __name__ == "__main__":
    batch_size = 16
    channels = 4
    image_height = 32
    image_width = 32

    x = torch.rand(batch_size, channels, image_height, image_width)
    noise_level = torch.rand(batch_size,)
    label = torch.rand(batch_size, channels, image_height, image_width)

    model = Denoiser(image_size=image_height,
                    noise_embed_dims=channels,
                    patch_size=2,
                    embed_dim=768,
                    dropout=0.2,
                    n_layers=12,
                     text_emb_size=4096)
    output = model(x, noise_level, label)
    print(output.shape)