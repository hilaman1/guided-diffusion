import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from diffusers.models import AutoencoderKL
import math
import time
import numpy as np


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout,
                              **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, add_bias_kv=True, batch_first=True,
                                                     dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # normed_x = self.norm2(x)
        # normed_y = self.norm4(y)
        # x = x + self.cross_attention(normed_y, normed_x, normed_y)[0]
        # x = (normed_x * normed_y) * y
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


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


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiffusionSegmentationTransformer(nn.Module):
    def __init__(self, height, patch_size, in_channels, hidden_size, num_heads, mlp_ratio, out_channels, depth, dropout):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.seg_image_embedder = PatchEmbed(height, patch_size, in_channels, hidden_size, bias=True)
        self.rgb_image_embedder = PatchEmbed(height, patch_size, in_channels, hidden_size, bias=True)
        self.time_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.seg_image_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        pos_embed = self.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.seg_image_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.DiT_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = torch.reshape(x, (x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, seg_image, rgb_image, t):
        x = torch.cat((rgb_image, seg_image), dim=1)
        x = self.seg_image_embedder(x) + self.pos_embed
        # embedded_rgb = self.rgb_image_embedder(rgb_image) + self.pos_embed
        embedded_t = self.time_embedder(t)

        for DiT_block in self.DiT_blocks:
            x = DiT_block(x, embedded_t)

        x = self.final_layer(x, embedded_t)
        x = self.unpatchify(x)
        return x

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False, extra_tokens=0):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 10
    seg_channels = 3
    rgb_channels = 3
    image_height = 256
    image_width = 256

    seg_image = torch.rand(batch_size, seg_channels, image_height, image_width)
    rgb_image = torch.rand(batch_size, rgb_channels, image_height, image_width)
    timestep = torch.tensor([10])
    seg_image = seg_image.to(device)
    rgb_image = rgb_image.to(device)
    timestep = timestep.to(device)

    embedded_seg_image = torch.rand(batch_size, rgb_channels, image_height // 8, image_width // 8, device=device)
    embedded_rgb_image = torch.rand(batch_size, rgb_channels, image_height // 8, image_width // 8, device=device)

    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    #
    # start = time.time()
    # embedded_seg_image = vae.encoder(seg_image)
    # embedded_rgb_image = vae.encoder(rgb_image)
    # end = time.time()
    #
    # print(f"VAE Running Time {end - start}Sec.")
    # print(f"VAE Output Shape {embedded_seg_image.shape}")

    hidden_size = 512
    in_channels = embedded_seg_image.shape[1]
    patch_size = 2
    height = embedded_seg_image.shape[2]
    num_heads = 16
    mlp_ratio = 4.0
    out_channels = 1
    depth = 12
    model = DiffusionSegmentationTransformer(height=height,
                                             patch_size=patch_size,
                                             in_channels=in_channels,
                                             hidden_size=hidden_size,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             out_channels=out_channels,
                                             depth=depth)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", total_params)

    model = model.to(device)

    start = time.time()
    output = model(embedded_seg_image, embedded_rgb_image, timestep)
    end = time.time()
    print(f"Model Running Time {end - start}Sec.")
    print(f"Output Shape {output.shape}")

    state = {
        "model": model.state_dict()
    }
    torch.save(state, "model.pth")
