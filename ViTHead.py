import torch
import torch.nn as nn

def build_2d_rope_angles(head_dim, height, width):
    """
    Precompute rotation angles for every patch position.
    Returns cos, sin of shape [height*width, head_dim // 2].
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    quarter = head_dim // 4

    # standard RoPE frequency schedule: low frequencies for later
    # dimensions so different dims capture position at different scales
    freqs = 1.0 / (10000 ** (torch.arange(quarter).float() / quarter))  # [quarter]

    rows = torch.arange(height).float()
    cols = torch.arange(width).float()

    row_angles = rows[:, None] * freqs[None, :]   # [height, quarter]
    col_angles = cols[:, None] * freqs[None, :]   # [width, quarter]

    # broadcast into a full [height, width, quarter] grid each, then
    # concatenate: first half of the angle vector encodes row, second
    # half encodes column
    grid_row = row_angles[:, None, :].expand(height, width, quarter)
    grid_col = col_angles[None, :, :].expand(height, width, quarter)
    angles = torch.cat([grid_row, grid_col], dim=-1)     # [height, width, head_dim/2]
    angles = angles.reshape(height * width, head_dim // 2)

    return angles.cos(), angles.sin()   # each: [num_patches, head_dim/2]


def apply_rope(x, cos, sin):
    """
    x: [batch, num_heads, num_patches, head_dim]
    cos, sin: [num_patches, head_dim/2]
    Rotates each adjacent (even, odd) pair of dims by the precomputed angle.
    """
    x1 = x[..., 0::2]   # even dims:  [batch, heads, num_patches, head_dim/2]
    x2 = x[..., 1::2]   # odd dims:   [batch, heads, num_patches, head_dim/2]

    # standard 2D rotation matrix applied to each (x1, x2) pair
    rotated1 = x1 * cos - x2 * sin
    rotated2 = x1 * sin + x2 * cos

    out = torch.stack([rotated1, rotated2], dim=-1)   # [..., head_dim/2, 2]
    return out.flatten(-2)                             # interleave back to head_dim


class ViTBlockRoPE(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        # manual Q/K/V projections instead of nn.MultiheadAttention,
        # since we need to rotate Q and K before the dot product
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, cos, sin):
        # x: [batch, num_patches, dim]
        B, N, D = x.shape
        h = self.norm1(x)

        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, batch, heads, num_patches, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # rotate q and k by position -- v is left alone, since RoPE only
        # needs to affect how attention SCORES depend on position, not
        # the content being aggregated
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        attn_out = attn_weights @ v                     # [batch, heads, num_patches, head_dim]

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size=16, dim=256, depth=4, num_heads=8):
        super().__init__()
        self.output_dim = dim   # NEW: SlotAttentionAutoencoder reads this to size slot attention's input projection

        img_h, img_w = image_size   # NEW: unpack instead of treating as one int
        assert img_h % patch_size == 0 and img_w % patch_size == 0

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.grid_h = img_h // patch_size
        self.grid_w = img_w // patch_size

        head_dim = dim // num_heads
        cos, sin = build_2d_rope_angles(head_dim, self.grid_h, self.grid_w)  # NEW: non-square grid
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.blocks = nn.ModuleList([ViTBlockRoPE(dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def encode(self, images):
        x = self.patch_embed(images)             # [batch, dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)          # [batch, grid_h*grid_w, dim]
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.norm(x)