import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPositionEmbed(nn.Module):
    """
    Dense positional embedding added at every spatial location of a grid.
    Builds a fixed 4-channel coordinate grid (x, 1-x, y, 1-y) and projects
    it with a learned linear layer into the feature dimension, then adds
    it elementwise to the input — same construction the paper uses for
    both the encoder and the broadcast decoder.
    """

    def __init__(self, hidden_dim: int, resolution: tuple):
        super().__init__()
        self.proj = nn.Linear(4, hidden_dim)
        self.register_buffer("grid", self._build_grid(resolution), persistent=False)

    @staticmethod
    def _build_grid(resolution):
        H, W = resolution
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, H),
            torch.linspace(0.0, 1.0, W),
            indexing="ij",
        )
        grid = torch.stack([x, y, 1.0 - x, 1.0 - y], dim=-1)  # [H, W, 4]
        return grid

    def forward(self, x: torch.Tensor):
        # x: [B, H, W, hidden_dim]
        return x + self.proj(self.grid)

class SpatialBroadcastDecoder(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        out_resolution: tuple = (128, 128),
        init_resolution: tuple = (8, 8),
        decoder_channels: int = 64,
        background_slot_idx: int = 0,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.init_resolution = init_resolution
        self.out_resolution = out_resolution
        self.background_slot_idx = background_slot_idx
        # NEW: no more bg_low/bg_high -- grayscale doesn't restrict brightness,
        # only forces R == G == B

        self.pos_embed = SoftPositionEmbed(slot_dim, init_resolution)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(slot_dim, decoder_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, 4, kernel_size=3, stride=1, padding=1),
        )

    def _apply_background_constraint(self, slot_rgb):
        """
        slot_rgb: [B, N, 3, H, W], raw/unbounded everywhere.
        Forces the background slot's R, G, B channels to be identical
        (grayscale) -- brightness is unrestricted, only hue/saturation is.
        """
        idx = self.background_slot_idx
        raw_bg = slot_rgb[:, idx]                        # [B, 3, H, W]

        gray_raw = raw_bg.mean(dim=1, keepdim=True)       # NEW: collapse 3 channels -> 1, averaging the raw logits
        gray = torch.sigmoid(gray_raw)                    # NEW: full [0,1] brightness range, no narrow band
        gray_rgb = gray.expand(-1, 3, -1, -1)             # NEW: broadcast the same value to all 3 channels

        before = slot_rgb[:, :idx]
        after = slot_rgb[:, idx + 1:]
        return torch.cat([before, gray_rgb.unsqueeze(1), after], dim=1)

    def forward(self, slots: torch.Tensor, slot_mask: torch.Tensor = None):
        B, N, D = slots.shape
        Hi, Wi = self.init_resolution
        Ho, Wo = self.out_resolution

        x = slots.reshape(B * N, 1, 1, D).expand(-1, Hi, Wi, -1)
        x = self.pos_embed(x)
        x = x.permute(0, 3, 1, 2)
        x = self.deconv(x)

        if x.shape[-2:] != (Ho, Wo):
            x = F.interpolate(x, size=(Ho, Wo), mode="bilinear", align_corners=False)

        x = x.reshape(B, N, 4, Ho, Wo)
        slot_rgb, alpha_logits = x[:, :, :3], x[:, :, 3:4]

        slot_rgb = self._apply_background_constraint(slot_rgb)

        if slot_mask is not None:
            bias = torch.zeros(B, N, 1, 1, 1, device=slots.device)
            bias.masked_fill_(~slot_mask.view(B, N, 1, 1, 1), float("-inf"))
            alpha_logits = alpha_logits + bias

        masks = torch.softmax(alpha_logits, dim=1)
        masks = torch.nan_to_num(masks, nan=0.0)
        recon = (slot_rgb * masks).sum(dim=1)

        return recon, masks, slot_rgb