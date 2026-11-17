from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from CLIP import CLIPImageEncoder, CLIPTextEncoder, IdentityImageEncoder, CachedTextEncoder
from Reconstruction import SpatialBroadcastDecoder
class PromptSlotInitializer(nn.Module):
    def __init__(self, text_encoder, text_dim, slot_dim, init_sigma=1.0, learnable_sigma=True):
        super().__init__()
        self.text_encoder = text_encoder
        self.proj = nn.Linear(text_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)
        self.generic_mu = nn.Parameter(torch.randn(1, slot_dim) * 0.02)
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.full((1, slot_dim), init_sigma)))
        else:
            self.register_buffer("log_sigma", torch.log(torch.full((1, slot_dim), init_sigma)))

    def forward(
        self,
        labels_per_sample,
        n_extra_per_sample=None,          # NEW: was silently hardcoded to zero before
        device: Optional[torch.device] = None,
    ):
        """
        labels_per_sample: batch of label lists, e.g.
            [["cat", "sofa"], ["dog", "table", "lamp"], []]
        n_extra_per_sample: number of extra/background slots per sample.
            These are placed FIRST (indices [0, n_extra)) so a fixed slot
            index -- e.g. index 0 -- reliably means "background" for
            every sample regardless of how many object labels follow.

        Returns:
            slots:      [B, N_max, D]  (padded)
            slot_mask:  [B, N_max]     bool, True where slot is real/valid
        """
        B = len(labels_per_sample)
        device = device or next(self.parameters()).device
        if n_extra_per_sample is None:                     # NEW: real default instead of hardcoded [0]*B
            n_extra_per_sample = [0] * B

        flat_labels = []
        offsets = []
        for labels in labels_per_sample:
            offsets.append(len(flat_labels))
            flat_labels.extend(labels)
        offsets.append(len(flat_labels))

        if len(flat_labels) > 0:
            text_emb = self.text_encoder.encode(flat_labels).to(device)
            text_emb = self.norm(self.proj(text_emb))
        else:
            text_emb = torch.empty(0, self.proj.out_features, device=device)

        n_per_sample = [
            n_extra_per_sample[i] + (offsets[i + 1] - offsets[i]) for i in range(B)
        ]
        n_max = max(n_per_sample) if n_per_sample else 0
        slot_dim = self.proj.out_features

        slots = torch.zeros(B, n_max, slot_dim, device=device)
        slot_mask = torch.zeros(B, n_max, dtype=torch.bool, device=device)
        sigma = self.log_sigma.exp()

        for i, labels in enumerate(labels_per_sample):
            n_lab = offsets[i + 1] - offsets[i]
            n_ext = n_extra_per_sample[i]
            n_tot = n_lab + n_ext

            mu = torch.empty(n_tot, slot_dim, device=device)
            if n_ext > 0:
                mu[:n_ext] = self.generic_mu.expand(n_ext, -1)              # NEW: extra/background FIRST
            if n_lab > 0:
                mu[n_ext:n_ext + n_lab] = text_emb[offsets[i]:offsets[i + 1]]  # NEW: labels shifted after n_ext

            eps = torch.randn(n_tot, slot_dim, device=device)
            slots[i, :n_tot] = mu + sigma * eps
            slot_mask[i, :n_tot] = True

        return slots, slot_mask
class GaussianSlotInitializer(nn.Module):
    """
    Vanilla Slot Attention initialization (Locatello et al. 2020): every
    slot, for every sample, is drawn IID from the SAME learned Gaussian
    (one shared mu, one shared sigma across all N slots). No text, no
    per-object identity, no symmetry-breaking beyond the random noise --
    this is the fully unsupervised baseline PromptSlotInitializer departs
    from.

    Unlike PromptSlotInitializer, num_slots is a fixed hyperparameter, not
    derived from label counts -- so there's no padding/masking needed;
    every sample gets exactly num_slots slots, and slot_mask is all True.
    """

    def __init__(self, slot_dim: int, num_slots: int, init_sigma: float = 1.0, learnable_sigma: bool = True):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        # ONE shared mu/sigma, not one per slot -- this is what makes
        # slots symmetric/interchangeable before any noise is added
        self.mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.full((1, 1, slot_dim), init_sigma)))
        else:
            self.register_buffer("log_sigma", torch.log(torch.full((1, 1, slot_dim), init_sigma)))

    def forward(self, labels_per_sample=None, n_extra_per_sample=None, device=None):
        """
        Signature matches PromptSlotInitializer.forward so this drops in
        with zero changes at the call site -- labels_per_sample and
        n_extra_per_sample are accepted but ignored (only used here to
        recover the batch size B via len()).
        """
        device = device or next(self.parameters()).device
        assert labels_per_sample is not None, "pass labels_per_sample (or any list of length B) just to get batch size"
        B = len(labels_per_sample)
        N = self.num_slots

        mu = self.mu.expand(B, N, -1)
        sigma = self.log_sigma.exp().expand(B, N, -1)
        eps = torch.randn(B, N, self.slot_dim, device=device)
        slots = mu + sigma * eps

        slot_mask = torch.ones(B, N, dtype=torch.bool, device=device)  # no padding -- every slot is always real
        return slots, slot_mask
class CompetitiveSlotAttention(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        input_dim: int,
        num_iters: int = 3,
        mlp_hidden_mult: int = 4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * mlp_hidden_mult),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * mlp_hidden_mult, slot_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,          # [B, L, D_in]
        slots: torch.Tensor,           # [B, N, D_slot]  (from PromptSlotInitializer)
        slot_mask: torch.Tensor,       # [B, N] bool, True = real slot
        input_mask: torch.Tensor = None,  # [B, L] bool, True = valid input
    ):
        B, N, D = slots.shape
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)  # [B, L, D]
        v = self.to_v(inputs)  # [B, L, D]

        # additive bias for padded slots: -inf so they never win competition
        slot_bias = torch.zeros(B, N, 1, device=slots.device)
        slot_bias.masked_fill_(~slot_mask.unsqueeze(-1), float("-inf"))

        if input_mask is not None:
            input_bias = torch.zeros(B, 1, inputs.shape[1], device=slots.device)
            input_bias.masked_fill_(~input_mask.unsqueeze(1), float("-inf"))
        else:
            input_bias = 0.0

        for _ in range(self.num_iters):
            slots_prev = slots
            slots_n = self.norm_slots(slots)
            q = self.to_q(slots_n) * self.scale  # [B, N, D]

            logits = torch.einsum("bnd,bld->bnl", q, k)  # [B, N, L]
            logits = logits + slot_bias + input_bias

            # *** competitive softmax: normalize over the SLOT axis ***
            attn = F.softmax(logits, dim=1)  # [B, N, L], competes over N
            # renormalize each slot's row into a weighted mean over inputs
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            attn = torch.nan_to_num(attn, nan=0.0)  # padded slots -> all -inf row -> nan, zero it

            updates = torch.einsum("bnl,bld->bnd", attn, v)  # [B, N, D]

            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D),
            ).reshape(B, N, D)
            slots = slots + self.mlp(self.norm_mlp(slots))

            # keep padded slots inert (avoid drifting from GRU/MLP on zeros)
            slots = torch.where(slot_mask.unsqueeze(-1), slots, slots_prev)

        return slots, attn  # attn useful for visualization / aux losses



class SlotSelfAttention(nn.Module):
    """
    Standard pre-norm transformer encoder block applied across slots
    (not across inputs). Lets slots exchange information after
    competitive attention has given each one a distinct identity —
    e.g. relational reasoning between objects.
    """

    def __init__(self, slot_dim: int, num_heads: int = 4, mlp_hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(slot_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=slot_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * mlp_hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(slot_dim * mlp_hidden_mult, slot_dim),
        )

    def forward(self, slots: torch.Tensor, slot_mask: torch.Tensor):
        # slots:     [B, N, D]
        # slot_mask: [B, N] bool, True = real slot

        # nn.MultiheadAttention's key_padding_mask wants True = IGNORE this key,
        # which is the opposite convention from our slot_mask (True = real) —
        # invert it here.
        key_padding_mask = ~slot_mask  # [B, N], True = pad, ignore as a key

        x = self.norm1(slots)
        attn_out, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
        )
        slots = slots + attn_out

        slots = slots + self.mlp(self.norm2(slots))

        # padded slots stay inert, same convention as CompetitiveSlotAttention
        slots = torch.where(slot_mask.unsqueeze(-1), slots, torch.zeros_like(slots))
        return slots

class SlotModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,                 # produces [B, L, D_in] tokens from raw input
        text_encoder,
        text_dim: int,
        slot_dim: int,
        input_dim: int,
        num_slot_iters: int = 3,
        num_self_attn_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.slot_init = PromptSlotInitializer(text_encoder, text_dim, slot_dim)
        self.competitive_attn = CompetitiveSlotAttention(slot_dim, input_dim, num_iters=num_slot_iters)
        self.self_attn_layers = nn.ModuleList([
            SlotSelfAttention(slot_dim, num_heads=num_heads) for _ in range(num_self_attn_layers)
        ])

    def forward(self, raw_input, labels_per_sample, n_extra_per_sample=None):
        inputs = self.backbone(raw_input)                                  # [B, L, D_in]
        slots, slot_mask = self.slot_init(
            labels_per_sample,
            device=inputs.device,
        )                                                  # [B, N, D], [B, N]
        slots, _ = self.competitive_attn(inputs, slots, slot_mask)         # [B, N, D]

        for layer in self.self_attn_layers:
            slots = layer(slots, slot_mask)                                # [B, N, D]

        return slots, slot_mask

class SlotAttentionAutoencoder(nn.Module):
    def __init__(
        self,
        cfg,
        image_encoder,
        text_encoder,
        slot_dim: int,
        num_attn_iters: int,
        decoder_resolution: tuple,
        decoder_init_resolution: tuple,
        n_extra_slots: int = 1,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.n_extra_slots = n_extra_slots
        if cfg.use_vanilla_slot_init:   # NEW: add this flag to Config.py
            self.slot_init = GaussianSlotInitializer(slot_dim=cfg.slot_dim, num_slots=cfg.num_slots)
        else:
            self.slot_init = PromptSlotInitializer(text_encoder, text_encoder.output_dim, cfg.slot_dim)
        self.slot_attn = CompetitiveSlotAttention(slot_dim, image_encoder.output_dim, num_iters=num_attn_iters)
        self.decoder = SpatialBroadcastDecoder(slot_dim, decoder_resolution, decoder_init_resolution, background_slot_idx=0)

    def forward(self, batch):
        if "clip_embedding" in batch:
            inputs = batch["clip_embedding"].to(next(self.parameters()).device)
        else:
            inputs = self.image_encoder.encode(batch["image"])

        labels_per_sample = batch["labels"]
        n_extra = [self.n_extra_slots] * len(labels_per_sample)

        slots, slot_mask = self.slot_init(labels_per_sample, n_extra, device=inputs.device)  # NEW: n_extra passed through
        slots, attn = self.slot_attn(inputs, slots, slot_mask)
        recon, masks, slot_rgb = self.decoder(slots, slot_mask)
        return recon, masks, slot_rgb
from ViTHead import ViTEncoder
from CNNHead import CNNEncoder
def build_model(cfg, device: str) -> SlotAttentionAutoencoder:
    encoder_type = getattr(cfg, "image_encoder_type", "clip")

    if encoder_type == "clip" and getattr(cfg, "use_precomputed_clip", False):
        meta = torch.load(cfg.clip_meta_path, map_location="cpu")
        image_encoder = IdentityImageEncoder(output_dim=meta["image_output_dim"])
        text_encoder = CachedTextEncoder(cfg.clip_meta_path, device=device)
        print(f"build_model: using precomputed CLIP embeddings from {cfg.clip_meta_path} "
              f"(no CLIP weights loaded)")
    else:
        # NEW: pick the image encoder based on cfg.image_encoder_type instead
        # of always using CLIP. Text still goes through CLIP either way --
        # only the image side changes.
        encoder_type = getattr(cfg, "image_encoder_type", "clip")

        if encoder_type == "cnn":
            image_encoder = CNNEncoder(
                hidden_dim=cfg.encoder_hidden_dim,
                image_size=cfg.image_size,
            ).to(device)
        elif encoder_type == "vit":
            image_encoder = ViTEncoder(
                image_size=cfg.image_size,
                patch_size=cfg.vit_patch_size,
                dim=cfg.encoder_hidden_dim,
                depth=cfg.vit_depth,
                num_heads=cfg.vit_num_heads,
            ).to(device)
        elif encoder_type == "clip":
            image_encoder = CLIPImageEncoder(cfg.clip_model_name, device)
        else:
            raise ValueError(f"unknown image_encoder_type: {encoder_type!r}")

        # text encoder is unchanged -- still CLIP, regardless of what the
        # image side is using
        text_encoder = CLIPTextEncoder(cfg.clip_model_name, device)

        print(f"build_model: image_encoder={encoder_type}, text_encoder=CLIP({cfg.clip_model_name})")

    model = SlotAttentionAutoencoder(
        cfg,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        slot_dim=cfg.slot_dim,
        num_attn_iters=cfg.num_attn_iters,
        decoder_resolution=cfg.image_size,
        decoder_init_resolution=cfg.decoder_init_resolution,
        n_extra_slots=cfg.n_extra_slots,
        
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"total:     {total_params:,}")
    print(f"trainable: {trainable_params:,}")
    print(f"frozen:    {frozen_params:,}")
    return model