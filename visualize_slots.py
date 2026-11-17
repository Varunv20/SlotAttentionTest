
import math
import os

import matplotlib
matplotlib.use("Agg")  # no display needed; must be set before pyplot is imported
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def run_model_with_attn(model, batch, device):
    """Re-implements SlotAttentionAutoencoder.forward, but also returns attn."""
    model.eval()
    images = batch["image"].to(device)

    if "clip_embedding" in batch:
        inputs = batch["clip_embedding"].to(device)
    else:
        inputs = model.image_encoder.encode(images)

    labels_per_sample = batch["labels"]
    n_extra = [model.n_extra_slots] * len(labels_per_sample)   # NEW: was missing -- this is what SlotAttentionAutoencoder.forward does

    slots, slot_mask = model.slot_init(labels_per_sample, n_extra, device=inputs.device)  # NEW: n_extra passed through
    slots, attn = model.slot_attn(inputs, slots, slot_mask)
    recon, masks, slot_rgb = model.decoder(slots, slot_mask)

    model.train()
    return recon, masks, slot_rgb, attn, slot_mask
def _patch_grid_size(num_patches: int) -> tuple[int, int]:
    side = int(math.isqrt(num_patches))
    assert side * side == num_patches, (
        f"expected a square patch grid, got {num_patches} patches "
        f"(CLIP ViT patch count = (image_size / patch_size)**2)"
    )
    return side, side


def visualize_batch(
    model,
    batch,
    device,
    num_samples: int = 2,
    max_slots: int = None,
    out_dir: str = "viz",
    step: int = None,
):
    """
    Saves a grid per sample as a PNG:
        row 1: [input image | composited recon | attn heatmap x N slots]
        row 2: [ (blank)    | (blank)           | masked slot RGB x N slots]

    Set max_slots to cap how many slot columns are shown (useful if
    n_max is large and mostly padding for this batch).

    Files are written to out_dir as "step_{step}_sample_{b}.png" (or
    "sample_{b}.png" if step is None), one per sample. Returns the list
    of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []
    recon, masks, slot_rgb, attn, slot_mask = run_model_with_attn(model, batch, device)
    # recon:     [B, 3, H, W]
    # masks:     [B, N, 1, H, W]
    # slot_rgb:  [B, N, 3, H, W]
    # attn:      [B, N, L]
    # slot_mask: [B, N] bool

    images = batch["image"].to(device)
    B, N = slot_mask.shape
    H, W = images.shape[-2:]
    n_show = min(num_samples, B)

    gh, gw = _patch_grid_size(attn.shape[-1])

    for b in range(n_show):
        real_slots = slot_mask[b].nonzero(as_tuple=True)[0].tolist()
        if max_slots is not None:
            real_slots = real_slots[:max_slots]
        n_cols = 2 + len(real_slots)

        fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 4.6))

        img_np = images[b].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        recon_np = recon[b].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("input")
        axes[1, 0].axis("off")

        axes[0, 1].imshow(recon_np)
        axes[0, 1].set_title("reconstruction")
        axes[1, 1].axis("off")

        for col, n in enumerate(real_slots, start=2):
            # --- attention heatmap: this slot's attn over CLIP patches, upsampled ---
            heat = attn[b, n].reshape(gh, gw).cpu()
            heat = torch.nn.functional.interpolate(
                heat[None, None], size=(H, W), mode="bilinear", align_corners=False
            )[0, 0].numpy()

            axes[0, col].imshow(img_np)
            axes[0, col].imshow(heat, cmap="jet", alpha=0.55)
            axes[0, col].set_title(f"slot {n} attn")

            # --- masked slot reconstruction: this slot's RGB * its alpha mask ---
            masked_rgb = (slot_rgb[b, n] * masks[b, n]).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            axes[1, col].imshow(masked_rgb)
            axes[1, col].set_title(f"slot {n} recon")

        for ax in axes.ravel():
            ax.axis("off")
        fig.suptitle(f"sample {b}" if step is None else f"step {step} — sample {b}")
        fig.tight_layout()

        fname = f"sample_{b}.png" if step is None else f"step_{step}_sample_{b}.png"
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)  # free the figure; important in a loop or it'll leak memory
        saved_paths.append(path)

    return saved_paths


if __name__ == "__main__":
    """
    Quick standalone check: builds the model fresh (random init, so the
    panels will look like noise) and runs it on one batch from val_dir,
    just to confirm the plotting code runs end-to-end before you wire it
    into training. Replace with a real checkpoint load once you have one.
    """
    import torch
    from torch.utils.data import DataLoader

    from Config import Config
    from SlotModel import build_model
    from data.clevr import CLEVRStream, clevr_collate

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, device)

    # Uncomment once you have a checkpoint:
    # ckpt = torch.load("checkpoints/step_10000.pt", map_location=device)
    # model.load_state_dict(ckpt["model"])

    val_loader = DataLoader(
        CLEVRStream(cfg.val_dir, image_size=cfg.image_size, max_objects=cfg.max_objects, shuffle_buffer=0),
        batch_size=4,
        collate_fn=clevr_collate,
    )
    batch = next(iter(val_loader))
    paths = visualize_batch(model, batch, device, num_samples=2, out_dir="viz")
    print("saved:", paths)