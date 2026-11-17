import torch
import torch.nn.functional as F


def cycle_consistency_train_step(
    model,        # SlotAttentionAutoencoder
    batch,
    optimizer,
    device,
    recon_weight: float = 1.0,          # weight on the primary pixel reconstruction loss
    slot_consistency_weight: float = 0.1,  # weight on the "re-encode the reconstruction" loss
):
    """
    1. encode image -> slots_1 -> decode -> recon              (pixel loss)
    2. re-encode recon -> slots_2 (same labels, so same slot identities)
    3. compare slots_2 to slots_1                               (consistency loss)
    """
    image = batch["image"].to(device)
    labels_per_sample = batch["labels"]

    if "clip_embedding" in batch:
        inputs_1 = batch["clip_embedding"].to(device)   # ok to use the cached embedding here
    else:
        inputs_1 = model.image_encoder.encode(image)

    slots_init_1, slot_mask = model.slot_init(labels_per_sample, device=device)
    slots_1, _ = model.slot_attn(inputs_1, slots_init_1, slot_mask)
    recon, masks, slot_rgb = model.decoder(slots_1, slot_mask)

    loss_recon = F.mse_loss(recon, image)

    inputs_2 = model.image_encoder.encode(recon)

    slots_init_2, _ = model.slot_init(labels_per_sample, device=device)  # same labels -> same slot identities
    slots_2, _ = model.slot_attn(inputs_2, slots_init_2, slot_mask)

    loss_consistency = F.mse_loss(slots_2, slots_1)

    total_loss = recon_weight * loss_recon + slot_consistency_weight * loss_consistency

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "total": total_loss.item(),
        "recon": loss_recon.item(),
        "consistency": loss_consistency.item(),
    }