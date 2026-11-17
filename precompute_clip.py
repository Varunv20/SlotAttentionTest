"""
One-time offline pass: run every CLEVR image through the frozen CLIP vision
tower once, and every possible CLEVR label string through the frozen CLIP
text tower once, and cache the results to disk. Training then reads these
cached tensors instead of recomputing CLIP forward passes on every step --
this removes the single biggest cost from the training loop, since CLIP is
frozen and its output for a given image/label never changes.

PREREQUISITE: Config.train_dir / Config.val_dir must already point at real,
non-overlapping tfrecords files (run make_val_split.py first if you haven't).

Run from modeling/:
    python precompute_clip.py

Writes:
    data/clevr6_clip_train/shard_00000.pt, shard_00001.pt, ...
    data/clevr6_clip_val/shard_00000.pt, ...
    data/clip_meta.pt   (text vocab embeddings + output dims, shared)

After this finishes, set in Config.py:
    use_precomputed_clip = True
(the cache dir / meta path fields already default to where this script
writes, so nothing else needs to change).
"""

import itertools
import os

import torch
from torch.utils.data import DataLoader

from CLIP import CLIPImageEncoder, CLIPTextEncoder
from Config import Config
from data.clevr import CLEVRStream, clevr_collate, COLORS, MATERIALS, SHAPES


def build_full_vocab():
    """Every (color, material, shape) label string the fixed CLEVR vocabulary
    can produce. "none" (code 0) is the padding code for each attribute and
    never appears in a real per-object label (see clevr_collate), so it's
    excluded here -- 8 colors x 2 materials x 3 shapes = 48 combinations."""
    return [
        f"{color} {material} {shape}"
        for color in COLORS[1:]
        for material in MATERIALS[1:]
        for shape in SHAPES[1:]
    ]


def precompute_split(split_name, tfrecord_path, cfg, image_encoder, out_dir, shard_size):
    os.makedirs(out_dir, exist_ok=True)
    ds = CLEVRStream(
        tfrecord_path,
        image_size=cfg.image_size,
        max_objects=cfg.max_objects,
        shuffle_buffer=0,  # deterministic single pass; shard order doesn't matter, PrecomputedCLEVRStream reshuffles later
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=clevr_collate)

    buf_image, buf_clip, buf_label_map, buf_vis, buf_nobj, buf_labels = [], [], [], [], [], []
    shard_idx = 0
    total = 0
    buffered = 0

    def flush():
        nonlocal shard_idx, buf_image, buf_clip, buf_label_map, buf_vis, buf_nobj, buf_labels
        if not buf_image:
            return
        shard = {
            "image": torch.cat(buf_image, dim=0),
            "clip_embedding": torch.cat(buf_clip, dim=0),
            "label_map": torch.cat(buf_label_map, dim=0),
            "visibility": torch.cat(buf_vis, dim=0),
            "num_true_objects": torch.cat(buf_nobj, dim=0),
            "labels": list(itertools.chain.from_iterable(buf_labels)),
        }
        path = os.path.join(out_dir, f"shard_{shard_idx:05d}.pt")
        torch.save(shard, path)
        print(f"  [{split_name}] wrote {path} ({shard['image'].shape[0]} examples)")
        shard_idx += 1
        buf_image, buf_clip, buf_label_map, buf_vis, buf_nobj, buf_labels = [], [], [], [], [], []

    for batch in loader:
        clip_emb = image_encoder.encode(batch["image"]).half().cpu()  # [B, L, D] fp16 -- halves storage vs fp32
        buf_image.append((batch["image"] * 255).round().to(torch.uint8).cpu())
        buf_clip.append(clip_emb)
        buf_label_map.append(batch["label_map"].to(torch.uint8).cpu())
        buf_vis.append(batch["visibility"].cpu())
        buf_nobj.append(batch["num_true_objects"].cpu())
        buf_labels.append(batch["labels"])

        n = batch["image"].shape[0]
        buffered += n
        total += n
        if buffered >= shard_size:
            flush()
            buffered = 0
        if total % 2000 < cfg.batch_size:
            print(f"  [{split_name}] {total:,} examples processed")

    flush()
    print(f"[{split_name}] done: {total:,} examples across {shard_idx} shards\n")


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    image_encoder = CLIPImageEncoder(cfg.clip_model_name, device)
    text_encoder = CLIPTextEncoder(cfg.clip_model_name, device)

    vocab = build_full_vocab()
    print(f"encoding {len(vocab)} vocab label strings...")
    text_emb = text_encoder.encode(vocab).cpu()
    meta = {
        "text_embeddings": {label: text_emb[i] for i, label in enumerate(vocab)},
        "text_output_dim": text_encoder.output_dim,
        "image_output_dim": image_encoder.output_dim,
    }
    os.makedirs(os.path.dirname(cfg.clip_meta_path) or ".", exist_ok=True)
    torch.save(meta, cfg.clip_meta_path)
    print(f"wrote {cfg.clip_meta_path} ({len(vocab)} labels)\n")

    precompute_split("train", cfg.train_dir, cfg, image_encoder, cfg.clip_train_cache_dir, cfg.clip_shard_size)
    precompute_split("val", cfg.val_dir, cfg, image_encoder, cfg.clip_val_cache_dir, cfg.clip_shard_size)

    print("Done. Set in Config.py:")
    print("  use_precomputed_clip = True")


if __name__ == "__main__":
    main()
