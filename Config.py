from dataclasses import dataclass


@dataclass
class Config:
    # --- data ---
    # These used to point at .npz directories (the old conversion-based
    # pipeline). If you're on CLEVRStream now, they need to point directly
    # at the .tfrecords files instead.
    train_dir: str = "data/clevr_train"   # -> data/clevr_train_images.npy, _masks.npy, _labels.pt
    val_dir: str = "data/clevr_val"       # -> data/clevr_val_images.npy, _masks.npy, _labels.pt
    image_size: tuple[int, int] = (128, 128)
    batch_size: int = 16
    num_workers: int = 4

    # new: CLEVRStream needs these, the old .npz Dataset didn't
    max_objects: int = 6          # CLEVR6 filter, applied per-record while streaming
    shuffle_buffer: int = 1000    # windowed shuffle size (see the streaming tradeoffs discussed earlier)

    # --- model ---
    clip_model_name: str = "openai/clip-vit-base-patch32"
    slot_dim: int = 64
    num_attn_iters: int = 3
    decoder_init_resolution: tuple[int, int] = (8, 8)
    n_extra_slots: int = 1

    # --- optimization ---
    base_lr: float = 4e-4
    warmup_steps: int = 10_000
    decay_steps: int = 100_000
    decay_rate: float = 0.5
    grad_clip_norm: float = 1.0
    total_steps: int = 500_000

    # --- logging / checkpointing ---
    log_every: int = 1000
    eval_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1000

    # --- precomputed CLIP embeddings (see precompute_clip.py) ---
    # Set use_precomputed_clip=True after running precompute_clip.py once.
    # Training then reads cached tensors instead of running the frozen CLIP
    # vision/text towers on every step -- see the throughput discussion.
    use_precomputed_clip: bool = False
    clip_train_cache_dir: str = "data/clevr6_clip_train"
    clip_val_cache_dir: str = "data/clevr6_clip_val"
    clip_meta_path: str = "data/clip_meta.pt"
    clip_shard_size: int = 500       # examples per shard file written by precompute_clip.py
    clip_cache_shuffle_buffer: int = 2000

    image_encoder_type: str = "vit"   # "clip" | "cnn" | "vit"
    encoder_hidden_dim: int = 64      # CNN's hidden_dim, or ViT's dim
    vit_patch_size: int = 16
    vit_depth: int = 4
    vit_num_heads: int = 8
    use_vanilla_slot_init: bool = True   # NEW: if True, use GaussianSlotInitializer instead of PromptSlotInitializer
