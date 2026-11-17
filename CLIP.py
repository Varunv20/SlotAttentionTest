from transformers import CLIPModel, CLIPProcessor
import torch
from torch import nn
import torch.nn.functional as F



class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, labels):
        tokens = self.processor(text=labels, return_tensors="pt", padding=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        text_outputs = self.model.text_model(**tokens)
        pooled_output = text_outputs.pooler_output          # [batch, hidden]
        text_features = self.model.text_projection(pooled_output)  # [batch, projection_dim]
        return text_features

    @property
    def output_dim(self):
        return self.model.config.projection_dim



class CLIPImageEncoder():
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False

        self.clip_image_size = self.model.config.vision_config.image_size  # 224 for this model
        # CLIP's fixed normalization constants (from its pretraining)
        self.register_normalization_buffers()

    def register_normalization_buffers(self):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        self.clip_mean = mean
        self.clip_std = std

    @torch.no_grad()
    def encode(self, images):
        images = images.to(self.device)

        # resize to whatever CLIP's vision tower actually expects, regardless
        # of what resolution the images arrived at (e.g. 128x128 for the decoder)
        if images.shape[-2:] != (self.clip_image_size, self.clip_image_size):
            images = F.interpolate(
                images, size=(self.clip_image_size, self.clip_image_size),
                mode="bilinear", align_corners=False,
            )

        images = (images - self.clip_mean) / self.clip_std  # CLIP's own normalization, not raw [0,1]

        output = self.model.vision_model(pixel_values=images)
        return output.last_hidden_state[:, 1:, :]

    @property
    def output_dim(self):
        return self.model.config.vision_config.hidden_size


class IdentityImageEncoder:
    """
    Drop-in replacement for CLIPImageEncoder when CLIP image embeddings have
    already been precomputed offline (see precompute_clip.py) and are being
    supplied directly via batch["clip_embedding"]. Loads NO weights, so it
    costs zero VRAM/RAM -- used by build_model when cfg.use_precomputed_clip
    is True.

    .encode() is intentionally not implemented: if something calls it, that
    means raw images are being run through this encoder instead of using the
    precomputed embeddings, which defeats the whole point.
    """

    def __init__(self, output_dim: int):
        self._output_dim = output_dim

    def encode(self, images):
        raise RuntimeError(
            "IdentityImageEncoder.encode() was called directly -- this means "
            "raw images are being passed through the encoder instead of using "
            "precomputed embeddings from batch['clip_embedding']. Either use "
            "PrecomputedCLEVRStream (which supplies clip_embedding), or set "
            "cfg.use_precomputed_clip = False to use a real CLIPImageEncoder."
        )

    @property
    def output_dim(self):
        return self._output_dim


class CachedTextEncoder:
    """
    Drop-in replacement for CLIPTextEncoder that looks embeddings up from a
    precomputed table (written by precompute_clip.py) instead of running the
    CLIP text tower. Loads NO CLIP weights -- used by build_model when
    cfg.use_precomputed_clip is True.

    meta_path: path to a .pt file containing
        {"text_embeddings": {label_str: 1D tensor}, "text_output_dim": int,
         "image_output_dim": int}
    """

    def __init__(self, meta_path: str, device: str = "cpu"):
        meta = torch.load(meta_path, map_location="cpu")
        self._output_dim = meta["text_output_dim"]
        self._embeddings = {label: emb.float() for label, emb in meta["text_embeddings"].items()}
        self.device = device

    def encode(self, labels):
        missing = [l for l in labels if l not in self._embeddings]
        if missing:
            raise KeyError(
                f"CachedTextEncoder has no precomputed embedding for {missing!r}. "
                f"Re-run precompute_clip.py if the label vocabulary changed."
            )
        stacked = torch.stack([self._embeddings[l] for l in labels], dim=0)
        return stacked.to(self.device)

    @property
    def output_dim(self):
        return self._output_dim
