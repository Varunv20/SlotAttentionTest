"""
Interfaces for the slot attention pipeline's pluggable components.

Anything that implements TextEncoder can be handed to PromptSlotInitializer;
anything that implements ImageEncoder can be handed to CompetitiveSlotAttention
(via SlotAttentionAutoencoder). Concrete implementations (CLIP, a plain CNN,
a learned word-embedding table, DINOv2, etc.) live in encoders.py and only
need to satisfy the contract defined here.
"""

from abc import ABC, abstractmethod

import torch


class TextEncoder(ABC):
    """
    Turns a list of label strings into a batch of vectors.

    Implementations: CLIPTextEncoder, SimpleWordEncoder, or any other text
    embedding model, so long as it can map List[str] -> Tensor.
    """

    @abstractmethod
    def encode(self, labels: list[str]) -> torch.Tensor:
        """
        labels: list of strings, e.g. ["large gray metal cube", "small red rubber sphere"]
        returns: Tensor[num_labels, output_dim]
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the vectors `encode` returns."""
        raise NotImplementedError


class ImageEncoder(ABC):
    """
    Turns a batch of images into a sequence of per-location tokens.

    Implementations: CLIPImageEncoder, a plain CNN + positional embedding,
    DINOv2 patch tokens, or any other visual backbone, so long as it maps
    a batch of images to a batch of token sequences.
    """

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: Tensor[B, 3, H, W]
        returns: Tensor[B, L, output_dim]  (L = number of tokens per image,
                 e.g. patches or pixels, depending on the backbone)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of each token `encode` returns (the D_in that
        CompetitiveSlotAttention needs to know about)."""
        raise NotImplementedError