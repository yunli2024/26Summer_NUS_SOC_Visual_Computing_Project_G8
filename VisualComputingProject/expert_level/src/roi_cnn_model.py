"""Multi-region CNN models for full face + landmark ROI expression recognition."""

from __future__ import annotations

import torch
from torch import nn

try:
    from . import config
except ImportError:
    import config


class FERRegionEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout: float = config.CNN_DROPOUT) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(256 * 6 * 6, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.projection(self.features(x))


class MultiRegionExpressionCNN(nn.Module):
    def __init__(self, variant: str, num_classes: int = 7, embedding_dim: int = 128) -> None:
        super().__init__()
        if variant not in config.ROI_CNN_VARIANTS:
            raise ValueError(f"Unknown ROI CNN variant: {variant}")
        self.variant = variant
        self.face_encoder = FERRegionEncoder(embedding_dim)
        self.eye_encoder = FERRegionEncoder(embedding_dim) if "eyes" in variant else None
        self.mouth_encoder = FERRegionEncoder(embedding_dim) if "mouth" in variant else None
        branch_count = 1 + int(self.eye_encoder is not None) + int(self.mouth_encoder is not None)
        fused_dim = embedding_dim * branch_count
        self.main_head = nn.Linear(fused_dim, num_classes)
        self.use_aux = variant.endswith("_aux")
        if self.use_aux:
            self.angry_head = nn.Linear(fused_dim, 4)
            self.fear_head = nn.Linear(fused_dim, 3)
            self.sad_head = nn.Linear(fused_dim, 3)

    def embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.face_encoder(batch["face"])]
        if self.eye_encoder is not None:
            parts.append(self.eye_encoder(batch["eye_brow"]))
        if self.mouth_encoder is not None:
            parts.append(self.mouth_encoder(batch["nose_mouth"]))
        return torch.cat(parts, dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = self.embeddings(batch)
        output = {"main": self.main_head(embedding), "embedding": embedding}
        if self.use_aux:
            output["angry_aux"] = self.angry_head(embedding)
            output["fear_aux"] = self.fear_head(embedding)
            output["sad_aux"] = self.sad_head(embedding)
        return output


def build_roi_cnn_model(variant: str) -> MultiRegionExpressionCNN:
    return MultiRegionExpressionCNN(variant=variant, num_classes=len(config.CLASSES))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
