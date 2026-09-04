"""Exact ResNet50 linear and C4 classifier-head definitions."""

from __future__ import annotations

import logging
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn


LOGGER = logging.getLogger(__name__)
STRIP_PREFIXES = (
    "student_backbone.", "teacher_backbone.", "backbone.", "module.", "model.",
)


def normalize_state_dict(checkpoint) -> OrderedDict:
    if isinstance(checkpoint, dict):
        for wrapper in ("state_dict", "model", "teacher", "student"):
            if wrapper in checkpoint and isinstance(checkpoint[wrapper], dict):
                checkpoint = checkpoint[wrapper]
                break
    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        new_key = key
        for prefix in STRIP_PREFIXES:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        state_dict[new_key] = value
    return state_dict


def load_pretrained(backbone: nn.Module, path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict(checkpoint)
    incompatible = backbone.load_state_dict(state_dict, strict=False)
    matched = len(state_dict) - len(incompatible.unexpected_keys)
    LOGGER.info(
        "Loaded %d/%d checkpoint tensors (%d missing, %d unexpected)",
        matched, len(state_dict), len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )
    if matched == 0:
        raise RuntimeError("no pretrained checkpoint tensors matched ResNet50")


class C4EquivariantHead(nn.Module):
    """C4-equivariant MLP over pooled features with invariant output logits.

    Only the classifier head is equivariant. The ResNet50 backbone is not.
    """

    def __init__(self, in_features: int = 2048, hidden_fields=(64, 64), classes=2):
        super().__init__()
        base_space = gspaces.rot2dOnR2(N=4)
        self.gspace = gspaces.no_base_space(base_space.fibergroup)
        regular_dimension = self.gspace.fibergroup.order()
        input_fields = max(in_features // regular_dimension, 1)
        projected_features = input_fields * regular_dimension
        self.input_proj = (
            nn.Identity() if projected_features == in_features
            else nn.Linear(in_features, projected_features)
        )
        self.input_type = enn.FieldType(
            self.gspace, input_fields * [self.gspace.regular_repr]
        )
        hidden_types = [
            enn.FieldType(self.gspace, count * [self.gspace.regular_repr])
            for count in hidden_fields
        ]
        self.output_type = enn.FieldType(
            self.gspace, classes * [self.gspace.trivial_repr]
        )
        types = [self.input_type, *hidden_types, self.output_type]
        layers = []
        for index, (input_type, output_type) in enumerate(zip(types, types[1:])):
            layers.append(enn.Linear(input_type, output_type))
            if index < len(types) - 2:
                layers.append(enn.ReLU(output_type, inplace=False))
        self.equivariant_net = enn.SequentialModule(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.input_proj(features)
        geometric = enn.GeometricTensor(features, self.input_type)
        return self.equivariant_net(geometric).tensor


class SubmissionResNet50(nn.Module):
    def __init__(self, family: str, pretrained_path):
        super().__init__()
        if family not in ("linear", "eqc4"):
            raise ValueError("family must be linear or eqc4")
        self.backbone = timm.create_model("resnet50", pretrained=False, num_classes=2)
        load_pretrained(self.backbone, pretrained_path)
        input_features = self.backbone.fc.in_features
        if family == "linear":
            self.backbone.fc = nn.Linear(input_features, 2)
            self.head = nn.Identity()
        else:
            self.backbone.fc = nn.Identity()
            self.head = C4EquivariantHead(
                in_features=input_features, hidden_fields=(64, 64), classes=2
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))
