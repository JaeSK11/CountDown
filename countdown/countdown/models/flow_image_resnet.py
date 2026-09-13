"""ResNet-18 over flow images -- the backbone arm of the image-member scorecard.

Subclasses the Okonkwo baseline so that *only the backbone changes*.  Training loop,
optimiser, batching, seeding, calibration and persistence are inherited verbatim, which is
what makes a scorecard row attributable: a difference against
``flow_image_cnn_baseline`` under the same protocol is the architecture, not the harness.

Two adaptations, both required and both consequential:

**Stem.** Stock ResNet-18 opens with a 7x7 stride-2 convolution and a stride-2 max-pool,
which throws away 3/4 of the spatial resolution before the first residual block.  That is
tuned for 224px natural images whose content is dense; on a flow image -- where a packet
may be a single 3x3 mark -- it discards the signal for the same reason the baseline's
stride-2-on-every-conv does.  ``adapt_stem=True`` (the default) replaces it with a 3x3
stride-1 convolution and drops the max-pool.

**Channels.** ``in_channels`` of 1 or 3 is handled by rebuilding the stem convolution.  With
ImageNet weights and 1 channel, the pretrained stem kernels are summed across RGB, which is
the standard way to keep a pretrained stem meaningful on single-channel input.

``pretrained`` is off by default.  ImageNet features are a genuinely open question on this
input -- a FlowPic is not a natural image and its low-level statistics are not ImageNet's --
so it is a scorecard variable, not an assumption baked into the member.
"""

from __future__ import annotations

from typing import Any

from countdown.config import get_logger
from countdown.models.base import register
from countdown.models.flow_image_cnn import FlowImageCNN, _torch

log = get_logger(__name__)


def build_resnet18(n_classes: int, in_channels: int = 1, adapt_stem: bool = True,
                   pretrained: bool = False):
    """ResNet-18 with a stem sized for flow images rather than photographs."""
    torch = _torch()
    from torch import nn

    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "flow_image_resnet18 needs torchvision: pip install torchvision"
        ) from exc

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = resnet18(weights=weights)

    if in_channels != 3 or adapt_stem:
        old = net.conv1
        if adapt_stem:
            new = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                w = old.weight                       # (64, 3, 7, 7)
                if in_channels == 1:
                    w = w.sum(dim=1, keepdim=True)   # collapse RGB, preserving response scale
                if adapt_stem:                       # 7x7 -> 3x3: keep the centre taps
                    w = w[:, :, 2:5, 2:5]
                new.weight.copy_(w)
        net.conv1 = new

    if adapt_stem:
        net.maxpool = nn.Identity()

    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


@register("flow_image_resnet18")
class FlowImageResNet18(FlowImageCNN):
    """ResNet-18 on ``flow_image`` tensors.  Same harness as the baseline, new backbone."""

    input_type = "flow_image"
    expects_ndim = 4
    supports = "*"

    def __init__(self, label_space=None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        p = self.params
        p.setdefault("adapt_stem", True)
        p.setdefault("pretrained", False)
        # ResNet-18 is ~110x the baseline's parameter count, so the paper's 1e-4 is not the
        # right default here; 3e-4 with the same Adam is the conventional starting point.
        p["lr"] = float(p.get("lr", 3e-4))

    def _build(self, n_classes: int, in_channels: int, size: int):
        return build_resnet18(
            n_classes, in_channels=in_channels,
            adapt_stem=bool(self.params["adapt_stem"]),
            pretrained=bool(self.params["pretrained"]),
        )
