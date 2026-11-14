"""The two noise levels of a DSF window.

Per window: ``t_low`` from ``dsf_low_timesteps`` (``uniform``: U[0, 1], the rule of the LTX WAM pilot; ``recipe``: the
policy's own training distribution, logit-logistic with shift 42, median ~0.98), then ``t_high ~ U[t_low, 1]``. The
hidden video tokens and the actions of their latent frames get ``t_high``, every other predicted token ``t_low``; the
conditioning streams (the context frame, the state) stay clean at 0 as always.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..processing import packing

LOW_TIMESTEPS = ("uniform", "recipe")


def draw_levels(
    n: int, mode: str, *, width: float, shift: float, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """(t_low, t_high), each (n,) float32 on the CPU, with t_low <= t_high <= 1."""
    if mode == "uniform":
        low = torch.rand(n, generator=generator)
    elif mode == "recipe":
        low = packing.sample_timesteps(n, generator, width, shift)
    else:
        raise ValueError(f"dsf_low_timesteps must be one of {LOW_TIMESTEPS}, got {mode!r}")
    high = low + (1 - low) * torch.rand(n, generator=generator)
    return low, high


def token_timesteps(
    low: Tensor, high: Tensor, video_high: Tensor, action_high: Tensor
) -> tuple[Tensor, Tensor]:
    """Per-token levels: ``video_high`` (n, Nv) and ``action_high`` (n, K) bool -> (n, Nv) and (n, K) float32."""
    low, high = low.float()[:, None], high.float()[:, None]
    video = torch.where(video_high.cpu(), high, low)
    action = torch.where(action_high.cpu(), high, low)
    return video, action
