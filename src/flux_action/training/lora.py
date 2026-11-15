"""LoRA on the DiT for the DSF fine-tune: adapters on the shared single-stream blocks and the video mode blocks only.

Placement (see ``models/transformer.py``): every content stream first runs its own ``depth`` mode blocks, then all
streams join one sequence for the ``depth_single_blocks`` shared blocks. The adapters go on the six linear layers
(``q_proj, k_proj, v_proj, attn_out, mlp_in, mlp_out``, plus ``gate_proj`` when the block has one) of

* every single-stream block (where video, actions, state and text attend to each other), and
* the mode blocks of the ``video`` and ``video_cond`` streams.

Everything else stays frozen at the policy's weights: the action and state streams (their mode blocks, ``emb_in``,
``final_layer``, modulations), the text path, ``time_in`` / ``vector_in``, the per-stream modulations, the VAE and
the text encoder. ``W x + (alpha / rank) * B A x`` with ``A`` Kaiming-uniform and ``B`` zero, so training starts from
the policy exactly; the adapters stay in float32 next to a base in the base's dtype.

A LoRA checkpoint holds the adapters only (the base is rebuilt from ``trunk_weights``); :func:`merge_lora` folds
them into plain ``nn.Linear`` layers, which is what an inference export holds.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch import Tensor, nn

LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out", "gate_proj")
DEFAULT_MODE_STREAMS = ("video", "video_cond")


class LoRALinear(nn.Module):
    """``base(x) + scale * (x A^T) B^T``; ``base`` is frozen, ``lora_A`` (rank, in) and ``lora_B`` (out, rank) train."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank, self.alpha, self.scale = int(rank), float(alpha), float(alpha) / int(rank)
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # PEFT's default; B = 0 keeps the base output
        base.requires_grad_(False)

    @property
    def weight(self) -> Tensor:  # read by code that inspects a layer's dtype or device
        return self.base.weight

    def forward(self, x: Tensor) -> Tensor:
        lora = (x @ self.lora_A.to(x.dtype).t()) @ self.lora_B.to(x.dtype).t()
        return self.base(x) + self.scale * lora

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """A plain ``nn.Linear`` with ``W + scale * B A`` (computed in float32, stored in the base dtype)."""
        weight = self.base.weight
        delta = self.scale * (self.lora_B.float() @ self.lora_A.float())
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        out = out.to(device=weight.device, dtype=weight.dtype)
        out.weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def target_blocks(dit: nn.Module, mode_streams: Sequence[str] = DEFAULT_MODE_STREAMS) -> Iterator[nn.Module]:
    """The blocks that get adapters: all single-stream blocks, then the mode blocks of ``mode_streams``."""
    yield from dit.single_blocks
    for stream in mode_streams:
        if stream not in dit.content_mode_blocks:
            raise ValueError(f"no mode blocks for stream {stream!r}; have {sorted(dit.content_mode_blocks)}")
        yield from dit.content_mode_blocks[stream]


def apply_lora(
    dit: nn.Module, rank: int, alpha: float, mode_streams: Sequence[str] = DEFAULT_MODE_STREAMS
) -> list[str]:
    """Freeze the whole DiT and wrap the target linear layers in :class:`LoRALinear`; returns the wrapped names.

    Call before activation checkpointing / FSDP wrap the blocks."""
    dit.requires_grad_(False)
    blocks = {id(block) for block in target_blocks(dit, mode_streams)}
    wrapped = []
    for block_name, block in dit.named_modules():
        if id(block) not in blocks:
            continue
        for name in LINEAR_NAMES:
            layer = getattr(block, name, None)
            if isinstance(layer, nn.Linear):
                setattr(block, name, LoRALinear(layer, rank, alpha))
                wrapped.append(f"{block_name}.{name}")
    if not wrapped:
        raise ValueError("no LoRA target layers found")
    return wrapped


def merge_lora(module: nn.Module) -> int:
    """Replace every :class:`LoRALinear` under ``module`` by its merged ``nn.Linear``; returns how many."""
    count = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
                count += 1
    return count


def lora_settings(config: dict | None) -> dict | None:
    """``TrainConfig.lora`` with defaults filled and checked: ``{"rank", "alpha", "mode_streams"}``."""
    if config is None:
        return None
    unknown = set(config) - {"rank", "alpha", "mode_streams"}
    if unknown:
        raise ValueError(f"unknown lora settings {sorted(unknown)}")
    rank = int(config.get("rank", 512))
    settings = {
        "rank": rank,
        "alpha": float(config.get("alpha", rank)),
        "mode_streams": list(config.get("mode_streams", DEFAULT_MODE_STREAMS)),
    }
    if settings["rank"] < 1 or settings["alpha"] <= 0:
        raise ValueError("lora rank and alpha must be positive")
    return settings
