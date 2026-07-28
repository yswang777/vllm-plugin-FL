# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU activation followed by element-wise multiplication."""
    d = x.shape[-1] // 2
    op = getattr(torch.ops._C, "silu_and_mul", None)
    if op is not None:
        out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
        op(out, x)
        return out

    x1, x2 = x[..., :d], x[..., d:]
    return F.silu(x1) * x2


def gelu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """GELU activation followed by element-wise multiplication."""
    d = x.shape[-1] // 2
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"

    if approximate == "tanh":
        op = getattr(torch.ops._C, "gelu_tanh_and_mul", None)
    else:
        op = getattr(torch.ops._C, "gelu_and_mul", None)

    if op is not None:
        out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
        op(out, x)
        return out

    x1, x2 = x[..., :d], x[..., d:]
    return F.gelu(x1, approximate=approximate) * x2
