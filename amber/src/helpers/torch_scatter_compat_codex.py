from __future__ import annotations

import torch
from torch_geometric.utils import scatter

try:
    from torch_scatter import scatter_add as _torch_scatter_add
    from torch_scatter import scatter_max as _torch_scatter_max
    from torch_scatter import scatter_mean as _torch_scatter_mean
    from torch_scatter import scatter_min as _torch_scatter_min
except (ImportError, OSError):
    _torch_scatter_add = None
    _torch_scatter_mean = None
    _torch_scatter_max = None
    _torch_scatter_min = None
def scatter_add_codex(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None) -> torch.Tensor:
    # [CodeX] 优先复用 torch_scatter；若 Windows 轮子不可用，则回退到 PyG 自带 scatter。
    if _torch_scatter_add is not None:
        return _torch_scatter_add(src=src, index=index, dim=dim, dim_size=dim_size)
    return scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_mean_codex(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None) -> torch.Tensor:
    # [CodeX] mean 聚合在 fallback 路径下直接委托给 PyG scatter，接口保持与 torch_scatter 一致。
    if _torch_scatter_mean is not None:
        return _torch_scatter_mean(src=src, index=index, dim=dim, dim_size=dim_size)
    return scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="mean")


def scatter_max_codex(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None) -> torch.Tensor:
    # [CodeX] 当前仓库的 max 聚合只消费聚合值本身，因此 fallback 只返回值张量即可。
    if _torch_scatter_max is not None:
        return _torch_scatter_max(src=src, index=index, dim=dim, dim_size=dim_size)[0]
    return scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="max")


def scatter_min_codex(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None) -> torch.Tensor:
    # [CodeX] 与 max 相同，fallback 仅返回当前调用链真正需要的聚合结果。
    if _torch_scatter_min is not None:
        return _torch_scatter_min(src=src, index=index, dim=dim, dim_size=dim_size)[0]
    return scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="min")
