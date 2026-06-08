"""
model.py — 模型1（Transport）多任务网络

结构：共享 MLP backbone  →  每个 endpoint 一个轻量 head
  8 endpoints: logKow, logKoc, pKa, water_solubility, Henry,
               vapor_pressure, logKoa, half_life

关键设计
--------
1. 掩码多任务损失：金标签是 long 格式、稀疏（不是每个分子都有全部 8 个 endpoint），
   缺失位置用 mask=0 跳过，避免把 NaN 当成 0 来监督。
2. MC-dropout 不确定性：预测时保持 dropout 打开，跑 T 次前向，输出 mean + std。
   std 就是给模型2用的"transport 预测不确定性"——级联误差的显式度量。
3. backbone 可冻结：金标签只有 25 PFAS，fine-tune 时可只动 head（freeze_backbone=True），
   或低学习率微调全网，二选一防过拟合。

依赖：torch
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MultiTaskTransportNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        endpoints: list[str],
        hidden: tuple[int, ...] = (512, 256, 128),
        head_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.endpoints = list(endpoints)

        # ---- 共享 backbone ----
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.backbone_out = prev

        # ---- 每个 endpoint 一个 head ----
        self.heads = nn.ModuleDict({
            ep: nn.Sequential(
                nn.Linear(prev, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, 1),
            )
            for ep in self.endpoints
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 [B, K]，K=len(endpoints)，列顺序与 self.endpoints 一致。"""
        z = self.backbone(x)
        return torch.cat([self.heads[ep](z) for ep in self.endpoints], dim=1)

    # ----- backbone 冻结 / 解冻 -----
    def set_backbone_trainable(self, flag: bool):
        for p in self.backbone.parameters():
            p.requires_grad = flag


def masked_multitask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    pred/target/mask : [B, K]    mask=1 表示该位置有观测值
    weights          : [K]       各 endpoint 的损失权重（来自 missing_priority）
    先按 endpoint 求观测样本上的均方误差，再做加权平均。
    """
    se = (pred - target) ** 2 * mask
    denom = mask.sum(dim=0).clamp(min=1.0)
    per_ep = se.sum(dim=0) / denom            # [K]
    w = weights / weights.sum().clamp(min=1e-8)
    return (per_ep * w).sum()


# --------------------------------------------------------------------------- #
# MC-dropout 预测
# --------------------------------------------------------------------------- #
def _enable_dropout(model: nn.Module):
    """只把 Dropout 层切回 train 模式，BatchNorm 等保持 eval。"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(
    model: MultiTaskTransportNet,
    X: torch.Tensor,
    n_samples: int = 30,
    batch_size: int = 4096,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (mean, std)，形状均为 [N, K]（标准化空间，反标准化在 train 脚本里做）。
    std 即 MC-dropout 认知不确定性，给模型2当特征 / 样本权重用。
    """
    model.eval()
    _enable_dropout(model)
    N = X.shape[0]
    K = len(model.endpoints)
    acc_sum = np.zeros((N, K), dtype=np.float64)
    acc_sq = np.zeros((N, K), dtype=np.float64)

    for _ in range(n_samples):
        out = np.empty((N, K), dtype=np.float32)
        for s in range(0, N, batch_size):
            xb = X[s : s + batch_size].to(device)
            out[s : s + batch_size] = model(xb).cpu().numpy()
        acc_sum += out
        acc_sq += out ** 2

    mean = acc_sum / n_samples
    var = np.clip(acc_sq / n_samples - mean ** 2, 0.0, None)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)
