"""EdgeAD 组件自检（不依赖 pytest，可直接运行）。

覆盖：
* U-Net 输入输出形状；
* Canny 边缘图取值与形状；
* 严格 RBI 掩码的「互不相交 + 完整覆盖」性质（多种 K / 非整除尺寸）；
* 边缘加权损失是否有限、是否可反传。

运行::

    conda activate env01
    python tests/test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import UNet, EdgeWeightedLoss, canny_edge, make_rbi_masks  # noqa: E402


def test_unet_shape() -> None:
    net = UNet(in_ch=3, out_ch=3, base=8)
    x = torch.rand(2, 3, 64, 64)
    y = net(x).detach()
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_canny() -> None:
    img = torch.zeros(1, 3, 64, 64)
    img[:, :, :, 32:] = 1.0
    edge = canny_edge(img)
    assert edge.shape == (1, 1, 64, 64)
    assert set(edge.unique().tolist()) <= {0.0, 1.0}
    assert float(edge.sum()) > 0.0


def test_rbi_masks_cover_and_disjoint() -> None:
    for height, width in [(64, 64), (60, 76), (256, 256)]:
        for k in (8, 16, 7):
            masks = make_rbi_masks(height, width, batch_size=3, k=k, n=3)
            assert len(masks) == 3
            stack = torch.stack(masks)
            coverage = stack.sum(dim=0)
            # 每个像素恰好被一个子集覆盖
            assert float(coverage.min()) == 1.0
            assert float(coverage.max()) == 1.0
            # 两两不相交
            for i in range(3):
                for j in range(i + 1, 3):
                    assert float((masks[i] * masks[j]).max()) == 0.0


def test_loss_backward() -> None:
    torch.manual_seed(0)
    target = torch.rand(2, 3, 64, 64)
    recon = (target + 0.05 * torch.randn_like(target)).clamp(0, 1).requires_grad_(True)
    edge = canny_edge(target)
    criterion = EdgeWeightedLoss(lam_g=1.0, lam_s=1.0)
    loss, parts = criterion(recon, target, edge)
    loss.backward()
    assert torch.isfinite(loss)
    assert recon.grad is not None and torch.isfinite(recon.grad).all()
    assert set(parts) == {"edge", "gmsd", "ssim"}


if __name__ == "__main__":
    test_unet_shape()
    test_canny()
    test_rbi_masks_cover_and_disjoint()
    test_loss_backward()
    print("All EdgeAD component tests passed.")
