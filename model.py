"""EdgeAD 纯 PyTorch 组件。

对应论文 *EdgeAD: Unsupervised Learning Model Based on Prior Knowledge Enhanced
Image Anomaly Detection of Heavy Railway Freight Cars* (IEEE TIM 2025) 的
Sec. II-B ~ II-E：

* ``UNet``：论文 Fig.5 的对称 U-Net 主干（编码 3 次下采样，解码带 skip）；
* ``canny_edge``：Canny 边缘提取模块（TH1=100, TH2=400），边缘像素权重为 1；
* ``make_rbi_masks``：RBI（reconstruction-by-inpainting）严格掩码划分，
  把 ``k x k`` 像素块随机分成 ``n`` 个互不相交的子集；
* ``EdgeWeightedLoss``：论文 Eq.5，``L = L_E + λ_G·L_G + λ_S·L_S``，
  其中 ``L_E`` 为边缘加权 SmoothL1（Eq.2），``L_G`` 为 GMSD（Eq.3），
  ``L_S`` 为 SSIM（Eq.4）。
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "UNet",
    "canny_edge",
    "make_rbi_masks",
    "smooth_l1_map",
    "gms_map",
    "gmsd_loss",
    "ssim_map",
    "ssim_loss",
    "anomaly_map",
    "EdgeWeightedLoss",
]


# --------------------------------------------------------------------------- #
# 边缘信息提取（论文 Sec. II-C）
# --------------------------------------------------------------------------- #
def canny_edge(x: torch.Tensor, th1: int = 100, th2: int = 400) -> torch.Tensor:
    """用 Canny 算子提取边缘权重图。

    Args:
        x: ``(B, C, H, W)``，取值范围 ``[0, 1]``（PreProcessor 不做 ImageNet 归一化）。
        th1: Canny 低阈值，论文最优值 100。
        th2: Canny 高阈值，论文最优值 400。

    Returns:
        ``(B, 1, H, W)`` 的 0/1 权重图，边缘像素为 1，其余为 0。
    """
    if x.ndim != 4:
        msg = f"expected (B, C, H, W) input, got {tuple(x.shape)}"
        raise ValueError(msg)

    with torch.no_grad():
        # 论文中的 TFDS 图像是灰度图；MVTec 等彩色数据先按 ITU-R BT.601 转灰度
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        elif x.shape[1] > 3:
            gray = x[:, :3].mean(dim=1)
        else:
            gray = x[:, 0]

        gray_np = (gray.detach().float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        edges = np.stack([cv2.Canny(img, th1, th2) for img in gray_np])
        edge = torch.from_numpy(edges).to(dtype=x.dtype, device=x.device).unsqueeze(1) / 255.0
    return edge


# --------------------------------------------------------------------------- #
# RBI 掩码（论文 Sec. II-E）
# --------------------------------------------------------------------------- #
def make_rbi_masks(
    height: int,
    width: int,
    batch_size: int,
    k: int,
    n: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """生成一组严格 RBI 掩码。

    把 ``H x W`` 图像按 ``k x k`` 的块划分成网格，再对每个样本随机地把网格块
    分配给 ``n`` 个互不相交的子集；返回 ``n`` 个 ``(B, 1, H, W)`` 的 0/1 掩码，
    它们的并集覆盖整幅图像且两两不重叠。

    Args:
        height: 图像高。
        width: 图像宽。
        batch_size: batch 大小。
        k: 像素块边长（论文中取 8、16）。
        n: 子集个数（论文中 n=3）。
        device: 掩码所在设备。
        dtype: 掩码数据类型。
        generator: 可选随机数生成器，便于复现实验。

    Returns:
        长度为 ``n`` 的掩码列表，每个元素形如 ``(B, 1, H, W)``。
    """
    if k <= 0 or n <= 0:
        msg = f"k and n must be positive, got k={k}, n={n}"
        raise ValueError(msg)

    grid_h = math.ceil(height / k)
    grid_w = math.ceil(width / k)
    size = (batch_size, 1, grid_h, grid_w)

    if generator is not None:
        sub = torch.randint(0, n, size, generator=generator, device=device)
    else:
        sub = torch.randint(0, n, size, device=device)

    masks: list[torch.Tensor] = []
    for s in range(n):
        mask = (sub == s).to(dtype=dtype)
        # 最近邻上采样到网格实际像素范围，再裁剪回 H x W
        mask = F.interpolate(mask, size=(grid_h * k, grid_w * k), mode="nearest")
        masks.append(mask[:, :, :height, :width].contiguous())
    return masks


# --------------------------------------------------------------------------- #
# U-Net 主干（论文 Fig.5）
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    """``Conv3x3-BN-ReLU`` 堆叠两次。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """对称 U-Net：3 次下采样，256x256 输入 -> 32x32 bottleneck。

    Args:
        in_ch: 输入通道数。
        out_ch: 输出通道数（与原图一致，用于重建）。
        base: 第一层通道数；论文未给出具体宽度，默认 64（标准 U-Net）。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 64) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)  # H
        self.enc2 = DoubleConv(base, base * 2)  # H/2
        self.enc3 = DoubleConv(base * 2, base * 4)  # H/4
        self.bottleneck = DoubleConv(base * 4, base * 8)  # H/8
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


# --------------------------------------------------------------------------- #
# 边缘加权损失（论文 Sec. II-D，Eq.1 ~ Eq.5）
# --------------------------------------------------------------------------- #
def smooth_l1_map(a: torch.Tensor, b: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """逐像素 SmoothL1（论文 Eq.1，β=1）。"""
    diff = (a - b).abs()
    return torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """逐通道 Sobel 梯度幅值。

    用 ``F.pad`` + 切片实现而不是 ``F.conv2d``，既等价又对 ONNX 导出友好
    （避免动态构造卷积核）。
    """
    padded = F.pad(x, (1, 1, 1, 1))
    top = padded[:, :, :-2, :]
    mid = padded[:, :, 1:-1, :]
    bot = padded[:, :, 2:, :]

    gx = (top[:, :, :, 2:] + 2.0 * mid[:, :, :, 2:] + bot[:, :, :, 2:]) - (
        top[:, :, :, :-2] + 2.0 * mid[:, :, :, :-2] + bot[:, :, :, :-2]
    )
    gy = (bot[:, :, :, :-2] + 2.0 * bot[:, :, :, 1:-1] + bot[:, :, :, 2:]) - (
        top[:, :, :, :-2] + 2.0 * top[:, :, :, 1:-1] + top[:, :, :, 2:]
    )
    return torch.sqrt(gx * gx + gy * gy + 1e-8) / 8.0


def gms_map(a: torch.Tensor, b: torch.Tensor, c: float = 0.0026) -> torch.Tensor:
    """逐像素梯度幅值相似度（GMS），取值 ``(0, 1]``。"""
    ga = _gradient_magnitude(a)
    gb = _gradient_magnitude(b)
    return (2.0 * ga * gb + c) / (ga * ga + gb * gb + c)


def gmsd_loss(a: torch.Tensor, b: torch.Tensor, c: float = 0.0026) -> torch.Tensor:
    """Gradient Magnitude Similarity Deviation loss（论文 Eq.3）。

    ``L_G = 1 - mean( (2·g_a·g_b + c) / (g_a² + g_b² + c) )``。
    ``c`` 为 GMSD 原文的稳定常数 0.0026。
    """
    return 1.0 - gms_map(a, b, c).mean()


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim_map(a: torch.Tensor, b: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """逐像素 SSIM 图（论文 Eq.4 的被平均项）。"""
    channels = a.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, a.device, a.dtype)
    pad = window_size // 2

    mu_a = F.conv2d(a, kernel, padding=pad, groups=channels)
    mu_b = F.conv2d(b, kernel, padding=pad, groups=channels)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = F.conv2d(a * a, kernel, padding=pad, groups=channels) - mu_a_sq
    sigma_b_sq = F.conv2d(b * b, kernel, padding=pad, groups=channels) - mu_b_sq
    sigma_ab = F.conv2d(a * b, kernel, padding=pad, groups=channels) - mu_ab

    c1 = 0.01**2
    c2 = 0.03**2
    return ((2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)) / (
        (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2) + 1e-12
    )


def ssim_loss(a: torch.Tensor, b: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM loss（论文 Eq.4）：``1 - mean SSIM``，11x11 高斯窗（σ=1.5）。"""
    return 1.0 - ssim_map(a, b, window_size, sigma).mean()


def anomaly_map(recon: torch.Tensor, target: torch.Tensor, mode: str = "gms") -> torch.Tensor:
    """由重建图与输入图计算 ``(B, 1, H, W)`` 异常图。

    Args:
        recon: 重建图 ``I_Y``。
        target: 输入图 ``I_X``。
        mode:
            * ``"gms"``（默认）：``1 - GMS``，即梯度幅值相似度偏差。EdgeAD 的
              训练目标本身由边缘/GMSD/SSIM 构成，用梯度结构差异做评分对
              局部缺陷最敏感（见 README 消融）；
            * ``"l1"``：``|I_X - I_Y|`` 的通道平均，即论文所述的重建差异；
            * ``"ssim"``：``1 - SSIM`` 图。
    """
    if mode == "gms":
        return 1.0 - gms_map(recon, target).mean(dim=1, keepdim=True)
    if mode == "l1":
        return (recon - target).abs().mean(dim=1, keepdim=True)
    if mode == "ssim":
        return 1.0 - ssim_map(recon, target).mean(dim=1, keepdim=True)
    msg = f"unknown anomaly map mode: {mode!r} (expected 'gms', 'l1' or 'ssim')"
    raise ValueError(msg)


class EdgeWeightedLoss(nn.Module):
    """论文 Eq.5 的联合损失。

    Args:
        lam_g: GMSD 权重 ``λ_G``，论文最优值 1.0。
        lam_s: SSIM 权重 ``λ_S``，论文最优值 1.0。
    """

    def __init__(self, lam_g: float = 1.0, lam_s: float = 1.0) -> None:
        super().__init__()
        self.lam_g = lam_g
        self.lam_s = lam_s

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        edge: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # L_E: 只在边缘像素上计算 SmoothL1，再除以 W*H（论文 Eq.2）
        smooth = smooth_l1_map(recon, target).mean(dim=1, keepdim=True)
        l_e = (edge * smooth).mean()
        l_g = gmsd_loss(recon, target)
        l_s = ssim_loss(recon, target)
        total = l_e + self.lam_g * l_g + self.lam_s * l_s
        parts = {"edge": l_e.item(), "gmsd": l_g.item(), "ssim": l_s.item()}
        return total, parts
