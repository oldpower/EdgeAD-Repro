"""EdgeAD 的 anomalib / PyTorch Lightning 封装。

设计要点（对应论文与 PRD）：

* 只使用正常样本训练（``LearningType.ONE_CLASS``）；
* 训练与推理都走**严格 RBI**：把图像划分成 ``k x k`` 块，随机分成 ``n`` 个
  互不相交子集，每次遮挡一个子集送入 U-Net，最后只用各次重建中“被遮挡区域”
  的像素拼出完整重建图 ``I_Y``；
* 多 ``K``（论文最优 ``(8, 16)``）分别重建，损失/异常图在多个 ``K`` 上取平均；
* 损失为论文 Eq.5 的边缘加权损失（``L_E + λ_G·L_G + λ_S·L_S``）；
* 推理时异常图取 ``|I_X - I_Y|`` 的通道平均，图像分数取异常图最大值，
  后续由 anomalib 的 PostProcessor / Evaluator 完成归一化、阈值和指标计算。
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchvision.transforms.v2 import Compose, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.models.components import AnomalibModule
from anomalib.pre_processing import PreProcessor

from model import UNet, EdgeWeightedLoss, anomaly_map, canny_edge, make_rbi_masks

__all__ = ["EdgeADLightning"]


class EdgeADLightning(AnomalibModule):
    """EdgeAD 的 Lightning 模块。

    Args:
        in_ch: 输入/输出通道数（MVTec RGB 为 3）。
        base: U-Net 第一层通道数。
        k_values: RBI 像素块边长集合，论文最优 ``(8, 16)``。
        n: RBI 子集个数，论文 n=3。
        th1: Canny 低阈值，论文最优 100。
        th2: Canny 高阈值，论文最优 400。
        lam_g: GMSD 损失权重，论文最优 1.0。
        lam_s: SSIM 损失权重，论文最优 1.0。
        lr: 学习率，论文 1e-4。
        image_size: 网络输入尺寸，论文 256x256。
        score_mode: 异常图定义，`gms`（默认）/ `l1` / `ssim`，见 :func:model.anomaly_map。
        pre_processor / post_processor / evaluator / visualizer:
            透传给 :class:`~anomalib.models.components.AnomalibModule`。
    """

    def __init__(
        self,
        in_ch: int = 3,
        base: int = 64,
        k_values: tuple[int, ...] = (8, 16),
        n: int = 3,
        th1: int = 100,
        th2: int = 400,
        lam_g: float = 1.0,
        lam_s: float = 1.0,
        lr: float = 1e-4,
        image_size: tuple[int, int] = (256, 256),
        score_mode: str = "gms",
        pre_processor: PreProcessor | bool = True,
        post_processor: Any = True,
        evaluator: Any = True,
        visualizer: Any = True,
    ) -> None:
        if isinstance(pre_processor, bool) and pre_processor:
            # EdgeAD 的输入必须保持在 [0,1]：U-Net 输出经过 Sigmoid，
            # Canny 阈值也按 0~255 的像素定义，因此不做 ImageNet 归一化。
            pre_processor = self.configure_pre_processor(image_size)

        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = UNet(in_ch=in_ch, out_ch=in_ch, base=base)
        self.loss_fn = EdgeWeightedLoss(lam_g=lam_g, lam_s=lam_s)
        self.k_values = tuple(int(k) for k in k_values)
        self.n = int(n)
        self.th1 = int(th1)
        self.th2 = int(th2)
        self.lr = float(lr)
        self.image_size = tuple(image_size)
        self.score_mode = score_mode

        # 手动优化：按 K 逐个反传，避免同时保留多个 K 的计算图
        self.automatic_optimization = False

    # ------------------------------------------------------------------ #
    # anomalib 组件配置
    # ------------------------------------------------------------------ #
    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """只做 Resize，不做归一化（保持 [0,1]），与 Canny / Sigmoid 输出匹配。"""
        image_size = image_size or (256, 256)
        return PreProcessor(transform=Compose([Resize(image_size, antialias=True)]))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @property
    def learning_type(self) -> LearningType:
        """无监督单类异常检测，只用正常样本训练。"""
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        # 训练前不做 sanity validation，节省 CPU 训练时间
        return {"num_sanity_val_steps": 0}

    # ------------------------------------------------------------------ #
    # 严格 RBI
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _rbi_masks(self, images: Tensor, k: int) -> list[Tensor]:
        return make_rbi_masks(
            height=images.shape[-2],
            width=images.shape[-1],
            batch_size=images.shape[0],
            k=k,
            n=self.n,
            device=images.device,
            dtype=images.dtype,
        )

    def _rbi_reconstruct_k(self, images: Tensor, k: int) -> Tensor:
        """对单个 ``K`` 做严格 RBI 重建。

        把 ``n`` 个子集的遮挡图拼成一个 batch 一次前向，等价于逐个子集前向，
        但在 CPU 上批量化能明显提升吞吐。
        """
        masks = self._rbi_masks(images, k)
        batch = images.shape[0]
        masked = torch.cat([images * (1.0 - mask) for mask in masks], dim=0)
        preds = self.model(masked)

        recon = torch.zeros_like(images)
        covered = torch.zeros_like(images[:, :1])
        for i, mask in enumerate(masks):
            recon = recon + preds[i * batch : (i + 1) * batch] * mask
            covered = covered + mask
        # 未被遮挡的像素直接使用原图（与论文“其余区域保持不变”一致）
        return recon + images * (1.0 - covered.clamp(max=1.0))

    def _rbi_reconstruct(self, images: Tensor) -> list[Tensor]:
        """对每个 ``K`` 做一次严格 RBI 重建，返回完整重建图列表。"""
        return [self._rbi_reconstruct_k(images, k) for k in self.k_values]

    @torch.no_grad()
    def predict_anomaly_map(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """推理：多 K RBI 重建 -> 平均异常图 -> 图像级分数。

        Returns:
            ``anomaly_map``: ``(B, 1, H, W)``。
            ``pred_score``: ``(B,)``。
        """
        maps = [anomaly_map(recon, images, mode=self.score_mode) for recon in self._rbi_reconstruct(images)]
        merged = torch.stack(maps, dim=0).mean(dim=0)
        pred_score = merged.amax(dim=(-2, -1)).flatten()
        return merged, pred_score

    # ------------------------------------------------------------------ #
    # Lightning steps
    # ------------------------------------------------------------------ #
    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """对一个 batch 做多 K、多子集的严格 RBI 训练。"""
        del batch_idx
        images = batch.image
        edge = canny_edge(images, self.th1, self.th2)

        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        optimizer.zero_grad(set_to_none=True)

        num_k = len(self.k_values)
        total: float = 0.0
        parts_sum = {"edge": 0.0, "gmsd": 0.0, "ssim": 0.0}

        for k in self.k_values:
            recon = self._rbi_reconstruct_k(images, k)
            loss, parts = self.loss_fn(recon, images, edge)
            self.manual_backward(loss / num_k)
            total += loss.detach().item()
            for name in parts_sum:
                parts_sum[name] += parts[name] / num_k

        optimizer.step()

        loss_mean = total / num_k
        self.log("train_loss", loss_mean, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=images.shape[0])
        self.log("train_edge", parts_sum["edge"], on_step=False, on_epoch=True, logger=False, batch_size=images.shape[0])
        self.log("train_gmsd", parts_sum["gmsd"], on_step=False, on_epoch=True, logger=False, batch_size=images.shape[0])
        self.log("train_ssim", parts_sum["ssim"], on_step=False, on_epoch=True, logger=False, batch_size=images.shape[0])
        return {"loss": torch.tensor(loss_mean, device=images.device)}

    def _predict_batch(self, batch: Batch) -> STEP_OUTPUT:
        anomaly_map, pred_score = self.predict_anomaly_map(batch.image)
        return batch.update(anomaly_map=anomaly_map, pred_score=pred_score)

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        del batch_idx
        return self._predict_batch(batch)

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        del batch_idx
        return self._predict_batch(batch)

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        del batch_idx, dataloader_idx
        return self._predict_batch(batch)
