"""EdgeAD 的 ONNX 导出与推理脚本。

依赖 env01 中已安装的 ``onnx`` / ``onnxruntime`` / ``onnxscript``，不需要改动
conda 环境。

导出：

    conda activate env01
    python export_onnx.py export \
        --checkpoint outputs/EdgeADLightning/MVTecAD/bottle/v0/weights/lightning/model.ckpt \
        --output outputs/edgead.onnx --verify

推理：

    python export_onnx.py infer --onnx outputs/edgead.onnx \
        --image datasets/mvtec/bottle/test/broken_large/000.png --save-dir outputs/onnx_pred

设计说明
--------
严格 RBI 的掩码在训练/推理时是随机划分的。ONNX 图里无法方便地表达“每次随机
生成互不相交的 n 个子集”，因此导出时把掩码**按固定种子烘焙成常量**：

* 每个 K 生成一组覆盖全图的 n 个掩码，写进 ONNX initializer；
* 导出后模型完全确定：同一输入永远得到同一输出，便于部署复现；
* PyTorch 端 ``EdgeADOnnxWrapper`` 使用相同掩码，所以可以逐元素对比验证。

输入  ``image``       float32 ``[B, 3, 256, 256]``，范围 ``[0, 1]``（RGB）
输出  ``anomaly_map`` float32 ``[B, 1, 256, 256]``
      ``pred_score``  float32 ``[B]``（异常图最大值）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lightning_model import EdgeADLightning  # noqa: E402
from model import anomaly_map as compute_anomaly_map  # noqa: E402
from model import make_rbi_masks  # noqa: E402


# --------------------------------------------------------------------------- #
# 导出图：把严格 RBI + 异常图打包成可导出的 nn.Module
# --------------------------------------------------------------------------- #
class EdgeADOnnxWrapper(nn.Module):
    """把 ``EdgeADLightning`` 的推理流程封装成单输入、双输出的 ONNX 图。

    Args:
        model: 已加载的 EdgeAD Lightning 模块（eval 模式）。
        mask_seed: 烘焙掩码的随机种子；导出与验证必须使用同一个种子。
    """

    def __init__(self, model: EdgeADLightning, mask_seed: int = 0) -> None:
        super().__init__()
        self.unet = model.model.eval()
        self.k_values = tuple(int(k) for k in model.k_values)
        self.n = int(model.n)
        self.score_mode = model.score_mode
        self.image_size = (int(model.image_size[0]), int(model.image_size[1]))

        for idx, k in enumerate(self.k_values):
            generator = torch.Generator().manual_seed(mask_seed + 1000 * idx)
            masks = make_rbi_masks(
                height=self.image_size[0],
                width=self.image_size[1],
                batch_size=1,
                k=k,
                n=self.n,
                generator=generator,
            )
            # (n, 1, H, W)，广播到任意 batch
            self.register_buffer(f"masks_{idx}", torch.cat(masks, dim=0), persistent=False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = image.shape[0]
        maps = []
        for idx in range(len(self.k_values)):
            masks = getattr(self, f"masks_{idx}")
            # n 个子集一次前向
            masked = torch.cat([image * (1.0 - masks[j]) for j in range(self.n)], dim=0)
            preds = self.unet(masked)

            recon = torch.zeros_like(image)
            covered = torch.zeros_like(image[:, :1])
            for j in range(self.n):
                recon = recon + preds[j * batch : (j + 1) * batch] * masks[j]
                covered = covered + masks[j]
            recon = recon + image * (1.0 - covered.clamp(max=1.0))
            maps.append(compute_anomaly_map(recon, image, mode=self.score_mode))

        anomaly_map = torch.stack(maps, dim=0).mean(dim=0)
        pred_score = anomaly_map.amax(dim=(-2, -1)).flatten()
        return anomaly_map, pred_score


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def load_model(checkpoint: Path | str, score_mode: str | None = None) -> EdgeADLightning:
    """从 checkpoint 加载 EdgeAD（CPU）。"""
    model = EdgeADLightning.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.eval()
    if score_mode is not None:
        model.score_mode = score_mode
    return model


def export_onnx(
    checkpoint: Path | str,
    output: Path | str,
    mask_seed: int = 0,
    opset: int = 17,
    score_mode: str | None = None,
) -> Path:
    """把 EdgeAD 导出为 ONNX，并在同名 ``.json`` 中写元数据。

    Returns:
        生成的 ONNX 文件路径。
    """
    import onnx

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint, score_mode=score_mode)
    wrapper = EdgeADOnnxWrapper(model, mask_seed=mask_seed).eval()
    dummy = torch.rand(1, 3, *wrapper.image_size, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["anomaly_map", "pred_score"],
            dynamic_axes={
                "image": {0: "batch"},
                "anomaly_map": {0: "batch"},
                "pred_score": {0: "batch"},
            },
            dynamo=False,
        )

    onnx.checker.check_model(str(output))

    thresholds = {}
    post_processor = getattr(model, "post_processor", None)
    if post_processor is not None:
        for name in ("image_threshold", "pixel_threshold"):
            value = getattr(post_processor, name, None)
            if value is not None:
                value = float(value)
                if math.isfinite(value):
                    thresholds[name] = value

    metadata = {
        "model": "EdgeAD",
        "checkpoint": str(checkpoint),
        "image_size": list(wrapper.image_size),
        "k_values": list(wrapper.k_values),
        "n": wrapper.n,
        "score_mode": wrapper.score_mode,
        "mask_seed": mask_seed,
        "opset": opset,
        "input": {"name": "image", "shape": ["B", 3, *wrapper.image_size], "range": [0.0, 1.0]},
        "outputs": {
            "anomaly_map": ["B", 1, *wrapper.image_size],
            "pred_score": ["B"],
        },
        "thresholds": thresholds,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ONNX saved: {output}  ({output.stat().st_size / 1e6:.2f} MB)")
    print(f"metadata  : {output.with_suffix('.json')}")
    return output


def verify_onnx(
    checkpoint: Path | str,
    onnx_path: Path | str,
    mask_seed: int = 0,
    score_mode: str | None = None,
) -> None:
    """对比 PyTorch wrapper 与 ONNX Runtime 在随机输入上的输出。"""
    import onnxruntime as ort

    model = load_model(checkpoint, score_mode=score_mode)
    wrapper = EdgeADOnnxWrapper(model, mask_seed=mask_seed).eval()

    x = torch.rand(2, 3, *wrapper.image_size, dtype=torch.float32)
    with torch.no_grad():
        torch_map, torch_score = wrapper(x)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_map, onnx_score = session.run(["anomaly_map", "pred_score"], {input_name: x.numpy()})

    map_diff = float(np.abs(torch_map.numpy() - onnx_map).max())
    score_diff = float(np.abs(torch_score.numpy() - onnx_score).max())
    print(f"verify batch=2  anomaly_map {tuple(onnx_map.shape)}  pred_score {tuple(onnx_score.shape)}")
    print(f"max |Δanomaly_map| = {map_diff:.3e}")
    print(f"max |Δpred_score|  = {score_diff:.3e}")
    if map_diff > 1e-4 or score_diff > 1e-4:
        raise RuntimeError("ONNX 与 PyTorch 输出差异过大，请检查导出设置")
    print("verify OK")


# --------------------------------------------------------------------------- #
# ONNX Runtime 推理
# --------------------------------------------------------------------------- #
class OnnxEdgeAD:
    """ONNX Runtime 推理封装。

    Args:
        onnx_path: 导出的 ``.onnx`` 文件。
        providers: onnxruntime providers，默认 CPU。
    """

    def __init__(self, onnx_path: Path | str, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        meta_path = self.onnx_path.with_suffix(".json")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.image_size = tuple(int(v) for v in self.metadata.get("image_size", (256, 256)))

    # ---- 预处理 / 推理 -------------------------------------------------- #
    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 HWC -> float32 [1,3,H,W]，RGB 且保持 [0,1]。"""
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        array = resized.astype(np.float32) / 255.0
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None])

    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """输入 ``[B,3,H,W]`` float32，返回 ``anomaly_map [B,1,H,W]`` 与 ``pred_score [B]``。"""
        anomaly_map, pred_score = self.session.run(self.output_names, {self.input_name: images.astype(np.float32)})
        return anomaly_map, pred_score

    def predict_path(self, image_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取单张图片并推理，返回 ``(anomaly_map[H,W], score, 原始 BGR 图)``。"""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"无法读取图片: {image_path}"
            raise FileNotFoundError(msg)
        anomaly_map, pred_score = self.predict(self.preprocess(image))
        return anomaly_map[0, 0], float(pred_score[0]), image


def save_visualization(
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    score: float,
    out_path: Path | str,
    threshold: float | None = None,
) -> Path:
    """保存三联图：原图 / 异常热力图叠加 / 归一化异常图。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original = cv2.resize(image_bgr, (anomaly_map.shape[1], anomaly_map.shape[0]))
    normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.55, heat, 0.45, 0)

    map_u8 = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if threshold is not None:
        mask = (normalized >= threshold).astype(np.uint8) * 255
        map_u8 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def title(image: np.ndarray, text: str) -> np.ndarray:
        image = image.copy()
        cv2.rectangle(image, (0, 0), (image.shape[1], 22), (30, 30, 30), -1)
        cv2.putText(image, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    panels = [
        title(original, f"image score={score:.4f}"),
        title(overlay, "ONNX anomaly map"),
        title(map_u8, "thresholded map" if threshold is not None else "normalized map"),
    ]
    cv2.imwrite(str(out_path), np.concatenate(panels, axis=1))
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EdgeAD ONNX export & inference")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="导出 EdgeAD 到 ONNX")
    export.add_argument("--checkpoint", type=Path, required=True, help="Lightning checkpoint 路径")
    export.add_argument("--output", type=Path, default=Path("outputs/edgead.onnx"), help="输出 .onnx 路径")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--mask-seed", type=int, default=0, help="烘焙 RBI 掩码的随机种子")
    export.add_argument("--score-mode", choices=["gms", "l1", "ssim"], default=None, help="覆盖 checkpoint 中的评分方式")
    export.add_argument("--verify", action="store_true", help="导出后用 onnxruntime 对比 PyTorch 输出")

    infer = sub.add_parser("infer", help="用 ONNX Runtime 推理")
    infer.add_argument("--onnx", type=Path, required=True, help="导出的 .onnx 路径")
    infer.add_argument("--image", type=Path, nargs="+", required=True, help="一张或多张图片")
    infer.add_argument("--save-dir", type=Path, default=Path("outputs/onnx_pred"), help="可视化输出目录")
    infer.add_argument("--threshold", type=float, default=None, help="可选：用于阈值化异常图的归一化阈值")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "export":
        output = export_onnx(
            checkpoint=args.checkpoint,
            output=args.output,
            mask_seed=args.mask_seed,
            opset=args.opset,
            score_mode=args.score_mode,
        )
        if args.verify:
            verify_onnx(args.checkpoint, output, mask_seed=args.mask_seed, score_mode=args.score_mode)
        return

    detector = OnnxEdgeAD(args.onnx)
    print(f"ONNX: {args.onnx}")
    print(f"input={detector.input_name} {detector.image_size}  outputs={detector.output_names}")
    for image_path in args.image:
        anomaly_map, score, image_bgr = detector.predict_path(image_path)
        print(f"{image_path}: score={score:.6f}  map={tuple(anomaly_map.shape)}  max={anomaly_map.max():.6f}")
        # 用父目录名区分同名文件（如 broken_large/000.png 与 good/000.png）
        stem = f"{Path(image_path).parent.name}_{Path(image_path).stem}_onnx.png"
        out_path = args.save_dir / stem
        save_visualization(image_bgr, anomaly_map, score, out_path, threshold=args.threshold)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
