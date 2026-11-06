"""Generate qualitative EdgeAD anomaly-map previews from a trained checkpoint.

Usage::

    python scripts/visualize_gms.py \
        --checkpoint outputs/EdgeADLightning/MVTecAD/bottle/v0/weights/lightning/model.ckpt \
        --category bottle --num-images 6

Each output image is a 4-panel strip: input, ground-truth mask, anomaly
heat-map overlay and the normalized anomaly map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomalib.data import MVTecAD  # noqa: E402

from lightning_model import EdgeADLightning  # noqa: E402
from model import anomaly_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render EdgeAD anomaly-map previews")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets" / "mvtec")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "qualitative")
    parser.add_argument("--score-mode", choices=["gms", "l1", "ssim"], default="gms")
    parser.add_argument("--num-images", type=int, default=6)
    return parser.parse_args()


def _title(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 22), (30, 30, 30), -1)
    cv2.putText(image, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = EdgeADLightning.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.score_mode = args.score_mode
    model.eval()
    k = model.k_values[0]

    datamodule = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )
    datamodule.setup()
    resize = T.Resize((256, 256), antialias=True)

    saved = 0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            images = resize(batch.image)
            recon = model._rbi_reconstruct_k(images, k)
            amaps = anomaly_map(recon, images, mode=args.score_mode)
            gts = F.interpolate(batch.gt_mask.float().unsqueeze(1), size=(256, 256), mode="nearest").squeeze(1)

            for i in range(images.shape[0]):
                if saved >= args.num_images:
                    break
                image_u8 = (np.clip(images[i].permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
                gt_u8 = (gts[i].numpy() * 255).astype(np.uint8)
                amap = amaps[i, 0].numpy()
                norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

                heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(image_u8, 0.55, heat, 0.45, 0)
                panels = [
                    _title(image_u8, f"image label={int(batch.gt_label[i].item())}"),
                    _title(cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR), "gt mask"),
                    _title(overlay, f"{args.score_mode.upper()} anomaly map"),
                    _title(cv2.cvtColor((norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), "normalized map"),
                ]
                path = args.output_dir / f"preview_{saved:02d}.png"
                cv2.imwrite(str(path), np.concatenate(panels, axis=1))
                print(f"saved {path}")
                saved += 1
            if saved >= args.num_images:
                break


if __name__ == "__main__":
    main()
