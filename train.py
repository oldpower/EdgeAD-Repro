"""EdgeAD 训练 / 测试 / 推理入口（基于 anomalib 2.7 + PyTorch Lightning）。

用法示例::

    conda activate env01
    python train.py --category bottle --epochs 300 --patience 40 --base 64

CPU 环境下的快速验证::

    python train.py --category bottle --epochs 5 --base 16 --k 8 --n 3 --batch-size 4

恢复/复现实验时固定随机种子即可；训练产物默认写入 ``outputs/EdgeADLightning/<category>``。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="anomalib.*")

import torch  # noqa: E402  (在 warnings 过滤之后导入)
from lightning.pytorch.callbacks import EarlyStopping  # noqa: E402
from lightning.pytorch import seed_everything  # noqa: E402

from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402

from lightning_model import EdgeADLightning  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EdgeAD: prior-knowledge enhanced unsupervised anomaly detection")
    # 数据
    parser.add_argument("--data-root", type=Path, default=Path("./datasets/mvtec"), help="MVTec AD 根目录")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD 类别")
    parser.add_argument("--batch-size", type=int, default=8, help="训练 batch size")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="验证/测试 batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（Windows 建议 0）")
    # 模型 / 论文超参
    parser.add_argument("--base", type=int, default=64, help="U-Net 第一层通道数")
    parser.add_argument("--k", type=int, nargs="+", default=[8, 16], help="RBI 像素块边长 K，论文最优 8 16")
    parser.add_argument("--n", type=int, default=3, help="RBI 子集个数，论文 n=3")
    parser.add_argument("--th1", type=int, default=100, help="Canny 低阈值")
    parser.add_argument("--th2", type=int, default=400, help="Canny 高阈值")
    parser.add_argument("--lam-g", type=float, default=1.0, help="GMSD 损失权重 λ_G")
    parser.add_argument("--lam-s", type=float, default=1.0, help="SSIM 损失权重 λ_S")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--image-size", type=int, default=256, help="输入边长（论文 256）")
    parser.add_argument("--score-mode", choices=["gms", "l1", "ssim"], default="gms",
                        help="异常图定义：gms=梯度结构偏差（默认/推荐），l1=重建强度差，ssim=结构相似度差")
    # 训练控制
    parser.add_argument("--epochs", type=int, default=300, help="最大 epoch 数（论文 300）")
    parser.add_argument("--max-steps", type=int, default=-1, help="最大优化步数，-1 表示不限制（便于 CPU 冒烟测试）")
    parser.add_argument("--patience", type=int, default=40, help="EarlyStopping 耐心值（论文 40），0 表示关闭")
    parser.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1, help="每 N 个 epoch 验证一次（CPU 上可设为 2 提速）")
    # 阶段控制
    parser.add_argument("--skip-test", action="store_true", help="只训练，不测试")
    parser.add_argument("--test-only", action="store_true", help="不训练，直接加载 --checkpoint 做测试")
    parser.add_argument("--checkpoint", type=Path, default=None, help="--test-only 时使用的 checkpoint 路径")
    parser.add_argument("--predict", action="store_true", help="训练/测试后额外保存可视化预测结果")
    parser.add_argument("--resume", type=Path, default=None, help="从 checkpoint 恢复训练")
    return parser


def build_datamodule(args: argparse.Namespace) -> MVTecAD:
    return MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )


def build_engine(args: argparse.Namespace) -> Engine:
    callbacks = []
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                patience=args.patience,
                mode="min",
                check_on_train_epoch_end=True,
            )
        )
    return Engine(
        accelerator=args.accelerator,
        devices=1,
        default_root_dir=args.output_dir,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
    )


def build_model(args: argparse.Namespace) -> EdgeADLightning:
    return EdgeADLightning(
        in_ch=3,
        base=args.base,
        k_values=tuple(args.k),
        n=args.n,
        th1=args.th1,
        th2=args.th2,
        lam_g=args.lam_g,
        lam_s=args.lam_s,
        lr=args.lr,
        image_size=(args.image_size, args.image_size),
        score_mode=args.score_mode,
    )


def main() -> None:
    # Windows 中文环境默认 GBK，把日志重定向到文件时进度条中的 “•” 会触发
    # UnicodeEncodeError；这里统一切到 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    seed_everything(args.seed, workers=True)

    datamodule = build_datamodule(args)
    engine = build_engine(args)

    print("=" * 78)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}")
    print(f"epochs={args.epochs} max_steps={args.max_steps} batch={args.batch_size} lr={args.lr}")

    if args.test_only:
        if args.checkpoint is None:
            raise SystemExit("--test-only 需要同时指定 --checkpoint")
        print(f"loading checkpoint: {args.checkpoint}")
        model = EdgeADLightning.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
        model.score_mode = args.score_mode
    else:
        model = build_model(args)
        engine.fit(model=model, datamodule=datamodule, ckpt_path=args.resume)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"EdgeAD | category={args.category} | K={model.k_values} n={model.n} params={n_params / 1e6:.2f}M")
    print("=" * 78)

    results = None
    if not args.skip_test:
        if args.test_only:
            # 重新跑一遍验证集，让 PostProcessor 的归一化/阈值统计与当前评分方式一致
            print("running validation to refresh post-processor statistics ...")
            engine.validate(model=model, datamodule=datamodule, verbose=False)
        results = engine.test(model=model, datamodule=datamodule)
        print("-" * 78)
        print("TEST RESULTS:", json.dumps(results, indent=2, default=float))
        print("-" * 78)

    if args.predict:
        predictions = engine.predict(model=model, datamodule=datamodule)
        print(f"prediction batches: {len(predictions) if predictions is not None else 0}")

    if results:
        out = Path(args.output_dir) / "EdgeADLightning" / args.category / "test_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        print(f"metrics saved to {out}")


if __name__ == "__main__":
    main()
