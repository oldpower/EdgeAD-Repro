基于anomalib 实现 EdgeAD
===

### 任务说明

基于anomalib 实现论文中的异常检测方法 EdgeAD

### 开发环境说明

使用当前的conda python环境
```bash
conda activate env01
```
 - 注意，当前env01已经配置好了anomalib 2.7.0版本。不要修改当前的env01环境，因为这个环境之前成功跑通了EfficientDA训练推理转onnx全流程。

### 目录结构

    .
    |—— PRD.md      # 需求说明
    ├── README.md
    ├── assets
    │   └── EdgeAD-Unsupervised-Learning-Model-Based-on-Prior-Knowledge-Enhanced-Image-Anomaly-Detection.pdf    # EdgeAD论文
    ├── datasets                                                                                                # mvtec数据集
    │   └── mvtec -> /mnt/d/dataset/mvtec_anomaly_detection
    ├── lightning_model.py   # AnomalyModule 封装 + 严格 RBI                                                                                     
    ├── model.py             # 纯 PyTorch：U-Net、Canny、三种损失         
    └── train.py            # 训练/测试/推理脚本

