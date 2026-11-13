import os

from ultralytics import YOLO

os.environ["TBE_PARALLEL_NUM"] = "1"
os.environ["TE_PARALLEL_COMPILER"] = "1"
# model = YOLO("yolo11n.pt")  # 或你的 YOLO 模型
model = YOLO("yolo26n.pt")  # load an official model
model.export(format="ascend", name="Ascend310P3")

"""
python export_npu.py 
Ultralytics 8.4.198 🚀 Python-3.11.18 torch-2.15.0+cu132 CPU (13th Gen Intel Core i5-13420H)
YOLO26n summary (fused): 120 layers, 2,408,932 parameters, 0 gradients, 5.5 GFLOPs

PyTorch: starting from 'yolo26n.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400) (5.3 MB)

ONNX: starting export with onnx 1.23.2 opset 17...
ONNX: slimming with onnxslim 0.1.99...
ONNX: export success ✅ 1.5s, saved as 'yolo26n.onnx' (9.5 MB)

Ascend: starting export with ATC for Ascend310P3...
Ascend: running 'atc --model=/mnt/d/workspace/work01/yolo26n.onnx --framework=5 --output=/mnt/d/workspace/work01/yolo26n_ascend_mod
el/yolo26n_Ascend310P3 --input_format=NCHW --input_shape=images:1,3,640,640 --soc_version=Ascend310P3 --precision_mode=force_fp16'
ATC start working now, please wait for a moment.
...
ATC run success, welcome to the next use.

Ascend: export success ✅ 68.3s, saved as '/mnt/d/workspace/work01/yolo26n_ascend_model' (6.2 MB)

Export complete (68.6s)
Results saved to /mnt/d/workspace/work01/yolo26n_ascend_model
Predict:         yolo predict task=detect model=/mnt/d/workspace/work01/yolo26n_ascend_model imgsz=640 quantize=16
Validate:        yolo val task=detect model=/mnt/d/workspace/work01/yolo26n_ascend_model imgsz=640  quantize=16
Visualize:       https://netron.app


tree yolo26n_ascend_model/
yolo26n_ascend_model/
├── metadata.yaml
└── yolo26n_Ascend310P3.om

"""
