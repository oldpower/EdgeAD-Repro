整理了从 WSL 环境准备、CANN 安装到 ONNX 转 ATC 的完整标准流程。建议按照这个步骤重新梳理或在新终端中操作，以排除之前的残留干扰。

---

### 第一阶段：linux环境准备与依赖安装(当前使用了WSL)

**1. 创建昇腾默认运行用户组**
在 WSL 终端中执行（如已创建可跳过）：
```bash
sudo groupadd HwHiAiUser
sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
```

**2. 安装系统级依赖**
```bash
sudo apt-get update
sudo apt-get install -y gcc python3 python3-pip
```

**3. 补齐 Python 环境依赖（极其重要）**
激活你用于模型转换的 conda 或 venv 环境（假设名字是 `cv1`）：
```bash
conda activate cv1
```
安装 ATC 底层编译引擎（TE）所需的 Python 库：
```bash
pip install decorator scipy absl-py attrs numpy
```
*(注：安装过程中出现无关包的依赖冲突 `ERROR` 可忽略，只要看到 `Successfully installed` 即可)*

---

### 第二阶段：安装 CANN Toolkit 与算子包

**注意：WSL 中不需要、也无法安装驱动。** 必须下载**独立的 Toolkit 包**和 **310P 算子包**（不要下带 `cann` 字样的合一包）。

**1. 下载独立包**
前往昇腾社区下载以下两个包（版本以 9.2.1 为例，x86_64 架构）：
*   `Ascend-cann-toolkit_9.2.1_linux-x86_64.run`
*   `Ascend-cann-310p-ops_9.2.1_linux-x86_64.run`

**2. 赋予执行权限并安装**
在包含这两个 `.run` 文件的目录下执行：
```bash
chmod +x Ascend-cann-toolkit_9.2.1_linux-x86_64.run
./Ascend-cann-toolkit_9.2.1_linux-x86_64.run --install

chmod +x Ascend-cann-310p-ops_9.2.1_linux-x86_64.run
./Ascend-cann-310p-ops_9.2.1_linux-x86_64.run --install
```
*(安装路径默认为 `/home/你的用户名/Ascend`)*

**3. 配置环境变量**
每次新开终端都需要执行（或者在 `~/.bashrc` 末尾添加这句）：
```bash
source /home/user/Ascend/cann/set_env.sh
```

---

### 第三阶段：ONNX 转 ATC 脚本编写与执行

**1. 确认模型输入层名称和尺寸**
用 Python 查一下你的 `edgead.onnx` 的输入层名字（因为动态 shape 必须被写死）：
```bash
python3 -c "import onnx; m=onnx.load('./outputs/edgead.onnx'); print([i.name for i in m.graph.input])"
```
假设输出是 `['input']`，且你的图像尺寸是 256x256。

**2. 编写执行脚本 `export_atc.sh`**
在 ONNX 文件所在的目录（或者保证路径正确）新建 `export_om.sh`，**强烈建议使用 WSL 原生路径**（如 `~/project/` 而不是 `/mnt/d/`），避免路径解析报错。

脚本内容如下（注意换行符 `\`）：
```bash
#!/bin/bash

# 1. 激活环境并加载 CANN 环境变量
source ~/miniconda3/bin/activate cv1
source /home/quan/Ascend/cann/set_env.sh  # 注意替换为你的实际路径

# 2. 执行 ATC 转换
atc --model=./outputs/edgead.onnx \
    --framework=5 \
    --output=test_model \
    --soc_version=Ascend310P3 \
    --input_shape="input:1,3,256,256" 
```
*(注：把 `input` 换成你查到的真实名字；把 `256,256` 换成你模型真实的输入宽高)*

**3. 执行脚本**
```bash
chmod +x export_om.sh
./export_atc.sh
```

---

### 常见报错排查指南

如果在第三阶段执行时报错，通常只有以下几种情况：

*   **报错仍提示 `input_shape` 无效**：说明脚本里的 `--input_shape` 名字写错了，或者反斜杠 `\` 后面有空格导致命令截断。仔细核对模型输入名。
*   **报错 `EZ3001` / `E19999` / `Op xxx not supported`**：这是**算子不支持**。Python 环境已经没问题了，是 anomalib 导出的模型里含有昇腾 310P 不支持的算子。此时需要查看 `atc.log` 寻找具体算子，可能需要修改 PyTorch 模型源码、替换该算子并重新导出 ONNX。
*   **生成成功后**：目录下会出现 `test_model.om` 文件。此时将 `.om` 文件拷贝到插有 Atlas 300I Pro 的物理服务器上，使用 AscendCL 或 MindX SDK 编写 C++/Python 推理代码即可。
