# 关键词唤醒的嵌入式高斯注意力

**一套 Clean、可复现的 CTGWP 与 Baseline 实现**

本项目提供：

* 数据预处理（Mel 频谱提取）
* Baseline 模型训练（4 层 CNN + GAP）
* CTGWP 模型训练（可学习高斯注意力）
* 调参版与高级版 CTGWP 变体
* 评估脚本（Accuracy、FAR、FRR）
* （可选）模型导出与嵌入式推理

> **提示：** 所有 GPU 命令 **默认使用 CUDA 设备 1**。你也可以通过 `--device cuda:<index>` 覆盖，或设置 `CUDA_VISIBLE_DEVICES=1`。

---

## 📋 前置环境

* Ubuntu / macOS / Windows WSL
* NVIDIA GPU + CUDA 12.2（设备编号 1）
* Conda 或 Virtualenv

### 1. 创建 Python 环境

```bash
conda create -n kws python=3.11 -y
conda activate kws
pip install --upgrade pip
pip install -r requirements.txt
```

**`requirements.txt`** 建议包含：

```
torch==2.5.1
torchaudio
numpy
scipy
transformers
tqdm
soundfile
onnx
onnxruntime
```

---

## 📂 目录结构

```
embedded-gaussian-attention/
├── data/
│   ├── raw/                # 原始 .wav 文件
│   └── feats/              # 提取后的 Mel 频谱 (.npy)
├── models/                 # 保存的模型 checkpoint
│   ├── baseline/
│   ├── ctgwp/
│   ├── ctgwp_tuned/
│   └── ctgwp_adv/
├── outputs/                # 评估与导出结果
├── scripts/                # 所有训练/评估/导出脚本
├── requirements.txt        # 依赖列表
└── README.md               # 英文版本文件
```

---

## 🛠️ 1. 数据预处理

下载并解压 SpeechCommands v0.02：

```bash
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data/raw && tar zxvf speech_commands_v0.02.tar.gz -C data/raw
```

提取 40 通道 Mel 频谱：

```bash
python scripts/preprocess.py \
    --input_dir data/raw \
    --output_dir data/feats
```

生成 `data/feats/train`、`data/feats/val`、`data/feats/test` 目录，以及标签映射 `mapping.json`。

---

## ⚙️ 2. 模型训练

GPU 默认使用设备 1：

```bash
export CUDA_VISIBLE_DEVICES=1
```

### 2.1 Baseline（CNN + GAP）

```bash
python scripts/train_baseline.py \
    --feats_dir data/feats \
    --out models/baseline \
    --device cuda:1
```

> 输出：`models/baseline/best.ckpt`

### 2.2 CTGWP（原版）

```bash
python scripts/train_ctgwp.py \
    --feats_dir data/feats \
    --out models/ctgwp \
    --device cuda:1
```

> 输出：`models/ctgwp/best.ckpt`

### 2.3 CTGWP Tuned（调参版）

```bash
python scripts/ctgwp_tuned.py \
    --feats_dir data/feats \
    --out models/ctgwp_tuned \
    --device cuda:1
```

> 输出：`models/ctgwp_tuned/best.ckpt`

### 2.4 CTGWP Advanced（推荐）

```bash
python scripts/ctgwp_advanced.py \
    --feats_dir data/feats \
    --baseline_ckpt models/baseline/best.ckpt \
    --out models/ctgwp_adv \
    --device cuda:1
```

> 输出：`models/ctgwp_adv/best.ckpt`

---

## ✅ 3. 模型评估

统一调用：

```bash
python scripts/evaluate.py \
    --model_path <CKPT_PATH> \
    --feats_dir data/feats \
    --split test \
    --device cuda:1 \
    --out outputs/result.json
```

**示例**：

* Baseline：

  ```bash
  python scripts/evaluate.py \
      --model_path models/baseline/best.ckpt \
      --feats_dir data/feats
  ```
* Advanced：

  ```bash
  python scripts/evaluate.py \
      --model_path models/ctgwp_adv/best.ckpt \
      --feats_dir data/feats \
      --out outputs/ctgwp_adv_test.json
  ```

输出：

```
Accuracy on test: 95.23%
Saved to outputs/ctgwp_adv_test.json
```

*若需 FAR/FRR，请在 `evaluate.py` 中启用。*

---

## 📦 4. 可选：模型导出与嵌入式推理

### 4.1 导出 ONNX

```bash
python scripts/export_onnx.py \
    --ckpt models/ctgwp_adv/best.ckpt \
    --onnx_out outputs/ctgwp_adv.onnx
```

### 4.2 转 TFLite

参见脚本注释：ONNX → TF → TFLite (FP32/INT8)。

### 4.3 Raspberry Pi 性能测试

```bash
python scripts/benchmark_pi.py \
    --model outputs/ctgwp_adv_int8.tflite \
    --feats_dir data/feats/test \
    --out outputs/pi_bench.json
```

结果示例：

```json
{"mean_ms": 4.3, "p95_ms": 5.7}
```

---

## 🛠️ 常见问题

| 问题             | 解决方案                                      |
| -------------- | ----------------------------------------- |
| GPU OOM        | 降低 `--bs` 或开启 AMP                         |
| val\_acc 不升    | 调整 `--lr`、延长 `--epochs`、检查 Baseline\_ckpt |
| 加载报错 field 不匹配 | 确保脚本与模型版本一致或用 `strict=False`加载            |