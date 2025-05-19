# 关键词唤醒的嵌入式高斯注意力

**一键复现 Baseline / CTGWP / Tuned / Advanced 的模型训练、评估、ONNX/TFLite 导出，并在树莓派上进行性能基准测试。**
GPU 默认使用 `cuda:1` 进行训练（可用 `--device cuda:0` 或 `--device cpu` 覆盖）。

---

## 项目流程概览

本项目包含以下主要步骤：

1.  **环境配置**: 分别为开发环境（用于训练和模型转换）和树莓派环境（用于性能测试）设置依赖。
2.  **数据预处理**: 生成用于模型训练的 Mel 频谱特征。
3.  **模型训练**: 在开发环境上训练 Baseline, CTGWP, CTGWP Tuned, CTGWP Advanced 四种模型。
4.  **准确性评估**: 在开发环境上评估训练好的 PyTorch 模型的准确率、FAR 和 FRR。
5.  **模型导出**: 将训练好的 PyTorch 模型导出为固定输入形状的 ONNX 格式，再转换为 TFLite FP32 和 TFLite INT8 格式。
6.  **树莓派性能基准测试**:
    *   准备与 TFLite 模型输入形状匹配的 NPY 数据。
    *   将模型和数据传输到树莓派。
    *   在树莓派上编译并运行 C++ 基准测试脚本。
    *   （可选）在树莓派上运行 Python 基准测试脚本进行对比。

---

## 1. 依赖环境配置

### 1.1 开发环境 (例如：带 GPU 的服务器或 PC)
用于数据预处理、模型训练、准确性评估和模型转换。

```bash
# 1. 创建并激活 Conda 环境 (推荐)
conda create -n kws python=3.11 -y
conda activate kws

# 2. 安装 PyTorch 及相关训练依赖
# 请根据你的 CUDA 版本选择合适的 PyTorch 安装命令，或使用 requirements.txt
# 例如: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt # 确保 requirements.txt 包含 torch, torchaudio, tqdm 等

# 3. 安装 ONNX 和 TensorFlow 相关库 (用于模型转换)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  onnx onnxruntime onnx-tf==1.14.0 \
  tensorflow-cpu==2.15.* tensorflow-probability==0.23.*
```

### 1.2 树莓派环境 (用于性能基准测试)

*   **操作系统**: Raspberry Pi OS (推荐 64-bit)。
*   **核心工具**: `git`, `g++` (C++17 支持), `python3`, `pip3`。
*   **TensorFlow Lite C++ 库**:
    *   你需要为树莓派编译或获取预编译的 `libtensorflow-lite.so` 及其所有依赖项（如 flatbuffers, Abseil 等的头文件和库）。
    *   将这些库存放在树莓派上的一个明确路径，例如 `/opt/tflite_build` (头文件通常在 `include` 子目录，库文件在 `lib` 子目录)。
*   **Python 依赖 (在树莓派上安装)**:
    ```bash
    sudo apt-get update && sudo apt-get install -y python3-pip libatlas-base-dev # libatlas 用于 numpy
    pip3 install numpy psutil onnxruntime
    # 安装 TFLite Python 运行时 (选择一个):
    pip3 install tflite-runtime # 轻量级，推荐 (从 piwheels 或 Google 查找适合你架构的 .whl)
    # 或者，如果 tflite-runtime 安装困难:
    # pip3 install tensorflow # 完整包，会比较大
    ```

---

## 2. 数据预处理 (在开发环境)

运行 `scripts/preprocess.py` 脚本以从原始音频数据生成 Mel 频谱特征和 `mapping.json` 文件。

```bash
# 示例命令 (在项目根目录执行)
python scripts/preprocess.py --data_dir path/to/your/speech_commands_v2 --out_dir data/feats
```
此步骤生成的特征将用于后续的模型训练。`MelDataset` 类（例如在 `scripts/train_baseline.py` 中定义）会处理这些特征，并将其统一到固定的时间长度（例如 `target_len=98`）。

---

## 3. 模型训练 (在开发环境)

所有训练脚本会将训练过程中验证集上表现最优的模型权重保存在各自指定的输出目录下的 `best.ckpt` 文件中。
以下命令均假设在**项目根目录**执行。

| 模型           | 训练脚本 (`scripts/`)      | 命令示例 (根据需求调整超参数)                                                                                                                                        | 输出模型路径 (示例)                  |
| -------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Baseline       | `train_baseline.py`        | `python scripts/train_baseline.py --feats_dir data/feats --out models/baseline --epochs 100 --bs 64 --lr 1e-3 --device cuda:1`                                  | `models/baseline/best.ckpt`          |
| CTGWP (原版)   | `train_ctgwp.py`         | `python scripts/train_ctgwp.py --feats_dir data/feats --out models/ctgwp --epochs 100 --bs 64 --lr 1e-3 --device cuda:1`                                        | `models/ctgwp/best.ckpt`             |
| CTGWP Tuned    | `ctgwp_tuned.py`         | `python scripts/ctgwp_tuned.py --feats_dir data/feats --out models/ctgwp_tuned --epochs 150 --bs 64 --lr 2e-3 --device cuda:1`                                   | `models/ctgwp_tuned/best.ckpt`       |
| CTGWP Advanced | `ctgwp_advanced.py`      | `python scripts/ctgwp_advanced.py --feats_dir data/feats --out models/ctgwp_adv --epochs 200 --bs 64 --lr 2e-3 --baseline_ckpt models/baseline/best.ckpt --device cuda:1` | `models/ctgwp_adv/best.ckpt`         |

**注意：**
*   `CTGWP Advanced` 的训练依赖于预训练的 `models/baseline/best.ckpt`。

---

## 4. 准确率 / FAR / FRR 评估 (在开发环境)

使用训练好的 PyTorch 模型 (`best.ckpt`) 在开发环境上评估模型的识别性能。

| 模型           | 命令示例 (在项目根目录执行)                                                                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `python scripts/evaluate.py --model_path models/baseline/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/baseline_all.json`              |
| CTGWP (原版)   | `python scripts/evaluate.py --model_path models/ctgwp/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_original_all.json`            |
| CTGWP Tuned    | `python scripts/evaluate.py --model_path models/ctgwp_tuned/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_tuned_all.json`          |
| CTGWP Advanced | `python scripts/evaluate.py --model_path models/ctgwp_adv/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_adv_all.json`            |

---

## 5. 模型导出 (ONNX 和 TFLite) (在开发环境)

为了在树莓派上进行可靠的基准测试，导出具有**固定输入时间轴**的ONNX模型至关重要。

### 5.1 导出固定形状的ONNX
对每个训练好的模型运行 `scripts/export_onnx.py`，**确保不使用 `--dynamic` 标志**。这将生成 `<model_type>_static_T.onnx` 文件。

### 5.2 从ONNX转换为TFLite (FP32 和 INT8)
使用 `scripts/export_tflite.py` 将上一步生成的 `_static_T.onnx` 文件转换为 `<model_type>_static_T_fp32.tflite` 和 `<model_type>_static_T_int8.tflite`。

**一键导出所有模型的脚本 (`export_all_static.sh`)**:
将以下内容保存为项目根目录下的 `export_all_static.sh`，赋予执行权限 (`chmod +x export_all_static.sh`)，然后运行 `./export_all_static.sh`。

```bash
#!/bin/bash
# export_all_static.sh - Export all models to ONNX (static T axis) and TFLite (FP32/INT8)

MODELS_BASE_DIR="models" # 模型检查点和导出模型的根目录

process_model() {
    MODEL_TYPE=$1
    MODEL_DIR="${MODELS_BASE_DIR}/${MODEL_TYPE}"
    ONNX_OUT="${MODEL_DIR}/${MODEL_TYPE}_static_T.onnx"
    TFLITE_FP32_OUT="${MODEL_DIR}/${MODEL_TYPE}_static_T_fp32.tflite"
    TFLITE_INT8_OUT="${MODEL_DIR}/${MODEL_TYPE}_static_T_int8.tflite"
    BEST_CKPT="${MODEL_DIR}/best.ckpt"

    echo "--- Processing Model: ${MODEL_TYPE} ---"

    if [ ! -f "${BEST_CKPT}" ]; then
        echo "ERROR: ${BEST_CKPT} not found. Skipping export for ${MODEL_TYPE}."
        return
    fi

    echo "Exporting ${MODEL_TYPE} to ONNX (Static Time Axis)..."
    python scripts/export_onnx.py \
        --model_path "${BEST_CKPT}" \
        --out "${ONNX_OUT}" \
        --model_type "${MODEL_TYPE}"

    if [ -f "${ONNX_OUT}" ]; then
        echo "Exporting ${MODEL_TYPE} ONNX to TFLite FP32..."
        python scripts/export_tflite.py \
            --onnx "${ONNX_OUT}" \
            --out "${TFLITE_FP32_OUT}"

        echo "Exporting ${MODEL_TYPE} ONNX to TFLite INT8..."
        python scripts/export_tflite.py \
            --onnx "${ONNX_OUT}" \
            --out "${TFLITE_INT8_OUT}" \
            --int8
    else
        echo "ERROR: ${ONNX_OUT} not found. Cannot export TFLite for ${MODEL_TYPE}."
    fi
    echo "--- Finished Exporting Model: ${MODEL_TYPE} ---"
    echo ""
}

# Export all defined models
process_model "baseline"
process_model "ctgwp"
process_model "ctgwp_tuned"
process_model "ctgwp_adv"

echo "All static model exports completed."
```
**验证**: 使用 Netron 打开生成的 `.tflite` 文件，确认输入形状是固定的（例如 `[1,1,40,98]`），并且 INT8 模型的输入类型仍为 `FLOAT32`（因为是动态范围量化）。

---

## 6. 准备树莓派基准测试NPY输入数据 (在开发环境)

使用 `scripts/make_sample.py` 脚本生成与TFLite模型固定输入时间长度匹配的NPY文件。

```bash
# 假设在项目根目录运行
# TARGET_LEN 应与模型训练/导出时使用的 target_len 一致 (例如 98)
TARGET_LEN=98
python scripts/make_sample.py \
    --feat_dir data/feats/val/yes \
    --num_samples 100 \
    --target_len ${TARGET_LEN} \
    --out_dir data \
    --out_name sample_len${TARGET_LEN}.npy # 例如 data/sample_len98.npy
```

---

## 7. 树莓派性能基准测试

将以下内容传输到树莓派：
*   `models` 目录中所有生成的 `_static_T.onnx`, `_static_T_fp32.tflite`, `_static_T_int8.tflite` 文件。
*   `data` 目录中生成的 `sample_lenXX.npy` 文件。
*   `pi-scripts` 目录（包含 `benchmark_tflite_all_cpp.cpp` 和 `benchmark_unified.py`）。

以下命令假设你在树莓派的 `pi-scripts` 目录下执行。

### 7.1 C++ 基准测试

#### 7.1.1 编译 C++ 脚本
假设你的C++脚本名为 `benchmark_tflite_all_cpp.cpp`。

```bash
# 确保以下路径与你的树莓派环境匹配
# TFLITE_SRC_HEADERS_DIR: 包含 TFLite C++ API 头文件的目录 (例如 tensorflow/lite/, tensorflow/lite/kernels/ 等)
# TFLITE_BUILD_LIB_DIR: 包含 libtensorflow-lite.so 和其他依赖库 (如 libfarmhash.so, libabsl_*.so) 的目录
TFLITE_SRC_HEADERS_DIR="/path/to/your/tflite_headers_on_pi"
TFLITE_BUILD_LIB_DIR="/path/to/your/tflite_lib_on_pi"

# 注意：以下编译命令需要根据你的TFLite库的实际构建情况进行精确调整
g++ benchmark_tflite_all_cpp.cpp -std=c++17 -O2 \
  -I"${TFLITE_SRC_HEADERS_DIR}" \
  -I"${TFLITE_SRC_HEADERS_DIR}/tensorflow/lite/kernels" \
  -I"${TFLITE_SRC_HEADERS_DIR}/tensorflow/lite/tools/evaluation" \
  -I"${TFLITE_BUILD_LIB_DIR}/flatbuffers_include" `#<-- 假设flatbuffers头文件路径` \
  -L"${TFLITE_BUILD_LIB_DIR}" \
  # 你可能需要为Abseil, farmhash等每个依赖库单独指定 -L 和 -I
  -Wl,-rpath,"${TFLITE_BUILD_LIB_DIR}" `#<-- 将库路径嵌入可执行文件，可能避免设置LD_LIBRARY_PATH` \
  -ltensorflow-lite \
  -lfarmhash `#<-- 示例依赖` \
  # ... (其他必要的 -labsl_... 库) ...
  -lpthread -ldl -lm \
  -o benchmark_cpp_runner
```
**强烈建议**: 仔细检查并调整上述编译命令中的 `-I`, `-L`, `-l` 和 `-rpath` 选项，使其与你在树莓派上TFLite库的实际文件结构和依赖关系完全匹配。`-rpath` 可以帮助程序在运行时找到共享库。

#### 7.1.2 运行 C++ 测试
1.  **设置共享库路径 (如果未使用 -rpath 或仍有问题)**:
    ```bash
    export LD_LIBRARY_PATH="/path/to/your/tflite_lib_on_pi:${LD_LIBRARY_PATH}"
    ```
2.  **运行**:
    ```bash
    INPUT_NPY="../data/sample_len98.npy" # 确认路径和文件名
    MODELS_TFLITE=(
        ../models/baseline/baseline_static_T_fp32.tflite
        ../models/baseline/baseline_static_T_int8.tflite
        ../models/ctgwp/ctgwp_static_T_fp32.tflite
        ../models/ctgwp/ctgwp_static_T_int8.tflite
        ../models/ctgwp_tuned/ctgwp_tuned_static_T_fp32.tflite
        ../models/ctgwp_tuned/ctgwp_tuned_static_T_int8.tflite
        ../models/ctgwp_adv/ctgwp_adv_static_T_fp32.tflite
        ../models/ctgwp_adv/ctgwp_adv_static_T_int8.tflite
    )
    WARMUP=1

    ./benchmark_cpp_runner \
        --models "${MODELS_TFLITE[@]}" \
        --input "${INPUT_NPY}" \
        --warmup ${WARMUP}
    ```

### 7.2 Python 基准测试

#### 7.2.1 运行 Python 测试
假设你的Python脚本名为 `benchmark_unified.py`。

```bash
INPUT_NPY_PYTHON="../data/sample_len98.npy"
MODELS_ALL_PYTHON=(
    # ONNX Models
    ../models/baseline/baseline_static_T.onnx
    ../models/ctgwp/ctgwp_static_T.onnx
    ../models/ctgwp_tuned/ctgwp_tuned_static_T.onnx
    ../models/ctgwp_adv/ctgwp_adv_static_T.onnx
    # TFLite Models
    ../models/baseline/baseline_static_T_fp32.tflite
    ../models/baseline/baseline_static_T_int8.tflite
    ../models/ctgwp/ctgwp_static_T_fp32.tflite
    ../models/ctgwp/ctgwp_static_T_int8.tflite
    ../models/ctgwp_tuned/ctgwp_tuned_static_T_fp32.tflite
    ../models/ctgwp_tuned/ctgwp_tuned_static_T_int8.tflite
    ../models/ctgwp_adv/ctgwp_adv_static_T_fp32.tflite
    ../models/ctgwp_adv/ctgwp_adv_static_T_int8.tflite
)
WARMUP_PYTHON=1
TFLITE_THREADS=1 # 推荐单线程进行公平对比

python3 benchmark_unified.py \
    --models "${MODELS_ALL_PYTHON[@]}" \
    --input "${INPUT_NPY_PYTHON}" \
    --warmup ${WARMUP_PYTHON} \
    --tflite_threads ${TFLITE_THREADS}
```

---

## 8. 常见问题
(调整编号，保留你之前的Q1-Q4，并补充Q5, Q6)

**Q1: 为什么训练时默认使用 cuda:1？**
...
**Q6: Python TFLite测试提示 `AttributeError: 'Interpreter' object has no attribute 'get_num_threads'`?**
...

---

**关键整合点：**

1.  **统一的项目流程概览**: 开头用一个概览来统领全文。
2.  **环境配置分开**: 将开发环境和树莓派环境的依赖分开说明。
3.  **模型导出步骤更清晰**: 分为ONNX导出（强调非动态）和TFLite转换两步，并提供了一个集成的shell脚本示例。
4.  **NPY数据准备**: 单独一节，强调时间长度匹配。
5.  **树莓派测试**: 成为一个主要章节，细分为C++和Python两部分，每部分都包含环境准备/编译和运行命令。
6.  **路径和文件名**: 尽可能使用相对路径，并提醒用户根据实际情况修改。C++编译命令部分特别强调了路径的复杂性和重要性。
7.  **你的C++脚本文件名**: 已更新为 `benchmark_tflite_all_cpp.cpp` (在7.1.1节) 和 `benchmark_cpp_runner` (可执行文件名，在7.1.2节)。

这个版本应该更符合一个完整的项目从头到尾的操作流程。请你仔细检查每一部分的命令、路径和文件名，确保它们与你的项目结构和实际操作一致。