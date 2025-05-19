# 关键词唤醒的嵌入式高斯注意力

**一键复现 Baseline / CTGWP / Tuned / Advanced + ONNX + TFLite + 树莓派基准**
GPU 默认使用 `cuda:1`（可用 `--device cuda:0` 覆盖）

---

## 目录

1.  [项目说明](#1--项目说明)
2.  [依赖环境](#2--依赖环境)
    *   [2.1 开发环境 (服务器/PC)](#21-开发环境-服务器pc)
    *   [2.2 树莓派环境 (用于基准测试)](#22-树莓派环境-用于基准测试)
3.  [数据预处理](#3--数据预处理)
4.  [模型训练](#4--模型训练)
5.  [准确率 / FAR / FRR 评估 (在开发环境)](#5--准确率--far--frr-评估-在开发环境)
6.  [模型导出 (ONNX 和 TFLite)](#6--模型导出-onnx-和-tflite)
    *   [6.1 导出固定形状的ONNX (用于TFLite转换)](#61-导出固定形状的onnx-用于tflite转换)
    *   [6.2 从ONNX转换为TFLite (FP32 和 INT8)](#62-从onnx转换为tflite-fp32-和-int8)
7.  [准备树莓派基准测试NPY输入数据](#7--准备树莓派基准测试npy输入数据)
8.  [树莓派性能基准测试](#8--树莓派性能基准测试)
    *   [8.1 C++ 基准测试](#81-c-基准测试)
        *   [8.1.1 编译 C++ 脚本](#811-编译-c-脚本)
        *   [8.1.2 运行 C++ 测试](#812-运行-c-测试)
    *   [8.2 Python 基准测试](#82-python-基准测试)
        *   [8.2.1 运行 Python 测试](#821-运行-python-测试)
9.  [常见问题](#9--常见问题)

---

## 1 · 项目说明

本项目实现了一套轻量级、可复现的嵌入式关键词唤醒流程，包括从数据预处理、模型训练、评估，到模型导出与实际部署（树莓派延迟测试）等全流程脚本支持。目标是方便研究生和工程师快速上手 CTGWP 机制，并可在资源受限的边缘设备上高效运行。

主要功能：
- **数据预处理**: 脚本自动提取 Mel 频谱并划分数据集。
- **多种模型**: 提供 Baseline、原版 CTGWP、Tuned 版、Advanced 版四种对比方案。
- **全面评估**: 计算整体准确率、FAR、FRR 及每个关键词的详情指标。
- **模型导出**: 支持 ONNX 与 TFLite（FP32/INT8）导出，方便多平台部署。
- **实例基准**: 树莓派延迟测试脚本（C++ 和 Python），展示实际运行时延。

---

## 2 · 依赖环境

### 2.1 开发环境 (服务器/PC)
用于数据预处理、模型训练、准确性评估和模型转换。

```bash
conda create -n kws python=3.11 -y
conda activate kws

# 基本训练 & 评估 (PyTorch, torchaudio, etc.)
pip install -r requirements.txt

# ONNX / TFLite 模型转换工具 (仅需一次)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  onnx onnxruntime onnx-tf==1.14.0 \
  tensorflow-cpu==2.15.* tensorflow-probability==0.23.*
```

### 2.2 树莓派环境 (用于基准测试)

*   **操作系统**: Raspberry Pi OS (推荐 64-bit)。
*   **C++ 编译器**: `g++` (通常已预装)。
*   **Python 3**: 及 `pip`。
*   **TensorFlow Lite C++ 库**:
    *   你需要为你的树莓派架构编译或获取预编译的 `libtensorflow-lite.so` (及其依赖，如 flatbuffers, Abseil 等的头文件和库)。
    *   假设这些文件位于树莓派的例如 `/path/to/your/tflite_build_on_pi` 目录。
*   **Python 依赖 (在树莓派上安装)**:
    ```bash
    pip install numpy psutil onnxruntime
    # 安装 TFLite Python 运行时 (选择一个):
    pip install tflite-runtime # 轻量级，推荐
    # 或者如果 tflite-runtime 安装困难，可以尝试安装完整的 tensorflow:
    # pip install tensorflow
    ```

---

## 3 · 数据预处理 (在开发环境)

运行 `scripts/preprocess.py` 脚本
```bash
python scripts/preprocess.py \
    --input_dir data/raw \
    --output_dir data/feats
```

---

## 4 · 模型训练

所有训练脚本会将最优模型 (`best.ckpt`) 保存在各自模型的输出目录中 (例如 `models/baseline/`)。

| 模型           | 脚本                         | 命令示例 (在项目根目录执行)                                                                                                             |
| -------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `scripts/train_baseline.py`  | `python scripts/train_baseline.py --feats_dir data/feats --out models/baseline --device cuda:1`                                       |
| CTGWP (原版)   | `scripts/train_ctgwp.py`     | `python scripts/train_ctgwp.py --feats_dir data/feats --out models/ctgwp --device cuda:1`                                             |
| CTGWP Tuned    | `scripts/ctgwp_tuned.py`     | `python scripts/ctgwp_tuned.py --feats_dir data/feats --out models/ctgwp_tuned --device cuda:1`                                       |
| CTGWP Advanced | `scripts/ctgwp_advanced.py`  | `python scripts/ctgwp_advanced.py --feats_dir data/feats --baseline_ckpt models/baseline/best.ckpt --out models/ctgwp_adv --device cuda:1` |

---

## 5 · 准确率 / FAR / FRR 评估 (在开发环境)

此步骤通常在开发环境（如服务器）上使用 PyTorch 模型 (`best.ckpt`) 进行。

运行 `scripts/evaluate.py` 计算各项指标并保存 JSON 结果。

| 模型           | 命令示例 (在项目根目录执行)                                                                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `python scripts/evaluate.py --model_path models/baseline/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/baseline_all.json`              |
| CTGWP (原版)   | `python scripts/evaluate.py --model_path models/ctgwp/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_original_all.json`            |
| CTGWP Tuned    | `python scripts/evaluate.py --model_path models/ctgwp_tuned/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_tuned_all.json`          |
| CTGWP Advanced | `python scripts/evaluate.py --model_path models/ctgwp_adv/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_adv_all.json`            |

---

## 6 · 模型导出 (ONNX 和 TFLite)

为了在树莓派上进行可靠的基准测试并获得与训练时一致的输入维度，我们需要从 PyTorch 模型导出具有**固定输入时间轴**的 ONNX 模型，然后将其转换为 TFLite。

### 6.1 导出固定形状的ONNX (用于TFLite转换)
对每个模型运行 `scripts/export_onnx.py`，**确保不使用 `--dynamic` 标志**。

| 模型           | ONNX 导出命令 (在项目根目录执行, 生成 `_static_T.onnx`)                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `python scripts/export_onnx.py --model_path models/baseline/best.ckpt --out models/baseline/baseline_static_T.onnx --model_type baseline`                                     |
| CTGWP (原版)   | `python scripts/export_onnx.py --model_path models/ctgwp/best.ckpt --out models/ctgwp/ctgwp_static_T.onnx --model_type ctgwp`                                                 |
| CTGWP Tuned    | `python scripts/export_onnx.py --model_path models/ctgwp_tuned/best.ckpt --out models/ctgwp_tuned/ctgwp_tuned_static_T.onnx --model_type ctgwp_tuned`                         |
| CTGWP Advanced | `python scripts/export_onnx.py --model_path models/ctgwp_adv/best.ckpt --out models/ctgwp_adv/ctgwp_adv_static_T.onnx --model_type ctgwp_adv`                                 |

### 6.2 从ONNX转换为TFLite (FP32 和 INT8)
使用 `scripts/export_tflite.py` 将上一步生成的 `_static_T.onnx` 文件转换为 TFLite。

| 模型           | TFLite FP32 导出 (生成 `_static_T_fp32.tflite`)                                                                                                                               | TFLite INT8 导出 (生成 `_static_T_int8.tflite`)                                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `python scripts/export_tflite.py --onnx models/baseline/baseline_static_T.onnx --out models/baseline/baseline_static_T_fp32.tflite`                                           | `python scripts/export_tflite.py --onnx models/baseline/baseline_static_T.onnx --out models/baseline/baseline_static_T_int8.tflite --int8`                                                      |
| CTGWP (原版)   | `python scripts/export_tflite.py --onnx models/ctgwp/ctgwp_static_T.onnx --out models/ctgwp/ctgwp_static_T_fp32.tflite`                                                       | `python scripts/export_tflite.py --onnx models/ctgwp/ctgwp_static_T.onnx --out models/ctgwp/ctgwp_static_T_int8.tflite --int8`                                                                   |
| CTGWP Tuned    | `python scripts/export_tflite.py --onnx models/ctgwp_tuned/ctgwp_tuned_static_T.onnx --out models/ctgwp_tuned/ctgwp_tuned_static_T_fp32.tflite`                               | `python scripts/export_tflite.py --onnx models/ctgwp_tuned/ctgwp_tuned_static_T.onnx --out models/ctgwp_tuned/ctgwp_tuned_static_T_int8.tflite --int8`                                          |
| CTGWP Advanced | `python scripts/export_tflite.py --onnx models/ctgwp_adv/ctgwp_adv_static_T.onnx --out models/ctgwp_adv/ctgwp_adv_static_T_fp32.tflite`                                     | `python scripts/export_tflite.py --onnx models/ctgwp_adv/ctgwp_adv_static_T.onnx --out models/ctgwp_adv/ctgwp_adv_static_T_int8.tflite --int8`                                                |

**强烈建议**: 将上述导出命令整合到一个shell脚本（如 `export_all_static.sh`）中以便一键执行。
**验证**: 使用 Netron 检查生成的 `.tflite` 文件，确认其输入形状是固定的（例如 `[1,1,40,98]`），并且 INT8 模型的输入类型是 `FLOAT32`。

---

## 7 · 准备树莓派基准测试NPY输入数据

为了与TFLite模型的固定输入形状匹配，你需要生成具有相应时间长度的NPY文件。
使用 `scripts/make_sample.py` 脚本 (确保它是我们讨论过的包含padding/cropping和`--target_len`参数的版本)。

```bash
# 在项目根目录运行
# TARGET_LEN 应与模型训练和导出时使用的 target_len 一致 (例如 98)
TARGET_LEN=98
python scripts/make_sample.py \
    --feat_dir data/feats/val/yes \
    --num_samples 100 \
    --target_len ${TARGET_LEN} \
    --out_dir data \
    --out_name sample_len${TARGET_LEN}.npy
```
这将生成一个例如 `data/sample_len98.npy` 的文件，其形状为 `(100, 1, 40, 98)`。

---

## 8 · 树莓派性能基准测试

将项目（或至少 `models` 目录包含生成的 `_static_T` 模型、`data` 目录包含生成的 `sample_lenXX.npy` 文件以及 `pi-scripts` 目录）复制到树莓派。以下命令需要在树莓派的 `pi-scripts` 目录下执行。

### 8.1 C++ 基准测试

#### 8.1.1 编译 C++ 脚本

```bash
# 将以下路径替换为与你的树莓实际环境匹配的路径
TENSORFLOW_SRC_DIR="/home/LLM/tensorflow"  # TensorFlow 源码根目录 (包含头文件)
TFLITE_BUILD_DIR="/home/LLM/tflite_shared_build" # TFLite 及其依赖的构建目录


g++ benchmark_tflite_all_cpp.cpp -std=c++17 -O2 \
  -I"${TENSORFLOW_SRC_DIR}" \
  -I"${TENSORFLOW_SRC_DIR}/tensorflow/lite/kernels" \
  -I"${TENSORFLOW_SRC_DIR}/tensorflow/lite/tools/evaluation" \
  -I"/path/to/your/flatbuffers_include_on_pi" \
  -L"${TFLITE_LIB_DIR}" \
  # ... (根据你的实际编译命令和依赖库位置，补全所有的 -I, -L, -l 选项) ...
  -ltensorflow-lite \
  # ... (其他 -l 库) ...
  -lpthread -ldl -lm \
  -o benchmark_cpp_runner
```
**注意**: 确保编译命令中的所有路径和链接库都正确指向你在树莓派上编译或存放的TFLite库及其依赖项。

#### 8.1.2 运行 C++ 测试
1.  **设置共享库路径**:
    ```bash
    export LD_LIBRARY_PATH="${TFLITE_BUILD_DIR}:${LD_LIBRARY_PATH}"
    # 例如:
    # export LD_LIBRARY_PATH=/home/LLM/tflite_shared_build:$LD_LIBRARY_PATH
    ```
2.  **运行**:
    ```bash
    INPUT_NPY="../data/sample_len98.npy"
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

### 8.2 Python 基准测试

#### 8.2.1 运行 Python 测试

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

python benchmark_unified.py \
    --models "${MODELS_ALL_PYTHON[@]}" \
    --input "${INPUT_NPY_PYTHON}" \
    --warmup ${WARMUP_PYTHON} \
    --tflite_threads ${TFLITE_THREADS}
```

---

## 9 · 常见问题

---

## X · 常见问题 (FAQ)

**Q1: 为什么训练时默认使用 `cuda:1`？**
A1: 服务器上可能有多块 GPU，如果编号 0 的 GPU 被其他进程或系统任务占用，建议使用编号 1 的 GPU 以获得更稳定的训练环境。用户可以通过 `--device cuda:0` 或 `--device cpu` 参数来覆盖默认设置。

**Q2: 导出 TFLite 时遇到 “Non-Converted Ops” 或类似警告怎么办？**
A2:
    *   **常量折叠算子**: 某些操作如常量折叠（Constant Folding）在模型转换时会被优化掉，它们不需要运行时的支持，这类警告通常可以放心忽略。
    *   **未支持的操作**: 如果提示某些操作（Ops）不被 TFLite 内置操作集支持，并且你没有启用 Flex Delegate (允许运行 TensorFlow 操作)，这可能会导致运行时错误。
        *   **检查依赖**: 确保你的模型转换环境安装了所有必要的库，例如 `tensorflow-probability` (如果模型中用到了相关操作) 和正确版本的 `onnx-tf`、`tensorflow`。
        *   **模型结构**: 你的模型中可能包含了一些 TFLite 内置操作集不支持的层或操作。你可能需要简化模型结构，替换这些操作，或者在转换 TFLite 时启用 Flex Delegate (但这会增加最终模型的依赖和大小)。
        *   **ONNX Opset版本**: 尝试使用不同的 ONNX opset 版本进行导出，有时某些版本与 TFLite 转换器的兼容性更好。

**Q3: 在服务器上评估准确率时，FAR/FRR 为何有时为 0？**
A3: 当评估使用的数据分割 (split) 中完全没有负样本（即非关键词样本，或者没有被标记为背景/未知类的样本）时，计算 FAR (False Acceptance Rate) 的分母会为零，脚本可能会将其处理为 FAR=0。类似地，如果完全没有正样本（目标关键词），FRR 也可能为0。可以通过调整 `evaluate.py` 脚本中的 `--target_keywords` 参数（例如，暂时排除所有已知关键词，将它们都视为“未知”来检查 FAR）来验证数据和标签的分布。

**Q4: 如何在其他嵌入式平台（如 ESP32）运行这些模型？**
A4: 
    *   **TFLite Micro**: 对于像 ESP32 这样的微控制器，你需要使用 TensorFlow Lite for Microcontrollers (TFLM) 框架。这通常涉及将 `.tflite` 模型转换为 C/C++ 数组，并将其集成到你的微控制器项目中。TFLM 对操作集和内存使用有更严格的限制。
    *   **模型优化**: 你可能需要对模型进行更深度的优化，例如更积极的量化（如全整型INT8，甚至INT4/二值化）、剪枝、或使用专为微控制器设计的更小模型架构。
    *   **SDK 和工具链**: 熟悉目标平台的 SDK 和交叉编译工具链是必要的。

**Q5: 在树莓派上运行 C++ 基准测试时，提示 `error while loading shared libraries: libtensorflow-lite.so: cannot open shared object file: No such file or directory`？**
A5: 这个错误意味着动态链接器在运行时找不到 `libtensorflow-lite.so` 这个共享库。
    *   **解决方案**: 在运行你的 C++ 可执行文件**之前**，在**同一个终端会话中**设置 `LD_LIBRARY_PATH` 环境变量，使其指向包含 `libtensorflow-lite.so` 及其依赖库的目录。例如：
        ```bash
        export LD_LIBRARY_PATH="/path/to/your/tflite_build_on_pi/lib:${LD_LIBRARY_PATH}"
        ```
        确保路径正确。或者，在编译 C++ 代码时使用 `-Wl,-rpath,"/path/to/your/tflite_build_on_pi/lib"` 链接选项将库路径嵌入到可执行文件中。

**Q6: 在树莓派上运行 C++ 或 Python 的 TFLite 基准测试时出现段错误 (Segmentation Fault)？**
A6: 段错误通常由以下原因引起：
    *   **输入数据与模型期望不匹配 (最常见)**:
        *   **形状不匹配**: TFLite 模型期望特定形状的输入 (例如 `[1,1,40,98]`)。你提供的 NPY 数据的每个样本（在去掉批次维度 N 后）必须具有完全相同的形状 (例如 `(1,40,98)`)。特别注意时间维度。
        *   **数据类型不匹配**: 大多数 TFLite 模型（即使是 INT8 动态范围量化模型）期望 `FLOAT32` 类型的输入。确保你的 NPY 数据在加载后转换为 `float32`。
    *   **模型文件问题**:
        *   **转换错误**: ONNX 到 TFLite 的转换过程可能不完美，导致生成的 `.tflite` 文件内部结构有问题。尝试使用**非动态 ONNX** (`export_onnx.py` 不加 `--dynamic` 标志) 进行转换，这通常更稳定。
        *   **模型损坏**: 确保模型文件在传输过程中没有损坏。
    *   **TFLite 运行时/库问题**: 你在树莓派上使用的 TFLite C++ 库或 Python 运行时可能与模型不兼容，或者本身存在 bug。
    *   **内存不足**: 虽然段错误不总是直接由内存不足引起，但如果模型非常大或系统资源紧张，也可能间接触发。
    *   **调试步骤**: 仔细检查 C++ 和 Python 脚本的调试输出，特别是关于加载的 NPY 形状和模型期望的输入张量细节。使用 Netron 查看 `.tflite` 文件的确切输入规格。

**Q7: Python TFLite 测试脚本提示 `AttributeError: 'Interpreter' object has no attribute 'get_num_threads'`？**
A7: 你使用的 TFLite Python `Interpreter` (无论是来自 `tensorflow.lite.python.interpreter` 还是特定版本的 `tflite_runtime`) 可能没有 `get_num_threads()` 这个方法。
    *   **解决方案**: 修改 Python 脚本，移除或注释掉调用 `get_num_threads()` 的代码。线程数是由传递给 `Interpreter` 构造函数的 `num_threads` 参数（来自命令行的 `--tflite_threads`）来**请求**的。你可以通过观察 CPU 使用率来间接判断是否使用了多线程。

**Q8: 为什么 ONNX 模型在树莓派上的延迟可能比 TFLite FP32 模型还低，但 TFLite INT8 应该更快？**
A8:
    *   **ONNX Runtime 优化**: ONNX Runtime 是一个高度优化的推理引擎，其 CPU 执行提供者可能对某些简单模型（如 Baseline）的优化非常有效。
    *   **TFLite 动态范围量化开销**: 你使用的 TFLite INT8 模型是通过动态范围量化生成的。这意味着权重是 INT8，但激活值在运行时需要从 FP32 动态量化到 INT8，计算后再反量化回 FP32。这个量化/反量化的过程本身有开销，对于非常轻量级的模型，这个开销可能部分抵消甚至超过 INT8 整数运算带来的加速。
    *   **比较**: TFLite INT8 通常会比**对应的 TFLite FP32** 版本快。与 ONNX Runtime 的比较则取决于两者各自的优化程度和量化开销。

**Q9: 在树莓派上，为什么增加 TFLite 的线程数（例如从1到8）反而导致延迟大幅增加？**
A9: 对于在边缘设备上进行单样本推理的轻量级模型，这可能是由于：
    *   **线程同步/管理开销过大**: 将任务拆分到多个线程并进行同步的开销超过了并行计算的收益。
    *   **缓存竞争**: 多个线程争抢有限的CPU缓存资源，导致缓存命中率下降。
    *   **硬件限制**: 树莓派的CPU核心性能和内存带宽有限，过多线程可能很快达到瓶颈。
    *   **结论**: 对于这类场景，单线程或少量线程（例如2个）可能是最优的。建议主要报告单线程性能，或通过实验找到最佳线程数。

---

## 附录A: 在树莓派上从源码编译TensorFlow Lite C++库 (可选)

如果你无法找到适用于你的树莓派的预编译 TensorFlow Lite C++ 库，或者希望获得针对你硬件的特定优化配置，可以尝试从 TensorFlow 源码进行编译。**此过程可能比较耗时且步骤繁琐，请谨慎操作并参考 TensorFlow 官方文档获取最新和最详细的指南。**

**A.1 安装编译依赖:**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git zip unzip # 基础编译工具
# 为了解决 TFLite 编译时可能遇到的 cpuinfo 子模块问题，
# 推荐预先安装系统提供的 libcpuinfo 开发包：
sudo apt-get install -y libcpuinfo-dev
# 其他可能需要的依赖（根据TFLite官方文档或编译错误提示补充）
# 例如: sudo apt-get install -y flatbuffers-compiler libflatbuffers-dev zlib1g-dev
```

**A.2 克隆 TensorFlow 仓库:**

```bash
cd ~ # 或者你选择的其他工作目录
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
# 建议切换到一个稳定的 release 分支，例如 r2.15 (请根据当前稳定版调整)
# git checkout r2.15
cd ..
```

**A.3 CMake 配置和编译 TFLite:**

```bash
# 1. 创建并进入构建目录 (在tensorflow源码目录之外)
mkdir tflite_from_source_build && cd tflite_from_source_build

# 2. 运行 CMake 配置
#    -S 指向 TensorFlow 源码中的 TensorFlow Lite 目录 (tensorflow/tensorflow/lite)
#    -DCMAKE_BUILD_TYPE=Release : 发布版本优化
#    -DBUILD_SHARED_LIBS=ON : 构建共享库 (.so)
#    -DTFLITE_ENABLE_XNNPACK=OFF : (可选) 如果你的C++基准测试不想默认使用XNNPACK，可以在此禁用。
#                                若要启用，设为ON (可能需要额外依赖或解决编译问题)。
#    -DTFLITE_ENABLE_RUY=ON : (推荐) 使用Ruy矩阵乘法库以获得较好的CPU性能。
#    -DTFLITE_ENABLE_GPU=OFF : 树莓派通常不使用其GPU进行TFLite Delegate。
#    确保不传递 -DCMAKE_DISABLE_FIND_PACKAGE_cpuinfo=TRUE，让CMake使用系统安装的cpuinfo。
cmake \
  -S ../tensorflow/tensorflow/lite \
  -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  -Wno-dev

# 3. 开始编译 (使用所有可用核心，这会非常耗时)
make -j$(nproc)
```

编译完成后，你需要的 `libtensorflow-lite.so` 通常会生成在当前构建目录 (`tflite_from_source_build`) 下。头文件位于 `../tensorflow/tensorflow/lite/` 及其子目录，以及构建目录中可能生成的 `flatbuffers_include` 等。你需要将这些路径正确配置到你的 C++ 编译命令的 `-I` 和 `-L` 参数中。依赖库 (Abseil, farmhash, Eigen 等) 通常会被一同构建在构建目录的 `_deps` 子文件夹下，你也可能需要为它们配置链接路径。

**注意**: 从源码编译大型项目如 TensorFlow Lite 可能会遇到各种依赖问题和平台特定的编译错误。请务必仔细阅读并遵循 TensorFlow 官方提供的针对 ARM 或树莓派的交叉编译或本地编译指南。