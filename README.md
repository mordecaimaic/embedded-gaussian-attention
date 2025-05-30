# Embedded Gaussian Attention for Keyword Spotting

**One-click Reproduction for Baseline / CTGWP / Tuned / Advanced + ONNX + TFLite + Raspberry Pi Benchmarks**
GPU defaults to `cuda:1` (can be overridden with `--device cuda:0`)

---

## Table of Contents

1.  [Project Description](#1--project-description)
2.  [Dependencies](#2--dependencies)
    * [2.1 Development Environment (Server/PC)](#21-development-environment-serverpc)
    * [2.2 Raspberry Pi Environment (for Benchmarking)](#22-raspberry-pi-environment-for-benchmarking)
3.  [Data Preprocessing](#3--data-preprocessing)
4.  [Model Training](#4--model-training)
5.  [Accuracy / FAR / FRR Evaluation (on Development Environment)](#5--accuracy--far--frr-evaluation-on-development-environment)
6.  [Model Export (ONNX and TFLite)](#6--model-export-onnx-and-tflite)
    * [6.1 Export Fixed-Shape ONNX (for TFLite Conversion)](#61-export-fixed-shape-onnx-for-tflite-conversion)
    * [6.2 Convert from ONNX to TFLite (FP32 and INT8)](#62-convert-from-onnx-to-tflite-fp32-and-int8)
7.  [Prepare NPY Input Data for Raspberry Pi Benchmark](#7--prepare-npy-input-data-for-raspberry-pi-benchmark)
8.  [Raspberry Pi Performance Benchmark](#8--raspberry-pi-performance-benchmark)
    * [8.1 C++ Benchmark](#81-c-benchmark)
        * [8.1.1 Compile C++ Script](#811-compile-c-script)
        * [8.1.2 Run C++ Test](#812-run-c-test)
    * [8.2 Python Benchmark](#82-python-benchmark)
        * [8.2.1 Run Python Test](#821-run-python-test)
9.  [Frequently Asked Questions (FAQ)](#9--frequently-asked-questions-faq)

---

## 1 · Project Description

This project implements a lightweight, reproducible workflow for embedded keyword spotting, including full script support from data preprocessing, model training, and evaluation, to model export and actual deployment (Raspberry Pi latency testing). The goal is to enable graduate students and engineers to quickly get started with the CTGWP mechanism and run it efficiently on resource-constrained edge devices.

Key Features:
- **Data Preprocessing**: Scripts automatically extract Mel spectrograms and partition datasets.
- **Multiple Models**: Provides four comparison schemes: Baseline, original CTGWP, Tuned version, and Advanced version.
- **Comprehensive Evaluation**: Calculates overall accuracy, FAR, FRR, and detailed metrics for each keyword.
- **Model Export**: Supports ONNX and TFLite (FP32/INT8) export for easy deployment on multiple platforms.
- **Instance Benchmark**: Raspberry Pi latency test scripts (C++ and Python) demonstrate real-world runtime latency.

---

## 2 · Dependencies

### 2.1 Development Environment (Server/PC)
Used for data preprocessing, model training, accuracy evaluation, and model conversion.

```bash
conda create -n kws python=3.11 -y
conda activate kws

# Basic training & evaluation (PyTorch, torchaudio, etc.)
pip install -r requirements.txt

# ONNX / TFLite model conversion tools (needed only once)
pip install -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple) \
  onnx onnxruntime onnx-tf==1.14.0 \
  tensorflow-cpu==2.15.* tensorflow-probability==0.23.*
```

### 2.2 Raspberry Pi Environment (for Benchmarking)

* **Operating System**: Raspberry Pi OS (64-bit recommended).
* **C++ Compiler**: `g++` (usually pre-installed).
* **Python 3**: and `pip`.
* **TensorFlow Lite C++ Library**:
    * You will need to compile or obtain a pre-compiled `libtensorflow-lite.so` (and its dependencies, such as headers and libraries for flatbuffers, Abseil, etc.) for your Raspberry Pi architecture.
    * Assume these files are located in a directory on the Raspberry Pi, for example, `/path/to/your/tflite_build_on_pi`.
* **Python Dependencies (install on Raspberry Pi)**:
    ```bash
    pip install numpy psutil onnxruntime
    # Install TFLite Python runtime (choose one):
    pip install tflite-runtime # Lightweight, recommended
    # Or if tflite-runtime installation is difficult, you can try installing full tensorflow:
    # pip install tensorflow
    ```

---

## 3 · Data Preprocessing (on Development Environment)

Run the `scripts/preprocess.py` script.
```bash
python scripts/preprocess.py \
    --input_dir data/raw \
    --output_dir data/feats
```

---

## 4 · Model Training

All training scripts will save the best model (`best.ckpt`) in their respective model's output directory (e.g., `models/baseline/`).

| Model           | Script                         | Example Command (execute from project root)                                                                                               |
| -------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `scripts/train_baseline.py`  | `python scripts/train_baseline.py --feats_dir data/feats --out models/baseline --device cuda:1`                                       |
| CTGWP (Original)   | `scripts/train_ctgwp.py`     | `python scripts/train_ctgwp.py --feats_dir data/feats --out models/ctgwp --device cuda:1`                                             |
| CTGWP Tuned    | `scripts/ctgwp_tuned.py`     | `python scripts/ctgwp_tuned.py --feats_dir data/feats --out models/ctgwp_tuned --device cuda:1`                                       |
| CTGWP Advanced | `scripts/ctgwp_advanced.py`  | `python scripts/ctgwp_advanced.py --feats_dir data/feats --baseline_ckpt models/baseline/best.ckpt --out models/ctgwp_adv --device cuda:1` |

---

## 5 · Accuracy / FAR / FRR Evaluation (on Development Environment)

This step is typically performed on the development environment (e.g., a server) using the PyTorch model (`best.ckpt`).

Run `scripts/evaluate.py` to calculate various metrics and save the JSON results.

| Model           | Example Command (execute from project root)                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `python scripts/evaluate.py --model_path models/baseline/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/baseline_all.json`              |
| CTGWP (Original)   | `python scripts/evaluate.py --model_path models/ctgwp/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_original_all.json`            |
| CTGWP Tuned    | `python scripts/evaluate.py --model_path models/ctgwp_tuned/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_tuned_all.json`          |
| CTGWP Advanced | `python scripts/evaluate.py --model_path models/ctgwp_adv/best.ckpt --feats_dir data/feats --split test --device cuda:1 --out outputs/ctgwp_adv_all.json`            |

---

## 6 · Model Export (ONNX and TFLite)

To perform reliable benchmarking on the Raspberry Pi and obtain input dimensions consistent with training, we need to export an ONNX model with a **fixed input time axis** from the PyTorch model, and then convert it to TFLite.

### 6.1 Export Fixed-Shape ONNX (for TFLite Conversion)
Run `scripts/export_onnx.py` for each model, **ensuring not to use the `--dynamic` flag**.

| Model           | ONNX Export Command (execute from project root, generates `_static_T.onnx`)                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `python scripts/export_onnx.py --model_path models/baseline/best.ckpt --out models/baseline/baseline_static_T.onnx --model_type baseline`                                     |
| CTGWP (Original)   | `python scripts/export_onnx.py --model_path models/ctgwp/best.ckpt --out models/ctgwp/ctgwp_static_T.onnx --model_type ctgwp`                                                 |
| CTGWP Tuned    | `python scripts/export_onnx.py --model_path models/ctgwp_tuned/best.ckpt --out models/ctgwp_tuned/ctgwp_tuned_static_T.onnx --model_type ctgwp_tuned`                         |
| CTGWP Advanced | `python scripts/export_onnx.py --model_path models/ctgwp_adv/best.ckpt --out models/ctgwp_adv/ctgwp_adv_static_T.onnx --model_type ctgwp_adv`                                 |

### 6.2 Convert from ONNX to TFLite (FP32 and INT8)
Use `scripts/export_tflite.py` to convert the `_static_T.onnx` files generated in the previous step to TFLite.

| Model           | TFLite FP32 Export (generates `_static_T_fp32.tflite`)                                                                                                                               | TFLite INT8 Export (generates `_static_T_int8.tflite`)                                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | `python scripts/export_tflite.py --onnx models/baseline/baseline_static_T.onnx --out models/baseline/baseline_static_T_fp32.tflite`                                           | `python scripts/export_tflite.py --onnx models/baseline/baseline_static_T.onnx --out models/baseline/baseline_static_T_int8.tflite --int8`                                                      |
| CTGWP (Original)   | `python scripts/export_tflite.py --onnx models/ctgwp/ctgwp_static_T.onnx --out models/ctgwp/ctgwp_static_T_fp32.tflite`                                                       | `python scripts/export_tflite.py --onnx models/ctgwp/ctgwp_static_T.onnx --out models/ctgwp/ctgwp_static_T_int8.tflite --int8`                                                                   |
| CTGWP Tuned    | `python scripts/export_tflite.py --onnx models/ctgwp_tuned/ctgwp_tuned_static_T.onnx --out models/ctgwp_tuned/ctgwp_tuned_static_T_fp32.tflite`                               | `python scripts/export_tflite.py --onnx models/ctgwp_tuned/ctgwp_tuned_static_T.onnx --out models/ctgwp_tuned/ctgwp_tuned_static_T_int8.tflite --int8`                                          |
| CTGWP Advanced | `python scripts/export_tflite.py --onnx models/ctgwp_adv/ctgwp_adv_static_T.onnx --out models/ctgwp_adv/ctgwp_adv_static_T_fp32.tflite`                                     | `python scripts/export_tflite.py --onnx models/ctgwp_adv/ctgwp_adv_static_T.onnx --out models/ctgwp_adv/ctgwp_adv_static_T_int8.tflite --int8`                                                |

**Strongly Recommended**: Consolidate the above export commands into a shell script (e.g., `export_all_static.sh`) for one-click execution.
**Verification**: Use Netron to inspect the generated `.tflite` files, confirming that their input shape is fixed (e.g., `[1,1,40,98]`) and that the input type for INT8 models is `FLOAT32`.

---

## 7 · Prepare NPY Input Data for Raspberry Pi Benchmark

To match the fixed input shape of the TFLite models, you need to generate NPY files with the corresponding time length.
Use the `scripts/make_sample.py` script (ensure it is the version we discussed that includes padding/cropping and the `--target_len` parameter).

```bash
# Run from the project root directory
# TARGET_LEN should be consistent with the target_len used during model training and export (e.g., 98)
TARGET_LEN=98
python scripts/make_sample.py \
    --feat_dir data/feats/val/yes \
    --num_samples 100 \
    --target_len ${TARGET_LEN} \
    --out_dir data \
    --out_name sample_len${TARGET_LEN}.npy
```
This will generate a file, for example, `data/sample_len98.npy`, with a shape of `(100, 1, 40, 98)`.

---

## 8 · Raspberry Pi Performance Benchmark

Copy the project (or at least the `models` directory containing the generated `_static_T` models, the `data` directory containing the generated `sample_lenXX.npy` file, and the `pi-scripts` directory) to the Raspberry Pi. The following commands need to be executed in the `pi-scripts` directory on the Raspberry Pi.

### 8.1 C++ Benchmark

#### 8.1.1 Compile C++ Script

```bash
# Replace the following paths with those matching your actual Raspberry Pi environment
TENSORFLOW_SRC_DIR="/home/LLM/tensorflow"  # TensorFlow source root directory (contains headers)
TFLITE_BUILD_DIR="/home/LLM/tflite_shared_build" # Build directory for TFLite and its dependencies


g++ benchmark_tflite_all_cpp.cpp -std=c++17 -O2 \
  -I"${TENSORFLOW_SRC_DIR}" \
  -I"${TENSORFLOW_SRC_DIR}/tensorflow/lite/kernels" \
  -I"${TENSORFLOW_SRC_DIR}/tensorflow/lite/tools/evaluation" \
  -I"/path/to/your/flatbuffers_include_on_pi" \
  -L"${TFLITE_LIB_DIR}" \
  # ... (complete all -I, -L, -l options based on your actual compile command and dependency locations) ...
  -ltensorflow-lite \
  # ... (other -l libraries) ...
  -lpthread -ldl -lm \
  -o benchmark_cpp_runner
```
**Note**: Ensure all paths and linked libraries in the compile command correctly point to the TFLite libraries and their dependencies that you compiled or stored on the Raspberry Pi.

#### 8.1.2 Run C++ Test
1.  **Set Shared Library Path**:
    ```bash
    export LD_LIBRARY_PATH="${TFLITE_BUILD_DIR}:${LD_LIBRARY_PATH}"
    # For example:
    # export LD_LIBRARY_PATH=/home/LLM/tflite_shared_build:$LD_LIBRARY_PATH
    ```
2.  **Run**:
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

### 8.2 Python Benchmark

#### 8.2.1 Run Python Test

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
TFLITE_THREADS=1 # Recommended single thread for fair comparison

python benchmark_unified.py \
    --models "${MODELS_ALL_PYTHON[@]}" \
    --input "${INPUT_NPY_PYTHON}" \
    --warmup ${WARMUP_PYTHON} \
    --tflite_threads ${TFLITE_THREADS}
```

---

## 9 · Frequently Asked Questions (FAQ)

**Q1: Why is `cuda:1` used by default during training?**
A1: Servers may have multiple GPUs. If GPU 0 is occupied by other processes or system tasks, it is recommended to use GPU 1 for a more stable training environment. Users can override the default setting with the `--device cuda:0` or `--device cpu` parameter.

**Q2: What to do if "Non-Converted Ops" or similar warnings are encountered when exporting TFLite?**
A2:
    * **Constant Folding Ops**: Some operations like Constant Folding are optimized away during model conversion and do not require runtime support. These warnings can usually be safely ignored.
    * **Unsupported Operations**: If it indicates that some operations (Ops) are not supported by the TFLite built-in operator set, and you have not enabled Flex Delegate (which allows running TensorFlow operations), this may lead to runtime errors.
        * **Check Dependencies**: Ensure your model conversion environment has all necessary libraries installed, such as `tensorflow-probability` (if the model uses related operations) and the correct versions of `onnx-tf` and `tensorflow`.
        * **Model Structure**: Your model may contain layers or operations not supported by the TFLite built-in operator set. You might need to simplify the model structure, replace these operations, or enable Flex Delegate when converting to TFLite (but this will increase the final model's dependencies and size).
        * **ONNX Opset Version**: Try exporting with different ONNX opset versions, as some versions may have better compatibility with the TFLite converter.

**Q3: Why are FAR/FRR sometimes 0 when evaluating accuracy on the server?**
A3: When the data split used for evaluation contains no negative samples (i.e., non-keyword samples, or samples not labeled as background/unknown), the denominator for calculating FAR (False Acceptance Rate) will be zero, and the script might handle this as FAR=0. Similarly, if there are no positive samples (target keywords), FRR might also be 0. You can verify the data and label distribution by adjusting the `--target_keywords` parameter in the `evaluate.py` script (e.g., temporarily exclude all known keywords, treating them all as "unknown" to check FAR).

**Q4: How to run these models on other embedded platforms (e.g., ESP32)?**
A4:
    * **TFLite Micro**: For microcontrollers like ESP32, you need to use the TensorFlow Lite for Microcontrollers (TFLM) framework. This usually involves converting the `.tflite` model into a C/C++ array and integrating it into your microcontroller project. TFLM has stricter limitations on operator sets and memory usage.
    * **Model Optimization**: You may need to perform deeper model optimization, such as more aggressive quantization (e.g., full integer INT8, or even INT4/binarization), pruning, or using smaller model architectures designed for microcontrollers.
    * **SDK and Toolchain**: Familiarity with the target platform's SDK and cross-compilation toolchain is necessary.

**Q5: When running C++ benchmarks on Raspberry Pi, I get `error while loading shared libraries: libtensorflow-lite.so: cannot open shared object file: No such file or directory`?**
A5: This error means the dynamic linker cannot find the `libtensorflow-lite.so` shared library at runtime.
    * **Solution**: **Before** running your C++ executable, set the `LD_LIBRARY_PATH` environment variable in the **same terminal session** to point to the directory containing `libtensorflow-lite.so` and its dependent libraries. For example:
        ```bash
        export LD_LIBRARY_PATH="/path/to/your/tflite_build_on_pi/lib:${LD_LIBRARY_PATH}"
        ```
        Ensure the path is correct. Alternatively, use the `-Wl,-rpath,"/path/to/your/tflite_build_on_pi/lib"` linker option when compiling your C++ code to embed the library path into the executable.

**Q6: Segmentation Fault when running C++ or Python TFLite benchmarks on Raspberry Pi?**
A6: Segmentation faults are usually caused by:
    * **Input data mismatch with model expectation (most common)**:
        * **Shape Mismatch**: TFLite models expect input of a specific shape (e.g., `[1,1,40,98]`). Each sample in your NPY data (after removing the batch dimension N) must have exactly the same shape (e.g., `(1,40,98)`). Pay special attention to the time dimension.
        * **Data Type Mismatch**: Most TFLite models (even INT8 dynamic range quantized models) expect `FLOAT32` input. Ensure your NPY data is converted to `float32` after loading.
    * **Model File Issues**:
        * **Conversion Error**: The ONNX to TFLite conversion process might have been imperfect, leading to issues in the internal structure of the generated `.tflite` file. Try converting using a **non-dynamic ONNX** (run `export_onnx.py` without the `--dynamic` flag), which is generally more stable.
        * **Model Corruption**: Ensure the model file was not corrupted during transfer.
    * **TFLite Runtime/Library Issues**: The TFLite C++ library or Python runtime you are using on the Raspberry Pi might be incompatible with the model or have bugs.
    * **Insufficient Memory**: While segmentation faults are not always directly caused by insufficient memory, it can be an indirect trigger if the model is very large or system resources are tight.
    * **Debugging Steps**: Carefully check the debug output of your C++ and Python scripts, especially regarding the loaded NPY shape and the model's expected input tensor details. Use Netron to view the exact input specifications of the `.tflite` file.

**Q7: Python TFLite test script shows `AttributeError: 'Interpreter' object has no attribute 'get_num_threads'`?**
A7: The TFLite Python `Interpreter` you are using (either from `tensorflow.lite.python.interpreter` or a specific version of `tflite_runtime`) might not have the `get_num_threads()` method.
    * **Solution**: Modify the Python script to remove or comment out the code calling `get_num_threads()`. The number of threads is **requested** by the `num_threads` parameter passed to the `Interpreter` constructor (from the command-line `--tflite_threads`). You can indirectly assess if multi-threading is used by observing CPU utilization.

**Q8: Why might ONNX model latency on Raspberry Pi be lower than TFLite FP32, even though TFLite INT8 should be faster?**
A8:
    * **ONNX Runtime Optimization**: ONNX Runtime is a highly optimized inference engine, and its CPU execution provider might be very effective for certain simple models (like the Baseline).
    * **TFLite Dynamic Range Quantization Overhead**: The TFLite INT8 models you are using are generated via dynamic range quantization. This means weights are INT8, but activations need to be dynamically quantized from FP32 to INT8 at runtime, and then de-quantized back to FP32 after computation. This quantization/de-quantization process itself has overhead, which for very lightweight models, might partially offset or even exceed the acceleration gained from INT8 integer operations.
    * **Comparison**: TFLite INT8 is generally faster than its **corresponding TFLite FP32** version. The comparison with ONNX Runtime depends on their respective optimization levels and quantization overhead.

**Q9: On Raspberry Pi, why does increasing TFLite threads (e.g., from 1 to 8) significantly increase latency?**
A9: For lightweight models performing single-sample inference on edge devices, this could be due to:
    * **Excessive Thread Synchronization/Management Overhead**: The cost of splitting the task among multiple threads and synchronizing them outweighs the benefits of parallel computation.
    * **Cache Contention**: Multiple threads compete for limited CPU cache resources, leading to a decrease in cache hit rates.
    * **Hardware Limitations**: The Raspberry Pi's CPU core performance and memory bandwidth are limited, and too many threads can quickly hit a bottleneck.
    * **Conclusion**: For such scenarios, single-threaded or a small number of threads (e.g., 2) might be optimal. It's recommended to primarily report single-thread performance or find the optimal number of threads through experimentation.

---

## Appendix A: Compiling TensorFlow Lite C++ Library from Source on Raspberry Pi (Optional)

If you cannot find a pre-compiled TensorFlow Lite C++ library suitable for your Raspberry Pi, or wish to obtain specific optimization configurations for your hardware, you can try compiling from TensorFlow source. **This process can be time-consuming and involve complex steps; please proceed with caution and refer to the official TensorFlow documentation for the latest and most detailed guidance.**

**A.1 Install Compilation Dependencies:**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git zip unzip # Basic compilation tools
# To resolve potential cpuinfo submodule issues during TFLite compilation,
# it's recommended to pre-install the system-provided libcpuinfo development package:
sudo apt-get install -y libcpuinfo-dev
# Other potential dependencies (supplement based on TFLite official docs or compilation errors)
# E.g.: sudo apt-get install -y flatbuffers-compiler libflatbuffers-dev zlib1g-dev
```

**A.2 Clone TensorFlow Repository:**

```bash
cd ~ # Or your chosen working directory
git clone [https://github.com/tensorflow/tensorflow.git](https://github.com/tensorflow/tensorflow.git)
cd tensorflow
# It's advisable to switch to a stable release branch, e.g., r2.15 (adjust based on current stable version)
# git checkout r2.15
cd ..
```

**A.3 CMake Configuration and TFLite Compilation:**

```bash
# 1. Create and enter the build directory (outside the tensorflow source directory)
mkdir tflite_from_source_build && cd tflite_from_source_build

# 2. Run CMake configuration
#    -S points to the TensorFlow Lite directory within the TensorFlow source (tensorflow/tensorflow/lite)
#    -DCMAKE_BUILD_TYPE=Release : Release version optimization
#    -DBUILD_SHARED_LIBS=ON : Build shared libraries (.so)
#    -DTFLITE_ENABLE_XNNPACK=OFF : (Optional) If you don't want your C++ benchmark to use XNNPACK by default, disable it here.
#                                To enable, set to ON (may require additional dependencies or resolving compile issues).
#    -DTFLITE_ENABLE_RUY=ON : (Recommended) Use Ruy matrix multiplication library for better CPU performance.
#    -DTFLITE_ENABLE_GPU=OFF : Raspberry Pi typically doesn't use its GPU for TFLite Delegates.
#    Ensure not to pass -DCMAKE_DISABLE_FIND_PACKAGE_cpuinfo=TRUE, to let CMake use the system-installed cpuinfo.
cmake \
  -S ../tensorflow/tensorflow/lite \
  -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  -Wno-dev

# 3. Start compilation (uses all available cores, this will be very time-consuming)
make -j$(nproc)
```

After compilation, the `libtensorflow-lite.so` you need will typically be generated in the current build directory (`tflite_from_source_build`). Header files are located in `../tensorflow/tensorflow/lite/` and its subdirectories, as well as potentially generated directories like `flatbuffers_include` in the build directory. You will need to correctly configure these paths in the `-I` and `-L` parameters of your C++ compilation command. Dependent libraries (Abseil, farmhash, Eigen, etc.) are usually built together in the `_deps` subfolder of the build directory, and you may also need to configure link paths for them.

**Note**: Compiling large projects like TensorFlow Lite from source can encounter various dependency issues and platform-specific compilation errors. Be sure to carefully read and follow the official TensorFlow guides for cross-compilation or native compilation on ARM or Raspberry Pi.
