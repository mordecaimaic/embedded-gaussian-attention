# 嵌入式高斯注意力机制的关键词识别

---

## 1 · 项目概述

本项目为嵌入式关键词识别 (Keyword Spotting, KWS) 提供了一个轻量级、可复现的完整工作流，特别关注CTGWP (Contextual Transformer with Gaussian Weighted Pooling) 机制。它涵盖了从数据预处理、多种模型（Baseline、CTGWP原始版/调整版/高级版）训练与评估，到模型导出 (ONNX, TFLite FP32/INT8) 并在树莓派上进行C++/Python性能基准测试的全过程。

**核心价值**: 快速搭建、评估和部署轻量级关键词识别模型到边缘设备。

## 2 · 环境简述

* **开发端 (PC/服务器)**: Python (推荐Conda), PyTorch, ONNX, TensorFlow (转换用) 等。详细依赖见 `requirements.txt`。
* **树莓派端**: Raspberry Pi OS (推荐64位), C++编译器, Python3, TFLite Runtime, ONNX Runtime等。

## 3 · 核心工作流概览

所有主要操作均通过 `scripts/` 目录下的Python脚本执行，树莓派测试脚本位于 `pi-scripts/`。

1.  **数据预处理 (`preprocess.py`)**: 自动提取梅尔频谱特征并划分数据集。
2.  **模型训练 (`train_*.py`)**: 训练Baseline及三种CTGWP变体模型。
3.  **模型评估 (`evaluate.py`)**: 计算准确率、误识率 (FAR)、拒识率 (FRR)。
4.  **模型导出**:
    * `export_onnx.py`: PyTorch -> 固定形状ONNX。
    * `export_tflite.py`: ONNX -> TFLite (FP32/INT8)。
5.  **树莓派样本准备 (`make_sample.py`)**: 生成匹配TFLite输入的NPY格式数据。
6.  **树莓派基准测试 (`pi-scripts/`)**:
    * C++ (`benchmark_tflite_all_cpp.cpp`): 编译后运行，需配置TFLite库。
    * Python (`benchmark_unified.py`): 直接运行，测试ONNX和TFLite模型。

## 4 · 常见问题解答 (FAQ)

此部分解答了GPU使用、模型转换警告、FAR/FRR为0、其他平台部署、树莓派运行时库错误、段错误、Python属性错误、ONNX与TFLite性能差异、树莓派多线程性能下降等常见问题。
*(详细问答内容请参见上一版或完整版README)*

## 5 · 附录A: 源码编译TFLite C++库 (可选)

此部分提供了在树莓派上从TensorFlow源码编译TensorFlow Lite C++库的步骤，供无法找到预编译库或需要特定优化的用户参考。

---

**提示**: 这个版本更加精简。如果需要每个步骤的具体命令示例或者FAQ和附录的详细内容，请参考之前更详细的版本或项目中的具体脚本。