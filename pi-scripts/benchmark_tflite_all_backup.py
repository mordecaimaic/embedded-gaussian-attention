#!/usr/bin/env python3
"""
pi-scripts/benchmark_unified.py — 批量评测多个 TFLite 模型在树莓派上的推理性能
输出 Markdown 表格：Model | Latency (ms) | CPU (%) | RAM (MB) | Size (MB)
"""
import argparse
import time
import os
from statistics import mean

import numpy as np
import psutil
# 选择轻量级 tflite_runtime 或系统 TensorFlow 的 Interpreter
# try:
from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
# except ImportError:
#     from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter


def measure(model_path, sample_path, warmup=1):
    # 1. 加载样本数据
    data = np.load(sample_path).astype(np.float32)
    N = data.shape[0]

    # 2. 初始化 Interpreter
    interpreter = TFLiteInterpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    idx_in = inp_detail['index']

    # 3. 预热
    for _ in range(warmup):
        interpreter.set_tensor(idx_in, data[:1])
        interpreter.invoke()

    # 4. 性能测量
    proc = psutil.Process()
    latencies, cpu_usages, mem_usages = [], [], []
    for x in data:
        t0 = time.perf_counter()
        interpreter.set_tensor(idx_in, x[np.newaxis, ...])
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        cpu_usages.append(psutil.cpu_percent(interval=None))
        mem_usages.append(proc.memory_info().rss / 1024 / 1024)  # MB

    # 5. 模型大小
    size_mb = os.path.getsize(model_path) / 1024 / 1024

    return {
        'latency': mean(latencies),
        'cpu': mean(cpu_usages),
        'ram': max(mem_usages),
        'size': size_mb,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--models', nargs='+', required=True,
        help='待评测的 .tflite 模型文件列表'
    )
    parser.add_argument(
        '--input', required=True,
        help='输入样本 .npy 文件，shape=[N,C,F,T]'
    )
    parser.add_argument(
        '--warmup', type=int, default=1,
        help='预热次数'
    )
    args = parser.parse_args()

    print('| Model | Latency (ms) | CPU (%) | RAM (MB) | Size (MB) |')
    print('|---|---|---|---|---|')
    for model_path in args.models:
        stats = measure(model_path, args.input, args.warmup)
        name = os.path.basename(model_path)
        print(
            f"| {name} | {stats['latency']:.2f} | {stats['cpu']:.1f} | {stats['ram']:.1f} | {stats['size']:.2f} |"
        )
