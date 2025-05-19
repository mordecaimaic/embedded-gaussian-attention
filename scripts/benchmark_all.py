#!/usr/bin/env python3
"""
benchmark_all.py — 批量评测多个 TFLite 模型
输出 Markdown 表格：Model | Latency (ms) | CPU (%) | RAM (MB) | Size (MB)
"""
import argparse, time, os
import numpy as np
import psutil
import tflite_runtime.interpreter as tflite
from statistics import mean

def measure(model_path, sample_path, warmup=1):
    data = np.load(sample_path).astype(np.float32)
    N = data.shape[0]

    # 加载模型
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]['index']

    # 预热
    for _ in range(warmup):
        interpreter.set_tensor(inp_idx, data[:1])
        interpreter.invoke()

    # 采样前重置计量
    proc = psutil.Process()
    cpu_readings = []
    mem_readings = []

    # 推理并采集指标
    latencies = []
    for x in data:
        start = time.perf_counter()
        interpreter.set_tensor(inp_idx, x[np.newaxis, ...])
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

        cpu_readings.append(psutil.cpu_percent(interval=None))
        mem_readings.append(proc.memory_info().rss / 1024 / 1024)

    # 模型文件大小
    size_mb = os.path.getsize(model_path) / 1024 / 1024

    return {
        "latency": mean(latencies),
        "cpu":    mean(cpu_readings),
        "ram":    max(mem_readings),
        "size":   size_mb,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
      "--models", nargs="+", required=True,
      help="要评测的 .tflite 模型列表"
    )
    p.add_argument(
      "--input", required=True,
      help="输入的 sample.npy 路径"
    )
    args = p.parse_args()

    print("| Model | Latency (ms) | CPU (%) | RAM (MB) | Size (MB) |")
    print("|---|---|---|---|---|")
    for m in args.models:
        stats = measure(m, args.input)
        name = os.path.basename(m)
        print(
          f"| {name} | "
          f"{stats['latency']:.2f} | "
          f"{stats['cpu']:.1f} | "
          f"{stats['ram']:.1f} | "
          f"{stats['size']:.2f} |"
        )

if __name__ == "__main__":
    main()
