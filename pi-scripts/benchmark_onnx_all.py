#!/usr/bin/env python3
"""
benchmark_onnx_all.py — 批量评测多个 ONNX 模型在 Pi 上的推理性能
输出 Markdown 表格：Model | Latency (ms) | CPU (%) | RAM (MB) | Size (MB)
"""
import argparse, time, os
import numpy as np
import psutil
import onnxruntime as ort
from statistics import mean

def measure(model_path, sample_path, warmup=1):
    # 加载样本
    data = np.load(sample_path).astype(np.float32)
    N = data.shape[0]

    # 创建 ONNX Runtime 会话
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # 预热
    for _ in range(warmup):
        sess.run(None, {inp_name: data[:1]})

    # 用 psutil 监控当前进程
    proc = psutil.Process(os.getpid())
    latencies, cpu_usages, mem_rss = [], [], []

    # 逐样本推理并采集指标
    for x in data:
        t0 = time.perf_counter()
        sess.run(None, {inp_name: x[np.newaxis, ...]})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        cpu_usages.append(psutil.cpu_percent(interval=None))
        mem_rss.append(proc.memory_info().rss / 1024 / 1024)  # 转为 MB

    # 模型文件大小
    size_mb = os.path.getsize(model_path) / 1024 / 1024

    return {
        "latency": mean(latencies),
        "cpu": mean(cpu_usages),
        "ram": max(mem_rss),
        "size": size_mb,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", nargs="+", required=True,
        help="待评测的 ONNX 模型文件列表"
    )
    p.add_argument(
        "--input", required=True,
        help="输入样本 npy 文件，shape=[N,C,F,T]"
    )
    p.add_argument(
        "--warmup", type=int, default=1,
        help="预热次数"
    )
    args = p.parse_args()

    # 输出 Markdown 表头
    print("| Model | Latency (ms) | CPU (%) | RAM (MB) | Size (MB) |")
    print("|---|---|---|---|---|")

    for m in args.models:
        stats = measure(m, args.input, args.warmup)
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
