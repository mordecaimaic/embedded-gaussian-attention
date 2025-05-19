#!/usr/bin/env python3
"""
benchmark_onnx.py — Raspberry Pi ONNX Runtime 推理基准
"""
import argparse, time, numpy as np
import onnxruntime as ort

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ONNX 模型文件")
    p.add_argument("--input", required=True, help="输入 .npy 文件，shape=[N,C,F,T]")
    p.add_argument("--warmup", type=int, default=1, help="预热次数")
    args = p.parse_args()

    data = np.load(args.input).astype(np.float32)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # 预热
    for _ in range(args.warmup):
        sess.run(None, {inp_name: data[:1]})

    # 计时
    t0 = time.perf_counter()
    for x in data:
        sess.run(None, {inp_name: x[np.newaxis,...]})
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    print(f"Model: {args.model}")
    print(f"Runs: {len(data)}  Total: {total_ms:.2f} ms  Avg: {total_ms/len(data):.2f} ms")

if __name__ == "__main__":
    main()
