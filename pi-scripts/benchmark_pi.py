#!/usr/bin/env python3
"""
benchmark_pi.py — Raspberry Pi TFLite 推理基准（禁用 XNNPACK）
---------------------------------------------------------------
用法:
  python benchmark_pi.py --model <model.tflite> --input <sample.npy> [--warmup 1]
"""
import argparse, time, sys
import numpy as np
import tflite_runtime.interpreter as tflite

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--model", required=True, help="TFLite 模型文件")
    ap.add_argument("--input", required=True, help="输入 Numpy 文件，shape=[N,C,F,T]")
    ap.add_argument("--warmup", type=int, default=1, help="预热次数")
    args = ap.parse_args()

    # 1. 加载输入
    data = np.load(args.input).astype(np.float32)
    N, C, F, T = data.shape
    print(f"Loaded {args.input}, shape={data.shape}", flush=True)

    # 2. 创建 Interpreter 时禁用所有 delegate
    interpreter = tflite.Interpreter(
        model_path=args.model,
        experimental_delegates=[]
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    idx_in = inp["index"]

    # 确认输入 shape
    print("Model input shape:", inp["shape"], flush=True)

    # 3. 预热
    for _ in range(args.warmup):
        interpreter.set_tensor(idx_in, data[:1])
        interpreter.invoke()
    print(f"Warm-up {args.warmup} runs done", flush=True)

    # 4. 测时
    t0 = time.perf_counter()
    for x in data:
        interpreter.set_tensor(idx_in, x[np.newaxis, ...])
        interpreter.invoke()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    print(f"Total time for {N} runs: {total_ms:.2f} ms", flush=True)
    print(f"Average latency: {total_ms/N:.2f} ms", flush=True)

    # 5. 退出
    sys.exit(0)

if __name__ == "__main__":
    main()
