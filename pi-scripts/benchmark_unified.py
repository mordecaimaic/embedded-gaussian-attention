#!/usr/bin/env python3
"""
pi-scripts/benchmark_unified.py — 批量评测多个 ONNX 和 TFLite 模型在树莓派上的推理性能
输出 Markdown 表格：Model | Format | Latency (ms) | CPU (%) | RAM (MB) | Size (MB)
"""
import argparse
import time
import os
import sys
from statistics import mean, stdev

# --- TFLite 配置 ---
os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"
print(f"DEBUG: TF_LITE_DISABLE_XNNPACK set to: {os.getenv('TF_LITE_DISABLE_XNNPACK')}")

import numpy as np
import psutil

# --- ONNX Runtime ---
try:
    import onnxruntime as ort
    print(f"INFO: ONNX Runtime version: {ort.__version__}")
    # sess_options = ort.SessionOptions() # 可选设置
except ImportError:
    print("WARNING: onnxruntime not found. ONNX models cannot be benchmarked.")
    ort = None

# --- TensorFlow Lite Interpreter ---
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    print("INFO: Using tflite_runtime.interpreter")
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter
        print("INFO: Using tensorflow.lite.python.interpreter (Fallback)")
    except ImportError:
        print("WARNING: TensorFlow Lite Interpreter not found. TFLite models cannot be benchmarked.")
        TFLiteInterpreter = None

# --- measure_tflite function (保持不变) ---
def measure_tflite(interpreter: TFLiteInterpreter, data: np.ndarray, warmup: int):
    N = data.shape[0]
    inp_detail = interpreter.get_input_details()[0]
    idx_in = inp_detail['index']
    expected_input_shape_tflite = tuple(inp_detail['shape'])

    print(f"DEBUG: TFLite Model Expected Input Shape: {expected_input_shape_tflite}")
    print(f"DEBUG: NPY Data Shape: {data.shape}")

    npy_sample_shape = data.shape[1:]
    model_input_shape_no_batch = expected_input_shape_tflite[1:]

    if npy_sample_shape != model_input_shape_no_batch:
        raise ValueError(f"NPY sample shape {npy_sample_shape} (C,H,W after N) does not match "
                         f"TFLite model expected input shape {model_input_shape_no_batch} (C,H,W after B=1). "
                         f"Please regenerate NPY data with correct dimensions matching the model.")
    print("DEBUG: TFLite input shape validation passed.")

    print("DEBUG: Starting TFLite warmup...")
    for i in range(warmup):
        interpreter.set_tensor(idx_in, data[:1])
        interpreter.invoke()
        # print(f"DEBUG: Warmup iteration {i} completed.") # Can be verbose
    print("DEBUG: TFLite warmup finished.")

    print("DEBUG: Starting TFLite benchmark...")
    proc = psutil.Process()
    latencies, cpu_usages, mem_usages = [], [], []
    for i, x in enumerate(data):
        input_data = x[np.newaxis, ...]
        t0 = time.perf_counter()
        interpreter.set_tensor(idx_in, input_data)
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        cpu_usages.append(psutil.cpu_percent(interval=None))
        mem_usages.append(proc.memory_info().rss / 1024 / 1024)
    print("DEBUG: TFLite benchmark finished.")
    return latencies, cpu_usages, mem_usages

# --- measure_onnx function (保持不变) ---
def measure_onnx(session: ort.InferenceSession, data: np.ndarray, warmup: int):
    N = data.shape[0]
    input_name = session.get_inputs()[0].name
    input_shape_onnx = session.get_inputs()[0].shape
    print(f"DEBUG: ONNX Model Expected Input Name: {input_name}")
    print(f"DEBUG: ONNX Model Expected Input Shape (from model): {input_shape_onnx}")
    print(f"DEBUG: NPY Data Shape: {data.shape}")
    batch_input_shape = (1,) + data.shape[1:]
    print(f"DEBUG: Effective ONNX Input Shape for this run: {batch_input_shape}")
    if all(isinstance(dim, int) for dim in input_shape_onnx):
        if tuple(input_shape_onnx) != batch_input_shape:
             raise ValueError(f"NPY derived shape {batch_input_shape} does not match "
                              f"fixed ONNX model input shape {tuple(input_shape_onnx)}.")
        print("DEBUG: ONNX fixed input shape validation passed.")

    print("DEBUG: Starting ONNX warmup...")
    input_feed_warmup = {input_name: data[:1]}
    for i in range(warmup):
        _ = session.run(None, input_feed_warmup)
        # print(f"DEBUG: Warmup iteration {i} completed.") # Can be verbose
    print("DEBUG: ONNX warmup finished.")

    print("DEBUG: Starting ONNX benchmark...")
    proc = psutil.Process()
    latencies, cpu_usages, mem_usages = [], [], []
    for i, x in enumerate(data):
        input_data = x[np.newaxis, ...]
        input_feed = {input_name: input_data}
        t0 = time.perf_counter()
        _ = session.run(None, input_feed)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        cpu_usages.append(psutil.cpu_percent(interval=None))
        mem_usages.append(proc.memory_info().rss / 1024 / 1024)
    print("DEBUG: ONNX benchmark finished.")
    return latencies, cpu_usages, mem_usages

# --- main function ---
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Benchmark ONNX and TFLite models."
    )
    parser.add_argument(
        '--models', nargs='+', required=True,
        help='List of .onnx and/or .tflite model files to benchmark.'
    )
    parser.add_argument(
        '--input', required=True,
        help='Input sample .npy file, shape should match model requirements (e.g., [N,1,40,98]).'
    )
    parser.add_argument(
        '--warmup', type=int, default=1,
        help='Number of warmup iterations.'
    )
    parser.add_argument(
        '--onnx_provider', type=str, default='CPUExecutionProvider',
        help='ONNX Runtime Execution Provider (e.g., CPUExecutionProvider, OpenVINOExecutionProvider).'
    )
    parser.add_argument(
        '--tflite_threads', type=int, default=1,
        help='Number of threads for TFLite interpreter (-1 for default).'
    )
    args = parser.parse_args()

    try:
        npy_data = np.load(args.input).astype(np.float32)
        if npy_data.ndim != 4:
            raise ValueError(f"Input NPY data must be 4-dimensional (N,C,H,T), but got {npy_data.ndim} dimensions.")
        print(f"INFO: Loaded input NPY data '{args.input}' with shape {npy_data.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load NPY input '{args.input}': {e}")
        sys.exit(1)

    results = []

    for model_path in args.models:
        # ... (模型路径检查) ...
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}. Skipping.")
            results.append({'name': os.path.basename(model_path), 'stats': None}) # Record error state
            continue

        model_name = os.path.basename(model_path)
        file_size_mb = os.path.getsize(model_path) / 1024 / 1024
        model_format = "Unknown"
        latencies, cpu_usages, mem_usages = [], [], []
        stats = None

        print(f"\n--- Benchmarking Model: {model_name} ---")

        try:
            if model_path.endswith(".tflite"):
                if TFLiteInterpreter is None:
                    print("ERROR: TFLite Interpreter not available. Skipping TFLite model.")
                    stats = None # Ensure error state
                else:
                    model_format = "TFLite"
                    print("INFO: Initializing TFLite Interpreter...")
                    num_threads = args.tflite_threads if args.tflite_threads > 0 else None
                    interpreter = TFLiteInterpreter(model_path=model_path, num_threads=num_threads, experimental_delegates=[])
                    interpreter.allocate_tensors()
                    print(f"INFO: TFLite interpreter initialized (requested threads: {num_threads if num_threads else 'default'}).") # 修改后的信息
                    latencies, cpu_usages, mem_usages = measure_tflite(interpreter, npy_data, args.warmup)
                    del interpreter # Explicit cleanup (optional)

            elif model_path.endswith(".onnx"):
                if ort is None:
                    print("ERROR: ONNX Runtime not available. Skipping ONNX model.")
                    stats = None # Ensure error state
                else:
                    model_format = "ONNX"
                    print("INFO: Initializing ONNX Runtime Session...")
                    sess_options = ort.SessionOptions()
                    # sess_options.intra_op_num_threads = args.tflite_threads if args.tflite_threads > 0 else 1 # Example
                    print(f"INFO: ONNX Runtime using provider: {args.onnx_provider}")
                    # Check if provider is available before creating session
                    available_providers = ort.get_available_providers()
                    if args.onnx_provider not in available_providers:
                         print(f"Warning: ONNX Provider '{args.onnx_provider}' not available. Available: {available_providers}. Falling back to CPUExecutionProvider.")
                         provider_to_use = 'CPUExecutionProvider'
                    else:
                         provider_to_use = args.onnx_provider

                    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=[provider_to_use])
                    latencies, cpu_usages, mem_usages = measure_onnx(session, npy_data, args.warmup)
                    del session # Explicit cleanup (optional)

            else:
                print(f"ERROR: Unknown model format for {model_name}. Skipping.")
                stats = None # Ensure error state

            # Calculate statistics only if measurements were successful
            if latencies: # Check if latencies list is not empty
                 avg_latency = mean(latencies)
                 std_latency = stdev(latencies) if len(latencies) > 1 else 0
                 avg_cpu = mean(cpu_usages) if cpu_usages else 0
                 max_ram = max(mem_usages) if mem_usages else 0
                 stats = {
                     'format': model_format,
                     'latency': avg_latency,
                     'latency_std': std_latency,
                     'cpu': avg_cpu,
                     'ram': max_ram,
                     'size': file_size_mb,
                 }
            elif stats is not False: # Ensure stats is None if no latencies and not already set to None by other errors
                 print(f"Warning: No measurements collected for {model_name}.")
                 stats = None

        except Exception as e:
            print(f"ERROR: Failed to benchmark {model_name}: {e}")
            import traceback
            traceback.print_exc()
            stats = None # Ensure stats is None on any exception during benchmark

        results.append({'name': model_name, 'stats': stats})

    # --- Print Markdown table ---
    print("\n--- Benchmark Results ---")
    print('| Model                           | Format | Latency (ms) (± std) | CPU (%) | RAM (MB) | Size (MB) |')
    print('|---------------------------------|--------|----------------------|---------|----------|-----------|')
    for res in results:
        name = res['name']
        stats = res['stats']
        # Try to get size even on error, using the original model_path which is more reliable
        model_exists = os.path.exists(next((m for m in args.models if os.path.basename(m) == name), None) or name) # Find full path or check basename
        full_model_path_for_size = next((m for m in args.models if os.path.basename(m) == name), name) if model_exists else None
        size_on_error = os.path.getsize(full_model_path_for_size) / 1024 / 1024 if full_model_path_for_size and os.path.exists(full_model_path_for_size) else 0.0

        if stats:
            lat_std_str = f"{stats['latency']:.2f} (± {stats['latency_std']:.2f})"
            print(
                f"| {name:<31} | {stats['format']:<6} | {lat_std_str:<20} | {stats['cpu']:<7.1f} | {stats['ram']:<8.1f} | {stats['size']:<9.2f} |"
            )
        else:
            # Determine format based on name if possible, otherwise 'ERROR'
            fmt = "TFLite" if name.endswith(".tflite") else "ONNX" if name.endswith(".onnx") else "ERROR"
            print(f"| {name:<31} | {fmt:<6} | ---                  | ---     | ---      | {size_on_error:<9.2f} |")

if __name__ == '__main__':
    main()