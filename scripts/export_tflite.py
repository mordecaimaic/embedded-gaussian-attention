#!/usr/bin/env python
"""
export_tflite.py — ONNX → TFLite (fp32 / int8)
----------------------------------------------
require: onnx, onnx-tf, tensorflow[‑cpu]
"""
import argparse, os, tempfile, onnx
import onnx_tf.backend as backend
import tensorflow as tf

def onnx_to_tf(onnx_path, tf_dir):
    model = onnx.load(onnx_path)
    tf_rep = backend.prepare(model, device="CPU")   # 纯 CPU
    tf_rep.export_graph(tf_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out",  required=True, help=".tflite file")
    ap.add_argument("--int8", action="store_true",
                    help="post‑training dynamic int8 quant")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = os.path.join(tmp, "saved_model")
        onnx_to_tf(args.onnx, saved_dir)         # 1) ONNX → TF

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        if args.int8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()       # 2) TF → TFLite

        with open(args.out, "wb") as f:
            f.write(tflite_model)
    print("TFLite saved to", args.out)

if __name__ == "__main__":
    main()