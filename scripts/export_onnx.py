#!/usr/bin/env python
"""
export_onnx.py  —  将已训练的 PyTorch KWS 模型导出为 ONNX
------------------------------------------------------------
支持 Baseline / ctgwp / ctgwp_tuned / ctgwp_adv；可选 fp16。
"""
import argparse, os, torch, onnx
from train_baseline   import BaselineKWS, MelDataset
from train_ctgwp      import CTGWP_KWS   as Orig  # 原版
from ctgwp_tuned      import CTGWP_KWS   as Tuned
from ctgwp_advanced   import CTGWP_KWS   as Adv

def build(model_type, n_cls, target_len):
    if model_type == "baseline":
        return BaselineKWS(n_cls)
    elif model_type == "ctgwp":
        return Orig(n_cls, target_len)
    elif model_type == "ctgwp_tuned":
        return Tuned(n_cls, target_len)
    elif model_type == "ctgwp_adv":
        return Adv(n_cls, target_len)
    else:
        raise ValueError(f"Unknown model_type {model_type}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out",        required=True, help="输出 onnx 文件名")
    ap.add_argument("--model_type", default="ctgwp_adv",
                    choices=["baseline","ctgwp","ctgwp_tuned","ctgwp_adv"])
    ap.add_argument("--feats_dir",  default="data/feats",
                    help="读取 mapping.json 以确定类别数/target_len")
    ap.add_argument("--half", action="store_true",
                    help="导出 fp16 – 需要 GPU 支持")
    ap.add_argument("--dynamic", action="store_true",
                    help="动态 batch / T 维")
    args = ap.parse_args()

    # 获取类别数 / target_len
    ds = MelDataset(args.feats_dir, "train")
    n_cls, target_len = len(ds.lut), ds.target_len

    # 构建并加载权重
    model = build(args.model_type, n_cls, target_len)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"),
                          strict=False)
    model.eval()

    if args.half:
        model.half()

    # 构造 dummy 输入
    dummy = torch.randn(1, 1, 40, target_len)
    if args.half: dummy = dummy.half()

    # 导出
    torch.onnx.export(
        model, dummy, args.out,
        input_names = ["mel"],
        output_names= ["prob"],
        dynamic_axes = {"mel": {0: "B", 3: "T"},
                        "prob": {0: "B"}} if args.dynamic else None,
        opset_version = 14
    )
    print("ONNX saved to", args.out)
    onnx.checker.check_model(args.out)
    print("✔ Model checked OK")

if __name__ == "__main__":
    main()