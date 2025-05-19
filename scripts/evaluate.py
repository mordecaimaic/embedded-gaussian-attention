# evaluate.py — supports baseline / ctgwp / ctgwp_tuned / ctgwp_adv
# Calculates Overall Acc/FAR/FRR and Per-Keyword FAR/FRR
# ---------------------------------------------------------------

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from collections import defaultdict # 用于存储 per-keyword 统计
from tqdm import tqdm

# --- 导入 ---
try:
    from train_baseline import MelDataset as FixedLenMelDataset
    from train_baseline import BaselineKWS
    from train_ctgwp   import CTGWP_KWS
    from ctgwp_tuned   import CTGWP_KWS as TunedKWS
    from ctgwp_advanced import CTGWP_KWS as AdvKWS
except ImportError as e:
    print(f"Error: Failed to import required modules: {e}")
    print("Ensure train_*.py files are accessible.")
    exit(1)
# ==========================================

def choose_model(path: str):
    """Chooses the correct model class based on the model path."""
    low = path.lower()
    if "ctgwp" not in low:
        print(f"Choosing model: BaselineKWS for path: {path}")
        return BaselineKWS
    if "adv" in low:
        print(f"Choosing model: AdvKWS for path: {path}")
        return AdvKWS
    if "tuned" in low:
        print(f"Choosing model: TunedKWS for path: {path}")
        return TunedKWS
    print(f"Choosing model: CTGWP_KWS (Original) for path: {path}")
    return CTGWP_KWS

# 使用 torch.no_grad 作为函数修饰器
@torch.no_grad()
def eval_model(model_path, feats_dir, split="test",
               device="cpu", batch_size=64,
               target_keywords=None):
    """
    Evaluates a KWS model.
    Returns a dictionary containing:
        - overall_accuracy
        - overall_far
        - overall_frr
        - per_keyword_metrics: {keyword: {"far": val, "frr": val}}
    """
    # --- 获取数据集信息和标签映射 ---
    try:
        ref_ds = FixedLenMelDataset(feats_dir, "train")
    except Exception as e:
        print(f"Error loading reference dataset: {e}")
        return None # 返回 None 表示失败

    n_cls = len(ref_ds.lut)
    t_len = ref_ds.target_len
    label_to_index = ref_ds.lut
    index_to_label = {v: k for k, v in label_to_index.items()}

    # --- 确定关键词和非关键词索引 ---
    if target_keywords is None:
        target_keywords = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        print(f"[Info] target_keywords not provided, using default: {target_keywords}")

    known_keywords_set = set(target_keywords)
    keyword_indices = {label_to_index[lbl] for lbl in known_keywords_set if lbl in label_to_index}
    missing_kws = known_keywords_set - {index_to_label[idx] for idx in keyword_indices}
    if missing_kws:
        print(f"[Warning] Target keywords not found in dataset labels: {missing_kws}")
    if not keyword_indices:
         print("[Error] No keyword indices identified!")
         return None

    all_indices = set(range(n_cls))
    non_keyword_indices = all_indices - keyword_indices

    print(f"Keyword indices: {keyword_indices} ({len(keyword_indices)} keywords)")
    print(f"Non-keyword indices: {non_keyword_indices} ({len(non_keyword_indices)} non-keywords)")
    print(f"Total classes: {n_cls}")

    # --- 加载模型 ---
    Model = choose_model(model_path)
    try:
      model = Model(n_cls=n_cls, target_len=t_len).to(device)
    except TypeError:
      model = Model(n_cls=n_cls).to(device)

    # --- 加载状态字典 ---
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
        print(f"[Info] Loaded state_dict (weights_only=True).")
    except Exception as e_safe:
        print(f"[Warning] weights_only=True failed ({e_safe}). Falling back.")
        try:
            state = torch.load(model_path, map_location=device, weights_only=False)
            print(f"[Warning] Loaded state_dict (weights_only=False).")
        except Exception as e_unsafe:
            print(f"[Error] Failed to load model weights from {model_path}.")
            print(f"  Error (safe): {e_safe}\n  Error (unsafe): {e_unsafe}")
            return None
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict: missing {len(missing)}, unexpected {len(unexpected)}")
    model.eval()

    # --- 创建 Dataloader ---
    try:
        eval_ds = FixedLenMelDataset(feats_dir, split, target_len=t_len)
        if len(eval_ds) == 0:
             print(f"Error: No samples found for split '{split}'.")
             return None
    except Exception as e:
        print(f"Error loading evaluation dataset for split '{split}': {e}")
        return None
    dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 初始化统计变量 ---
    # Overall metrics
    total_samples = 0
    correct_samples = 0
    overall_false_acceptances = 0
    overall_false_rejections = 0
    total_keywords_samples = 0
    total_nonkeywords_samples = 0
    # Per-keyword metrics (TP, TN, FP, FN for binary classification view)
    per_keyword_counts = {kw_idx: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for kw_idx in keyword_indices}

    # --- 评估循环 ---
    pbar_eval = tqdm(dl, desc=f"Evaluating on {split}", unit="batch")
    for x, y_true_idx_batch in pbar_eval:
        x, y_true_idx_batch = x.to(device), y_true_idx_batch.to(device)
        try:
            y_pred_logits = model(x)
            y_pred_idx_batch = y_pred_logits.argmax(dim=1)
        except Exception as e:
            print(f"\nError during model inference: {e}")
            return None

        # 逐个样本进行统计
        for i in range(y_true_idx_batch.size(0)):
            true_idx = y_true_idx_batch[i].item()
            pred_idx = y_pred_idx_batch[i].item()

            # --- Overall Metrics Calculation ---
            is_true_keyword_overall = true_idx in keyword_indices
            is_pred_keyword_overall = pred_idx in keyword_indices

            total_samples += 1
            if is_true_keyword_overall:
                total_keywords_samples += 1
            else:
                total_nonkeywords_samples += 1

            if pred_idx == true_idx:
                correct_samples += 1

            # Overall FA: True Non-Keyword -> Pred Keyword
            if not is_true_keyword_overall and is_pred_keyword_overall:
                overall_false_acceptances += 1

            # Overall FR: True Keyword -> Pred Non-Keyword
            if is_true_keyword_overall and not is_pred_keyword_overall:
                overall_false_rejections += 1

            # --- Per-Keyword Metrics Calculation ---
            for kw_idx in keyword_indices:
                # Treat current kw_idx as the "positive" class
                is_true_target = (true_idx == kw_idx)
                is_pred_target = (pred_idx == kw_idx)

                if is_true_target and is_pred_target:
                    per_keyword_counts[kw_idx]['TP'] += 1
                elif is_true_target and not is_pred_target:
                    per_keyword_counts[kw_idx]['FN'] += 1
                elif not is_true_target and is_pred_target:
                    per_keyword_counts[kw_idx]['FP'] += 1
                elif not is_true_target and not is_pred_target:
                    per_keyword_counts[kw_idx]['TN'] += 1

    # --- 计算最终指标 ---
    if total_samples == 0: return None # Should have been caught earlier

    # Overall Metrics
    overall_accuracy = correct_samples / total_samples
    overall_far = (overall_false_acceptances / total_nonkeywords_samples) if total_nonkeywords_samples > 0 else 0.0
    overall_frr = (overall_false_rejections / total_keywords_samples) if total_keywords_samples > 0 else 0.0

    # Per-Keyword Metrics
    per_keyword_metrics = {}
    for kw_idx, counts in per_keyword_counts.items():
        kw_label = index_to_label.get(kw_idx, f"idx_{kw_idx}")
        TP, TN, FP, FN = counts['TP'], counts['TN'], counts['FP'], counts['FN']
        kw_far = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        kw_frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        per_keyword_metrics[kw_label] = {"far": kw_far, "frr": kw_frr}
        # Optionally calculate per-keyword accuracy if needed:
        # kw_acc = (TP + TN) / total_samples if total_samples > 0 else 0.0
        # per_keyword_metrics[kw_label]["accuracy"] = kw_acc

    # 整理返回结果
    results = {
        "overall_accuracy": overall_accuracy,
        "overall_far": overall_far,
        "overall_frr": overall_frr,
        "per_keyword_metrics": per_keyword_metrics
    }
    return results

# ---------------- CLI ----------------
if __name__ == "__main__":
    # 使用一个 ArgumentParser
    pa = argparse.ArgumentParser(description="Evaluate KWS model: Overall Acc/FAR/FRR and Per-Keyword FAR/FRR.")
    pa.add_argument("--model_path", required=True, help="Path to the saved model checkpoint (.ckpt).")
    pa.add_argument("--feats_dir", required=True, help="Path to the directory containing preprocessed features.")
    pa.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate on.")
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for evaluation.")
    pa.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    pa.add_argument("--out", help="Optional JSON file to save the metrics.")
    pa.add_argument("--target_keywords", nargs='+',
                    default=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"], # GSC v1/v2 核心10词
                    help="List of target keyword labels. These are used to define keywords vs non-keywords.")

    args = pa.parse_args()

    # 检查路径
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.isdir(args.feats_dir):
        raise NotADirectoryError(f"Features directory not found: {args.feats_dir}")

    # 执行评估
    eval_results = eval_model(args.model_path, args.feats_dir,
                              split=args.split, device=args.device,
                              batch_size=args.batch_size,
                              target_keywords=args.target_keywords)

    if eval_results is None:
        print("Evaluation failed.")
        exit(1)

    # 准备并打印/保存结果
    print(f"\n--- Evaluation Results on split '{args.split}' ---")
    print(f"Model: {args.model_path}")
    print(f"Overall Accuracy         : {eval_results['overall_accuracy']:.4%}")
    print(f"Overall FAR (vs Non-KW)  : {eval_results['overall_far']:.4%}")
    print(f"Overall FRR (vs Non-KW)  : {eval_results['overall_frr']:.4%}")
    print("\n--- Per-Keyword Metrics ---")
    if eval_results['per_keyword_metrics']:
        max_kw_len = max(len(kw) for kw in eval_results['per_keyword_metrics'])
        print(f"{'Keyword'.ljust(max_kw_len)} |   FAR (%) |   FRR (%) |")
        print(f"{'-'*(max_kw_len+1)}|-----------|-----------|")
        for kw, metrics in sorted(eval_results['per_keyword_metrics'].items()):
            print(f"{kw.ljust(max_kw_len)} | {metrics['far']*100:9.4f} | {metrics['frr']*100:9.4f} |")
    else:
        print("No per-keyword metrics calculated (maybe no target keywords found?).")
    print("---------------------------------------------")

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            # 保存所有结果到一个 JSON
            with open(args.out, "w") as f:
                json.dump(eval_results, f, indent=4)
            print(f"All metrics saved to {args.out}")
        except IOError as e:
            print(f"Error saving metrics to {args.out}: {e}")