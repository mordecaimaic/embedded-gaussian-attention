# scripts/evaluate.py  — 统计 Accuracy / FAR / FRR
# ---------------------------------------------------------------
import argparse, json, os, torch
from torch.utils.data import DataLoader
from train_baseline import MelDataset, collate_pad, BaselineKWS
from train_ctgwp   import CTGWP_KWS            # 原版
from ctgwp_tuned   import CTGWP_KWS as TunedKWS    # 轻调参
from ctgwp_advanced import CTGWP_KWS as AdvKWS     # advanced

def choose_model(path: str):
    low = path.lower()
    if "ctgwp" not in low:
        return BaselineKWS
    if "adv" in low:
        return AdvKWS
    if "tuned" in low:
        return TunedKWS
    return CTGWP_KWS

@torch.no_grad()
def eval_model(model_path, feats_dir, split="test",
               device="cpu", batch_size=64, target_label=None):
    if target_label is None:
        raise ValueError("请通过 --target_label 指定要评估的关键词标签")
    # 载入数据集元信息
    ref = MelDataset(feats_dir, "train")
    n_cls, t_len = len(ref.lut), ref.target_len

    Model = choose_model(model_path)
    model = Model(n_cls).to(device)

    # 加载权重（兼容不同版本）
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 数据加载
    ds = MelDataset(feats_dir, split, target_len=t_len)
    dl = DataLoader(ds, batch_size=batch_size,
                    collate_fn=collate_pad, num_workers=4)

    # 确认并映射评估目标标签
    lut = ref.lut
    if isinstance(target_label, str):
        if target_label not in lut:
            raise ValueError(f"标签 '{target_label}' 未在训练集中找到")
        target_idx = lut[target_label]
    else:
        target_idx = int(target_label)

    # 统计指标
    TP = TN = FP = FN = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        # 假设类别 0 是 “非唤醒词”，其他 1..K-1 是 唤醒词
        for p, gt in zip(pred.cpu(), y.cpu()):
            if gt == target_idx:
                if p == target_idx: TP += 1
                else:              FN += 1
            else:
                if p == target_idx: FP += 1
                else:              TN += 1

    total = TP + TN + FP + FN
    if total == 0:
        raise RuntimeError(f"No samples in split '{split}'")

    accuracy = (TP + TN) / total
    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    return accuracy, FAR, FRR

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate Accuracy / FAR / FRR for a keyword")
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--feats_dir",    required=True)
    parser.add_argument("--split",        default="test", choices=["train","val","test"])
    parser.add_argument("--device",       default="cuda:1")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--target_label", required=True, help="指定要评估的关键词标签（字符串或索引）")
    parser.add_argument("--out",          help="optional JSON output file")
    args = parser.parse_args()

    acc, far, frr = eval_model(
        args.model_path, args.feats_dir,
        split=args.split, device=args.device,
        batch_size=args.batch_size,
        target_label=args.target_label
    )

    print(f"\nResults on {args.split}, target='{args.target_label}':")
    print(f"  Accuracy = {acc*100:.2f}%")
    print(f"  FAR      = {far*100:.2f}%")
    print(f"  FRR      = {frr*100:.2f}%")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"accuracy": acc, "FAR": far, "FRR": frr}, open(args.out, "w"), indent=4)
        print("Saved to", args.out)