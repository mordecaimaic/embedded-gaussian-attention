# evaluate.py — supports baseline / ctgwp / ctgwp_tuned / ctgwp_adv
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
    return CTGWP_KWS            # 原版

@torch.no_grad()
def eval_model(model_path, feats_dir, split="test",
               device="cpu", batch_size=64):
    # dataset meta
    ref = MelDataset(feats_dir, "train")
    n_cls, t_len = len(ref.lut), ref.target_len

    Model = choose_model(model_path)
    model = Model(n_cls).to(device)

    # ---- 兼容旧版 torch.load ----
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    # 用 strict=False 兼容不同字段
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict: missing {len(missing)}, "
              f"unexpected {len(unexpected)}")

    model.eval()

    # dataloader
    ds = MelDataset(feats_dir, split, target_len=t_len)
    dl = DataLoader(ds, batch_size=batch_size,
                    collate_fn=collate_pad, num_workers=4)

    correct = tot = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        tot += y.size(0)

    if tot == 0:
        raise RuntimeError(f"No samples in split '{split}'")
    return correct / tot

# ---------------- CLI ----------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser("Evaluate accuracy (baseline / CTGWP variants)")
    pa.add_argument("--model_path", required=True)
    pa.add_argument("--feats_dir", required=True)
    pa.add_argument("--split", default="test",
                    choices=["train", "val", "test"])
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--out", help="optional json output file")
    args = pa.parse_args()

    acc = eval_model(args.model_path, args.feats_dir,
                     split=args.split, device=args.device)

    print(f"\nAccuracy on {args.split}: {acc:.4%}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"accuracy": acc}, open(args.out, "w"), indent=4)
        print("Saved to", args.out)