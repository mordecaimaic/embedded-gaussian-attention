# scripts/ctgwp_advanced.py
# ------------------------------------------------------------
# Advanced CTGWP training script — 目标：Val / Test 精度 ≥ Baseline
# ------------------------------------------------------------
import os, argparse, math, random, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_baseline import MelDataset, collate_pad, BaselineKWS


# -------------- CTGWP 模块（动态 T，自带 log_sigma） --------------
class CTGWP(nn.Module):
    def __init__(self, init_T: int = 98):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_T / 6.0)))

    def forward(self, F):  # F:[B,C,T]
        _, _, T = F.shape
        sigma = self.log_sigma.exp() + 1e-6
        t = torch.arange(T, device=F.device, dtype=F.dtype)
        center = (T - 1) / 2
        w = torch.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        return (F * w).sum(dim=2)  # [B,C]


# ---------------- KWS 模型（4×CNN + CTGWP） -----------------------
class CTGWP_KWS(nn.Module):
    def __init__(self, n_cls: int, target_len=98, dropout=0.1):
        super().__init__()
        # 4‑layer CNN（与 baseline 结构一致，便于迁移卷积权重）
        in_ch, layers = 1, []
        for out in [16, 32, 64, 128]:
            layers += [nn.Conv2d(in_ch, out, 3, 1, 1),
                       nn.BatchNorm2d(out),
                       nn.ReLU(),
                       nn.MaxPool2d((2, 1))]  # 只池化频率轴
            in_ch = out
        self.conv = nn.Sequential(*layers)

        self.ctgwp = CTGWP(target_len)
        self.bn1d = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):  # x:[B,1,40,T]
        f = self.conv(x).mean(dim=2)  # [B,128,T]  ↓freq GAP
        v = self.ctgwp(f)  # [B,128]
        v = self.drop(self.bn1d(v))
        return self.fc(v)


# -------- 迁移卷积权重（不依赖 Baseline 类内部实现） ---------------
def load_conv_weights(model_conv, ckpt_path, device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("No baseline ckpt provided — training conv from scratch.")
        return
    state = torch.load(ckpt_path, map_location=device)
    # 只提取名含 ".conv." 的权重
    conv_w = {k.split("conv.", 1)[-1]: v for k, v in state.items()
              if ".conv." in k or k.startswith("conv.")}
    missing, _ = model_conv.load_state_dict(conv_w, strict=False)
    print(f"Loaded {len(conv_w) - len(missing)} conv params from baseline.")


# ---------------------- SpecAug 时间遮挡 --------------------------
def time_mask(batch, max_frac=0.1):
    B, _, _, T = batch.shape
    ml = int(T * max_frac)
    for i in range(B):
        s = random.randint(0, T - ml - 1)
        batch[i, :, :, s:s + ml] = 0.0


# --------------------------- Train --------------------------------
def train(args):
    os.makedirs(args.out, exist_ok=True)
    train_ds, val_ds = MelDataset(args.feats_dir, "train"), MelDataset(args.feats_dir, "val")
    train_dl = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collate_pad,
        num_workers=4,
        pin_memory=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_pad,
        num_workers=4,
        pin_memory=True,
    )

    model = CTGWP_KWS(len(train_ds.lut)).to(args.device)
    load_conv_weights(model.conv, args.baseline_ckpt, args.device)

    # Optimizer + CosineLR（含 5 epoch warmup）
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_dl)
    warmup_steps = 5 * len(train_dl)
    scheduler = optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / warmup_steps if s < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) /
                                 max(1, total_steps - warmup_steps)))
    )
    crit = nn.CrossEntropyLoss()

    # 先冻结卷积 5 epoch
    for p in model.conv.parameters(): p.requires_grad = False
    unfreeze_epoch = 5

    best, patience = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        if epoch == unfreeze_epoch:
            for p in model.conv.parameters(): p.requires_grad = True
            print("❄ Unfroze conv layers")

        pbar = tqdm(train_dl, desc=f"Ep{epoch}")
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            time_mask(x, 0.1)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step();
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # -------- Validation --------
        model.eval();
        correct = tot = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item();
                tot += y.size(0)
        acc = correct / tot
        print(f"Ep {epoch}  val_acc={acc:.4f}")

        if acc > best + 1e-3:  # 若提升超过 0.1 pp
            best = acc;
            patience = 0
            torch.save(model.state_dict(), os.path.join(args.out, "best.ckpt"))
            print(f"  ✔ new best {best:.4f}")
        else:
            patience += 1
            if patience >= 15:
                print("Early stop (val no improve 15 epochs)")
                break

    print("Finished. Best val acc:", best)


# --------------------------- CLI ----------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feats_dir", required=True)
    p.add_argument("--out", default="models/ctgwp_adv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--baseline_ckpt", default="models/baseline/best.ckpt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(p.parse_args())
