"""
ctgwp_tuned.py
小幅调参版：log_sigma + CosineAnnealing + 150 epoch
"""

import os, argparse, torch, math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_baseline import MelDataset, collate_pad

# ---------------- CTGWP ----------------
class CTGWP(nn.Module):
    def __init__(self, init_T: int = 98):
        super().__init__()
        # log_sigma 让梯度平滑些
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_T / 6.0)))

    def forward(self, F):
        B, C, T = F.shape
        sigma = self.log_sigma.exp() + 1e-6
        t = torch.arange(T, device=F.device, dtype=F.dtype)
        center = (T - 1) / 2
        w = torch.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        return (F * w).sum(dim=2)

# --------------- Model -----------------
class CTGWP_KWS(nn.Module):
    def __init__(self, n_cls: int, target_len=98):
        super().__init__()
        in_ch, layers = 1, []
        for out in [16, 32, 64, 128]:
            layers += [nn.Conv2d(in_ch, out, 3, 1, 1),
                       nn.BatchNorm2d(out), nn.ReLU(),
                       nn.MaxPool2d((2, 1))]
            in_ch = out
        self.conv = nn.Sequential(*layers)
        self.ctgwp = CTGWP(target_len)
        self.bn1d = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        f = self.conv(x).mean(dim=2)      # [B,128,T]
        v = self.bn1d(self.ctgwp(f))      # [B,128]
        return self.fc(v)

# -------------- Train Loop -------------
def train(args):
    os.makedirs(args.out, exist_ok=True)
    train_ds, val_ds = MelDataset(args.feats_dir, "train"), MelDataset(args.feats_dir, "val")
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          collate_fn=collate_pad, num_workers=4)
    val_dl   = DataLoader(val_ds,  batch_size=64, shuffle=False,
                          collate_fn=collate_pad, num_workers=4)

    model = CTGWP_KWS(len(train_ds.lut)).to(args.device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Ep{epoch}")
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # ---- val ----
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); tot += y.size(0)
        acc = correct / tot
        print(f"Ep {epoch}  val_acc={acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.out, "best.ckpt"))
            print(f"  ✔ new best {best:.4f}")

    print("best val acc:", best)

# -------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats_dir", required=True)
    ap.add_argument("--out", default="models/ctgwp_tuned")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(ap.parse_args())