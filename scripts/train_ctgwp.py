"""
train_ctgwp.py  -
Gaussian Attention (CTGWP) 复现脚本
-----------------------------------------------------
* 依赖  : train_baseline.MelDataset, collate_pad
* 模型  : 4‑layer CNN + CTGWP 权重池化

用法示例:
python scripts/train_ctgwp.py \
    --feats_dir data/feats \
    --out models/ctgwp \
    --epochs 100 --bs 64 --lr 1e-3
"""
import os, argparse, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_baseline import MelDataset, collate_pad  # 直接复用 Dataset

# --------------------------------------------------
#  模型定义
# --------------------------------------------------
class CTGWP(nn.Module):
    """可训练高斯权重池化 (Central‑Tendency Gaussian Weight Pooling)"""
    def __init__(self, T: int):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(float(T / 4)))   # learnable σ
        self.register_buffer("t", torch.arange(T).float())

    def forward(self, F: torch.Tensor):        # F:[B,C,T]
        sigma = self.sigma.abs() + 1e-6
        center = (F.size(2) - 1) / 2
        w = torch.exp(-((self.t - center) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        return (F * w).sum(dim=2)              # [B,C]

class CTGWP_KWS(nn.Module):
    """4×CNN + CTGWP + FC"""
    def __init__(self, n_cls: int, target_len: int = 98):
        super().__init__()
        in_ch, layers = 1, []
        for out in [16, 32, 64, 128]:
            layers += [
                nn.Conv2d(in_ch, out, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(out),
                nn.ReLU(),
                # 只在频率维度 (40→20→10…) 做池化，不缩短时间维度
                nn.MaxPool2d(kernel_size=(2, 1))
            ]
            in_ch = out
        self.conv = nn.Sequential(*layers)
        self.ctgwp = CTGWP(T=target_len)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):            # x:[B,1,40,T]
        f = self.conv(x)             # [B,128,H,T']
        # 仍可能有 H>1，先在频率维度做平均池化，保持时间帧数不变
        f = f.mean(dim=2)            # [B,128,T']
        v = self.ctgwp(f)            # [B,128]
        return self.fc(v)

# --------------------------------------------------
#  训练循环
# --------------------------------------------------
def train(args):
    os.makedirs(args.out, exist_ok=True)
    train_ds = MelDataset(args.feats_dir, "train")
    val_ds   = MelDataset(args.feats_dir, "val")
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          collate_fn=collate_pad, num_workers=4)
    val_dl   = DataLoader(val_ds, batch_size=64, collate_fn=collate_pad, num_workers=4)

    model = CTGWP_KWS(len(train_ds.lut)).to(args.device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    crit  = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); loss = crit(model(x), y)
            loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # -------- validation --------
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
        acc = correct / tot
        print(f"Epoch {epoch}  val_acc={acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.out, "best.ckpt"))
            print(f"  ✔ new best saved ({best:.4f})")

    print("Training finished. best val acc:", best)

# --------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats_dir", required=True)
    ap.add_argument("--out", default="models/ctgwp")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(ap.parse_args())