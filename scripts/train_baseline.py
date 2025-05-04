"""
train_baseline.py

Dataset + collate function that guarantee each audio feature tensor
 fed to the model has the same time dimension length.

Author: <your‑name>
Date  : 2025‑05‑04
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MelDataset(Dataset):
    """
    MelDataset 负责从预先保存的 Mel 频谱文件（.npy）中读取数据，
    并在 __getitem__ 阶段进行 **长度标准化**：
        - 若 T < target_len  → 右侧零填充
        - 若 T > target_len  → 以中心为基准裁剪
    这样就能保证 batch 内张量尺寸一致。
    """
    def __init__(self,
                 feats_root: str,
                 split: str,
                 target_len: int = 98) -> None:
        """
        Args
        ----
        feats_root : str
            预处理生成的特征根目录（下有 train/ val/ test/）
        split : str
            "train" | "val" | "test"
        target_len : int
            统一对齐的时间帧长度（论文里 1 s ≈ 98 帧）
        """
        super().__init__()
        self.root = Path(feats_root)
        self.split = split
        self.target_len = target_len

        # 加载标签查找表和文件清单
        self.lut = self._load_lut()         # label → index
        self.data = self._load_data()       # [(relative_path, label_idx), ...]

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _load_lut(self) -> dict:
        """
        读取 mapping.json 并生成 label→index 映射表

        mapping.json 格式:
        {
            "train/yes/0a7c2a8d_nohash_0.wav.npy": "yes",
            ...
        }
        """
        mapping_path = self.root / "mapping.json"
        if not mapping_path.exists():
            raise FileNotFoundError("mapping.json not found. "
                                    "Did you run preprocess.py ?")
        mapping = json.load(open(mapping_path))
        labels = sorted({v for v in mapping.values()})
        return {lbl: idx for idx, lbl in enumerate(labels)}

    def _load_data(self) -> List[Tuple[str, int]]:
        """
        根据 split 过滤 mapping.json 条目
        """
        mapping = json.load(open(self.root / "mapping.json"))
        items = []
        for rel, lbl in mapping.items():
            if rel.startswith(self.split):
                items.append((rel, self.lut[lbl]))
        if not items:
            raise RuntimeError(f"No data found for split='{self.split}'")
        return items

    # ------------------------------------------------------------------
    # torch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        rel_path, label = self.data[idx]
        # 载入 numpy → torch
        feat = np.load(self.root / rel_path)           # [40, T] float32
        feat = torch.from_numpy(feat).unsqueeze(0)     # [1, 40, T]

        # -------- 统一长度 ------------
        T = feat.size(-1)
        if T < self.target_len:                       # pad right
            pad_len = self.target_len - T
            feat = F.pad(feat, (0, pad_len))
        elif T > self.target_len:                     # center crop
            start = (T - self.target_len) // 2
            feat = feat[..., start:start + self.target_len]

        return feat, label


# ----------------------------------------------------------------------
# DataLoader collate function (变长对齐)
# ----------------------------------------------------------------------
def collate_pad(batch):
    """
    将同 batch 内的 tensor 按最大帧数零填充
    用于非固定长度输入的 DataLoader

    batch : List[(tensor,label)]
    returns
    -------
    feats  : FloatTensor  [B, 1, 40, max_T]
    labels : LongTensor   [B]
    """
    feats, labels = zip(*batch)
    max_T = max(f.size(-1) for f in feats)
    feats = [F.pad(f, (0, max_T - f.size(-1))) if f.size(-1) < max_T else f
             for f in feats]
    feats = torch.stack(feats)          # [B, 1, 40, max_T]
    labels = torch.tensor(labels)
    return feats, labels


# ----------------------------------------------------------------------
# Quick sanity check (run directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ds = MelDataset("data/feats", "train")
    print("train samples:", len(ds), "label vocab:", len(ds.lut))
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_pad)
    x, y = next(iter(dl))
    print("batch shape:", x.shape, "labels:", y)


# ----------------------------------------------------------------------
# Training loop (baseline CNN + GAP)
# ----------------------------------------------------------------------
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaselineKWS(nn.Module):
    """4‑layer CNN  + GAP baseline."""
    def __init__(self, n_cls: int):
        super().__init__()
        chs = [16, 32, 64, 128]
        layers = []
        in_ch = 1
        for out in chs:
            layers += [
                nn.Conv2d(in_ch, out, 3, 1, 1),
                nn.BatchNorm2d(out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        x = self.conv(x)           # [B,128,H=1,W]
        x = self.gap(x).flatten(1) # [B,128]
        return self.fc(x)


def train_baseline(args):
    """train and save best.ckpt to args.out"""
    os.makedirs(args.out, exist_ok=True)

    train_ds = MelDataset(args.feats_dir, "train")
    val_ds   = MelDataset(args.feats_dir, "val")
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          collate_fn=collate_pad, num_workers=4)
    val_dl   = DataLoader(val_ds, batch_size=64, collate_fn=collate_pad, num_workers=4)

    model = BaselineKWS(len(train_ds.lut)).to(args.device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        # ---------------- validation ----------------
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
        acc = correct / tot
        print(f"Epoch {epoch}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.out, "best.ckpt"))
            print(f"  ✔ new best saved ({best_acc:.4f})")

    print("Training done. best val acc:", best_acc)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--feats_dir", required=True, help="directory produced by preprocess.py")
    ap.add_argument("--out", default="models/baseline")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_baseline(args)