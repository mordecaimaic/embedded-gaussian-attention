import os, json, numpy as np, torch
from torch.utils.data import Dataset

class MelDataset(Dataset):
    def __init__(self, feats_root, split):
        self.root = feats_root
        self.mapping = json.load(open(os.path.join(feats_root,"mapping.json")))
        # 只保留当前 split 的条目
        self.items = [(k,v) for k,v in self.mapping.items() if k.startswith(split)]
        # label -> idx
        labels = sorted(set(self.mapping.values()))
        self.lut = {l:i for i,l in enumerate(labels)}
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        feat = np.load(os.path.join(self.root, path))        # [40,T]
        feat = torch.from_numpy(feat).unsqueeze(0)           # [1,40,T]
        return feat, self.lut[label]