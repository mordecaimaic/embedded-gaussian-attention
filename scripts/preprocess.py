import argparse, os, json, soundfile as sf, numpy as np, torch
import torchaudio.transforms as T
from tqdm import tqdm, trange

mel = T.MelSpectrogram(sample_rate=16000,
                       n_fft=512, hop_length=160, win_length=400,
                       n_mels=40)
db  = T.AmplitudeToDB()

def wav2mel(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)        # 单声道
    wav = torch.from_numpy(wav).float()
    if sr != 16000:
        wav = T.Resample(sr, 16000)(wav)
    spec = db(mel(wav))               # [40,T]
    return spec.numpy().astype(np.float32)

def main(raw_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    mapping = {}
    # 按官方 txt 划分
    splits = {"validation_list.txt":"val",
              "testing_list.txt":"test"}
    validation = set(open(os.path.join(raw_root, "validation_list.txt")).read().splitlines())
    testing    = set(open(os.path.join(raw_root, "testing_list.txt")).read().splitlines())

    for subdir, _, files in os.walk(raw_root):
        for f in files:
            if not f.endswith(".wav"): continue
            rel = os.path.relpath(os.path.join(subdir, f), raw_root)
            if rel in validation: split = "val"
            elif rel in testing: split = "test"
            else: split = "train"
            feat = wav2mel(os.path.join(raw_root, rel))
            save_dir = os.path.join(out_root, split, os.path.dirname(rel))
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f+".npy"), feat)
            mapping[os.path.join(split, rel+".npy")] = rel.split("/")[0]   # label
    json.dump(mapping, open(os.path.join(out_root,"mapping.json"),"w"))
    print("Done. {} items".format(len(mapping)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_dir)