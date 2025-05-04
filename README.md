# Embedded Gaussian Attention for Keyword Spotting

**A clean, reproducible implementation of CTGWP and baseline models**

This repository provides end-to-end scripts for:

* Data preprocessing (Mel-spectrogram extraction)
* Baseline model training (4-layer CNN + GAP)
* CTGWP model training (Gaussian Attention)
* Tuned and advanced CTGWP variants
* Evaluation (Accuracy, FAR, FRR)
* (Optional) Model export and embedded inference

> **Note:** All GPU commands assume **CUDA device index 1** by default. You can override with `--device cuda:<index>`, but setting `CUDA_VISIBLE_DEVICES=1` ensures scripts use GPU 1.

---

## 📋 Prerequisites

* Ubuntu / macOS / Windows WSL
* NVIDIA GPU with CUDA 12.2 (device index 1)
* Conda or virtualenv

### 1. Create environment

```bash
conda create -n kws python=3.11 -y
conda activate kws
pip install --upgrade pip
pip install -r requirements.txt
```

**`requirements.txt`** should include:

```
torch==2.5.1
torchaudio
numpy
scipy
transformers
tqdm
soundfile
onnx
onnxruntime
```

---

## 📂 Directory Structure

```
embedded-gaussian-attention/
├── data/
│   ├── raw/                # Original .wav files
│   └── feats/              # Extracted Mel spectrograms (.npy)
├── models/                 # Saved checkpoints
│   ├── baseline/
│   ├── ctgwp/
│   ├── ctgwp_tuned/
│   └── ctgwp_adv/
├── outputs/                # Evaluation & export outputs
├── scripts/                # All training/eval/export scripts
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🛠️ 1. Data Preprocessing

Download and extract SpeechCommands v0.02:

```bash
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data/raw && tar zxvf speech_commands_v0.02.tar.gz -C data/raw
```

Extract 40-channel Mel-spectrograms:

```bash
# Defaults: CPU
python scripts/preprocess.py \
    --input_dir data/raw \
    --output_dir data/feats
```

This creates `data/feats/train`, `data/feats/val`, `data/feats/test` as `.npy` files and a `mapping.json`.

---

## ⚙️ 2. Training

All training commands run on **GPU 1** by default. If necessary, set:

```bash
export CUDA_VISIBLE_DEVICES=1
```

### 2.1 Baseline (CNN + GAP)

```bash
python scripts/train_baseline.py \
    --feats_dir data/feats \
    --out models/baseline \
    --device cuda:1
```

* **Output:** `models/baseline/best.ckpt`

### 2.2 CTGWP (Original)

```bash
python scripts/train_ctgwp.py \
    --feats_dir data/feats \
    --out models/ctgwp \
    --device cuda:1
```

* **Output:** `models/ctgwp/best.ckpt`

### 2.3 CTGWP Tuned (log σ + CosineLR)

```bash
python scripts/ctgwp_tuned.py \
    --feats_dir data/feats \
    --out models/ctgwp_tuned \
    --device cuda:1
```

* **Output:** `models/ctgwp_tuned/best.ckpt`

### 2.4 CTGWP Advanced (Transfer + SpecAug + EarlyStop)

```bash
python scripts/ctgwp_advanced.py \
    --feats_dir data/feats \
    --baseline_ckpt models/baseline/best.ckpt \
    --out models/ctgwp_adv \
    --device cuda:1
```

* **Output:** `models/ctgwp_adv/best.ckpt`

---

## ✅ 3. Evaluation

Run evaluation (Accuracy by default):

```bash
python scripts/evaluate.py \
    --model_path <PATH_TO_CKPT> \
    --feats_dir data/feats \
    --split test \
    --device cuda:1 \
    --out outputs/<name>.json
```

**Examples:**

* Baseline:

  ```bash
  python scripts/evaluate.py \
      --model_path models/baseline/best.ckpt \
      --feats_dir data/feats \
      --device cuda:1
  ```

* CTGWP Advanced:

  ```bash
  python scripts/evaluate.py \
      --model_path models/ctgwp_adv/best.ckpt \
      --feats_dir data/feats \
      --device cuda:1 \
      --out outputs/ctgwp_adv_test.json
  ```

This prints:

```
Accuracy on test: 95.23%
Saved to outputs/ctgwp_adv_test.json
```

*To enable FAR/FRR, modify `evaluate.py` accordingly.*

---

## 📦 4. Optional Export & Embedded Inference

### 4.1 Export to ONNX

```bash
python scripts/export_onnx.py \
    --ckpt models/ctgwp_adv/best.ckpt \
    --onnx_out outputs/ctgwp_adv.onnx
```

### 4.2 Convert to TFLite (FP32/INT8)

Follow `export_onnx.py` comments to convert ONNX → TensorFlow → TFLite.

### 4.3 Raspberry Pi Benchmark

```bash
python scripts/benchmark_pi.py \
    --model outputs/ctgwp_adv_int8.tflite \
    --feats_dir data/feats/test \
    --out outputs/pi_benchmark.json
```

Outputs mean & p95 latency.

---

## 🛠️ Troubleshooting

| Issue                   | Solution                                              |
| ----------------------- | ----------------------------------------------------- |
| GPU OOM                 | `--bs 32` or use AMP (`autocast`)                     |
| No accuracy improvement | Increase `--lr`, extend epochs, check `baseline_ckpt` |
| Length mismatch errors  | Ensure correct `collate_pad` usage                    |