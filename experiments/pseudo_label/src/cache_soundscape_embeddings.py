"""Cache Perch v2 embeddings for ALL train_soundscape windows (one-time, GPU).

Output:
  embeddings.npy   float16 (N_windows, 1536)   ~393 MB for 127,896 windows
  index.parquet    row i -> (filename, window_idx, site, hour, end_sec, is_labeled)

Resumable: if embeddings.npy + index.parquet exist and cover all files, skips.

Run (GPU via reservation 1312):
  export LD_LIBRARY_PATH=/home/st6324034/miniconda3/envs/sd-webui/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=8
  srun --reservation=1312 --partition=A100.80gb -w A100.80gb-2 --gres=gpu:1 --cpus-per-task=8 \\
      /home/st6324034/Bird/.venv-bird/bin/python -u \\
      experiments/pseudo_label/src/cache_soundscape_embeddings.py --device cuda --batch 512
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
import torch

REPO = Path(__file__).resolve().parents[3]
WAVE_CACHE = REPO / "external/kaggle_kernels/eos5/datasets/tuckerarrants__birdclef-2026-waveform-cache/waveform_cache"
PERCH_ONNX = REPO / "external/kaggle_kernels/eos5/datasets/rishikeshjani__perch-onnx-for-birdclef-2026/perch_v2.onnx"
DATA = REPO / "birdclef-2026"
OUT_DIR = Path(__file__).resolve().parents[1] / "cache"

SR = 32000
WIN_SAMPLES = SR * 5      # 160000
N_WINDOWS = 12


def parse_hour(fn: str) -> int:
    return int(Path(fn).stem.split("_")[-1][:2])


def parse_site(fn: str) -> str:
    for p in Path(fn).stem.split("_"):
        if p.startswith("S") and p[1:].isdigit():
            return p
    return "S00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512, help="windows per Perch call")
    ap.add_argument("--limit", type=int, default=0, help="limit n files (debug)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fm = pd.read_csv(WAVE_CACHE / "soundscape_file_meta.csv")
    if args.limit:
        fm = fm.iloc[: args.limit]
    labeled = set(pd.read_csv(DATA / "train_soundscapes_labels.csv").filename.unique())

    n_files = len(fm)
    n_win = n_files * N_WINDOWS
    print(f"[init] {n_files} files -> {n_win} windows; device={args.device} batch={args.batch}")

    providers = ([("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(PERCH_ONNX), providers=providers)
    print(f"[init] ort providers: {sess.get_providers()}")

    emb_out = np.zeros((n_win, 1536), dtype=np.float16)
    index_rows = []

    # Stream windows in batches across files
    buf_waves = []   # list of (160000,) arrays
    buf_pos = []     # global window index for each buffered wave
    t0 = time.time()
    written = 0

    def flush():
        nonlocal written
        if not buf_waves:
            return
        x = np.stack(buf_waves, axis=0).astype(np.float32)
        out = sess.run(["embedding"], {"inputs": x})[0]  # (B, 1536)
        for j, gpos in enumerate(buf_pos):
            emb_out[gpos] = out[j].astype(np.float16)
        written += len(buf_waves)
        buf_waves.clear()
        buf_pos.clear()

    for fi, row in enumerate(fm.itertuples(index=False)):
        cache_path = WAVE_CACHE / row.cache_file
        wave = torch.load(cache_path, weights_only=False, map_location="cpu").numpy().astype(np.float32) / 32768.0
        total = N_WINDOWS * WIN_SAMPLES
        if wave.shape[0] < total:
            wave = np.pad(wave, (0, total - wave.shape[0]))
        else:
            wave = wave[:total]
        windows = wave.reshape(N_WINDOWS, WIN_SAMPLES)
        site = parse_site(row.filename)
        hour = parse_hour(row.filename)
        is_lab = row.filename in labeled
        for w in range(N_WINDOWS):
            gpos = fi * N_WINDOWS + w
            buf_waves.append(windows[w])
            buf_pos.append(gpos)
            index_rows.append({
                "row": gpos, "filename": row.filename, "window_idx": w,
                "end_sec": (w + 1) * 5, "site": site, "hour": hour, "is_labeled": is_lab,
            })
            if len(buf_waves) >= args.batch:
                flush()
        if (fi + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  [{fi+1}/{n_files}] files, {written} windows embedded, {el:.1f}s ({written/max(el,1):.0f} win/s)")
    flush()

    np.save(OUT_DIR / "embeddings.npy", emb_out)
    pd.DataFrame(index_rows).to_parquet(OUT_DIR / "index.parquet", index=False)
    el = time.time() - t0
    print(f"[done] {written} windows in {el:.1f}s -> {OUT_DIR/'embeddings.npy'} ({emb_out.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
