"""Test-Time Augmentation (time-shift) for Perch v2 → Proto-SSM on labeled train_soundscapes.

Consistent-shift TTA: for each shift offset, the whole 60s file is shifted by the
same amount, re-windowed into 12 × 5s crops, embedded by Perch, and run through
Proto-SSM as one (12,1536) sequence. Sigmoid probabilities are averaged across
offsets. Offset 0 alone == the no-TTA baseline, so a single run yields the A/B delta.

Run:
    LD_LIBRARY_PATH=<cudnn8>:$LD_LIBRARY_PATH OMP_NUM_THREADS=8 \\
    python experiments/tta/src/run_tta_inference.py \\
        --offsets 0 --offsets-tta -1.0 -0.5 0.0 0.5 1.0 --device cuda --batch 8

The script always computes BOTH:
  - baseline = offset {0.0} only
  - tta      = the --offsets-tta set
and reports macro-AUC for each plus the delta.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
SINGLE = REPO / "experiments" / "single_proto_ssm"
sys.path.insert(0, str(SINGLE / "src"))
from proto_ssm_model import load_proto_ssm  # noqa: E402

DATA = REPO / "birdclef-2026"
WAVE_CACHE = REPO / "external/kaggle_kernels/eos5/datasets/tuckerarrants__birdclef-2026-waveform-cache/waveform_cache"
PERCH_ONNX = REPO / "external/kaggle_kernels/eos5/datasets/rishikeshjani__perch-onnx-for-birdclef-2026/perch_v2.onnx"
PROTO_WEIGHTS = SINGLE / "models" / "proto_ssm_best.pt"
OUT_DIR = Path(__file__).resolve().parents[1] / "results"

SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC          # 160000
N_WINDOWS = 12
FILE_SAMPLES = WIN_SAMPLES * N_WINDOWS  # 1920000


def hms_to_sec(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def parse_hour(fn: str) -> int:
    return int(Path(fn).stem.split("_")[-1][:2])


def parse_site_idx(fn: str, site_to_idx: dict) -> int:
    for p in Path(fn).stem.split("_"):
        if p.startswith("S") and p[1:].isdigit():
            return site_to_idx.get(p, 0)
    return 0


def load_wave(cache_path: Path) -> np.ndarray:
    t = torch.load(cache_path, weights_only=False, map_location="cpu")
    return t.numpy().astype(np.float32) / 32768.0


def windows_shifted(wave: np.ndarray, shift_samples: int) -> np.ndarray:
    """Pad wave so a shifted 60s read stays valid, return (12, WIN_SAMPLES)."""
    pad = abs(shift_samples) + 1
    padded = np.pad(wave, (pad, pad))
    start = pad + shift_samples
    seg = padded[start:start + FILE_SAMPLES]
    if seg.shape[0] < FILE_SAMPLES:
        seg = np.pad(seg, (0, FILE_SAMPLES - seg.shape[0]))
    return seg.reshape(N_WINDOWS, WIN_SAMPLES)


def perch_embed(sess, waves: np.ndarray) -> np.ndarray:
    return sess.run(["embedding"], {"inputs": waves.astype(np.float32, copy=False)})[0]


def macro_auc(y: np.ndarray, p: np.ndarray, class_order, tax):
    aucs, per_class = [], {}
    for c in range(y.shape[1]):
        col = y[:, c]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        try:
            a = roc_auc_score(col, p[:, c]); aucs.append(a); per_class[class_order[c]] = a
        except Exception:
            pass
    by_taxon = defaultdict(list)
    cls_name = {str(r.primary_label): r.class_name for r in tax.itertuples(index=False)}
    for lbl, a in per_class.items():
        by_taxon[cls_name.get(lbl, "?")].append(a)
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs),
            {t: round(float(np.mean(v)), 4) for t, v in by_taxon.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets-tta", nargs="+", type=float, default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    providers = ([("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    print(f"[init] device={args.device} tta_offsets={args.offsets_tta}")
    sess = ort.InferenceSession(str(PERCH_ONNX), providers=providers)
    print(f"[init] ort providers: {sess.get_providers()}")
    proto = load_proto_ssm(str(PROTO_WEIGHTS), device=args.device)
    dev = torch.device(args.device)

    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    labels_df["end_sec"] = labels_df["end"].map(hms_to_sec)
    labels_df["primary_label"] = labels_df["primary_label"].astype(str)
    labeled_files = sorted(labels_df["filename"].unique())
    win_labels = defaultdict(set)
    for r in labels_df.itertuples(index=False):
        labs = r.primary_label.split(";") if ";" in r.primary_label else [r.primary_label]
        for lbl in labs:
            win_labels[(r.filename, r.end_sec)].add(lbl.strip())

    tax = pd.read_csv(DATA / "taxonomy.csv")
    class_order = list(tax.primary_label.astype(str))
    label_to_col = {l: i for i, l in enumerate(class_order)}
    n_classes = len(class_order)

    file_meta = pd.read_csv(WAVE_CACHE / "soundscape_file_meta.csv")
    cache_path_for = {r.filename: WAVE_CACHE / r.cache_file for r in file_meta.itertuples(index=False)}
    seen = []
    for r in file_meta.itertuples(index=False):
        if r.site not in seen:
            seen.append(r.site)
    site_to_idx = {s: i + 1 for i, s in enumerate(seen)}

    eligible = [f for f in labeled_files if f in cache_path_for]
    print(f"[init] {len(eligible)} labeled files with cache")

    # offsets union: ensure 0.0 present for baseline
    offsets = sorted(set(args.offsets_tta) | {0.0})
    # accumulate per-offset probabilities so we can compose baseline (offset 0) and TTA (mean)
    n_files = len(eligible)
    probs_by_offset = {off: np.zeros((n_files, N_WINDOWS, n_classes), dtype=np.float32) for off in offsets}
    y_true = np.zeros((n_files * N_WINDOWS, n_classes), dtype=np.float32)

    t0 = time.time()
    for off in offsets:
        shift = int(round(off * SR))
        B = args.batch
        for start in range(0, n_files, B):
            chunk = eligible[start:start + B]
            wave_batch = np.stack([windows_shifted(load_wave(cache_path_for[f]), shift) for f in chunk], axis=0)
            bsz = wave_batch.shape[0]
            emb_flat = perch_embed(sess, wave_batch.reshape(bsz * N_WINDOWS, WIN_SAMPLES))
            emb = torch.from_numpy(emb_flat.reshape(bsz, N_WINDOWS, 1536)).to(dev)
            site = torch.tensor([parse_site_idx(f, site_to_idx) for f in chunk], dtype=torch.long, device=dev)
            hour = torch.tensor([parse_hour(f) for f in chunk], dtype=torch.long, device=dev)
            with torch.no_grad():
                logits, _, _ = proto(emb, site_ids=site, hours=hour)
                p = torch.sigmoid(logits).cpu().numpy()
            probs_by_offset[off][start:start + bsz] = p
        print(f"  offset {off:+.2f}s done ({time.time()-t0:.1f}s)")

    # y_true (offset-independent)
    for fi, fname in enumerate(eligible):
        for w in range(N_WINDOWS):
            for lbl in win_labels.get((fname, (w + 1) * WIN_SEC), ()):
                col = label_to_col.get(lbl)
                if col is not None:
                    y_true[fi * N_WINDOWS + w, col] = 1.0

    baseline = probs_by_offset[0.0].reshape(-1, n_classes)
    tta = np.mean([probs_by_offset[o] for o in args.offsets_tta], axis=0).reshape(-1, n_classes)

    b_auc, b_n, b_tax = macro_auc(y_true, baseline, class_order, tax)
    t_auc, t_n, t_tax = macro_auc(y_true, tta, class_order, tax)

    print(f"\n=== RESULT (66 files, {b_n} eval classes) ===")
    print(f"  baseline (offset 0)      macro-AUC = {b_auc:.4f}")
    print(f"  TTA {args.offsets_tta}  macro-AUC = {t_auc:.4f}")
    print(f"  Δ (TTA - baseline)       = {t_auc - b_auc:+.4f}")
    print(f"\n  per-taxon baseline: {b_tax}")
    print(f"  per-taxon TTA:      {t_tax}")

    summary = {
        "offsets_tta": args.offsets_tta,
        "n_eval_classes": b_n,
        "baseline_macro_auc": round(b_auc, 4),
        "tta_macro_auc": round(t_auc, 4),
        "delta": round(t_auc - b_auc, 4),
        "baseline_by_taxon": b_tax,
        "tta_by_taxon": t_tax,
        "wall_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "tta_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved → {OUT_DIR/'tta_summary.json'}")


if __name__ == "__main__":
    main()
