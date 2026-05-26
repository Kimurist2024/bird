"""End-to-end Perch v2 → Proto-SSM inference on labeled train_soundscapes.

Reads pre-decoded waveforms from the waveform_cache (int16 .pt files), runs Perch v2
ONNX in 5-second windows, feeds the per-file (12, 1536) embedding tensor through
the loaded Proto-SSM, then computes macro-AUC against train_soundscapes_labels.csv.

Run:
    python experiments/single_proto_ssm/src/run_full_inference.py [--device cuda|cpu] [--batch N]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto_ssm_model import load_proto_ssm  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
DATA = REPO / "birdclef-2026"
WAVE_CACHE = REPO / "external" / "kaggle_kernels" / "eos5" / "datasets" / "tuckerarrants__birdclef-2026-waveform-cache" / "waveform_cache"
PERCH_ONNX = REPO / "external" / "kaggle_kernels" / "eos5" / "datasets" / "rishikeshjani__perch-onnx-for-birdclef-2026" / "perch_v2.onnx"
PROTO_WEIGHTS = EXP / "models" / "proto_ssm_best.pt"
OUT_DIR = EXP / "results"

SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC  # 160000
N_WINDOWS = 12
FILE_SAMPLES = WIN_SAMPLES * N_WINDOWS  # 1920000 (60s file)


def hms_to_sec(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def parse_hour(filename: str) -> int:
    # BC2026_Train_0001_S08_20250606_030007.ogg → 03
    stem = Path(filename).stem
    time_part = stem.split("_")[-1]  # '030007'
    return int(time_part[:2])


def parse_site(filename: str) -> str:
    # ..._S08_20250606_... → 'S08'
    parts = Path(filename).stem.split("_")
    for p in parts:
        if p.startswith("S") and p[1:].isdigit():
            return p
    return "S00"


def load_wave_cache_segment(cache_path: Path) -> np.ndarray:
    """Read int16 .pt → float32 / 32768, return (n_samples,) array."""
    t = torch.load(cache_path, weights_only=False, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"unexpected cache content type: {type(t)} for {cache_path}")
    arr = t.numpy().astype(np.float32) / 32768.0
    return arr


def windows_from_file(wave: np.ndarray, n_windows: int = N_WINDOWS) -> np.ndarray:
    """Return (n_windows, WIN_SAMPLES) zero-padded as needed."""
    total = n_windows * WIN_SAMPLES
    if wave.shape[0] < total:
        wave = np.pad(wave, (0, total - wave.shape[0]))
    else:
        wave = wave[:total]
    return wave.reshape(n_windows, WIN_SAMPLES)


def perch_embed(sess: ort.InferenceSession, waves: np.ndarray) -> np.ndarray:
    """waves: (B, WIN_SAMPLES) float32 → embeddings (B, 1536)."""
    out = sess.run(["embedding"], {"inputs": waves.astype(np.float32, copy=False)})
    return out[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="cuda → onnxruntime CUDAExecutionProvider + torch on cuda:0")
    ap.add_argument("--batch", type=int, default=1,
                    help="number of files to process per forward pass (Perch batch = batch*12)")
    ap.add_argument("--out-suffix", default=None,
                    help="suffix added to result filenames (default: device tag)")
    args = ap.parse_args()
    suffix = args.out_suffix if args.out_suffix is not None else f"_{args.device}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    providers = (
        [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        if args.device == "cuda" else ["CPUExecutionProvider"]
    )
    print(f"[init] device={args.device}  batch={args.batch}  providers={[p if isinstance(p,str) else p[0] for p in providers]}")
    print(f"[init] loading Perch v2 ONNX from {PERCH_ONNX}")
    sess = ort.InferenceSession(str(PERCH_ONNX), providers=providers, sess_options=ort.SessionOptions())
    print(f"[init] ort actual providers: {sess.get_providers()}")
    print(f"[init] loading Proto-SSM from {PROTO_WEIGHTS}")
    proto = load_proto_ssm(str(PROTO_WEIGHTS), device=args.device)

    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    labels_df["end_sec"] = labels_df["end"].map(hms_to_sec)
    labels_df["primary_label"] = labels_df["primary_label"].astype(str)
    labeled_files = sorted(labels_df["filename"].unique())
    print(f"[init] {len(labeled_files)} labeled train_soundscape files")

    win_labels: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in labels_df.itertuples(index=False):
        labs = r.primary_label.split(";") if ";" in r.primary_label else [r.primary_label]
        for lbl in labs:
            win_labels[(r.filename, r.end_sec)].add(lbl.strip())

    tax = pd.read_csv(DATA / "taxonomy.csv")
    class_order = list(tax.primary_label.astype(str))
    label_to_col = {lbl: i for i, lbl in enumerate(class_order)}
    n_classes = len(class_order)

    file_meta = pd.read_csv(WAVE_CACHE / "soundscape_file_meta.csv")
    cache_path_for: dict[str, Path] = {
        row.filename: WAVE_CACHE / row.cache_file for row in file_meta.itertuples(index=False)
    }
    # Build site_to_idx in cache-file appearance order (matches training notebook's unique() order)
    seen_sites: list[str] = []
    for row in file_meta.itertuples(index=False):
        if row.site not in seen_sites:
            seen_sites.append(row.site)
    site_to_idx = {s: i + 1 for i, s in enumerate(seen_sites)}
    print(f"[init] discovered {len(seen_sites)} sites → ids 1..{len(seen_sites)}")

    # Filter to labeled files that have a cache entry
    eligible = [f for f in labeled_files if f in cache_path_for]
    print(f"[init] {len(eligible)} of {len(labeled_files)} labeled files have a waveform cache entry")

    n_files = len(eligible)
    all_preds = np.zeros((n_files, N_WINDOWS, n_classes), dtype=np.float32)
    y_true = np.zeros((n_files * N_WINDOWS, n_classes), dtype=np.float32)
    row_keys: list[tuple[str, int]] = []

    t_perch = 0.0
    t_proto = 0.0
    t0 = time.time()
    B = max(1, int(args.batch))
    dev = torch.device(args.device)
    for start in range(0, n_files, B):
        chunk = eligible[start:start + B]
        # Stack waveforms → (B*12, 160000)
        wave_batch = np.stack([windows_from_file(load_wave_cache_segment(cache_path_for[f])) for f in chunk], axis=0)
        bsz = wave_batch.shape[0]
        wave_flat = wave_batch.reshape(bsz * N_WINDOWS, WIN_SAMPLES)
        tp = time.time()
        emb_flat = perch_embed(sess, wave_flat)  # (B*12, 1536)
        t_perch += time.time() - tp
        emb = emb_flat.reshape(bsz, N_WINDOWS, 1536)

        emb_t = torch.from_numpy(emb).to(dev)
        site_idx = torch.tensor([site_to_idx.get(parse_site(f), 0) for f in chunk], dtype=torch.long, device=dev)
        hour_idx = torch.tensor([parse_hour(f) for f in chunk], dtype=torch.long, device=dev)
        tp = time.time()
        with torch.no_grad():
            logits, _, _ = proto(emb_t, site_ids=site_idx, hours=hour_idx)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 12, 234)
        t_proto += time.time() - tp
        all_preds[start:start + bsz] = probs

        for bi, fname in enumerate(chunk):
            fi = start + bi
            for w in range(N_WINDOWS):
                end_sec = (w + 1) * WIN_SEC
                row_keys.append((fname, end_sec))
                for lbl in win_labels.get((fname, end_sec), ()):
                    col = label_to_col.get(lbl)
                    if col is not None:
                        y_true[fi * N_WINDOWS + w, col] = 1.0

        done = start + bsz
        elapsed = time.time() - t0
        rate = done / elapsed
        print(f"  [{done:>3}/{n_files}] elapsed={elapsed:.1f}s rate={rate:.2f} files/s perch={t_perch:.1f}s proto={t_proto:.1f}s")

    total_t = time.time() - t0
    print(f"[done] {n_files} files in {total_t:.1f}s  perch={t_perch:.1f}s  proto={t_proto:.1f}s")

    y_pred_flat = all_preds.reshape(-1, n_classes)
    # Macro-AUC
    aucs, per_class = [], {}
    for c in range(n_classes):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        try:
            a = roc_auc_score(col, y_pred_flat[:, c])
            aucs.append(a)
            per_class[class_order[c]] = a
        except Exception:
            pass
    overall = float(np.mean(aucs)) if aucs else float("nan")
    print(f"\n=== FULL PIPELINE MACRO-AUC ===")
    print(f"  n_windows={y_true.shape[0]}  n_files={n_files}  n_eval_classes={len(aucs)}/{n_classes}")
    print(f"  macro-AUC = {overall:.4f}")

    # By taxon
    print(f"\n=== By taxon ===")
    by_taxon = {}
    for cl in ["Aves", "Amphibia", "Mammalia", "Reptilia", "Insecta"]:
        cls_lbls = set(tax[tax.class_name == cl].primary_label.astype(str))
        cls_aucs = [a for lbl, a in per_class.items() if lbl in cls_lbls]
        if cls_aucs:
            m = float(np.mean(cls_aucs))
            print(f"  {cl:<10} mean AUC = {m:.4f}  evaluated {len(cls_aucs)}/{len(cls_lbls)}")
            by_taxon[cl] = {"mean_auc": m, "n_evaluated": len(cls_aucs), "n_total": len(cls_lbls)}

    # Worst 10
    print(f"\n=== Worst 10 classes ===")
    worst10 = []
    for lbl, a in sorted(per_class.items(), key=lambda kv: kv[1])[:10]:
        cnt = int(y_true[:, label_to_col[lbl]].sum())
        row = tax[tax.primary_label.astype(str) == lbl].iloc[0]
        print(f"  {lbl:>10}  {row.scientific_name[:28]:<28} {row.class_name:<10}  AUC={a:.3f}  pos={cnt}")
        worst10.append({"label": lbl, "name": row.scientific_name, "class": row.class_name, "auc": a, "positives": cnt})

    # Save summary + raw predictions
    summary = {
        "n_files": n_files,
        "n_windows": int(y_true.shape[0]),
        "n_eval_classes": len(aucs),
        "macro_auc": overall,
        "by_taxon": by_taxon,
        "worst_10": worst10,
        "timing": {
            "total_sec": total_t,
            "perch_sec": t_perch,
            "proto_sec": t_proto,
            "files_per_sec": n_files / total_t,
        },
    }
    summary_path = OUT_DIR / f"full_inference_summary{suffix}.json"
    preds_path = OUT_DIR / f"full_inference_preds{suffix}.npz"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez_compressed(
        preds_path,
        preds=all_preds, y_true=y_true.reshape(n_files, N_WINDOWS, n_classes),
        files=np.array(eligible),
    )
    print(f"\nsaved → {summary_path}")
    print(f"saved → {preds_path}")


if __name__ == "__main__":
    main()
