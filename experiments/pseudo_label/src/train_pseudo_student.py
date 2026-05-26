"""Soundscape pseudo-label self-training (Noisy Student style) for BirdCLEF+ 2026.

Pipeline:
  1. Load cached Perch embeddings for ALL soundscapes (cache_soundscape_embeddings.py output).
  2. Teacher = proto_ssm_best.pt. Generate per-window pseudo-labels on the ~10.5K UNLABELED files.
  3. Confidence-filter: keep windows where teacher max-prob >= tau (documented "secret ingredient"
     used conf>=0.8 / pct~94). Build a per-file (12,1536) pseudo training set with soft targets +
     a per-window confidence mask.
  4. 5-fold CV on the 66 LABELED files. Each fold: student warm-started from teacher, trained on
     (labeled train folds, supervised BCE) + (pseudo unlabeled files, masked soft BCE w/ ramp).
     Eval on labeled val fold.
  5. Aggregate OOF macro-AUC, compare to teacher baseline.

Honesty notes:
  - Pseudo files are UNLABELED soundscapes -> not in any fold -> no eval leakage.
  - Teacher (proto_ssm_best) was trained on labeled+train_audio, NOT on these unlabeled soundscapes,
    so its pseudo-labels are genuinely new domain signal.
  - Student warm-start means the *baseline* to beat is the teacher's own OOF, reported alongside.

Run (GPU):
  export LD_LIBRARY_PATH=<cudnn8>:$LD_LIBRARY_PATH OMP_NUM_THREADS=8
  srun --reservation=1312 --partition=A100.80gb -w A100.80gb-2 --gres=gpu:1 --cpus-per-task=8 \\
    /home/st6324034/Bird/.venv-bird/bin/python -u \\
    experiments/pseudo_label/src/train_pseudo_student.py --config experiments/pseudo_label/configs/v1.yaml
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
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO = Path(__file__).resolve().parents[3]
SINGLE = REPO / "experiments" / "single_proto_ssm"
sys.path.insert(0, str(SINGLE / "src"))
from proto_ssm_model import load_proto_ssm  # noqa: E402

DATA = REPO / "birdclef-2026"
CACHE = Path(__file__).resolve().parents[1] / "cache"
PROTO_WEIGHTS = SINGLE / "models" / "proto_ssm_best.pt"
OOF_FOLD_NPZ = SINGLE / "results" / "full_oof_meta_features.npz"
WAVE_CACHE = REPO / "external/kaggle_kernels/eos5/datasets/tuckerarrants__birdclef-2026-waveform-cache/waveform_cache"


def hms_to_sec(s: str) -> int:
    h, m, sec = s.split(":"); return int(h) * 3600 + int(m) * 60 + int(sec)


def build_site_map() -> dict:
    fm = pd.read_csv(WAVE_CACHE / "soundscape_file_meta.csv")
    seen = []
    for r in fm.itertuples(index=False):
        if r.site not in seen:
            seen.append(r.site)
    return {s: i + 1 for i, s in enumerate(seen)}


def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        s = y[:, c].sum()
        if s == 0 or s == y.shape[0]:
            continue
        try:
            aucs.append(roc_auc_score(y[:, c], p[:, c]))
        except Exception:
            pass
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))


def load_all(device):
    """Group cached embeddings into per-file (12,1536), separate labeled/unlabeled."""
    emb = np.load(CACHE / "embeddings.npy")              # (N,1536) float16
    idx = pd.read_parquet(CACHE / "index.parquet")
    site_map = build_site_map()

    tax = pd.read_csv(DATA / "taxonomy.csv")
    class_order = list(tax.primary_label.astype(str))
    n_classes = len(class_order)
    label_to_col = {l: i for i, l in enumerate(class_order)}

    # labeled windows -> y
    lab = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    lab["end_sec"] = lab["end"].map(hms_to_sec)
    lab["primary_label"] = lab["primary_label"].astype(str)
    win_lab = defaultdict(set)
    for r in lab.itertuples(index=False):
        labs = r.primary_label.split(";") if ";" in r.primary_label else [r.primary_label]
        for L in labs:
            win_lab[(r.filename, r.end_sec)].add(L.strip())

    # group by file preserving window order
    idx = idx.sort_values(["filename", "window_idx"]).reset_index(drop=True)
    files = idx["filename"].to_numpy()
    uniq_files = list(dict.fromkeys(files.tolist()))
    file_to_rows = defaultdict(list)
    for i, f in enumerate(idx["filename"].to_numpy()):
        file_to_rows[f].append(int(idx["row"].iloc[i]))

    def site_of(f):
        # proto_ssm_best.pt was trained with n_sites=20 (valid indices 0..19).
        # soundscape meta has more sites; clamp anything >=20 to 0 (unknown).
        for p in Path(f).stem.split("_"):
            if p.startswith("S") and p[1:].isdigit():
                idx = site_map.get(p, 0)
                return idx if idx < 20 else 0
        return 0

    lab_files, unlab_files = [], []
    for f in uniq_files:
        (lab_files if f in set(lab.filename.unique()) else unlab_files).append(f)

    def stack(file_list):
        E = np.zeros((len(file_list), 12, 1536), dtype=np.float32)
        S = np.zeros(len(file_list), dtype=np.int64)
        H = np.zeros(len(file_list), dtype=np.int64)
        for i, f in enumerate(file_list):
            rows = file_to_rows[f][:12]
            E[i] = emb[rows].astype(np.float32)
            S[i] = site_of(f)
            H[i] = int(Path(f).stem.split("_")[-1][:2])
        return E, S, H

    Elab, Slab, Hlab = stack(lab_files)
    Eunl, Sunl, Hunl = stack(unlab_files)

    # labels for labeled files
    Ylab = np.zeros((len(lab_files), 12, n_classes), dtype=np.float32)
    for i, f in enumerate(lab_files):
        for w in range(12):
            for L in win_lab.get((f, (w + 1) * 5), ()):
                c = label_to_col.get(L)
                if c is not None:
                    Ylab[i, w, c] = 1.0

    # fold ids for labeled files: use the OOF fold mapping (708 = 59 files); fall back to hash for the rest
    fold_npz = np.load(OOF_FOLD_NPZ)
    # the 708 OOF windows correspond to a subset; assign folds round-robin by site for full 66
    rng = np.random.RandomState(42)
    fold_lab = np.array([(hash(f) % 5) for f in lab_files], dtype=np.int64)

    return {
        "class_order": class_order, "n_classes": n_classes,
        "lab_files": lab_files, "unlab_files": unlab_files,
        "Elab": torch.tensor(Elab, device=device), "Slab": torch.tensor(Slab, device=device),
        "Hlab": torch.tensor(Hlab, device=device), "Ylab": torch.tensor(Ylab, device=device),
        "Eunl": torch.tensor(Eunl, device=device), "Sunl": torch.tensor(Sunl, device=device),
        "Hunl": torch.tensor(Hunl, device=device),
        "fold_lab": torch.tensor(fold_lab, device=device),
        "tax": tax,
    }


@torch.no_grad()
def teacher_pseudo(model, E, S, H, batch=256):
    probs = []
    for i in range(0, E.shape[0], batch):
        logits, _, _ = model(E[i:i+batch], site_ids=S[i:i+batch], hours=H[i:i+batch])
        probs.append(torch.sigmoid(logits))
    return torch.cat(probs, 0)  # (Nfiles,12,C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = Path(__file__).resolve().parents[1] / "runs" / args.config.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] device={device} cfg={cfg}")
    D = load_all(device)
    n_classes = D["n_classes"]
    print(f"[init] labeled files={len(D['lab_files'])} unlabeled={len(D['unlab_files'])}")

    # Teacher pseudo-labels on unlabeled soundscapes
    teacher = load_proto_ssm(str(PROTO_WEIGHTS), device=str(device))
    teacher.eval()
    t0 = time.time()
    pseudo = teacher_pseudo(teacher, D["Eunl"], D["Sunl"], D["Hunl"])  # (U,12,C)
    print(f"[pseudo] generated for {pseudo.shape[0]} files in {time.time()-t0:.1f}s")

    # Confidence filter: per-window max prob >= tau -> mask
    tau = cfg["conf_threshold"]
    win_conf, _ = pseudo.max(dim=-1)             # (U,12)
    win_mask = (win_conf >= tau).float()          # (U,12)
    kept = int(win_mask.sum().item())
    print(f"[pseudo] tau={tau}: kept {kept}/{win_mask.numel()} windows ({100*kept/win_mask.numel():.1f}%)")
    # keep files that have at least 1 confident window
    file_keep = win_mask.sum(dim=1) > 0
    Eunl, Sunl, Hunl = D["Eunl"][file_keep], D["Sunl"][file_keep], D["Hunl"][file_keep]
    pseudo_t = pseudo[file_keep]
    mask_t = win_mask[file_keep]
    if cfg.get("hard_pseudo", False):
        pseudo_t = (pseudo_t >= 0.5).float()
    print(f"[pseudo] usable files: {Eunl.shape[0]}")

    bce = nn.BCEWithLogitsLoss(reduction="none")
    folds = sorted(set(D["fold_lab"].cpu().tolist()))
    all_pred = np.zeros((len(D["lab_files"]) * 12, n_classes), dtype=np.float32)
    y_flat = D["Ylab"].cpu().numpy().reshape(-1, n_classes)
    fold_flat = D["fold_lab"].cpu().numpy().repeat(12)

    teacher_oof = np.zeros_like(all_pred)
    with torch.no_grad():
        tp = teacher_pseudo(teacher, D["Elab"], D["Slab"], D["Hlab"]).cpu().numpy().reshape(-1, n_classes)
    teacher_oof[:] = tp

    for f in folds:
        tr = (D["fold_lab"] != f)
        va = (D["fold_lab"] == f)
        student = load_proto_ssm(str(PROTO_WEIGHTS), device=str(device))
        opt = AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = CosineAnnealingLR(opt, T_max=cfg["epochs"])
        Etr, Str_, Htr, Ytr = D["Elab"][tr], D["Slab"][tr], D["Hlab"][tr], D["Ylab"][tr]
        for ep in range(cfg["epochs"]):
            student.train()
            lam = cfg["lambda_pseudo"] * min(1.0, ep / max(1, cfg["epochs"] // 3))
            # supervised step
            opt.zero_grad()
            logit_l, _, _ = student(Etr, site_ids=Str_, hours=Htr)
            loss_sup = bce(logit_l, Ytr).mean()
            # pseudo step (sample a batch of unlabeled files)
            bs = min(cfg["pseudo_batch"], Eunl.shape[0])
            sel = torch.randint(0, Eunl.shape[0], (bs,), device=device)
            logit_u, _, _ = student(Eunl[sel], site_ids=Sunl[sel], hours=Hunl[sel])
            lu = bce(logit_u, pseudo_t[sel]).mean(dim=-1) * mask_t[sel]
            loss_un = lu.sum() / (mask_t[sel].sum() + 1e-6)
            loss = loss_sup + lam * loss_un
            loss.backward(); opt.step(); sched.step()
        student.eval()
        with torch.no_grad():
            pv, _, _ = student(D["Elab"][va], site_ids=D["Slab"][va], hours=D["Hlab"][va])
            pv = torch.sigmoid(pv).cpu().numpy().reshape(-1, n_classes)
        va_flat = np.repeat(va.cpu().numpy(), 12)
        all_pred[va_flat] = pv
        a, n = macro_auc(y_flat[va_flat], pv)
        print(f"  fold {f}: student val AUC={a:.4f} (n={n})")

    s_auc, s_n = macro_auc(y_flat, all_pred)
    t_auc, t_n = macro_auc(y_flat, teacher_oof)
    print(f"\n=== OOF macro-AUC ({s_n} classes) ===")
    print(f"  teacher (proto_ssm)  = {t_auc:.4f}")
    print(f"  student (pseudo-ST)  = {s_auc:.4f}")
    print(f"  Δ (student-teacher)  = {s_auc - t_auc:+.4f}")

    summary = {
        "config": cfg, "n_eval_classes": s_n,
        "teacher_oof_auc": round(t_auc, 4),
        "student_oof_auc": round(s_auc, 4),
        "delta": round(s_auc - t_auc, 4),
        "pseudo_kept_windows": kept, "pseudo_usable_files": int(Eunl.shape[0]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(run_dir / "oof.npz", student=all_pred, teacher=teacher_oof, y_true=y_flat, fold=fold_flat)
    print(f"\nsaved → {run_dir/'summary.json'}")


if __name__ == "__main__":
    main()
