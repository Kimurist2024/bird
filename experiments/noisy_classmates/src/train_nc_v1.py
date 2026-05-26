"""Noisy Classmates v1 — co-evolutionary self-training on cached Perch embeddings.

Scope (v1):
- K=2 classmates with different architectures (default: proto_ssm + mlp_mixer)
- Uses the existing 708-window OOF embedding cache (full_perch_arrays.npz)
- 5-fold CV matching the cache's fold_id (so AUC is comparable to baseline 0.7999)
- Joint training: each classmate's loss = supervised BCE + λ_u * masked BCE on the
  OTHER classmate's pseudo-labels (anti-confirmation-bias)
- Embedding-level strong augmentation (gaussian noise + window shuffle + mixup)
- Initialize proto_ssm classmate from existing proto_ssm_best.pt warm start

v1 explicitly skips:
- Self-training on unlabeled soundscapes (no cached embeddings yet)
- Re-Perching with audio-level augmentation
- More than 2 classmates

Run:
    /home/st6324034/orbit/claws-orbit/.venv/bin/python -u \\
        experiments/noisy_classmates/src/train_nc_v1.py \\
        --config experiments/noisy_classmates/configs/v1_default.yaml
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
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architectures import build_classmate, load_proto_ssm, PRODUCTION_CONFIG  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
DATA = REPO / "birdclef-2026"

# Reuse the cached 708-window Perch arrays + meta from the eos5 kernel.
PERCH_ARRAYS = REPO / "external" / "kaggle_kernels" / "eos5" / "datasets" / "jaejohn__perch-meta" / "full_perch_arrays.npz"
PERCH_META = REPO / "external" / "kaggle_kernels" / "eos5" / "datasets" / "jaejohn__perch-meta" / "full_perch_meta.parquet"
PROTO_WEIGHTS = EXP.parent / "single_proto_ssm" / "models" / "proto_ssm_best.pt"
OOF_FOLD_NPZ = EXP.parent / "single_proto_ssm" / "results" / "full_oof_meta_features.npz"


def hms_to_sec(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def load_data(device: torch.device):
    """Returns (emb, labels, fold_id, site_id, hour, class_order)."""
    arrays = np.load(PERCH_ARRAYS)
    emb_full = arrays["emb_full"]                   # (708, 1536)
    perch_scores = arrays["scores_full_raw"]        # (708, 234) — Perch's own preds (used later)
    meta = pd.read_parquet(PERCH_META)
    meta = meta.copy()
    meta["end_sec"] = meta["row_id"].str.rsplit("_", n=1).str[1].astype(int)
    meta["file_key"] = meta["row_id"].str.rsplit("_", n=1).str[0] + ".ogg"

    fold_id = np.load(OOF_FOLD_NPZ)["fold_id"]      # (708,)
    assert len(fold_id) == 708, fold_id.shape

    # Build labels from train_soundscapes_labels.csv
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    labels_df["end_sec"] = labels_df["end"].map(hms_to_sec)
    labels_df["primary_label"] = labels_df["primary_label"].astype(str)
    win_labels: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in labels_df.itertuples(index=False):
        labs = r.primary_label.split(";") if ";" in r.primary_label else [r.primary_label]
        for lbl in labs:
            win_labels[(r.filename, r.end_sec)].add(lbl.strip())

    tax = pd.read_csv(DATA / "taxonomy.csv")
    class_order = list(tax.primary_label.astype(str))
    label_to_col = {lbl: i for i, lbl in enumerate(class_order)}
    y = np.zeros((len(meta), len(class_order)), dtype=np.float32)
    for i, row in meta.iterrows():
        for lbl in win_labels.get((row["file_key"], int(row["end_sec"])), ()):
            col = label_to_col.get(lbl)
            if col is not None:
                y[i, col] = 1.0

    # site + hour for metadata embeddings (must match training-time site_to_idx if warm-starting proto_ssm)
    site_csv = REPO / "external" / "kaggle_kernels" / "eos5" / "datasets" / "tuckerarrants__birdclef-2026-waveform-cache" / "waveform_cache" / "soundscape_file_meta.csv"
    file_meta = pd.read_csv(site_csv)
    seen_sites: list[str] = []
    for row in file_meta.itertuples(index=False):
        if row.site not in seen_sites:
            seen_sites.append(row.site)
    site_to_idx = {s: i + 1 for i, s in enumerate(seen_sites)}

    def site_of(filename: str) -> int:
        for p in filename.split("_"):
            if p.startswith("S") and p[1:].isdigit():
                return site_to_idx.get(p, 0)
        return 0
    site_ids = np.array([site_of(f) for f in meta["file_key"]], dtype=np.int64)
    hours = np.array([int(f.split("_")[-1][:2]) for f in meta["file_key"].str.replace(".ogg", "", regex=False)], dtype=np.int64)

    # Group into per-file (12-window) chunks. The 708 rows are 59 files × 12.
    # Rows for the same file are contiguous and ordered by end_sec ascending.
    files = meta["file_key"].to_numpy()
    n_files = 708 // 12
    assert n_files * 12 == 708, "data not cleanly 12-window aligned"
    # Validate the assumption: every file appears in exactly 12 consecutive rows
    for fi in range(n_files):
        block = files[fi * 12:(fi + 1) * 12]
        assert (block == block[0]).all(), f"file block boundary broken at fi={fi}: {set(block)}"

    emb_grp = emb_full.reshape(n_files, 12, 1536)
    y_grp = y.reshape(n_files, 12, -1)
    fold_grp = fold_id.reshape(n_files, 12)[:, 0]  # per-file fold (constant within file)
    # per-file metadata (file-constant)
    site_grp = site_ids.reshape(n_files, 12)[:, 0]
    hour_grp = hours.reshape(n_files, 12)[:, 0]

    return (
        torch.from_numpy(emb_grp).float().to(device),
        torch.from_numpy(y_grp).float().to(device),
        torch.from_numpy(fold_grp).long().to(device),
        torch.from_numpy(site_grp).long().to(device),
        torch.from_numpy(hour_grp).long().to(device),
        class_order,
    )


def strong_aug_embeddings(emb: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Embedding-level augmentations:
    - gaussian noise per-feature
    - window order shuffle
    - random window dropout (mask whole window to zeros)"""
    x = emb.clone()
    if cfg.get("noise_sigma", 0) > 0:
        x = x + torch.randn_like(x) * (cfg["noise_sigma"] * x.std(dim=(1, 2), keepdim=True))
    if cfg.get("p_window_dropout", 0) > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= cfg["p_window_dropout"]).float()
        x = x * mask
    if cfg.get("p_shuffle", 0) > 0:
        if torch.rand(1, device=x.device).item() < cfg["p_shuffle"]:
            perm = torch.randperm(x.shape[1], device=x.device)
            x = x[:, perm, :]
    return x


def mixup_batch(emb: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return emb, y
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(emb.shape[0], device=emb.device)
    return lam * emb + (1 - lam) * emb[perm], lam * y + (1 - lam) * y[perm]


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    aucs = []
    for c in range(y.shape[1]):
        col = y[:, c]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        try:
            aucs.append(roc_auc_score(col, p[:, c]))
        except Exception:
            pass
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))


def make_classmate(arch: str, warm_init: bool, device: torch.device) -> nn.Module:
    if arch == "proto_ssm" and warm_init:
        m = load_proto_ssm(str(PROTO_WEIGHTS), device=str(device))
        return m
    return build_classmate(arch).to(device)


def train_fold(
    fold: int,
    emb: torch.Tensor,
    y: torch.Tensor,
    fold_grp: torch.Tensor,
    site: torch.Tensor,
    hour: torch.Tensor,
    cfg: dict,
    device: torch.device,
):
    train_mask = fold_grp != fold
    val_mask = fold_grp == fold
    emb_tr, y_tr, site_tr, hour_tr = emb[train_mask], y[train_mask], site[train_mask], hour[train_mask]
    emb_va, y_va, site_va, hour_va = emb[val_mask], y[val_mask], site[val_mask], hour[val_mask]
    print(f"  fold {fold}: train={emb_tr.shape[0]} files, val={emb_va.shape[0]} files")

    archs = cfg["classmates"]                            # e.g. ["proto_ssm", "mlp_mixer"]
    warm = cfg.get("warm_start", [True] * len(archs))
    models = [make_classmate(a, w, device) for a, w in zip(archs, warm)]
    opts = [AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) for m in models]
    scheds = [CosineAnnealingLR(o, T_max=cfg["epochs"]) for o in opts]

    bce = nn.BCEWithLogitsLoss(reduction="none")
    n_steps_per_epoch = max(1, emb_tr.shape[0] // cfg["batch_size"])
    best_per = [None] * len(models)
    best_score = [-1.0] * len(models)
    history = []

    for epoch in range(cfg["epochs"]):
        # Lambda_u ramp-up
        lam_u = cfg["lambda_u"] * min(1.0, max(0.0, (epoch - cfg["lam_u_warmup_epochs"]) / max(1, cfg["epochs"] // 4)))
        epoch_loss = [0.0] * len(models)
        for step in range(n_steps_per_epoch):
            # Sample labeled batch (everything we have is labeled in v1)
            idx = torch.randint(0, emb_tr.shape[0], (cfg["batch_size"],), device=device)
            x_l = emb_tr[idx]; y_l = y_tr[idx]; s_l = site_tr[idx]; h_l = hour_tr[idx]

            x_l_mix, y_l_mix = mixup_batch(x_l, y_l, cfg.get("mixup_alpha", 0.4))
            x_l_strong = strong_aug_embeddings(x_l_mix, cfg["aug"])
            x_l_weak = x_l  # weak = identity for now

            # Compute pseudo labels from each classmate on weak input (no grad)
            with torch.no_grad():
                pseudo_logits = [m(x_l_weak, site_ids=s_l, hours=h_l)[0] for m in models]
                pseudo_probs = [torch.sigmoid(z) for z in pseudo_logits]

            # Each classmate's loss = supervised + λ_u * pseudo from OTHER classmates
            for k, model in enumerate(models):
                model.train()
                logits_k, _, _ = model(x_l_strong, site_ids=s_l, hours=h_l)
                # supervised
                loss_sup = bce(logits_k, y_l_mix).mean()
                # pseudo from others (anti-confirmation-bias)
                others = [p for j, p in enumerate(pseudo_probs) if j != k]
                if len(others) > 0 and lam_u > 0:
                    pseudo_target = torch.stack(others).mean(0)
                    # sharpen
                    T = cfg.get("sharpen_T", 0.5)
                    pseudo_target = pseudo_target.pow(1 / T) / (
                        pseudo_target.pow(1 / T) + (1 - pseudo_target).pow(1 / T) + 1e-9
                    )
                    # confidence mask: only windows whose ensemble max prob > τ count
                    conf, _ = pseudo_target.max(dim=-1)
                    mask = (conf > cfg.get("conf_threshold", 0.7)).float()  # (B, T)
                    loss_un = (bce(logits_k, pseudo_target.detach()).mean(dim=-1) * mask).mean()
                else:
                    loss_un = torch.tensor(0.0, device=device)
                loss = loss_sup + lam_u * loss_un
                opts[k].zero_grad(); loss.backward(); opts[k].step()
                epoch_loss[k] += float(loss.detach().cpu())

        for s in scheds: s.step()

        # Validation
        val_aucs = []
        val_preds = []
        for k, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(emb_va, site_ids=site_va, hours=hour_va)
                p = torch.sigmoid(logits).cpu().numpy().reshape(-1, y.shape[-1])
            y_va_flat = y_va.cpu().numpy().reshape(-1, y.shape[-1])
            auc, n = macro_auc(y_va_flat, p)
            val_aucs.append(auc)
            val_preds.append(p)
            if auc > best_score[k]:
                best_score[k] = auc
                best_per[k] = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
        # Ensemble = simple mean of probabilities
        ens = np.mean(val_preds, axis=0)
        ens_auc, _ = macro_auc(y_va_flat, ens)
        history.append({
            "epoch": epoch, "lam_u": lam_u,
            "loss": [round(L / n_steps_per_epoch, 4) for L in epoch_loss],
            "val_auc_per_classmate": [round(a, 4) for a in val_aucs],
            "val_auc_ensemble": round(ens_auc, 4),
        })
        print(f"    epoch {epoch:>2}  λu={lam_u:.2f}  loss={[round(L/n_steps_per_epoch,3) for L in epoch_loss]}  AUC k={[round(a,3) for a in val_aucs]}  ens={ens_auc:.4f}")

    # Return validation predictions at best per-classmate AUC for ensemble eval
    best_preds = []
    for k, model in enumerate(models):
        model.load_state_dict(best_per[k])
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(emb_va, site_ids=site_va, hours=hour_va)
            p = torch.sigmoid(logits).cpu().numpy().reshape(-1, y.shape[-1])
        best_preds.append(p)
    y_va_flat = y_va.cpu().numpy().reshape(-1, y.shape[-1])
    ens_best = np.mean(best_preds, axis=0)
    final = {
        "fold": fold,
        "per_classmate_best_auc": [round(a, 4) for a in best_score],
        "ensemble_best_auc": round(macro_auc(y_va_flat, ens_best)[0], 4),
        "history": history,
    }
    return final, val_mask.cpu().numpy(), best_preds, y_va_flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    run_dir = args.out or (EXP / "runs" / args.config.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}  config={args.config.name}  archs={cfg['classmates']}")

    emb, y, fold_grp, site, hour, class_order = load_data(device)
    print(f"[init] data: emb={tuple(emb.shape)} y={tuple(y.shape)} fold_grp uniques={fold_grp.unique().cpu().tolist()}")

    fold_results = []
    n_classes = y.shape[-1]
    all_preds_per_cm = [np.zeros((emb.shape[0] * emb.shape[1], n_classes), dtype=np.float32) for _ in cfg["classmates"]]
    y_flat_all = y.cpu().numpy().reshape(-1, n_classes)
    fold_id_flat = fold_grp.cpu().numpy().repeat(emb.shape[1])

    folds = sorted(fold_grp.unique().cpu().tolist())
    t0 = time.time()
    for f in folds:
        print(f"\n=== fold {f} ===")
        res, val_mask_files, best_preds, y_va_flat = train_fold(f, emb, y, fold_grp, site, hour, cfg, device)
        fold_results.append(res)
        # scatter best_preds (val of this fold) into the global flat array
        val_mask_flat = np.repeat(val_mask_files, emb.shape[1])
        for k, p in enumerate(best_preds):
            all_preds_per_cm[k][val_mask_flat] = p

    # Global OOF AUC per classmate + ensemble
    print(f"\n=== Global OOF AUC ===")
    per_cm_auc = []
    for k, arch in enumerate(cfg["classmates"]):
        auc, n_eval = macro_auc(y_flat_all, all_preds_per_cm[k])
        per_cm_auc.append({"arch": arch, "macro_auc": round(auc, 4), "n_eval_classes": n_eval})
        print(f"  classmate {k} ({arch})  macro-AUC={auc:.4f}  n_eval={n_eval}")
    ens = np.mean(all_preds_per_cm, axis=0)
    ens_auc, n_ens = macro_auc(y_flat_all, ens)
    print(f"  ENSEMBLE                   macro-AUC={ens_auc:.4f}  n_eval={n_ens}")

    baseline = 0.7999
    print(f"\n  vs single ProtoSSM baseline ({baseline}): Δ = {ens_auc - baseline:+.4f}")
    print(f"  wall time: {time.time() - t0:.1f}s")

    summary = {
        "config": cfg,
        "fold_results": fold_results,
        "global_oof": {
            "per_classmate": per_cm_auc,
            "ensemble": {"macro_auc": round(ens_auc, 4), "n_eval_classes": n_ens},
            "vs_baseline_delta": round(ens_auc - baseline, 4),
            "baseline_single_proto_ssm": baseline,
        },
        "wall_time_sec": round(time.time() - t0, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        run_dir / "oof_predictions.npz",
        **{f"preds_classmate_{k}_{arch}": all_preds_per_cm[k] for k, arch in enumerate(cfg["classmates"])},
        ensemble=ens, y_true=y_flat_all, fold_id=fold_id_flat,
    )
    print(f"\nsaved → {run_dir/'summary.json'}  +  oof_predictions.npz")


if __name__ == "__main__":
    main()
