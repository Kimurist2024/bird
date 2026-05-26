"""Pre-submission safety screen for a sidecar masked-rank correction.

Goal: honor the "tiny + is OK, regression is NOT" criterion by measuring, on the
708-window OOF, whether adding a candidate sidecar as a masked rank correction
helps or hurts — PER CLASS and overall — BEFORE spending an LB submission.

Mirrors the production logic in
  external/.../pilkwang__birdclef26-sidecar-exp002b-5s-weakaudio/sidecar_src/inference/masked_rank_blend.py
(per-class percentile rank, top-k mask, gated interpolation toward sidecar, d-budget cap).

Anchor proxy: rank-blend of Perch + Proto-SSM OOF (the components we have OOF for).
Candidate:    exp002b PCEN-ConvNeXt OOF.

Outputs which classes the correction would move and the net OOF macro-AUC delta.
A NEGATIVE overall delta => do NOT submit this blend. A class-gated variant that
keeps only net-positive classes is also reported (guaranteed >=0 on OOF by construction,
with the honest caveat that the gate is fit and evaluated on the same 708 windows).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "birdclef-2026"
EOS5 = REPO / "external/kaggle_kernels/eos5/datasets"
EOS7 = REPO / "external/kaggle_kernels/eos7sz/datasets"
OUT = Path(__file__).resolve().parent / "results"


def rank_pct(p: np.ndarray) -> np.ndarray:
    return pd.DataFrame(p).rank(axis=0, pct=True).to_numpy(np.float32)


def per_class_auc(y, p):
    out = {}
    for c in range(y.shape[1]):
        s = y[:, c].sum()
        if s == 0 or s == y.shape[0]:
            continue
        try:
            out[c] = roc_auc_score(y[:, c], p[:, c])
        except Exception:
            pass
    return out


def macro(y, p):
    d = per_class_auc(y, p)
    return (float(np.mean(list(d.values()))) if d else float("nan")), len(d)


def topk_mask(arr, k):
    m = np.zeros_like(arr, dtype=bool)
    idx = np.argsort(-arr, axis=1)[:, :k]
    rows = np.arange(arr.shape[0])[:, None]
    m[rows, idx] = True
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # --- load OOF components (708 x 234) ---
    z = np.load(EOS7 / "pilkwang__birdclef26-sidecar-exp002b-5s-weakaudio/oof_predictions.npz", allow_pickle=True)
    y = z["y_true"].astype(np.float32)
    exp002b = z["pred_oof"].astype(np.float32)
    proto = np.load(REPO / "experiments/single_proto_ssm/results/full_oof_meta_features.npz")["oof_base"].astype(np.float32)
    perch = np.load(EOS5 / "jaejohn__perch-meta/full_perch_arrays.npz")["scores_full_raw"].astype(np.float32)

    tax = pd.read_csv(DATA / "taxonomy.csv")
    class_order = list(tax.primary_label.astype(str))
    cls_name = {i: tax.iloc[i].class_name for i in range(len(tax))}

    # --- anchor = rank-blend(Perch, Proto) 40/60 ---
    rP, rPr, rE = rank_pct(perch), rank_pct(proto), rank_pct(exp002b)
    anchor = 0.4 * rP + 0.6 * rPr
    a_auc, a_n = macro(y, anchor)
    print(f"anchor (Perch40+Proto60, rank) OOF macro-AUC = {a_auc:.4f} (n={a_n})")

    # --- production-style masked correction toward exp002b ---
    TAU, ATOPK, STOPK = 0.90, 30, 30
    mask = topk_mask(anchor, ATOPK) | (topk_mask(rE, STOPK) & (anchor >= TAU))
    print(f"correction active on {100*mask.mean():.1f}% of cells")

    results = {"anchor_auc": round(a_auc, 4), "n_eval": a_n, "sweeps": []}
    # global weight sweep (uniform), measure delta
    print("\n=== uniform-weight masked correction (delta vs anchor) ===")
    best = (0.0, 0.0)
    for w in [0.02, 0.05, 0.10, 0.20, 0.30]:
        b = anchor.copy()
        b[mask] = anchor[mask] + w * (rE[mask] - anchor[mask])
        au, _ = macro(y, b)
        d = au - a_auc
        flag = "OK +" if d >= 0 else "REGRESS -"
        print(f"  w={w:.2f}: {au:.4f}  Δ={d:+.4f}  [{flag}]")
        results["sweeps"].append({"w": w, "auc": round(au, 4), "delta": round(d, 4)})
        if au > best[1]:
            best = (w, au)

    # --- per-class gated variant: apply correction ONLY where it improves per-class OOF AUC ---
    base_pc = per_class_auc(y, anchor)
    w_test = best[0] if best[0] > 0 else 0.10
    b_full = anchor.copy()
    b_full[mask] = anchor[mask] + w_test * (rE[mask] - anchor[mask])
    corr_pc = per_class_auc(y, b_full)
    # keep correction only for classes where it helps
    help_classes = [c for c in base_pc if c in corr_pc and corr_pc[c] > base_pc[c] + 1e-9]
    hurt_classes = [c for c in base_pc if c in corr_pc and corr_pc[c] < base_pc[c] - 1e-9]
    b_gated = anchor.copy()
    keep = np.zeros(anchor.shape[1], dtype=bool)
    keep[help_classes] = True
    cell_mask = mask & keep[None, :]
    b_gated[cell_mask] = anchor[cell_mask] + w_test * (rE[cell_mask] - anchor[cell_mask])
    g_auc, _ = macro(y, b_gated)
    print(f"\n=== per-class gated (keep only net-positive classes, w={w_test:.2f}) ===")
    print(f"  classes helped={len(help_classes)} hurt={len(hurt_classes)}")
    print(f"  gated OOF macro-AUC = {g_auc:.4f}  Δ={g_auc - a_auc:+.4f}")

    # where does exp002b help, by taxon?
    from collections import defaultdict
    by_tax = defaultdict(lambda: [0, 0])
    for c in help_classes:
        by_tax[cls_name[c]][0] += 1
    for c in hurt_classes:
        by_tax[cls_name[c]][1] += 1
    print("\n  helped/hurt by taxon:")
    for t, (h, hu) in sorted(by_tax.items()):
        print(f"    {t:<10} helped={h} hurt={hu}")

    results["gated"] = {
        "w": w_test, "helped": len(help_classes), "hurt": len(hurt_classes),
        "gated_auc": round(g_auc, 4), "gated_delta": round(g_auc - a_auc, 4),
        "by_taxon": {t: {"helped": h, "hurt": hu} for t, (h, hu) in by_tax.items()},
    }
    (OUT / "exp002b_screen.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {OUT/'exp002b_screen.json'}")
    print("\nNOTE: gated delta is OOF-fit-and-eval on the same 708 windows (optimistic).")
    print("      Treat as a screen, not a guarantee. Co-submit frozen 0.959 sentinel.")


if __name__ == "__main__":
    main()
