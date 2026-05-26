"""Evaluate the Proto-SSM single model on the OOF train_soundscapes subset.

Uses the cached OOF predictions shipped with the kernel so no GPU inference
is required. Outputs per-class, per-fold, and per-taxon macro-AUC plus
worst/best class breakdown.

Run:
    python experiments/single_proto_ssm/src/eval_oof.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
DATA = REPO / "birdclef-2026"

OOF_NPZ = EXP / "results" / "full_oof_meta_features.npz"
META_PARQUET = EXP / "results" / "full_perch_meta.parquet"
OUT_JSON = EXP / "results" / "oof_eval_summary.json"


def hms_to_sec(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def macro_auc(y: np.ndarray, p: np.ndarray, class_order: list[str]):
    aucs, per_class = [], {}
    for c in range(y.shape[1]):
        col = y[:, c]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        try:
            a = roc_auc_score(col, p[:, c])
            aucs.append(a)
            per_class[class_order[c]] = a
        except Exception:
            pass
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs), per_class)


def main() -> None:
    oof = np.load(OOF_NPZ, allow_pickle=True)
    oof_base = oof["oof_base"]
    oof_prior = oof["oof_prior"]
    fold_id = oof["fold_id"]

    meta = pd.read_parquet(META_PARQUET)
    meta = meta.copy()
    meta["end_sec"] = meta["row_id"].str.rsplit("_", n=1).str[1].astype(int)
    meta["file_key"] = meta["row_id"].str.rsplit("_", n=1).str[0] + ".ogg"

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

    y_true = np.zeros_like(oof_base)
    matched = 0
    for i, row in meta.iterrows():
        key = (row["file_key"], int(row["end_sec"]))
        if key in win_labels:
            matched += 1
            for lbl in win_labels[key]:
                col = label_to_col.get(lbl)
                if col is not None:
                    y_true[i, col] = 1.0
    print(f"matched {matched}/{len(meta)} windows to expert labels")

    summary: dict = {"n_windows": int(len(meta)), "matched": int(matched), "n_classes_total": 234}

    print("\n=== PROTO-SSM SINGLE MODEL OOF MACRO-AUC ===")
    for name, arr in [("oof_base", oof_base), ("oof_prior", oof_prior)]:
        auc, n, _ = macro_auc(y_true, arr, class_order)
        print(f"  {name:<12} macro-AUC = {auc:.4f}  (n_eval_classes = {n})")
        summary[name] = {"macro_auc": auc, "n_eval_classes": n}

    auc_base, _, pc_base = macro_auc(y_true, oof_base, class_order)

    print("\n=== Per-fold (oof_base) ===")
    summary["per_fold"] = {}
    for f in sorted(set(fold_id.tolist())):
        mask = fold_id == f
        a, n, _ = macro_auc(y_true[mask], oof_base[mask], class_order)
        print(f"  fold {f}: macro-AUC = {a:.4f}  windows={int(mask.sum())}, eval_classes={n}")
        summary["per_fold"][int(f)] = {"macro_auc": a, "n_windows": int(mask.sum()), "n_eval_classes": n}

    print("\n=== By taxonomic class (oof_base) ===")
    summary["by_taxon"] = {}
    for cl in ["Aves", "Amphibia", "Mammalia", "Reptilia", "Insecta"]:
        cls_lbls = set(tax[tax.class_name == cl].primary_label.astype(str))
        aucs = [a for lbl, a in pc_base.items() if lbl in cls_lbls]
        if aucs:
            mean = float(np.mean(aucs))
            print(f"  {cl:<10} mean AUC = {mean:.4f}  evaluated {len(aucs)}/{len(cls_lbls)}")
            summary["by_taxon"][cl] = {"mean_auc": mean, "n_evaluated": len(aucs), "n_total": len(cls_lbls)}
        else:
            print(f"  {cl:<10} (no positives in OOF subset)")
            summary["by_taxon"][cl] = None

    print("\n=== Worst 10 classes (oof_base) ===")
    worst = sorted(pc_base.items(), key=lambda kv: kv[1])[:10]
    summary["worst_10"] = []
    for lbl, a in worst:
        cnt = int(y_true[:, label_to_col[lbl]].sum())
        row = tax[tax.primary_label.astype(str) == lbl].iloc[0]
        print(f"  {lbl:>10}  {row.scientific_name[:28]:<28} {row.class_name:<10} AUC={a:.3f} pos={cnt}")
        summary["worst_10"].append({"label": lbl, "name": row.scientific_name,
                                     "class": row.class_name, "auc": a, "positives": cnt})

    print("\n=== Best 10 classes (oof_base) ===")
    best = sorted(pc_base.items(), key=lambda kv: kv[1])[-10:]
    summary["best_10"] = []
    for lbl, a in best:
        cnt = int(y_true[:, label_to_col[lbl]].sum())
        row = tax[tax.primary_label.astype(str) == lbl].iloc[0]
        print(f"  {lbl:>10}  {row.scientific_name[:28]:<28} {row.class_name:<10} AUC={a:.3f} pos={cnt}")
        summary["best_10"].append({"label": lbl, "name": row.scientific_name,
                                    "class": row.class_name, "auc": a, "positives": cnt})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsummary written to {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
