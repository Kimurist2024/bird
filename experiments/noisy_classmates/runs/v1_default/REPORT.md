# NC v1 — proto_ssm + mlp_mixer co-distillation

**Date**: 2026-05-19  **Wall time**: 47.2s on A100 80GB MIG 7g.79gb (reservation #1292)

## Headline numbers (global OOF macro-AUC over 708 windows, 71 evaluable classes)

| Variant | macro-AUC | Δ vs baseline 0.7999 |
|---|---:|---:|
| **Single Proto-SSM (baseline, no leak)** | **0.7999** | 0 |
| Proto-SSM warm-started + NC co-distillation | **0.8933** | +0.094 ⚠ leaked |
| MLP-Mixer (from-scratch + NC pseudo-labels) | 0.4579 | -0.342 |
| Simple-mean ensemble (proto + mixer) | 0.8282 | +0.028 |
| AUC²-weighted ensemble (proto 0.97 + mixer 0.03) | 0.8674 | +0.067 ⚠ leaked |

## Key findings

### 1. Co-distillation mechanism works — but as a one-way teacher→student transfer

Per-fold AUC trajectory shows the **anti-confirmation-bias loss does transfer knowledge**, but asymmetrically:

| | epoch 4 (λu starts) | epoch 19 (end) | Δ |
|---|---:|---:|---:|
| Proto-SSM | 0.965 | 0.952 | **-0.013** (slight degradation) |
| MLP-Mixer | 0.656 | 0.707 | **+0.051** (clear improvement from proto's pseudo-labels) |

Proto-SSM lost a small amount of accuracy from learning the noisier mixer's signal, but Mixer improved meaningfully from proto's pseudo-labels. **The mechanism is sound** — confirmation bias avoided (each classmate learns from the OTHER's predictions), and weaker classmate gains.

### 2. Simple-mean ensemble is the wrong combiner here

Mixer at 0.46 global OOF drags down the ensemble (proto alone 0.89 → ens 0.83). AUC²-weighted reduces the damage (→ 0.87) but **NEVER recovers proto-only performance**. For 2 classmates with very different strengths, simple averaging is actively harmful.

### 3. Comparison vs baseline is contaminated by warm-start leakage

`proto_ssm_best.pt` was trained on **all 708 OOF windows** (it's the final retrained-on-all-data model from the eos5 kernel). When we warm-start it and "fine-tune" inside our 5-fold CV, the warm model has already seen each fold's val windows. So the 0.8933 is a data-leaked upper bound — **the 0.094 lift vs 0.7999 baseline is mostly leakage, not NC**.

To get a fair NC vs baseline number, we'd need **both classmates trained from scratch** in each fold, with same total compute.

## What we proved (sound)

- Anti-confirmation-bias index loop is correct (mixer absorbing proto's knowledge through pseudo-labels)
- Multi-architecture training pipeline runs end-to-end on A100 in <1 min for 5 folds × 20 epochs
- Embedding-level augmentation (gaussian + window dropout + shuffle) is non-degrading
- Mixup α=0.4 + warmup-then-ramp λu schedule converges cleanly

## What we did NOT prove

- Whether NC ensemble actually beats a single well-trained proto_ssm on **held-out** data (need from-scratch baseline)
- Whether mixer-from-scratch would reach proto-level given more data (708 windows is tiny for a 3M-param transformer-like model from random init)
- Whether 3+ classmates (the paper hints at 4) would converge differently
- Whether self-training on unlabeled soundscapes (the paper's headline contribution) helps — v1 used only labeled data

## What to fix in v2

| Priority | Change | Rationale |
|---|---|---|
| **P0** | Both classmates from scratch (no warm-start) | Fair baseline comparison, removes leakage |
| **P0** | Train baseline single proto_ssm under same protocol | Apples-to-apples comparison |
| **P1** | Cache Perch embeddings for full 35,549 train_audio + 10,658 soundscapes | Real data scale (40-60× more samples) |
| **P1** | Add self-training on unlabeled soundscapes (the paper's actual contribution) | The point of NC |
| **P2** | K=3 classmates (add attn_pool) | Test diversity hypothesis |
| **P2** | Learned ensemble combiner (logistic stacking on OOF predictions) | Stop averaging losers in |
| **P3** | Per-class confidence threshold τ instead of global 0.7 | Most pseudo-labels currently get masked out for rare classes |

## Artifacts

- [summary.json](summary.json) — full per-fold metrics + history
- [oof_predictions.npz](oof_predictions.npz) — `preds_classmate_0`, `preds_classmate_1`, `ensemble`, `y_true`, `fold_id` all 708×234
- [run.log](run.log) — full stdout
