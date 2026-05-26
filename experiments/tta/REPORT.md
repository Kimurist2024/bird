# TTA Experiment — time-shift TTA on Perch → Proto-SSM

**Date**: 2026-05-26  **Device**: CPU (GPU reservation 1292 expired 05-25; no active reservation)

## Setup
- Consistent time-shift TTA: shift whole 60s file by {−1.0, −0.5, 0, +0.5, +1.0}s, re-window into 12×5s, Perch-embed, Proto-SSM, sigmoid, average across offsets.
- Eval: 66 labeled train_soundscapes, 75 evaluable classes, same OOF set as single-model baseline.
- Baseline = offset {0} only (== no TTA). One run yields the A/B delta.

## Result

| | macro-AUC |
|---|---:|
| Baseline (no TTA) | 0.9313 |
| 5-offset time-shift TTA | 0.9316 |
| **Δ** | **+0.0002** |

Per-taxon delta is within noise everywhere (Aves +0.0005, Mammalia +0.001, Insecta 0, Reptilia −0.002).

## Conclusion: time-shift TTA does NOT move this pipeline

**Negative result, with a clear mechanism**: Perch v2 pools over each 5-second window, so shifting the crop by ≤1s produces a near-identical 1536-dim embedding. Averaging near-identical predictions adds no diversity → ~0 gain.

This is consistent with where TTA's BirdCLEF reputation actually comes from: **frame-level SED models** (EfficientNet on mel patches), not pooled-embedding models like Perch. TTA helps when the augmentation actually changes the model input meaningfully.

> Caveat: baseline here is the data-leaked warm-started Proto-SSM (0.9313, it saw these files in pretraining). But the delta mechanism (shift-invariance) would hold on clean data too.

## Implication for the +0.01 goal

Time-shift TTA is **not** the lever for the Perch-based anchor. Reallocate to:
1. **TTA on the SED branch** (frame-level EfficientNet) — where TTA historically pays off — if a controllable SED inference path exists.
2. **Augmentation-diversity TTA** (gain / noise / band-EQ on waveform), which changes Perch input more than a sub-second shift — worth one more probe.
3. Skip TTA on Perch entirely and move to the higher-evidence lever: **soundscape pseudo-labeling** (the documented "secret ingredient", every-year winner technique).

## Artifacts
- [src/run_tta_inference.py](src/run_tta_inference.py)
- [results/tta_summary.json](results/tta_summary.json)
- [results/tta_run_cpu.log](results/tta_run_cpu.log)
