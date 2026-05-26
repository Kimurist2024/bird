# BirdCLEF+ 2026 — experiments

Working repo for BirdCLEF+ 2026 (multi-label bioacoustic species recognition, 234 classes
across Aves / Amphibia / Insecta / Mammalia / Reptilia; macro-averaged ROC-AUC over
5-second windows of 60-second Pantanal soundscapes).

## Baseline

[`baseline/birdclef-2026-eos-7-sz.ipynb`](baseline/birdclef-2026-eos-7-sz.ipynb) — the
**LB 0.959** inference notebook (EoS.7-sz). Rank-space masked-correction ensemble:

```
Perch v2 + Proto-SSM + Residual-SSM  (yukiZ anchor)
  ⊕ SED EfficientNet (rank blend)
  ⊕ BirdNET v2.4 sidecar     (masked rank correction, Aves only)
  ⊕ exp001/exp002b PCEN-ConvNeXt sidecars (masked rank correction, OOF-gated)
```

Sidecar gate weights are ultra-conservative (~0.004 avg) — the authors found aggressive
blending overfits the small public LB.

## Repo layout

| Path | What |
|---|---|
| `baseline/` | The 0.959 baseline notebook + kernel metadata |
| `scripts/` | Polite Xeno-canto (`fetch_xc.py`) + iNaturalist (`fetch_inat.py`) data fetchers |
| `experiments/single_proto_ssm/` | Extracted Proto-SSM single-model baseline (OOF macro-AUC 0.7999) |
| `experiments/noisy_classmates/` | Multi-arch co-evolutionary self-training (v1) |
| `experiments/tta/` | Time-shift TTA probe |
| `experiments/pseudo_label/` | Soundscape pseudo-label self-training |
| `knowledge/` | XC/iNat fetch logs, target species CSV |
| `docs/references/` | Design notes |
| `.claude/skills`, `.claude/agents` | Project skills + agents (rules, technique catalog, trainers) |

Large data (115GB kaggle datasets, 16GB competition data, model weights, embedding caches,
venvs) and secrets are **git-ignored** — see [`.gitignore`](.gitignore). Reproduce them via
the Kaggle CLI from the dataset/model handles in `baseline/eos-7-sz.kernel-metadata.json`.

## Experiment findings so far (toward +0.01 over 0.959)

| Experiment | Result | Note |
|---|---|---|
| Time-shift TTA on Perch anchor | **+0.0002** (no effect) | Perch pools each 5s window → shift-invariant; TTA helps frame-level SED, not pooled embeddings |
| Pseudo-label self-training (conf≥0.8) | **−0.0293** (regression) | conf≥0.8 keeps only strong-class windows; Insecta (weakest) gets no pseudo signal; teacher/student share Perch features → confirmation bias |

**Validation constraint:** the only labeled eval set is 708 windows / 71 of 234 classes —
+0.01 is within OOF noise and can only be confirmed on the real LB.

**Test-distribution proxy (labeled soundscapes):** Amphibia 66.8%, Insecta 18.2%, Aves 13.2%,
Mammalia 1.3%, Reptilia 0.4%. The ensemble is bird-optimized but the test is frog/insect-dominated;
the weakest taxon (Insecta, OOF AUC 0.694) is also the 2nd-most-prevalent — a real, non-phantom
headroom target.

See [`.claude/skills/birdclef-techniques-catalog/SKILL.md`](.claude/skills/birdclef-techniques-catalog/SKILL.md)
for the full technique catalog and failure log.
