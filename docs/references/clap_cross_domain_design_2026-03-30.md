# CLAP Cross-Domain Contrastive Bridge

**Source**: Tom (Kaggle 13位) BirdCLEF 2026 Experiment Design Report, 2026-03-30
**Status**: Reference document — preserved verbatim as design reference for ensemble integration.

## Motivation

Core challenge of BirdCLEF 2026 is **domain gap**:
- `train_audio`: clean close-range recordings, single species, high SNR
- `soundscape`: multi-species overlap, low SNR, background noise, ~66.8% Amphibia / 32.2% Aves
- Current best val AUC ~0.919 (SED + Perch ensemble)

**Goal**: Train a domain-invariant audio encoder so that embeddings of the same species in clean vs. wild domains are as close as possible in latent space.

## 1. CLAP Architecture (LAION laion/clap-htsat-unfused)

- **Audio encoder**: HTSAT (Hierarchical Token-Semantic Audio Transformer), 27.5M params
- **Text encoder**: RoBERTa, 124.6M params (always frozen)
- **Both project to 512-dim**
- **InfoNCE loss** with learnable temperature (init 1/0.07 ≈ 14.3)

### HTSAT input spec
| Setting | Value |
|--|--|
| Sample rate | **48 kHz** (BirdCLEF is 32 kHz → resample required) |
| Max input | 10 s |
| Mel bins | 64 |
| n_fft / hop | 1024 / 480 |
| Freq range | 50 – 14000 Hz |
| Stages depths | [2, 2, 6, 2] (freeze 0-1, fine-tune 2-3) |

> Use `torchaudio.functional.resample(wav, 32000, 48000)` before `ClapProcessor`.

## 2. Core Innovation: Audio-Audio Cross-Domain Contrastive

Replace audio-text pairs with **audio-audio cross-domain pairs**.
- **Positive**: same species, different domain (clean train_audio vs. wild soundscape pseudo)
- **Negative**: in-batch, different species

### Frozen CLAP + Trainable Head (preferred design)

```
   CLAP Audio Encoder (HTSAT, 27.5M) — FULLY FROZEN, eval+no_grad
          ↓ 512-dim audio embedding (pre-computed & cached)
   ════════════════════════════════════════════════════════
   TRAINABLE COMPONENTS ONLY:
   • Projection Head (512 → 256), BN+ReLU+Linear+L2 norm — ~200K params (SupCon)
   • Classification Head (512 → 234), Dropout(0.1)+Linear — ~120K params (FocalBCE)

   Loss = λ₁ × SupConLoss(proj, labels) + λ₂ × FocalBCE(logits, labels)
```

**Benefits of frozen CLAP**:
1. No risk of destroying pretrained features
2. Pre-compute all embeddings once → training is just MLP forward passes (very fast)
3. Much lower GPU memory
4. No 48 kHz resample overhead at training time

### SupConLoss (Cross-Domain)

```python
class CrossDomainSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, z_clean, z_wild, labels):
        # z_*: (B, D) L2-normalized
        z = torch.cat([z_clean, z_wild], dim=0)         # (2B, D)
        labels_2x = labels.repeat(2)
        sim = torch.mm(z, z.T) / self.temp
        sim.fill_diagonal_(-1e9)
        pos_mask = (labels_2x.unsqueeze(0) == labels_2x.unsqueeze(1))
        pos_mask.fill_diagonal_(False)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        n_pos = pos_mask.sum(1).clamp(min=1).float()
        loss = -(log_prob * pos_mask).sum(1) / n_pos
        return loss.mean()
```

## 3. Metadata-Enriched Text Prompts (train.csv)

| Field | Coverage | Value |
|--|--|--|
| primary_label | 100% | Main species |
| secondary_labels | 12.3% | Background species — ✅ High value |
| type | 63.5% | call type (advertisement/alarm) — ✅ High value |
| latitude / longitude | 100% | 🔶 Medium (all Pantanal, limited variation) |
| scientific_name / common_name | 100% | Species semantics |
| class_name | 100% | Aves / Amphibia / etc. |

### Prompt template (recommended v3)

```python
def build_text_prompt(row, taxonomy_df):
    sp = taxonomy_df.loc[row.primary_label]
    prompt = f"sound of {sp['common_name']} ({sp['scientific_name']}, {sp['class_name']})"
    if row['type'] not in ['[]', "['uncertain']", '']:
        call_type = eval(row['type'])[0]
        prompt += f", {call_type}"
    if row['secondary_labels'] != '[]':
        secs = eval(row['secondary_labels'])
        sec_names = [taxonomy_df.loc[s]['common_name'] for s in secs[:2] if s in taxonomy_df.index]
        if sec_names:
            prompt += f", with background sounds of {' and '.join(sec_names)}"
    prompt += ", recorded in Pantanal, Brazil"
    return prompt
```

### Soundscape prompt (wild domain)

```python
def build_soundscape_prompt(pseudo_scores, taxonomy_df, conf_thr=0.8):
    top = pseudo_scores.nlargest(3)
    primary = top.index[0]
    sp = taxonomy_df.loc[primary]
    prompt = f"wild soundscape recording containing {sp['common_name']} ({sp['scientific_name']})"
    others = [taxonomy_df.loc[s]['common_name']
              for s in top.index[1:] if top[s] > conf_thr and s in taxonomy_df.index]
    if others:
        prompt += f" and {', '.join(others)}"
    prompt += ", in Pantanal, Brazil wetland environment"
    return prompt
```

Intentional asymmetry: clean prompts say "sound of ...", wild prompts say "wild soundscape containing ..." — teaches text encoder the domain distinction while forcing audio encoder to learn domain-invariant species reps.

### Prompt variant priority

| Version | Content | Priority |
|--|--|--|
| v1 | `sound of {common_name}` | First (baseline) |
| v2 | + scientific_name + class_name | First |
| **v3** ★ | + call_type + secondary_labels + Pantanal | **Key experiment** |
| v4 | + T5 caption augmentation | Second |
| v5 | + time-of-day | Later |

## 4. Data Preparation

### Clean domain (train_audio)
- 206 species with train_audio
- ~35,549 recordings
- 5s random crop, resampled to 48 kHz for CLAP
- Augmentation: SumixFreq, Mixup (lam=0.5), time shift

### Wild domain (soundscape pseudo clips)
- Source: R4 pseudo labels (cleanest, threshold pct=94)
- Filter: confidence ≥ 0.8, max 200 clips per species
- 5s centered on pseudo offset, resampled to 48 kHz
- Augmentation: background noise, EQ, time stretch

For early experiments before R4 is ready: R1/R2 pseudo labels with threshold ≥ 0.85.

## 5. Training Strategy (Two-Stage)

### Stage 1 — Contrastive Pre-training (2-3 weeks)
- Load `laion/clap-htsat-unfused`
- Text encoder: `eval() + no_grad()` always
- HTSAT stages 0-1 frozen; 2-3 + projection head fine-tuned
- Loss: SupConLoss (cross-domain only)
- LR 1e-4, warmup 500, cosine decay
- Batch 128 (large batch critical for contrastive)
- 32 kHz → resample 48 kHz → ClapProcessor → HTSAT

### Stage 2 — Classification Fine-tuning (1-2 weeks)
- Classification head 768→234 on Stage 1 ckpt
- Loss: FocalBCE + λ_con × SupConLoss (λ=0.1 as regularizer)
- LR 5e-5 (smaller to avoid destroying contrastive features)
- Full BirdCLEF train_audio + R4 pseudo labels

### Inference
- eval() + no_grad()
- Sliding window 20s → 5s (4 overlapping chunks, same as SED NS)
- 5-fold ensemble

## 6. Experiment Variants

| Experiment | Description | Expected AUC | Priority |
|--|--|--|--|
| Baseline | SED NS R4 (EfficientNet-B0) | ~0.92 | — |
| CLAP-v1 | HTSAT direct fine-tune (no contrastive) | 0.91-0.93 | First |
| **CLAP-v2** | HTSAT + SupCon Stage 1 + FocalBCE Stage 2 | **0.93-0.95** | **Key** |
| CLAP-v2+prompt | v2 with enriched prompts (v3) | 0.93-0.95+ | Key |
| CLAP-v3 | v2 + Noisy Student (CLAP-generated pseudo) | 0.95+ | Second |
| BioLingual | BioLingual pretrained + fine-tune | 0.92-0.94 | Second |
| **Ensemble** | CLAP-v2 + SED NS R4 + Perch | **0.95+** | **Target** |

## 7. Risks & Mitigation

| Risk | Mitigation |
|--|--|
| Noisy pseudo labels | conf ≥ 0.85, Aves-only first, use R4 |
| 48 kHz overhead | Pre-extract and cache 48 kHz mel features |
| GPU memory (B=128) | Gradient accumulation (4×32), checkpointing |
| HTSAT overfitting | Strong aug, freeze more stages, weight decay |

## 8. Integration

```
Current:  Perch TFLite + SED NS R4 → ~0.926 LB
Target:   Perch (0.33) + SED NS R4 (0.33) + CLAP-v2 (0.33) → > 0.935 LB
Stretch:  + CLAP-NS (CLAP-v3) → > 0.940 LB
```

HTSAT transformer has very different inductive biases from EfficientNet-B0 CNN. Combined with CLAP's pre-trained cross-modal semantics → genuinely complementary 3-feature-space ensemble.

## 9. References

| Resource | Link |
|--|--|
| LAION CLAP | github.com/LAION-AI/CLAP |
| HF CLAP | laion/clap-htsat-unfused |
| BioLingual | davidrrobinson/BioLingual |
| NatureLM-audio | arxiv.org/abs/2411.07186 |
| CLAP paper | arxiv.org/abs/2211.06687 |
| SupConLoss | arxiv.org/abs/2004.11362 |
| HTSAT | arxiv.org/abs/2107.13228 |
| Microsoft CLAP | zenodo.org/record/8378278 |

## 10. Recommended Next Steps

1. Wait for R4 (~1 week) for cleanest pseudo labels
2. **Week 1**: Install deps, verify 32→48 kHz pipeline, run CLAP-v1 baseline
3. If CLAP-v1 ≥ SED-v5 AUC → proceed to CLAP-v2 (SupCon Stage 1)
4. Run prompt variants v1 → v2 → v3
5. Build 3-model ensemble: Perch + SED NS R4 + CLAP-v2
