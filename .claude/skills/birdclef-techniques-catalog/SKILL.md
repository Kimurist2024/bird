---
name: birdclef-techniques-catalog
description: BirdCLEF+ 2026 で公開されている実証済み手法カタログ。SED (Sound Event Detection)、Perch v2 head、CLAP Cross-Domain Contrastive、Distillation from embeddings、Noisy Student / Noisy Classmates、ensemble、aug 戦略を体系化。実験設計、モデル選定、ジョブキュー追加、ablation 計画時に参照。
origin: project
---

# BirdCLEF Techniques Catalog

BirdCLEF 2026 で 0.92-0.94 LB を実証した手法群。新規 experiment を queue に積む前にここを確認し、未試行 / 弱い領域を特定する。

> 関連: [birdclef-workflow-pattern](../birdclef-workflow-pattern/SKILL.md)、[birdclef-cv-lb-strategy](../birdclef-cv-lb-strategy/SKILL.md)
> 詳細設計: [CLAP cross-domain design](../../../docs/references/clap_cross_domain_design_2026-03-30.md)

## 0. Score Landscape (Tom 13位の実証進捗)

| 日付 | アプローチ | LB |
|---|---|---|
| 2026-03-15 | 単一 SED ベースライン | 0.849 |
| 2026-03-17 | SED 改良 | 0.893 |
| 2026-03-19 | distillation 導入後 | 0.918 |
| 2026-03-22 | SED + Perch ensemble | 0.926 |
| 2026-03-25 | secret ingredient 発見 | 0.929 |
| 2026-04-04 | CLAP + Perch + SED 3-ensemble | 0.938 |
| 2026-04-07+ | Noisy Classmates 実験中 | — |

> 「secret ingredient」は forum で明示されていない。soundscape pseudo の活用方法（threshold pct=94、conf≥0.8）が候補。

## 1. SED (Sound Event Detection) ベースライン

### Architecture
- Backbone: EfficientNet-B0 / B2 が王道（CPU 推論 90分制約で B2 が上限）
- Input: mel spectrogram, 32 kHz, 5s window
- Output: per-frame scores → attention pooling → clip-level scores → 234 classes

### Training
- Loss: FocalBCE (alpha=0.5, gamma=2)
- Augmentation: SumixFreq + Mixup (lam=0.5) + time shift + background noise
- Optimizer: AdamW, lr 1e-3, cosine schedule, warmup 500 steps
- 5-fold by site (緯度経度クラスタリング)

### Inference (CPU 90min 制約!)
- Sliding window: 60s ファイル → 12 segments × 5s
- Or: 20s window → 4 overlapping 5s chunks (Tom が採用)
- ONNX export + INT8 量子化を検討

### Known pitfalls
- **データリーク**: train_audio と soundscape の同一録音を別 split に分けると CV 暴騰
- **secondary_labels 無視**: multi-label で扱わないと soft target loss が崩れる
- **rating=0 (iNat)** の扱い: 重み付けの判断が必要

## 2. Perch v2 Head

### Concept
Google の Perch v2 を frozen feature extractor として使う。

```
audio (5s, 32kHz)
  → Perch v2 frozen (~15k-dim logits over Perch's species set)
  → FC(15k → 234) projection
  → FC(256) hidden
  → FC(234) classifier
```

### Why it works
- Perch は数千種の鳥でプリトレ済み → BirdCLEF 234 種への転移が極めて強い
- frozen なので学習は MLP 2-3 層のみ → 数分で 1 epoch
- ensemble 多様性に貢献（CNN 系 SED とは全く別の特徴空間）

### Training detail
- Perch 出力を pre-compute して disk cache（一度作れば学習は数分）
- Head は Dropout(0.2) + Linear、FocalBCE で訓練

## 3. CLAP Cross-Domain Contrastive Bridge

完全な設計書: [docs/references/clap_cross_domain_design_2026-03-30.md](../../../docs/references/clap_cross_domain_design_2026-03-30.md)

### Quick summary
- HTSAT (CLAP audio encoder, 27.5M) を完全 frozen で feature extractor 化
- 学習対象は projection head (512→256) + classifier (512→234) のみ
- **Cross-domain SupConLoss**: 同種の clean clip と wild pseudo clip を positive pair に
- 32 kHz → 48 kHz resample が必須（CLAP の要求 sample rate）
- Pre-compute して disk cache（学習は MLP forward のみ → 高速）

### Two-stage training
| Stage | Loss | LR | Note |
|---|---|---|---|
| 1: Contrastive pre-train | SupCon (cross-domain only) | 1e-4 | 2-3 週、batch 128 |
| 2: Classifier fine-tune | FocalBCE + 0.1×SupCon | 5e-5 | 1-2 週、SupCon は regularizer |

### Prompt 戦略
推奨 v3: `f"sound of {common_name} ({scientific_name}, {class_name}), {call_type}, with background sounds of {secondary}, recorded in Pantanal, Brazil"`

clean prompt と wild prompt を意図的に非対称にする（"sound of" vs "wild soundscape containing"）。

### Variants
| Variant | Description | Target AUC |
|---|---|---|
| CLAP-v1 | 直接 fine-tune (no contrastive) | 0.91-0.93 |
| **CLAP-v2** | SupCon Stage 1 + FocalBCE Stage 2 | **0.93-0.95** |
| CLAP-v3 | v2 + Noisy Student | 0.95+ |
| BioLingual | 生物音響特化 CLAP | 0.92-0.94 |

## 4. Distillation from Embeddings (hengck23's tip)

> 「unlabeled soundscape を大量に投げて埋め込み抽出してデータベースを作り、好みの PyTorch モデルに蒸留する」

### Pipeline
1. **Embedding DB 構築**
   - 入力: 今年の unlabeled soundscape + 過去 BirdCLEF (2021-2025) の wav
   - 教師: CLAP (HTSAT) と Perch v2 の両方の埋め込み
   - 保存: `.npy` 形式、ファイル名 = wav stem
2. **Soft target 生成**: 教師の logits（または埋め込みからの cosine similarity）を target に
3. **Student の蒸留**
   - 軽量 CNN（MobileNetV3, EfficientNet-B0）を student に
   - Loss: KL(student_logits, teacher_logits / T) + α × BCE(student, hard_label)
   - T=4, α=0.5 が出発点

### Why it helps inference budget
CPU 90分制約下では CLAP/Perch を直接動かすのは厳しい。student の軽量 CNN を提出 notebook に載せられる。

## 5. Noisy Student (BirdCLEF 標準)

### Iterations
- R1: clean train_audio のみで teacher 訓練
- R2: teacher で soundscape に pseudo label → conf≥0.8 のみ → student 訓練（train_audio + pseudo）
- R3-R4: 反復。R4 が最もクリーン（threshold pct=94）

### Tom's note
R4 を作るには 1 週間程度。それまでは R2 で代用可（conf≥0.85 に上げる）。

## 6. Noisy Classmates (Tom's novel approach, 2026-04-07)

Noisy Student の拡張。詳細は forum 添付の図のみで論文無し → 推測再構成:

- 「Classmate」= 同一録音内で **同時に鳴いている他種** を pseudo 教師として扱う
- 1 つの soundscape clip から複数種の soft target を抽出
- secondary_labels の補完にも効く
- multi-label / co-occurrence をネイティブに学習

実装は CLAP/Perch teacher を soundscape に流して top-K species の logits を全部 soft target として使う。

## 7. Ensemble Composition

### 3-model target (LB 0.935-0.940)
```
final = w1 × Perch_head_logits
      + w2 × SED_NS_R4_logits
      + w3 × CLAP_v2_logits
weights ≈ [0.33, 0.33, 0.33]
```

### Why diverse
| Model | Backbone | Inductive bias |
|---|---|---|
| Perch v2 head | Conformer (Google's pretrained) | Bird-specific, frozen |
| SED EfficientNet-B0 | CNN | Mel patches, attention pooling |
| CLAP HTSAT | Swin-Transformer | Cross-modal pretraining, token-semantic |

### Weight search
- Bayesian optimization on holdout val
- Or: grid search 0.1 step（3 model なら 36 通り）
- Per-class weighting（クラス別に重み）も検討に値する

## 8. Augmentation Stack

### Time domain
- Random time shift (±2s)
- Speed perturbation (0.9-1.1)
- Time stretch (0.9-1.1)
- Background noise mixing (other soundscape -10 to -20 dB)

### Frequency / spec domain
- SpecAugment (time mask + freq mask)
- SumixFreq (sum two random clips in freq)
- PinkNoise add
- EQ random (boost/cut random band)

### Sample-level
- Mixup (lam=0.5, multi-label aware)
- CutMix on spectrogram
- Label smoothing (0.05)

## 9. Loss Functions

### Standard
- **FocalBCE** (主流): alpha=0.5, gamma=2
- BCEWithLogits + class weights (1/sqrt(freq))

### Contrastive
- **SupConLoss** (cross-domain): temp=0.07
- ArcFace head（角度マージン、稀種に効くケースあり）

### Knowledge distillation
- KL(student/T, teacher/T) × T² + (1-α) × BCE(student, hard)

## 10. Post-Processing (重要、+0.01 級効果あり)

### Per-class threshold calibration
- Validation で best F1 となる threshold をクラス別に決定
- ROC-AUC 評価では threshold は影響しないが、確率較正は影響する

### Temperature scaling
- val NLL を最小化する T を grid search

### Geometric mean of multi-window
- 5s sliding window の N 個の予測を geometric mean（arithmetic より stable）

### Site/time-aware smoothing
- 同一サイト・近い時刻の clip 間で予測を smooth（一晩で完全に種構成が変わらない事前知識）

## 11. What NOT to Do (Tom's failure log)

| やったこと | 結果 |
|---|---|
| 「Keep improving」で blending 暴走 | CV 0.999, LB 崩壊（完全リーク） |
| LB 結果を全部 Claude に渡す | imagination CV をハルシネート |
| 1ファイル巨大セッションで全部 | コンテキスト管理破綻、コスト爆発 |
| 探索を Claude に丸投げ | ローカル小規模で絞ってから渡すべき |
| **時間シフト TTA を Perch anchor に** (2026-05-26) | **+0.0002（無効）。Perch は 5s 窓をプーリングするので ±1s シフトで embedding ほぼ不変。TTA は frame-level SED にのみ効く** |
| **pseudo-label self-training conf≥0.8 で proto_ssm fine-tune** (2026-05-26) | **−0.0293（悪化）。高信頼窓 16.9% しか通過せず全て強クラス(Aves)→ Insecta(0.694) は pseudo に1つも入らず弱クラスを飢えさせる。teacher と student が Perch 特徴共有で confirmation bias。warm-start からのドリフトも交絡** |

> 教訓 (council 2026-05-26): OOF 708 windows / 71 of 234 classes では +0.01 を検証不能。安易な単一レバー (TTA / 単純 pseudo) は効かないと実証済み。次は (1) test に Insecta クラスが存在するか diff-probe で確認 → (2) 存在すれば Insecta-only ロジット補正 (macro 平均の数学上最大 ROI、ただし blast radius を昆虫クラスに限定)。pseudo を再挑戦するなら decorrelated student (異種特徴) + 弱クラス優先サンプリングが必須。

## Backlog of Untested Ideas (forum 由来)

- [ ] BioLingual pretrained CLAP fine-tune
- [ ] NatureLM-audio (arxiv 2411.07186)
- [ ] Per-class threshold + post-hoc calibration
- [ ] Test-time augmentation (TTA): 5s × 3 shifts → mean
- [ ] Site-aware pseudo label confidence boost
- [ ] Insect sonotype-only head (Aves と別に学習)
- [ ] Amphibia の比率 66.8% に合わせた class-balanced sampling
- [ ] Geographic embedding (lat/lon → MLP → concat)

## See Also

- [birdclef-cv-lb-strategy](../birdclef-cv-lb-strategy/SKILL.md) — どの手法で probing するか
- [birdclef-workflow-pattern](../birdclef-workflow-pattern/SKILL.md) — どう自動化するか
