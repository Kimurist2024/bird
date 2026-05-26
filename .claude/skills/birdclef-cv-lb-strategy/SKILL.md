---
name: birdclef-cv-lb-strategy
description: BirdCLEF+ 2026 における CV-LB ギャップの扱い方、submission probing、"imagination CV" 運用、過学習防止。CV 信頼性が極めて低いコンペで、限られた 1日5回の submission slot を情報利得最大化に使うための戦略。新規実験の評価設計、ベスト notebook の選定、提出 candidate のランキング時に参照。
origin: project
---

# CV-LB Strategy for BirdCLEF 2026

BirdCLEF 2026 は CV の信頼性が極めて低い:
- train_audio (clean, single species) vs. soundscape (wild, multi-species, noisy) のドメインギャップが巨大
- 種の分布が不均衡、稀種は test に出るか不明
- macro ROC-AUC は per-class の極端な失敗に弱い

Tom（13位）も「CV が一番難しい、1週間悩んだ」と明言。**CV を信じすぎてはいけない、LB probing で補正する**。

> 関連: [birdclef-techniques-catalog](../birdclef-techniques-catalog/SKILL.md)、[birdclef-workflow-pattern](../birdclef-workflow-pattern/SKILL.md)

## When to Use

- 新規実験の評価指標（CV 何を見るか）を決めるとき
- 提出 candidate を 2-3 個に絞るとき
- 「CV 上がったけど LB 上がらない」と気づいたとき
- Anti-overfit critic を起動するとき
- 残り submission slot をどう使うか決めるとき（特に終盤）

## Core Principles

### 1. CV は per-class で見る、macro 1 数字を信じない
- macro ROC-AUC は per-class AUC の単純平均
- 1 クラスの AUC が 0.5 でも他で補える → macro は 0.92 でも特定クラスは壊れている
- **必ず per-class AUC を `knowledge/cv_lb/per_class_auc_<exp_id>.csv` に保存**

### 2. CV-LB 相関が崩れたら CV を捨てる
- 直近 5 submission で「CV+0.005 で LB ±0」「CV+0.001 で LB+0.01」のような不整合があれば CV は overfit
- そのときは LB probing を CV 代わりに使う

### 3. Submission は情報利得で配分
1日5枠を「ベスト微改善」で使うのは無駄。次の質問に置き換える:
- **「どの submission を投げれば現在の戦略・自信・LB 予測が最も大きく変わるか?」**（Tom の inverse submission guidance）
- 仮説が外れた時こそ corrective signal として最も価値が高い

### 4. ~80% confidence を狙う
- Claude に「現行ベスト 0.926 を ~80% confidence で 0.938-0.94 に拡張」と指定
- 「Keep improving」は CV 0.999 / LB 崩壊 を再現する罠

### 5. LB スコアを全部 Claude に渡さない
- Tom 実証: Claude が「imagination CV」を hallucinate する
- 直近 3-5 件の LB と、それに対応する CV の差分のみ共有

## CV Schemes — 信頼度ランキング

| Scheme | 信頼度 | 用途 |
|---|---|---|
| **train_soundscapes_labels の hold-out** | ★★★★★ | LB に最も近い分布。最重要 |
| Site-based 5-fold (site_id で split) | ★★★★ | Domain shift を再現 |
| Stratified by primary_label | ★★★ | 稀種を確実に val に含める |
| Random K-fold on train_audio | ★ | リーク注意、参考程度 |
| train_audio + soundscape 混合 split | ✗ | 厳禁。リーク確実 |

**推奨**: train_soundscapes_labels（ラベル付き soundscape）を完全 hold-out として常に評価。これが LB に最も近い。

## Probing Strategies (submission を CV 代わりに使う)

Submission は CSV の predict 値。これを差分計算すると per-class / per-site / per-time の partial ROC が漏れる場合がある。

### Strategy A: 単一実験の per-class probe
- 1つの実験の予測を提出 → LB スコア取得
- そのモデルの val per-class AUC と LB の差分を分析
- 「val で高い種」「val で低い種」の LB 寄与を推定

### Strategy B: 2-submission diff probe
- Submission 1: ベースライン
- Submission 2: ベースラインから特定種だけ予測値を 0.5 → 0.0 に変更
- LB の差分から、その種の test set 寄与（その種が test に存在するか、test での AUC）を推測
- **限られた slot を消費するので、影響の大きい候補種を絞ってから**

### Strategy C: Site probe
- train_soundscapes と test_soundscapes は一部 site が重複する事前知識あり
- 提出予測を site 別に微調整 → LB 差分から site 別 ROC を推定

### Strategy D: Ensemble weight probe
- 同じ model 3つ、重みだけ変えた 3 submission
- 重み変化 → LB 変化から「どの model component が最重要か」を回帰推定

## CV-LB Mismatch Diagnostic Flow

```
LB スコア取得
    ↓
CV と LB の差分計算
    ↓
直近 5 件の差分推移を確認
    ↓
┌─ 一貫性あり（±0.005 以内）─→ CV を信じてよい、CV で次の実験ランク付け
│
└─ 不整合 ─→ ┌─ CV ↑ / LB → 0 ─→ overfitting 確定
            │                     - anti-overfit-critic 起動
            │                     - 同じ実験を別 seed で再評価
            │                     - 提出候補を多様性高い方に変更
            │
            ├─ CV → 0 / LB ↑ ─→ CV scheme が test distribution と乖離
            │                     - train_soundscapes_labels hold-out に切替
            │                     - probing で per-class 確認
            │
            └─ CV ↑ / LB ↓ ─→ critical bug or 完全 leak
                                 - 直近の data pipeline diff を全件レビュー
                                 - submission stop
```

## "Imagination CV" Concept (Tom's observation)

> 「LLM の中に imagination CV は確かに存在する。本物の統計的検証ではないが、reward を与えれば perception-based CV proxy を発達させる」

実用化:
- Claude に毎週「現状のベスト stack を 1-10 で評価して」と聞く
- そのスコアと実際の LB の相関を取る
- 相関が出てきたら、Claude の自己評価を soft signal として使える（決して main metric にはしない）

## Anti-Overfit Probes (anti-overfit-critic に投げる)

実験結果を見たら以下を必ず確認:

1. **CV-LB 差分の推移**: 直近 5 件で発散していないか
2. **Per-class AUC 分散**: 一部クラスだけ極端に高くないか（リーク疑い）
3. **Re-train with different seed**: 同 config で seed を変えて再学習 → CV が ±0.005 内に収まるか
4. **Train/Val loss curve**: train loss が val loss より大きく低い epoch から overfit が始まっている
5. **Top-K most confident predictions**: それらが train data の copy ではないか
6. **Soundscape val のみ** での AUC: train_audio val が高くて soundscape val が低ければ domain gap

Claude に質問する形:
- 「この結果が overfit でない証拠は?」
- 「どの probing submission で general化を確認できる?」
- 「CV と LB が乖離した理由は?」

これは Tom も hengck23 から助言を受けて実践していたパターン。

## Submission Slot Budgeting (1日5枚)

### 終盤（残り 7 日）の枠配分例

| 枠 | 用途 |
|---|---|
| 1 | 現行ベスト stack（毎日固定、LB drift 検知） |
| 2 | 最も期待値が高い新規 candidate |
| 3 | Inverse submission probe（仮説が外れた時こそ高情報） |
| 4 | Per-class / site probing（戦略変更に必要なら） |
| 5 | 予備（緊急 hotfix or 別 candidate） |

毎日全枠使い切る必要は無い（提出文化として全部使うが、本コンペは 1日5枠 × 残日数 が有限資源）。

## Per-Class Analysis (hengck23 の助言)

> 「クラスごとに分析せよ。CV と LB の相関のあるクラスとないクラスを分けろ」

実装:
```python
# val per-class AUC
val_auc = roc_auc_score(y_val, p_val, average=None)  # (234,)
# LB per-class はわからないが、過去 LB と現在 LB の差分から間接推定
# クラスごとに「CV 上がったが LB 動かない」「LB 上がるが CV 動かない」を分類
```

- LB 寄与が大きいクラスにフォーカス
- リークしているクラスを発見

`knowledge/cv_lb/correlation_<date>.md` に保管。

## Failure Modes Checklist

実験の結果を見る前にこれをチェック:

- [ ] CV scheme が train_audio 単独 K-fold ではない（→ soundscape hold-out 使え）
- [ ] secondary_labels を target に組み込んでいる（multi-label）
- [ ] Augmentation が train のみで val には適用していない
- [ ] mixup の評価時は無効化
- [ ] Sample rate, normalization が train/val/test で同一
- [ ] Pseudo label の生成元と val が重複していない
- [ ] Ensemble の weight 探索が val 単独で行われている（test に染み出していない）

## Reporting Format

`knowledge/cv_lb/<date>_analysis.md`:

```markdown
# CV-LB Analysis — 2026-05-18

## Recent submissions
| date | exp_id | cv | lb | delta_cv | delta_lb | corr |
|---|---|---|---|---|---|---|
| 5/16 | exp-021 | 0.921 | 0.929 | +0.003 | +0.003 | ✓ |
| 5/17 | exp-024 | 0.928 | 0.928 | +0.007 | -0.001 | ✗ |
| 5/18 | exp-027 | 0.925 | 0.932 | -0.003 | +0.004 | ?? |

## Diagnosis
- exp-024 で CV-LB 不整合 → mixup の augmentation が val にも漏れていた
- exp-027 で逆方向（CV ↓ LB ↑） → val が test より厳しい split になっている

## Action
- val を train_soundscapes_labels hold-out に統一
- exp-024 の mixup 漏れを修正して再評価
- per-class AUC 出力を全実験で必須化
```

## See Also

- [birdclef-techniques-catalog](../birdclef-techniques-catalog/SKILL.md) — 何を実験するか
- [birdclef-workflow-pattern](../birdclef-workflow-pattern/SKILL.md) — どう自動化するか
