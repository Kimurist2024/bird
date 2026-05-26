---
name: birdclef-workflow-pattern
description: BirdCLEF+ 2026 向け Kaggle 自動化ワークフロー設計パターン。Tom（13位）の公開ワークフローをベースに、知識管理（md/html レポート）、実験ジョブチェイン、論文/フォーラムスクレイピング、提出キュレーション、自己改善ループを構成する。新規パイプライン構築、ジョブ追加、レポート設計、Cron/watcher 設定時に参照。
origin: project
---

# BirdCLEF Automation Workflow Pattern

Tom（Kaggle BirdCLEF 2026 13位）が公開した自動化ループを土台にした、本リポジトリ用の Claude Code 駆動 ML パイプライン設計。

> 関連: [birdclef-2026-rules](../birdclef-2026-rules/SKILL.md)、[birdclef-techniques-catalog](../birdclef-techniques-catalog/SKILL.md)、[birdclef-cv-lb-strategy](../birdclef-cv-lb-strategy/SKILL.md)

## When to Use

- 新しい実験ジョブを実験チェインに追加するとき
- 知識ベース（`knowledge/`）の md/html レポート構造を設計・更新するとき
- スクレイピング/論文収集ジョブを設定するとき
- 提出ノートブック生成・キュレーションパイプラインを構築するとき
- Cron / watcher / handler を追加するとき
- 「自動化ワークフローはどうなってる?」と聞かれたとき

## Non-Negotiable Rules

1. **すべての実験結果は `knowledge/` 配下に md と html の両方で残す** — Claude Code が次回参照する checkpoint になる。
2. **10分ごとの進捗報告と K 分ごとのレポート更新を必ず設定** — 長時間ジョブで Claude が状況を見失わないため。
3. **「Keep improving the result」プロンプトは禁止** — Tom 実証: CV 0.999 まで暴走するがリークで LB 崩壊。代わりに「~80% confidence で 0.938-0.94 LB を目指して現行ベストを拡張」と指定。
4. **submission のオンライン投入は人間確認を挟む** — Claude が自動提出するのではなく、候補ノートブックを生成して人間がレビュー・提出。
5. **知識更新と実験実行は分離した job/agent に持たせる** — 1つの長セッションで全部やらせない（context 汚染、コスト爆発、メモリ管理失敗の温床）。

## Directory Layout

```
/home/st6324034/Bird/
├── birdclef-2026/              # コンペデータ（実コピー、16GB）
├── .claude/
│   ├── skills/
│   │   ├── birdclef-2026-rules/
│   │   ├── birdclef-workflow-pattern/   # ← this
│   │   ├── birdclef-techniques-catalog/
│   │   └── birdclef-cv-lb-strategy/
│   └── agents/
│       ├── birdclef-experiment-runner.md
│       ├── birdclef-paper-scraper.md
│       ├── birdclef-submission-curator.md
│       └── birdclef-anti-overfit-critic.md
├── docs/
│   └── references/             # 設計ドキュメント保管
├── knowledge/                  # 自動更新される md/html レポート
│   ├── MASTER_REPORT.md        # ロールアップ。最新ベスト・直近実験・open issues
│   ├── experiments/            # 実験ごとに 1 md + 1 html
│   ├── ideas/                  # スクレイピング由来のアイデアプール
│   ├── submissions/            # 提出候補とLB結果
│   └── cv_lb/                  # クラス別 CV-LB 相関分析
├── jobs/                       # 実験ジョブの YAML 定義チェイン
│   ├── queue.yaml              # pending / in_progress / completed
│   └── archived/
├── notebooks/                  # Kaggle 提出用 *.ipynb
└── src/                        # 学習・推論・特徴抽出コード
    ├── data/
    ├── models/
    ├── train/
    └── infer/
```

## The 8-Step Loop (Tom's Tips)

### Start (one-shot)
1. **EDA**: train/test の ogg を数本読んでスペクトログラム描画、メタデータ統計
2. **ML workflow scaffold**: data → model → train → val → infer の最小構成
3. **First simple experiment**: SED ベースライン or Perch v2 head の単発実行

### Building System (continuous)
4. **Knowledge updater**: `MASTER_REPORT.md` を K 分ごとに自動更新（最新ベスト/失敗/open issues）
5. **Deep research job**: 論文・公開 notebook・フォーラムを巡回して `ideas/` に蓄積
6. **Experiment job chain**: `jobs/queue.yaml` から pop して実行 → 結果を `experiments/` に書き出し
7. **Watcher/handler**: 実行中ジョブの進捗を 10 分ごとに stdout & md 化（OOM / NaN / 学習停止検知）
8. **Submission curator**: `experiments/` を監視して N 時間ごとに提出ノートブック候補を生成

## Job Queue Schema (`jobs/queue.yaml`)

```yaml
- id: exp-2026-05-20-001
  status: pending        # pending | in_progress | completed | failed
  title: "SED EfficientNet-B0 baseline, 5-fold, mixup 0.5"
  config_path: configs/sed_b0_baseline.yaml
  expected_runtime_min: 240
  expected_val_auc: 0.91
  hypothesis: "再現性確認。SED v0 の素直な再実装"
  depends_on: []
  artifacts_dir: knowledge/experiments/exp-2026-05-20-001/
  created_at: 2026-05-18T12:00:00Z
  notes: |
    - LR 1e-3, AdamW, cosine
    - SumixFreq + Mixup
    - 5-fold by site
```

## Knowledge File Conventions

### `MASTER_REPORT.md` (always-on rollup, 1 file)

```markdown
# Master Report — updated 2026-05-18 12:34

## Best so far
- LB: 0.926  (SED NS R4 + Perch ensemble)
- CV: 0.919 soundscape val
- Notebook: notebooks/sub_20260322_ensemble.ipynb

## Last 5 experiments
| id | result | delta vs best | takeaway |
|---|---|---|---|
| exp-...-007 | 0.918 CV | -0.001 | mixup 0.7 で過剰 |
| ... |

## Open issues
- [ ] CV-LB ギャップ調査（クラス別 ROC probing）
- [ ] CLAP-v2 Stage 1 学習開始待ち

## Next 3 jobs (from queue)
1. exp-...-010 — CLAP-v1 ベースライン
2. exp-...-011 — Perch head ablation
3. exp-...-012 — noisy classmates prototype
```

### `experiments/<id>/` (per-experiment, 1 md + 1 html)

- `report.md`: hypothesis、config、結果（CV/LB/per-class AUC）、学んだこと、次の一手
- `report.html`: 同じ内容を可視化（loss curve、per-class AUC bar、confusion）
- `config.yaml`: 完全な再現用設定
- `metrics.jsonl`: epoch ごとのメトリクス
- `ckpts/`: 最良 checkpoint（容量管理: top-1 のみ）

## Reporting Cadence

| 種類 | 頻度 | 担当 agent |
|---|---|---|
| 実験進捗 stdout & 短い md update | 10 min | experiment-runner |
| `MASTER_REPORT.md` 更新 | 30 min または experiment 完了時 | experiment-runner / submission-curator |
| `ideas/` 追加 | 1 hour | paper-scraper |
| 提出候補生成 | N hours（人間 trigger 推奨） | submission-curator |
| CV-LB 分析更新 | LB スコア取得直後 | anti-overfit-critic |

## Prompt Patterns That Work (Tom's experiments)

✅ **Use**:
- 「現行ベスト notebook を ~80% confidence で 0.938-0.94 LB に拡張して」（incremental, bounded）
- 「次のうちどの submission が最も戦略・自信・LB 予測を変えるか?」（inverse submission guidance — 情報利得最大化）
- 「クラスごとの CV と LB を見て相関しているか確認して。していなければ proxy を提案」

❌ **Avoid**:
- 「Keep improving the result」（CV 0.999 までブレンド暴走 → 完全リーク）
- 「とにかく LB を上げて」（過学習方向に最適化）
- LB 結果を全部 Claude に渡す（記憶しすぎてカスタム CV をハルシネート）

## Cost Discipline (from hengck23 in Tom's thread)

> 「お金は探索ではなくスケールアップにのみ使え。事前にローカルで小規模実験して、性能向上が確実な search space を絞ってから自動探索を回せ」

- 探索（どのアーキ?どの aug?）はローカル小規模で
- 高信頼の方向が見えたら Claude に大量実験させる
- 提出スロット（1日5回）は inverse submission guidance で配分

## Anti-Patterns to Reject

- 1つの巨大な Claude セッションで全部やる（コンテキスト 1M でもダメ。job 分割必須）
- LB スコアの未確認の CV を「ベスト」として MASTER_REPORT に書き込む
- experiment 完了前に次の experiment を pop（同時実行は GPU 競合）
- `knowledge/` を Claude が編集できない場所に置く（更新できなくなる）
- 提出ノートブックを Claude が直接 Kaggle CLI で投入する（事故防止のため人間確認）

## See Also

- [birdclef-techniques-catalog](../birdclef-techniques-catalog/SKILL.md) — どんな experiment を queue に積むか
- [birdclef-cv-lb-strategy](../birdclef-cv-lb-strategy/SKILL.md) — CV-LB ギャップ運用
- [docs/references/clap_cross_domain_design_2026-03-30.md](../../../docs/references/clap_cross_domain_design_2026-03-30.md) — CLAP 設計書
