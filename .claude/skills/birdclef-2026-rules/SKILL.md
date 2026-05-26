---
name: birdclef-2026-rules
description: BirdCLEF+ 2026 Kaggle コンペティションの公式ルール、データセット仕様、評価指標、提出制約、タイムラインの完全リファレンス。Pantanal湿地帯の音声から鳥・両生類・哺乳類・爬虫類・昆虫を識別するタスクに取り組む際に参照する。
origin: project
---

# BirdCLEF+ 2026 Competition Rules

ブラジル・パンタナール湿地帯の連続音声データから野生生物（鳥類、両生類、哺乳類、爬虫類、昆虫）を識別するKaggleコンペティション。

## Goal

パッシブ音響モニタリング（PAM）から種を識別するMLモデルを構築する。生物多様性モニタリングの自動化が目的。

- 対象地域: Pantanal湿地帯（150,000+ km²、650+ 鳥種）
- 1,000台の音響レコーダーが連続録音
- 234種類のクラスを予測

## Timeline (UTC 23:59)

| 日付 | イベント |
|------|----------|
| 2026-03-11 | Start |
| 2026-05-27 | Entry Deadline / Team Merger |
| **2026-06-03** | **Final Submission Deadline** |
| 2026-06-17 | Working Note Submission |
| 2026-06-24 | Notification of Acceptance |
| 2026-07-06 | Camera-ready |

## Evaluation Metric

**macro-averaged ROC-AUC**（真陽性ラベルがないクラスはスキップ）

## Submission Format

- `row_id` ごとに各種の存在確率を予測
- 1行 = 5秒ウィンドウ
- 234種のカラム
- ファイル名: `submission.csv`
- `row_id` 形式: `[soundscape_filename]_[end_time]`
  - 例: `BC2026_Test_0001_S05_20250227_010002_20` (00:15-00:20のセグメント)

## Code Competition Constraints (CRITICAL)

- **CPU Notebook のみ**: ≤ **90分** ランタイム
- **GPU Notebook は無効化**（1分のみ → 実質提出不可）
- **インターネットアクセス禁止**
- 公開済み外部データ・事前学習モデルは使用可
- 提出は Kaggle Notebook 経由のみ

> **重要**: GPU推論禁止のため、推論時はCPUで90分以内に約600個のテストサウンドスケープを処理する必要がある。モデルサイズ・推論速度の最適化が必須。

## Prizes

| 順位 | 賞金 |
|------|------|
| 1st | $15,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $7,000 |
| 5th | $5,000 |
| Best Working Note (×2) | $2,500 each |

## Dataset Files

データ配置: `/home/st6324034/Bird/birdclef-2026/`

### `train_audio/`
- xeno-canto.org / iNaturalist 由来の個別種の短い録音
- 32 kHz リサンプリング済み、ogg形式
- ファイル名: `[collection][file_id_in_collection].ogg`
- 208サブディレクトリ（種ごと）

### `train_soundscapes/`
- test_soundscapes とほぼ同じ場所の追加音声
- ファイル名: `BC2026_Test_<file ID>_<site>_<date>_<time UTC>.ogg`
- 一部に専門家アノテーション（`train_soundscapes_labels.csv`）
- **重要**: 一部の種は train_audio になく、ここのラベル付き部分にしか存在しない

### `test_soundscapes/`
- 提出時に約600個のレコーディングが配置される
- 1分長、ogg形式、32 kHz
- ロード時間: 約5分

### `train.csv` (メタデータ)

| カラム | 内容 |
|--------|------|
| `primary_label` | 種コード (eBird/iNaturalist taxon ID) |
| `secondary_labels` | 副次的に出現する種（不完全な可能性あり） |
| `latitude`, `longitude` | 録音座標（方言差に注意） |
| `author` | 録音者 |
| `filename` | 音声ファイル名 |
| `rating` | 1-5 品質スコア（XC のみ、0.5減点=背景種あり、0=評価なし） |
| `collection` | `XC` または `iNat` |

### `taxonomy.csv`
- 234行 = 234クラス
- クラス: Aves, Amphibia, Mammalia, Insecta, Reptilia
- 昆虫の多くは sonotype として識別（例: `47158son16`）

### `sample_submission.csv`
- 234種カラム + row_id

### `recording_location.txt`
- 録音地点（Pantanal, Brazil）のメタ情報

## Key Strategic Considerations

1. **CPU推論90分制約** が最大の技術的制約
   - 軽量モデル（MobileNet / EfficientNet-B0 など）
   - ONNX / OpenVINO 量子化を検討
   - バッチ処理最適化

2. **データ分布のミスマッチ**
   - train_audio: 個別種の短い録音（クリーン）
   - test_soundscapes: 1分の連続音声（多種混在、ノイズ）
   - ドメインギャップを埋める必要

3. **train_soundscapes_labels** を必ず活用
   - test と同じ分布の貴重なラベル
   - train_audio にない種が含まれる

4. **secondary_labels の扱い**
   - multi-label 設計を検討
   - 不完全性を考慮（焦点損失等）

5. **地理的多様性と方言**
   - 緯度経度を特徴に
   - 同一種でも地域差あり

6. **昆虫 sonotype**
   - 種同定なしのクラスがある
   - 評価対象に含まれるため無視できない

## External Data Collection: Xeno-canto Etiquette (CRITICAL)

主催者 Stefan Kahl から「Xeno-canto への負荷軽減」要請あり（2026 競技中、XC 側でトラフィック異常が観測された）。XC は対応として **Anubis bot 防御** と **API v3 + API キー必須** を導入済み。Web スクレイピングは Anubis に弾かれるため事実上不可能。**必ず API v3 のみ使用すること**。

### API v3 仕様

- **エンドポイント**: `https://xeno-canto.org/api/3/recordings`
- **必須クエリ**: `key`（必須）, `query`（検索式）, `page`（1始まり）
- **キー取得**: https://xeno-canto.org/account でアカウント作成後に発行（1ユーザー1キー）
- **v2 (`/api/2/`) は 404 シャットダウン済み**
- 検索式は v2 と概ね同じ:`gen:Turdus sp:merula`, `cnt:Brazil`, `q:A` (quality A only) など
- レスポンスは `{numRecordings, numSpecies, numPages, recordings:[...]}`
  - `recordings[i].file` が音声ダウンロード URL（ID から直接組み立て可: `https://xeno-canto.org/{id}/download`）

### 取得時の遵守事項（プロジェクト固有ルール）

1. **キーはコードに埋めない**: `$XC_API_KEY` 環境変数または `~/.xenocanto_key`（mode 600）から読む。git にコミットしない。
2. **レート制限**: 1 req/sec を上限とする。メタデータ問い合わせとファイルダウンロードを合算してカウント。
3. **直列**: 並列ダウンロード禁止（concurrency=1）。
4. **User-Agent**: `BirdCLEF2026-prep/<ver> (research; contact: <user-email>)` の形で識別情報を入れる。
5. **再ダウンロード禁止**: 既に同 ID のファイルがローカルにあるならスキップ。`{species}/XC{id}.ogg` で再開可能に。
6. **失敗時の指数バックオフ**: 429/5xx は 5s, 15s, 45s で最大 3 リトライ。連続 5 件失敗なら停止して報告。
7. **取得上限**: 1 セッションあたり 200 ファイル、1 種あたり `fetch_cap`（`knowledge/xc_targets.csv` 参照）を超えない。
8. **既存 train_audio を尊重**: 既に train.csv にある XC ID と重複させない（`train.csv.filename` の `XC{id}.ogg` を除外）。
9. **取得対象**: `knowledge/xc_targets.csv` で `fetch_cap > 0` の種のみ。Insect sonotype (`47158sonNN`) は XC に存在しないので除外。
10. **取得記録**: 全リクエストを `knowledge/xc_fetch_log.jsonl` に追記（timestamp, species, query, status, n_recordings, downloaded_ids）。

### 違反時の対応

XC からの 429/403 が増えたら **即停止**し、24 時間以上空けてから再開を検討する。主催者の信頼を損なう行為はチーム全体に影響する。

## External Data Collection: iNaturalist API Etiquette

### API 仕様

- **エンドポイント**: `https://api.inaturalist.org/v1/observations`
- **認証**: 読み取り専用は **不要**。トークン (`Authorization: Bearer <jwt>`) は任意で `~/.inat_token` から読む。
- **トークン有効期限**: JWT は ~24h で失効する短命型。スクリプトはトークン無くても動く設計を維持。
- 主要パラメータ: `taxon_id`, `sounds=true`, `quality_grade=research[,needs_id]`, `per_page` (max 200), `page`, `order_by=created_at`
- レスポンス: `{total_results, results:[{id, license_code, sounds:[{id, file_url, license_code, attribution}]}]}`
- `sounds[i].file_url` は mp3/wav/m4a 等。BirdCLEF train_audio は ogg 32kHz なので必要なら下流で resample。

### 取得時の遵守事項

1. **レート**: 公式は 60 req/min 推奨・100 req/min 上限だが、実測で 1.1s/req でも 429 が出る。**1.5 req/sec 上限 (= 1.5s 間隔)** を採用。
2. **直列**: 並列禁止。
3. **429 バックオフ**: 30s, 120s, 300s の指数バックオフ。連続 3 回 429 で停止。
4. **User-Agent**: `BirdCLEF2026-prep/<ver> (research; contact: <email>)` で識別情報を入れる。
5. **再ダウンロード禁止**: 既存ローカル iNat ID と train.csv の iNat ID を skip。
6. **取得対象**: XC でカバーされなかった種を優先。`knowledge/xc_targets.csv` の `fetchable_via_xc` が False または XC ヒット 0 の種。
7. **昆虫 sonotype は不可**: 25 サブクラスがすべて `taxon_id=47158` (Cicadidae) に丸まるため、iNat では区別できない。スキップ必須。
8. **保護種の扱い**: iNat の `geoprivacy=obscured` で座標が伏せられた観察も音声 URL は通常公開。ライセンス (`cc-by`, `cc-by-nc`, `cc-by-nc-sa` 等) を必ず記録し、商用流通禁止クラス (`cc-by-nc*`) は研究目的内で利用。
9. **取得記録**: `knowledge/inat_fetch_log.jsonl` に追記。各 sound に license / attribution / observation_id を保存。
10. **bulk export の優先順位**: 数千件規模なら API ではなく https://www.inaturalist.org/observations/export を使うのが iNat 推奨。今回の規模 (数百件) は API で OK。

## Citation

Stefan Kahl, Tom Denton, Larissa Sugai, Liliana Piatti, Ryan Holbrook, Holger Klinck, and Ashley Oldacre. BirdCLEF+ 2026. https://kaggle.com/competitions/birdclef-2026, 2026. Kaggle.
