# ドキュメント案内

> **2026-09-01 検証結果更新** — 実効B64/B256/B1024、microbatch 64、accum 1/4/16の設定確認4件とCPU unit test 64件がユーザー実行で成功（11.626秒）。I-06完了、次は変更後コードのGPU確認です。変更前B64の5epoch・再開記録とは区別します。全フェーズでConvNeXt V2-Tiny、Phase 0・1・2はスクラッチ、SoupだけSSL初期値という境界は維持します。

[プロジェクト概要に戻る](../README.md)

**作業順・進捗は [ToDo List](TODO.md) を参照してください。**

## 資料一覧

| 資料 | 役割 | 状態 |
| --- | --- | --- |
| [プロジェクト概要](../README.md) | 目的、全体の流れ、検証コマンド | スクラッチ／Soup境界と新レシピを反映 |
| [ToDo List](TODO.md) | 依存関係、担当、完了条件、進捗 | I-06完了。変更後のGPU確認を含むS-03は未完了 |
| [実験計画](EXPERIMENT_PLAN.md) | 研究上の問い、比較条件、評価、可視化、解釈の方針 | ConvNeXt V2・AdamWの初期レシピを採用、Phase 2・3詳細は未確定 |
| [実装仕様](IMPLEMENTATION_SPEC.md) | モジュール構成、保存形式、処理手順、検証方法 | Phase 0・1の詳細契約を22節に記録、後段はDraft |
| [参考文献・参考実装](REFERENCES.md) | 提供資料と補足資料のリンク、参照する論点 | 採用範囲を明記・その他は参照用 |
| [設定例](examples/config_example.yaml) | Phase 1設定の説明用コピー | schema v3、microbatch 64、100epoch・固定LR 1e-3（warmupなし） |
| [Phase 0設定](../configs/phase0.yaml) / [Phase 1設定](../configs/phase1.yaml) | 実行設定の正本 | schema v3の設定確認4件・CPUテスト64件成功を受領 |

## 現在のフォルダ構成

```text
LossLandscape/
├── AGENTS.md
├── README.md
├── .gitignore
├── docs/
│   ├── README.md
│   ├── TODO.md
│   ├── EXPERIMENT_PLAN.md
│   ├── IMPLEMENTATION_SPEC.md
│   ├── REFERENCES.md
│   └── examples/config_example.yaml
├── configs/
│   ├── phase0.yaml
│   └── phase1.yaml
├── src/landscape_exp/
│   ├── __init__.py
│   ├── config.py
│   ├── checkpoints.py
│   ├── data.py
│   ├── evaluate.py
│   ├── logging_utils.py
│   ├── models.py
│   ├── seeds.py
│   └── train.py
├── scripts/
│   ├── check_config.py
│   ├── create_init_checkpoint.py
│   ├── prepare_splits.py
│   └── run_train.py
└── tests/
    ├── test_config.py
    ├── test_data.py
    ├── test_models.py
    ├── test_seeds.py
    └── test_training.py
```

ルートには概要、詳細資料は `docs/`、説明用の設定例は `docs/examples/` に置きます。実行設定の正本は `configs/` です。資料数が少ないため、研究用・実装用の細かなサブフォルダにはまだ分けません。

[AGENTS.md](../AGENTS.md)は変更していません。[旧環境確認記録](IMPLEMENTATION_SPEC.md#21-environment-check-2026-08-31)はDINO構造での測定であり、新しいConvNeXt V2のメモリ・時間の根拠には使いません。

## 将来のフォルダ構成について

全フェーズを見通した配置案は[実装仕様の Suggested Repository Layout](IMPLEMENTATION_SPEC.md#3-suggested-repository-layout)に残しています。一括作成せず、必要な実装から追加します。

I-01〜I-04を実装し、今回設定schema・勾配蓄積・テストを改訂しました。プログラム実行と新しい成果物の生成はユーザーが行います。[変更後の検証コマンド](../README.md#今回の検証コマンド)はルートREADMEに記載。S-01のsplitと共通初期重みは再利用します。

## 残る実装と確認

- 勾配蓄積版の実データでの成立性・再開の再現性、B256/B1024のGPUメモリ・時間の測定。今回の64件CPU成功と変更前B64の結果を区別する。
- 共通PCA・両背景・毎epochのGIFの実装と、可読性・各3 MB以下の確認。
- Phase 2のSWA/FGEは80〜100epochの共通4epoch三角周期を5回、同じ最低LRの5点での重み平均と予測確率平均を採用済み。毎epochの記録から採取し、半epochの追加保存は行わない。実装はこれから。Model Soupの固定LR 1e-4は確定し、残る条件・実装はその段階で進める。

2026-09-01のユーザー訂正を優先し、旧DINO・全フェーズfine-tuningという前提を撤回しました。毎epoch記録・両背景・GIF各3 MB以下・保存容量上限なしという要件は維持します。設計、実装、実行検証は区別して記録します。
