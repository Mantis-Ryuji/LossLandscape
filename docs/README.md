# ドキュメント案内

> **現在地** — Phase 0と可視化pipelineの確認は完了しています。現在はM-01として、seed 0のB64/B256/B1024を各100epoch実行します。

[プロジェクト概要に戻る](../README.md)

**作業順・進捗は [ToDo List](TODO.md) を参照してください。**

## 資料一覧

| 資料 | 役割 | 状態 |
| --- | --- | --- |
| [プロジェクト概要](../README.md) | 目的、全体の流れ、実行手順 | 現行条件とM-01の手順を記載 |
| [ToDo List](TODO.md) | 依存関係、担当、完了条件、進捗 | V-04/V-05完了、次はM-01 |
| [実験計画](EXPERIMENT_PLAN.md) | 研究上の問い、比較条件、評価、可視化、解釈の方針 | Phase 1を確定、Phase 2・3の一部詳細は未確定 |
| [実装仕様](IMPLEMENTATION_SPEC.md) | モジュール構成、保存形式、処理手順、検証方法 | Phase 0・1の詳細契約を22節に記録、後段はDraft |
| [参考文献・参考実装](REFERENCES.md) | 提供資料と補足資料のリンク、参照する論点 | 現行設計に関係する資料を整理 |
| [設定例](examples/config_example.yaml) | Phase 1設定の説明用コピー | schema v3、microbatch 64、100epoch・固定LR 1e-3（warmupなし） |
| [Phase 0設定](../configs/phase0.yaml) / [Phase 1設定](../configs/phase1.yaml) | 実行設定の正本 | schema v3、Phase 0確認済み、Phase 1実行中 |

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
│   ├── train.py
│   ├── landscape.py
│   ├── projection.py
│   ├── loss_surface.py
│   └── animation.py
├── scripts/
│   ├── check_config.py
│   ├── create_init_checkpoint.py
│   ├── compute_projection.py
│   ├── compute_loss_surfaces.py
│   ├── prepare_splits.py
│   ├── render_animation.py
│   └── run_train.py
└── tests/
    ├── test_config.py
    ├── test_data.py
    ├── test_animation.py
    ├── test_landscape.py
    ├── test_loss_surface.py
    ├── test_models.py
    ├── test_projection.py
    ├── test_seeds.py
    └── test_training.py
```

ルートには概要、詳細資料は `docs/`、説明用の設定例は `docs/examples/` に置きます。実行設定の正本は `configs/` です。資料数が少ないため、研究用・実装用の細かなサブフォルダにはまだ分けません。

[AGENTS.md](../AGENTS.md)に従い、プログラム・テスト・学習・解析はユーザーが実行します。

## 将来のフォルダ構成について

全フェーズを見通した配置案は[実装仕様の Suggested Repository Layout](IMPLEMENTATION_SPEC.md#3-suggested-repository-layout)に残しています。一括作成せず、必要な実装から追加します。

Phase 1に必要な設定、データ、初期化、学習、再開、射影、損失平面、GIF生成は実装済みです。実行手順はルートREADMEの[M-01](../README.md#m-01-seed-0の3run)に記載し、共有splitと共通初期重みを全runで再利用します。

## 残る実装と確認

- M-01のseed 0・B64/B256/B1024各100epoch学習と、その後の共通PCA・両背景・GIF生成。
- Phase 2のSWA/FGEは80〜100epochの共通4epoch三角周期を5回、同じ最低LRの5点での重み平均と予測確率平均を採用済み。毎epochの記録から採取し、半epochの追加保存は行わない。実装はこれから。Model Soupの固定LR 1e-4は確定し、残る条件・実装はその段階で進める。

設計、実装、実行検証を区別し、未実装または未検証の条件は各文書で明示します。
