# Loss Landscape Dynamics on CIFAR-10

> **検討中（Draft）** — 現在は資料を整理している段階です。研究内容、実験条件、実装仕様、フォルダ構成は未確定であり、本文中の「必須」「既定値」「完了条件」も現時点の案として扱います。実装・実験にはまだ着手しません。

資料の一覧と配置方針は [ドキュメント案内](docs/README.md) を参照してください。

作業順と進捗は [ToDo List](docs/TODO.md) で管理します。現在はPhase 1の実装前に設計・設定の整合を取る段階です。

## 目的

自己教師あり事前学習済みモデルを CIFAR-10 に fine-tune したとき、**最適化軌跡が損失地形上をどう移動するか**をアニメーション中心で可視化する。

現時点では、次の順序で実験を進めることを想定している。

1. **Phase 1: Optimization Dynamics**
   - まず通常の fine-tuning が損失地形をどう降りるかを見る。
   - batch size の違いで trajectory の揺らぎ、到達領域、局所的な捕捉のされ方がどう変わるかを観察する。
2. **Phase 2: LR Manipulation / SWA / FGE**
   - 学習後半で LR を操作し、trajectory が同じ low-loss region 内をどう移動するかを見る。
   - checkpoint 間の barrier と、実際の weight averaging / ensemble の性能を評価する。
3. **Phase 3: Same SSL Checkpoint → Multiple Fine-tuning Runs**
   - 同じ SSL checkpoint から複数条件で fine-tune したモデルの軌跡を比較する。
   - linear connectivity と Model Soup の成否を調べる。

**主成果物はアニメーション。静止図は補助。**

---

## 基本構成（候補）

- Dataset: CIFAR-10
- Backbone: ImageNet-1k で自己教師あり事前学習済み ViT-S 系を既定値とする
- Default candidate: DINO ViT-S/16
- Fine-tuning: full fine-tuning
- Optimizer: AdamW
- Visualization: 2D projected loss surface + checkpoint trajectory animation
- Framework: Python 3.10+, PyTorch
- Plotting: matplotlib
- Config: YAML

CIFAR-10 は 32x32 だが、pretrained ViT に合わせて既定では 224x224 に resize する。高解像度化そのものに意味を持たせず、同一条件間比較のために固定する。

---

## 設計上の方針（案）

### 1. 同一初期値

すべての比較 run は、classifier head の初期値まで含めて完全に同じ初期 checkpoint `theta_0` から開始する。

seed で変えてよいものは、原則として

- DataLoader shuffle
- stochastic augmentation
- dropout 等の stochasticity

のみ。

### 2. 可視化を主張の根拠にしすぎない

2D loss landscape は高次元空間の一断面・一射影にすぎない。

したがって、結論は必ず以下を組み合わせる。

- trajectory animation
- linear interpolation loss
- weight averaging / ensemble performance
- final validation / test metrics

### 3. “same basin” を乱用しない

- `linear barrier ≈ 0` → “linearly connected”
- 2D 上で近い → “close in the chosen projection”
- weight average が壊れない → “weight-space averaging compatible”

と書き分ける。

「同じ basin」と断定するには慎重であること。

### 4. 実験順序を崩さない

Phase 1 を先に完成させる。Phase 1 のアニメーションと保存系が安定する前に SWA/FGE/Soup を実装しない。

---

## ファイル構成

- [AGENTS.md](AGENTS.md): Agent との共通作業規約（計画・仕様の草案とは別に扱う）
- [docs/README.md](docs/README.md): 資料の一覧、配置方針、今後の検討事項
- [docs/TODO.md](docs/TODO.md): 作業順、完了条件、進捗、未決定事項
- [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md): 研究・評価設計の草案
- [docs/IMPLEMENTATION_SPEC.md](docs/IMPLEMENTATION_SPEC.md): 実装仕様と将来のフォルダ構成の案
- [docs/REFERENCES.md](docs/REFERENCES.md): 参考論文・解説記事・公式実装・可視化ツールのリンク集
- [docs/examples/config_example.yaml](docs/examples/config_example.yaml): 検討用の設定例（確定した実行設定ではない）

---

## 最初の完了条件（案）

最初に完成させるべき MVP は **Phase 1 の batch-size 比較アニメーション**。

最低条件:

- CIFAR-10 を読み込める
- 同じ `theta_0` から `batch_size = 16, 64, 256` を実行できる
- checkpoint を一定間隔で保存できる
- 全 trajectory を共通 2D 座標へ射影できる
- 固定 loss contour 上に複数 trajectory を重ねた MP4/GIF を生成できる
- train/val loss, accuracy, LR, gradient norm をログできる

ここまで動いてから Phase 2 へ進む。
