# Experiment Plan: Optimization Dynamics, SWA/FGE, and Model Soup on CIFAR-10

> **検討中（Draft）** — 研究・評価設計の草案です。実験対象、条件、フェーズ構成、成果物は未確定です。本文中の要件や数値は提案として残しており、実験の開始や内容の確定を意味しません。

[ドキュメント案内に戻る](README.md)

## 1. Research Goal

自己教師あり事前学習済みの同一 checkpoint から CIFAR-10 に fine-tuning するとき、重み空間上の optimization trajectory を観察し、以下を調べる。

1. 通常の optimization が損失地形をどう降りるか。
2. batch size により trajectory の stochasticity、探索範囲、到達領域がどう変わるか。
3. SWA / FGE のための LR 操作によって、学習後半に trajectory がどのように移動するか。
4. SWA/FGE checkpoint 間に高い linear loss barrier があるか。
5. trajectory 内 weight averaging が実際に validation/test performance を改善するか。
6. 同じ SSL checkpoint から分岐した複数 fine-tuning run が linearly connected な low-loss region を共有するか。
7. Model Soup の成否と pairwise connectivity が対応するか。

本実験の中心は **アニメーションによる軌跡観察** であり、最終解だけを比較する実験ではない。

---

# 2. Dataset

## CIFAR-10

- 10 classes
- train: 50,000
- test: 10,000

公式 test set は最終評価にのみ使用する。

train set から stratified split で validation set を切る。

既定値:

- train: 45,000
- validation: 5,000
- test: 10,000

split index は固定し、全 run で共有する。

## Input preprocessing

pretrained ViT を利用するため、既定では 224x224 に resize する。

Training augmentation:

- RandomResizedCrop(224)
- RandomHorizontalFlip
- pretrained model に対応する normalization

Validation/Test:

- Resize / CenterCrop など deterministic transform
- pretrained model に対応する normalization

Mixup / CutMix / RandAugment は初期実験では無効。

理由: 最適化挙動の解釈を単純化するため。

---

# 3. Model

## Default backbone

DINO ViT-S/16 を既定候補とする。

要件:

- ImageNet 系データで SSL pretrained
- BatchNorm 依存が少ない
- 20M〜30M parameters 程度
- classifier head を CIFAR-10 用に置換可能

## Initialization

pretrained backbone を読み込み、10-class head を一度だけ初期化して、その完全状態を

`artifacts/init/theta_0.pt`

として保存する。

全比較 run はこの `theta_0.pt` をロードして開始する。

head initialization を run ごとに変えない。

---

# 4. Shared Fine-tuning Defaults

- optimizer: AdamW
- epochs: 50
- base LR: 1e-4
- weight decay: 0.05
- scheduler: cosine decay
- warmup: 5 epochs
- precision: bf16 AMP where supported
- loss: CrossEntropyLoss
- gradient clipping: disabled by default
- deterministic validation

この値は既定値であり、最終的な LR は Phase 0 の sanity check で loss divergence がないことだけ確認して調整してよい。

SOTA accuracy を目的としない。

---

# 5. Phase 0: Sanity Check

目的は experimental pipeline の検証だけ。

- batch size = 64
- seed = 0
- epochs = 5

確認事項:

- loss が減少する
- accuracy が上昇する
- checkpoint save/load が完全に一致する
- trajectory 用に model weights を抽出できる
- validation evaluator が deterministic

ここでは可視化結果を解釈しない。

---

# 6. Phase 1: Optimization Dynamics and Batch Size

## 6.1 Core question

通常の fine-tuning が損失地形上をどう進むかを観察する。

特に batch size により

- trajectory の揺らぎ
- 同じ低損失領域への入り方
- local trapping 的な挙動
- 最終解周辺での wandering

がどう変わるかを見る。

「small batch は local minima を回避する」と最初から仮定しない。

---

## 6.2 Conditions

Primary comparison:

- B16
- B64
- B256

seeds:

- 0
- 1
- 2

合計 9 runs。

初期 checkpoint は完全に同一。

### Learning-rate policy

Phase 1A ではまず **same base LR** を使う。

これにより、batch size を変えた training configuration の観察を行う。

結果が興味深い場合のみ Phase 1B として以下を追加する。

- matched update count
- LR scaling rule を変えた control

Phase 1A の結果だけで gradient noise の因果効果を断定しない。

---

## 6.3 Checkpoint frequency

少なくとも epoch 単位で保存する。

推奨:

- epoch 0〜10: every 0.5 epoch
- epoch 10〜50: every 1 epoch

可能なら optimizer update 数も保存する。

---

## 6.4 Metrics

各 checkpoint で保存:

- train loss
- validation loss
- train accuracy
- validation accuracy
- LR
- global gradient L2 norm
- parameter displacement from theta_0

\[
 d_t = \|\theta_t - \theta_0\|_2
\]

任意:

- update norm
- cosine similarity between successive updates

---

## 6.5 Primary animation

### Animation A: Batch-size optimization trajectories

背景:

- fixed 2D loss contour

前景:

- B16 trajectory
- B64 trajectory
- B256 trajectory

同一 seed を1画面で比較し、seed ごとに動画を作る。

追加で、全 9 run を薄く重ねた summary animation を作る。

各 frame に表示:

- epoch
- LR
- validation loss
- validation accuracy
- gradient norm

### 観察したいもの

- 初期に trajectory がどれほど共通方向へ進むか
- batch size により path smoothness が変わるか
- small batch が広い領域を探索するか
- large batch が狭い path に収束するか
- final region が batch size で分離するか

---

# 7. Phase 2: LR Manipulation, SWA, and FGE

## 7.1 Core question

学習後半で LR を操作すると、trajectory が low-loss region 内をどのように移動するかを見る。

さらに、その複数 checkpoint を平均したときに performance が改善するか評価する。

---

## 7.2 Starting point

Phase 1 の representative run を1つ選ぶ。

既定候補:

- batch size = 64
- seed = 0

epoch 30 checkpoint を共通 branch point とする。

\[
\theta_{30}
\]

ここから3本に分岐する。

---

## 7.3 Branches

### Normal

通常 cosine schedule を epoch 50 まで継続。

### SWA

epoch 30 以降、比較的高めの LR で学習を継続する。

初期候補:

- SWA start: epoch 35
- SWA LR: 5e-5
- averaging frequency: every epoch

\[
\theta_{\mathrm{SWA}}
=
\frac{1}{K}\sum_{k=1}^K \theta_{t_k}
\]

### FGE

epoch 30 以降、cyclic LR を使用する。

初期候補:

- total: 20 epochs
- cycles: 5
- LR min: 1e-5
- LR max: 1e-4
- snapshot: cycle minima

保存:

\[
\theta^{\mathrm{FGE}}_1,\ldots,\theta^{\mathrm{FGE}}_5
\]

---

## 7.4 Phase 2 animation

### Animation B: Normal vs SWA vs FGE

同じ 2D coordinate system 上で、branch point から3本がどう分岐するかを表示する。

FGE は LR cycle と同期して現在 LR を表示する。

SWA では

- raw trajectory
- running average point

を同時に表示する。

特に、running average point が valley のどこへ移動するかを見たい。

---

## 7.5 Connectivity evaluation

SWA averaging targets と FGE snapshots の pair に対して linear interpolation を評価する。

\[
\theta_\lambda=(1-\lambda)\theta_a+\lambda\theta_b
\]

\[
\lambda \in \{0,0.05,\ldots,1\}
\]

barrier:

\[
\Delta(a,b)=
\max_{\lambda\in[0,1]}
\left[
L(\theta_\lambda)-
\{(1-\lambda)L(\theta_a)+\lambda L(\theta_b)\}
\right]
\]

---

## 7.6 Performance comparison

必須:

- best normal checkpoint
- last normal checkpoint
- SWA model
- best FGE snapshot
- FGE prediction ensemble
- FGE weight average

評価:

- validation loss / accuracy
- test loss / accuracy

FGE では prediction ensemble と weight average を必ず分ける。

---

# 8. Phase 3: Same SSL Checkpoint → Multiple Fine-tuning Runs

## 8.1 Core question

同じ SSL checkpoint + 同じ classifier initialization から開始した複数 fine-tuning run は、本当に linearly connected な low-loss region に留まるか。

また、それらを重み平均した Model Soup が性能を改善するか。

---

## 8.2 Candidate runs

### Seed variation

- R_seed0
- R_seed1
- R_seed2

### Mild hyperparameter variation

必要なら追加:

- LR low: 5e-5
- LR high: 2e-4
- WD high: 0.1 or 0.2

条件差は mild にする。

目的は別世界の解を作ることではなく、同一 initialization 周辺の fine-tuning solution family を見ること。

---

## 8.3 Phase 3 animation

### Animation C: Multiple fine-tuning trajectories

同一 2D contour 上に複数 run を同時表示する。

表示:

- trajectory
- current point
- final solution
- uniform soup point
- greedy soup point

可能なら greedy soup でモデルを追加するたびに soup point がどう移動するかも別アニメーションにする。

---

## 8.4 Model Soup

### Uniform Soup

\[
\theta_{\mathrm{uniform}}=
\frac1K\sum_{k=1}^{K}\theta_k
\]

### Greedy Soup

validation performance に基づいて逐次追加する。

選択に test set を使わない。

---

## 8.5 Pairwise barrier matrix

全 candidate pair で barrier を計算する。

\[
B_{ij}=\Delta(\theta_i,\theta_j)
\]

静止 heatmap を作る。

主に確認したい関係:

- barrier が小さい pair は averaging compatibility が高いか
- greedy soup が barrier の大きい model を避ける傾向があるか
- high-accuracy model でも geometry が悪いと soup を壊すか

---

# 9. 2D Loss Landscape Construction

## 9.1 Principle

2D visualization は phase ごとに目的に合う座標系を構成する。

単一の global 2D projection にすべてを押し込まない。

---

## 9.2 Recommended projection

### Phase 1

全 batch-size trajectory checkpoints の displacement

\[
x_t = \mathrm{vec}(\theta_t-\theta_0)
\]

に PCA を適用する。

### Phase 2

branch point 以降の Normal / SWA / FGE checkpoints を使って PCA。

### Phase 3

複数 fine-tuning run の final solution と trajectory checkpoints を使って PCA。

---

## 9.3 Loss surface evaluation

平面上の

\[
\theta(a,b)=\theta_{ref}+a v_1+b v_2
\]

を評価する。

### Landscape subset

計算量削減のため、train/validation とは別概念として固定評価 subset を作る。

推奨:

- validation から class-balanced 1,000 samples
- deterministic preprocessing
- 全 phase で subset index を保存

### Grid

初期値:

- coarse: 21 x 21
- final: 31 x 31

最初から 51 x 51 以上にしない。

trajectory が contour 外へ出たら range を拡張して再計算する。

---

# 10. Animation Requirements

動画は本実験の主成果物なので、後付けにしない。

## Required outputs

### Phase 1

- `phase1_seed0_batch_compare.mp4`
- `phase1_seed1_batch_compare.mp4`
- `phase1_seed2_batch_compare.mp4`
- `phase1_all_runs_summary.mp4`

### Phase 2

- `phase2_normal_swa_fge.mp4`
- `phase2_swa_running_average.mp4`

### Phase 3

- `phase3_multi_run_trajectories.mp4`
- `phase3_soup_formation.mp4`

## Frame design

1つの動画内で座標系を途中変更しない。

表示内容:

- loss contour
- past trajectory
- current point
- epoch or update number
- current LR
- val loss
- val accuracy

Phase 2 では必要に応じて SWA running average point も表示。

---

# 11. Interpretation Rules

## Allowed claims

- “B16 showed a broader trajectory in this projection.”
- “The two solutions were linearly connected under the evaluated interpolation path.”
- “Weight averaging improved validation accuracy.”
- “FGE snapshots remained within a low-loss region in the chosen 2D slice.”

## Avoid

- “Small batch escapes local minima” from animation alone.
- “These models are definitely in the same basin” from PCA alone.
- “Flat minima generalize better” from visual width alone.

batch size experiment は特に exploratory として扱う。

---

# 12. Go / No-Go Criteria

## After Phase 1

Phase 2 に進む条件:

- trajectory animation が安定して生成できる
- 2D projection が主要 trajectory を十分含む
- checkpoint interpolation が再現可能
- batch-size ごとに少なくとも何らかの trajectory difference が観察できる

batch-size 差がほぼ無くても Phase 2 には進んでよい。

## After Phase 2

Phase 3 に進む条件:

- SWA/FGE trajectory が可視化できる
- averaging / ensemble evaluator が正しく動く
- linear interpolation evaluator が正しく動く

---

# 13. Final Deliverables

最低限:

1. Phase 1 batch-size trajectory animations
2. Phase 2 SWA/FGE trajectory animations
3. Phase 3 multi-run trajectory animation
4. pairwise interpolation plots
5. pairwise barrier matrix
6. performance table
7. reproducible configs
8. all checkpoint metadata

本実験の中心メッセージは、**最終精度の順位ではなく、optimization / LR manipulation / multi-run averaging が重み空間上でどう見えるか**である。
