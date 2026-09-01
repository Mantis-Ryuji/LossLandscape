# Experiment Plan: Optimization Dynamics, SWA/FGE, and Model Soup on CIFAR-10

[ドキュメント案内に戻る](README.md)

## 1. Research Goal

CIFAR-10でConvNeXt V2-Tinyをスクラッチ学習するとき、重み空間上のoptimization trajectoryを観察する。Phase 1・2では外部の事前学習重みを使わない。最後のPhase 3でのみ、自己教師あり事前学習済みの同一checkpointからCIFAR-10へfine-tuningするModel Soup実験を行う。

モデル構造を全フェーズで統一し、初期化方式の違いは明示的に分離して、以下を調べる。

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

全フェーズでConvNeXt V2-Tinyのモデル構造を共通に保つため、既存の224×224入力を維持する。前処理設定の利用と事前学習重みの利用は別であり、Phase 0・1・2では重みをロードしない。

Training augmentation:

- RandomResizedCrop(224)
- RandomHorizontalFlip
- ConvNeXt V2の設定から記録したnormalization

Validation/Test:

- Resize / CenterCrop など deterministic transform
- ConvNeXt V2の設定から記録したnormalization

Mixup / CutMix / RandAugment は初期実験では無効。

理由: 最適化挙動の解釈を単純化するため。

---

# 3. Model

## Default backbone

ConvNeXt V2-Tinyを全フェーズ共通のモデルとする。Phase 0・1のtimm識別子は`convnextv2_tiny`、`pretrained=False`を固定する。

要件:

- スクラッチ学習と、最後のModel Soup用のSSL checkpointの両方を同一モデル構造で扱える
- BatchNorm 依存が少ない
- 20M〜30M parameters 程度
- classifier head を CIFAR-10 用に置換可能

## Initialization

Phase 0・1ではinit seed 0でbackboneと10-class headをまとめてモデル既定の方式で初期化し、その完全状態を

`artifacts/init/convnextv2_tiny_scratch/theta_0.pt`

として保存する。

Phase 1の全比較runはこの同一のスクラッチ`theta_0.pt`をロードして開始する。Phase 2は、そのスクラッチ学習途中のcheckpointから分岐する。

backbone・headともrun seedごとに初期化し直さない。Phase 3は純粋なFCMAE事前学習済みcheckpoint（`convnextv2_tiny.fcmae`）と共通headから別の初期値を作り、独立したfine-tuning群を実行する。Phase 1・2の学習済みcheckpointをSoup候補に再利用しない。

---

# 4. Scratch Training Recipe

Phase 0・1はAdamW・全期間100epoch・固定LR 1e-3を使う。warmupとLR decayは行わない。SWA/FGE以外の実験では各run内のLRを固定し、Model Soupは別の固定値1e-4を使う。

| 項目 | 設定 | 選定理由・制約 |
| --- | --- | --- |
| optimizer | AdamW | betas=(0.9, 0.999)、eps=1e-8、全parameterへweight decayを一様に適用する。 |
| 全学習期間 | 100epoch | epoch 0と毎epoch終了時の101時点を全て記録する。収束の保証ではない。 |
| learning rate | 1e-3固定 | 初回から最終更新まで、全バッチで同じ値を使う。自動スケーリングやrun別調整をしない。 |
| scheduler / warmup | constant / 0epoch | 軌跡の時間変化からLR操作という要因を除く。最終更新のLRも1e-3。 |
| weight decay | 0.05 | 全runで共通とする。 |

100epoch・固定LR 1e-3は軌跡観察のための実験条件であり、CIFAR-10に対する最適性や収束は未検証。[公式TRAINING.md](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md)はImageNetでのFCMAE事前学習・fine-tuning手順なので、このスクラッチ学習の実証根拠とはしない。

維持する実装条件:

- scheduler: constant、epochを共通時間軸にする
- precision: bf16 AMP where supported
- loss: CrossEntropyLoss
- gradient clipping: disabled by default
- deterministic validation

Phase 0でpipelineの成立性と資源を確認し、Phase 1で同じ学習条件を100epoch実行する。runごとの精度を見てLR等を調整しない。schedule・augmentation・再現性の詳細は[実装仕様22節](IMPLEMENTATION_SPEC.md#22-phase-0--1-operational-contract-d-05--d-06)に従う。ConvNeXt V2とAdamWへ手法を適用する探索実験であり、参考論文のモデル・optimizer・条件をそのまま再現した実験とは区別する。

SOTA accuracy を目的としない。

---

# 5. Phase 0: Sanity Check

目的は experimental pipeline の検証だけ。

- 基本sanity runは実効batch size 64
- seed = 0
- scheduleは本比較と同じ100epoch・固定LR 1e-3を保持し、5epoch終了時に停止する（`stop_after_epoch=5`）。warmupは行わない

基本B64の学習・保存・再開確認後、同じPhase 0契約を使うGPU probeとして実効B256（microbatch 64・accum 4）とB1024（microbatch 64・accum 16）も各5epoch実行する。比較実験の本runではなく、実データ・bf16・端数実効batch・CUDAメモリ・成果物保存が成立するかの確認とする。seed 0、共通theta_0、train 45,000件、LR・optimizer・評価・記録間隔は変えない。別の実験名に保存し、既存B64やPhase 1成果物を上書きしない。

確認事項:

- loss / accuracyの推移を確認し、非有限値・更新の不成立がない
- checkpoint save/load が完全に一致する
- trajectory 用に model weights を抽出できる
- validation evaluator が deterministic

ここでは可視化結果を解釈しない。5epochの初期動作と、100epochを通した安定性・最終精度の確認を区別する。

5epochでの精度向上そのものは合否条件にしない。学習・評価・保存・PCA・両格子・GIFの時間と資源を分けて測り、本比較前に実行規模を見積もる。壁時計時間の自動打ち切りは設けず、実行はユーザーが行う。

---

# 6. Phase 1: Optimization Dynamics and Batch Size

## 6.1 Core question

同一のスクラッチ初期値から始める学習が、損失地形上をどう進むかを観察する。

特に batch size により

- trajectory の揺らぎ
- 同じ低損失領域への入り方
- local trapping 的な挙動
- 最終解周辺での wandering

がどう変わるかを見る。

「small batch は local minima を回避する」と最初から仮定しない。

---

## 6.2 Conditions

実効バッチはB64/B256/B1024、全条件でmicrobatch 64、勾配蓄積1/4/16回を使う。モデルはConvNeXt V2-Tinyをスクラッチ学習し、4節のAdamW・100epoch条件を適用する。

Primary comparison:

- B64
- B256
- B1024

seeds:

- 0
- 1
- 2

合計 9 runs。

初期 checkpoint は完全に同一。

Phase 1の最小版では、まずseed 0のB64/B256/B1024を完成させる。その後、同じ共通条件でseed 1・2を追加する。

| 実効バッチ | microbatch | accumulation steps | AdamW更新数/epoch | 100epochの更新数 |
| --- | --- | --- | --- | --- |
| B64 | 64 | 1 | 704 | 70,400 |
| B256 | 64 | 4 | 176 | 17,600 |
| B1024 | 64 | 16 | 44 | 4,400 |

GPUへ渡す枚数は共通にし、同じ重みで計算したmicrobatchの勾配を画像数で平均してからAdamWを1回更新する。train 45,000件を捨てずに使うため、各epochの最後の実効バッチはそれぞれ8/200/968件となる。端数も実画像数で平均し、次epochへ勾配を持ち越さない。LRの自動スケーリングはしない。

この比較範囲だけで、小さいbatchが不要であることやB1024の収束を主張しない。勾配蓄積と物理的な大バッチは、同じ入力・重みでの画像数平均という定義を共有するが、浮動小数点演算のbitwise一致は保証しない。モデルはbatch統計を使うBatchNormを含まず、dropout/drop pathは0とする。

共通条件はCIFAR-10、ConvNeXt V2-Tiny、224×224、全parameterのスクラッチ学習とする。分類headまで含む同一のスクラッチ初期checkpoint `theta_0`を全runで共有する。SWA、FGE、Model SoupはPhase 1の対象外とする。

AdamW・100epoch・固定LR 1e-3・weight decay 0.05・warmupなしを全runで共有する。Phase 0で初期の動作を確認し、個々のrunの都合では条件を変更しない。

### Learning-rate policy

Phase 1Aでは **全run・全更新で同じ固定LR 1e-3** を使う。

これにより、LRを時間変化させずにbatch sizeを変えたtraining configurationを観察する。LR減衰によって軌跡の動きが小さくなる要因は除けるが、AdamWの勾配履歴に基づく適応的な更新は残る。固定LRは固定の更新幅を意味しない。[PyTorch 2.6 AdamW](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/adamw.py)

batch sizeごとに更新回数が異なり、AdamWの履歴・weight decayの累積適用も異なる。固定LRだけで勾配ノイズの因果効果を切り分けたとは主張しない。終盤にも移動や揺れが続く可能性を許容し、精度・収束を動画の合否条件にしない。

結果が興味深い場合のみ Phase 1B として以下を追加する。

- matched update count
- LR scaling rule を変えた control

Phase 1A の結果だけで gradient noise の因果効果を断定しない。

---

## 6.3 Checkpoint frequency

記録は全期間1 epoch単位とする。

- 共通初期状態 `theta_0`をepoch 0、optimizer step 0の起点として記録する。
- 解析用checkpointと指標は、毎epoch終了時に記録する。初期だけ保存間隔を細かくすることはしない。
- 保存時のepochと、累積optimizer更新回数 `global_step`を記録する。
- Phase 0の短い確認にも同じ記録間隔を適用する。

全期間E epochsのrunではepoch 0〜EのE+1時点、5 epochsのPhase 0ではepoch 0〜5の6時点となる。

保存容量に設計上の上限を設けず、初期状態と毎epochの解析用checkpointをすべて保持する。容量節約を理由に記録を間引いたり既存成果物を自動削除したりしない。RAM・VRAMの制約に合わせた分割処理と、GIFの各ファイル3 MB以下という要件を適用する。

解析用checkpointはFP32の`.pt`形式とする。再開用checkpointは別に毎epoch終了時に保存し、重み・optimizer・scheduler・使用する場合のscaler・乱数状態・データ順序の状態・完了epoch・累積optimizer更新回数・設定を保持する。最後に保存を完了したepochの次から再開し、中断したepochは先頭からやり直す。詳細は[保存仕様](IMPLEMENTATION_SPEC.md#7-checkpoint-format)を参照する。

設定構造・パス・PCAの計算方法・再開の詳細手順・予算の扱いはD-05で確定し、[実装仕様22節](IMPLEMENTATION_SPEC.md#22-phase-0--1-operational-contract-d-05--d-06)へ記録した。

動画は全runを共通のepoch時間軸で比較し、各runのoptimizer stepも併記する。同じepochでもbatch sizeにより更新回数は異なる。checkpoint間を結んだ線は保存点の接続であり、epoch内の経路や更新単位の揺らぎを表すものではない。

---

## 6.4 Metrics

毎epoch終了時のcheckpointに対応して保存:

- 学習中のtrain loss / accuracy（epoch単位の学習ログ）
- 実checkpointのtrain-subset loss / accuracy（固定1,000件・評価用前処理）
- 実checkpointのvalidation loss / accuracy（validation全体）
- LR
- global gradient L2 norm
- parameter displacement from theta_0

\[
 d_t = \|\theta_t - \theta_0\|_2
\]

実checkpointの評価はD-04に従い、初期状態（epoch 0）でも実施する。オンラインtrain lossは画像数で重み付けした平均、accuracyは正解数/画像数、gradient normは各更新の勾配L2 normのepoch平均とする。epoch 0のオンライン指標・gradient norm・使用済みLRはJSONでnull、CSVで空欄、図でN/A。実checkpoint評価はepoch 0にも保存する。学習中の指標、固定subsetでの評価、平面上の背景損失を同じ値として扱わない。

任意:

- update norm
- cosine similarity between successive updates

---

## 6.5 Primary animation

### Animation A: Batch-size optimization trajectories

背景:

- 主表示: trainの固定subsetによる2D loss contour
- 補助版: validationの固定subsetによる2D loss contourを使った別GIF

前景:

- B64 trajectory
- B256 trajectory
- B1024 trajectory

同一 seed を1画面で比較し、seed ごとに動画を作る。

追加で、全 9 run を薄く重ねた summary animation を作る。

seedごとの比較とsummaryの両方で、train背景・validation背景の2種類を生成する。対になるGIFは、PCAの原点・基底、格子範囲、軌跡、epochの進行、色尺度を共通にする。背景データの違いだけを比較できるようにし、追加の学習runは行わない。

各 frame に表示:

- epoch
- LR
- 実checkpointのtrain-subset loss / accuracy
- 実checkpointのvalidation loss / accuracy（validation全体）
- gradient norm
- 背景のデータ区分（train subset / validation subset）と件数

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

さらに、同じcheckpoint群を使って、SWAの重み平均とFGEの予測確率平均を比較する。平均方法以外に学習軌跡や採取時点の差を混ぜない。

---

## 7.2 Starting point

Phase 1のスクラッチ学習runを1つ選ぶ。外部SSL checkpointからのfine-tuningに切り替えない。

既定候補:

- batch size = 64
- seed = 0

epoch 80終了時の重み・AdamW状態・乱数状態を共通branch pointとする。epoch 81〜100の20epochで、4epoch周期を5回行う。

\[
\theta_{\mathrm{branch}}
\]

ここからNormalの固定LR系列と、SWA/FGE共通の周期LR系列を比較する。SWAとFGEは同じraw checkpointを参照し、独立した2回の学習は必要としない。

---

## 7.3 Branches

### Normal

Phase 1と同じ固定LR 1e-3を継続し、SWA/FGEと同じ開始・終了epochで比較する。

### SWA

FGEと同じ周期LRの軌跡から採取した5点の重みを等重み平均する。LRと採取時点をFGEに揃え、平均方法だけを比較する。原論文の設定と本実験の条件は後述のとおり区別する。

採用条件（未実装）:

- LR: 1e-3 → 1e-5 → 1e-3の三角周期、4epoch/cycleを5回
- averaging targets: FGEと同じ各周期中央の最低LR地点、計5点。branch point自体は含めない
- averaging frequency: 採取時だけ更新。全epochの記録は維持し、採取間では平均モデルを保持する。最初の採取前は平均モデル未定義
- 平均モデルを学習側へ戻さず、AdamW状態も平均・リセットしない

\[
\theta_{\mathrm{SWA}}
=
\frac{1}{K}\sum_{k=1}^K \theta_{t_k}
\]

### FGE

SWAと同じ周期LRの学習記録を使い、同じ5点のsoftmax後の予測確率を等重み平均する。logit平均とは区別する。

採用条件（未実装）:

- total: 分岐後20epoch、4epoch/cycleを5回
- LR: SWAと共通の1e-3 → 1e-5 → 1e-3の三角周期
- snapshot: SWAと同じ周期中央の最低LR地点、計5点

保存:

\[
\theta^{\mathrm{FGE}}_1,\ldots,\theta^{\mathrm{FGE}}_K
\]

### 原論文の設定と今回への適用

以下は代表的なCIFAR実験の参照値であり、今回のConvNeXt V2＋AdamWの確定値ではない。

| 手法・対象 | 原論文でのLRと採取 | 出典 |
| --- | --- | --- |
| SWA：CIFAR-10のVGG / PreAct-ResNet / WRN | 通常学習の約75%地点から固定LR 0.01。固定LR版はepoch末の重みを等重み平均。 | [SWA §3.2・§4.1・Appendix A.1](https://arxiv.org/pdf/1803.05407) |
| FGE：VGG | LR 0.01 → 0.0005 → 0.01の三角周期、2epoch/cycle。周期中央の低LR地点で採取。 | [FGE §5・§6](https://arxiv.org/pdf/1802.10026) |
| FGE：PreAct-ResNet / WRN | LR 0.05 → 0.0005 → 0.05、4epoch/cycle。同じく周期中央で採取し、予測確率を平均する。 | [FGE §5・§6](https://arxiv.org/pdf/1802.10026)・[著者実装](https://github.com/timgaripov/dnn-mode-connectivity/blob/master/fge.py) |

FGEは通常学習の約80%地点から開始する説明だが、実験の分岐・予算はモデル依存。Appendix A.9ではResNetはepoch 125から22epoch、VGG/WRNはepoch 120と156から各22epochの2系列を使う。単に全モデルを「最後の20%」で1系列だけ回した実験とは区別する。

著者のCIFAR実装はmomentum 0.9のSGDを使う。論文のLRをAdamWへそのまま移植する根拠はない。モデル・optimizer・通常学習のscheduleも今回と異なるため、以下は今回の比較用の採用条件であり、原実験の再現条件とは呼ばない。SWAの約75%開始は代表的な実験設定であって、常にepoch 75で開始するという手法の要件ではない。

**共通条件（未実装）:** epoch 80終了時の同じ重み・AdamW・RNGから分岐し、epoch 100終了まで比較する。B64・seed 0を既定候補とする。

- Normal：固定LR 1e-3で同じ20epochを比較する。
- SWA/FGE共通学習：最大1e-3・最小1e-5、4epochの三角周期を5回。前半2epochで低下、後半2epochで上昇させ、epoch 100終了で5周期を完了する。
- 共通採取点：epoch 82・86・90・94・98終了時の最低LR地点、計5点。同じcheckpoint IDと順序を両方式で使う。
- SWA：この5点の重みを等重み平均した単一モデルを評価する。
- FGE：この5モデルそれぞれのsoftmax確率を等重み平均して評価する。

B64候補では1epoch=ceil(45000/64)=704更新なので、1周期=2,816更新、半周期=1,408更新となり、最低LR地点がepoch終了時と一致する。step内のLR適用・端点、評価・射影の残る詳細はA-01で確定する。

checkpoint・指標・再開記録は毎epochのまま維持し、平均用の5点もその記録から選ぶ。半epochでの追加保存は行わない。SWAとFGEのraw trajectoryは共通であり、同じ周期学習を別々に実行しない。平均処理は共通checkpointから行い、単体・重み平均・予測平均の性能改善は未検証として扱う。

---

## 7.4 Phase 2 animation

### Animation B: Normal vs SWA vs FGE

同じ2D coordinate system上で、Normalの軌跡、SWA/FGE共通のraw trajectory、SWA平均点を区別して表示する。SWAとFGEが別の学習軌跡に分岐したようには描かない。

SWA/FGE共通のLR cycleと同期して現在LR・採取点・平均対象数を表示する。GIFは1epoch単位を維持し、SWA平均点は採取を完了したepochのframeから更新する。未来の採取点を先に平均へ含めない。予測ensembleには単一の重み座標がないため、FGEの平均位置をモデルの座標として描かない。

SWA では

- raw trajectory
- running average point

を同時に表示する。

特に、running average point が valley のどこへ移動するかを見たい。

---

## 7.5 Connectivity evaluation

SWAとFGEが共有する5つの採取checkpointの異なる点のpairに対してlinear interpolationを評価する。同じcheckpointをSWA用とFGE用として二重に列挙しない。

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
- FGE weight average（同じ5点・等重みならSWAと同一。独立手法ではなく一致確認として扱う）

評価:

- validation loss / accuracy
- test loss / accuracy

FGEではprediction ensembleとweight averageを必ず分ける。主要な平均方法の比較は、同じ5点を使うSWAとFGE prediction ensembleの間で行う。

---

# 8. Phase 3: Same SSL Checkpoint → Multiple Fine-tuning Runs

ここから初めてSSL事前学習済み重みを使う。共通モデルはConvNeXt V2-Tiny、初期checkpointは教師ありfine-tuning前の`convnextv2_tiny.fcmae`を候補とする。これは`fcmae_ft_in1k`等の教師ありfine-tuning済み重みとは異なる。[モデルカード](https://huggingface.co/timm/convnextv2_tiny.fcmae)

同一FCMAE backboneと同一の10-class headを保存し、全候補runがその完全な初期stateからCIFAR-10へfine-tuningする。Phase 1・2のスクラッチ学習済み重みはSoup候補として再利用しない。各runはAdamW・固定LR 1e-4・warmupなし・LR decayなしを基本とする。初期値・run・projectionの識別をスクラッチ系列と分離する。取得・変換・head初期化・epoch・weight decay等の残る条件はU-01で確定し、Phase 0・1のCLIへ先行実装しない。

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

- weight decay等、LR以外の候補差: U-01で確定。基本の全runは固定LR 1e-4を共通にし、LR値が異なる追加runは別途判断する

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

## 9.2 Projection

### Phase 1

比較対象runをまとめた中心化PCAを使い、共通平均を原点、上位2主成分を直交単位基底とする。

対象は、最小版ではseed 0のB64/B256/B1024の3 runs、拡張版ではseed 0・1・2の全9 runsとする。各runのepoch 0と毎epoch終了時のcheckpointを使い、同じepoch範囲・間隔で揃える。共通の `theta_0`も各runの起点として含め、各run・epochの組を等重みで扱う。checkpointを100件以下へ間引くことはしない。

対象のparameter vectorを \(w_k = \mathrm{vec}(\theta_k)\)、対象時点数を \(N\) とすると、

\[
\mu = \frac{1}{N}\sum_{k=1}^{N} w_k,\qquad X_{k,:} = (w_k-\mu)^\top
\]

を用いてPCAの上位2方向 \(v_1,v_2\) を求める。\(V=[v_1,v_2]\)、\(V^\top V=I_2\) とし、全runに同じ原点・基底を使う。

各checkpointの座標と、2D平面からの射影残差は

\[
z_k = V^\top(w_k-\mu),\qquad
e_k = \|w_k-\mu-Vz_k\|_2
\]

とする。2軸それぞれの寄与率、全checkpointの座標・射影残差、原点・基底、対象runとepochの対応を保存する。残差は重み空間で失われた成分の大きさであり、損失の誤差そのものではない。

計算はcheckpointの順次読み込み・分割処理で行い、全重みのRAM/GPU常駐を避ける。seed 1・2の追加時は全9 runsでPCAを再計算し、別の識別子で保存する。3 runs版と9 runs版の座標を混在させず、動画内では座標系を固定する。

ディスク上のFP32重みを16,384 parameterずつ読み、FP64で中心化Gram行列を累積・固有値分解する。100epoch・全9runでは909時点を扱う。平均・基底・座標・残差もFP64で保存し、記録点の間引きやIncrementalPCAは使わない。計算手順・数値検証・rank不足時の扱いは実装仕様22節を参照する。

**参考と今回の設計の区別:** 軌跡からPCAの上位2方向を選ぶ方法と寄与率の表示は、[Li et al.第7.2節・Figure 9（R05）](https://arxiv.org/pdf/1712.09913v3#page=10)を基礎とする。論文は最終checkpointとの差を並べてPCAを計算し、[著者実装](https://github.com/tomgoldstein/loss-landscape/blob/master/projection.py)は最終モデルを基準に軌跡を射影する。複数runをまとめること、共通平均を原点にすること、射影残差の保存、メモリ制約に合わせた分割処理は今回の比較向けの設計であり、原論文の実験をそのまま再現するものではない。

従来案の `theta - theta_0`を入力にしても、その後に同じ対象の平均を引く中心化PCAなら方向は同じになる。ただし、背景損失を評価する平面の原点は別に指定する必要がある。Phase 1ではその原点も \(\mu\) に揃え、`theta_0`や特定runの最終checkpointには置かない。

### Phase 2

branch point以降のNormalとSWA/FGE共通のraw checkpointを使う案。共有checkpointを別手法名で二重登録せず、SWA平均点をPCAのfit対象へ含めるかなどの詳細はA-01で確定する。

### Phase 3

複数 fine-tuning run の final solution と trajectory checkpoints を使って PCA。

---

## 9.3 Loss surface evaluation

平面上の

\[
\theta(a,b)=\theta_{ref}+a v_1+b v_2
\]

を評価する。**Phase 1の原点はD-03で確定した共通平均 \(\mu\)** とし、parameter vector上の平面 \(\mu+a v_1+b v_2\)を使う。

train由来の背景を主表示とし、validation由来の背景も同じ座標系の別GIFとして作成する。以下の評価条件はPhase 1と、その基盤を確認するPhase 0に適用する。後段の評価条件はDraftとする。

### Landscape subset

- 学習用とvalidation用のsplitを分けた後、それぞれから各クラス100件、計1,000件の固定subsetを抽出する。split間でデータを混ぜず、testは使用しない。
- train subsetを主背景、validation subsetを補助背景に使う。subsetの元split・index・抽出条件を保存し、全run・epochで共有する。3 runs版から9 runs版へPCAを作り直す場合も、subsetは変更しない。
- 背景損失は、それぞれのsubsetで計算した画像ごとのCross Entropyの平均とする。

### Preprocessing and precision

- モデル指定の評価用前処理を、224×224入力で固定する。Resize / Crop / interpolation / normalizationなど、解決された具体的な設定を保存する。
- 背景と実checkpointの評価に同じ前処理を使い、学習用のランダムaugmentationは使用しない。
- 評価はFP32で行い、AMP・TF32を無効にする。学習時のbf16とは分けて記録する。
- モデルを評価モードにし、勾配を計算しない。独立した評価loaderを使い、評価前後で学習の乱数状態・model modeを退避復元する。D-05の詳細手順は実装仕様22節に従う。

### Grid

- Phase 1はtrain・validationともに21×21点とする。31×31への拡大はこの確定範囲に含めない。
- 同一比較ではPCA artifactとx/yの格子座標を共有し、背景データだけを変えて損失を評価する。
- 各背景441点、計882点を1座標系につき事前計算する。これは格子点数であり、実行時間の実測値ではない。
- 対になるtrain版・validation版は、両方の格子の損失範囲を覆う共通の色尺度を使う。動画内で軸・色尺度を変更しない。
- 格子範囲は全軌跡の各軸min/maxに、その幅の10%ずつ余白を加える。幅0の場合は中心±1e-6。21点の等間隔とし、両背景の全lossを覆う20段階の線形色尺度を使う。再計算は両背景を同じ新projection_idへ保存し、既存artifactは上書きしない。

### Actual checkpoint metrics and interpretation

- epoch 0と毎epoch終了時の実checkpointを、背景と同じtrain subsetおよびvalidation全体で評価し、loss / accuracyを保存する。
- 実測値は元の高次元のcheckpointで計算する。平面への射影・復元後のモデルや、格子上の損失を代用しない。
- validation背景は固定1,000件、実checkpointのvalidation指標はvalidation全体であり、評価対象の範囲を表示・metadataで区別する。
- 学習中のtrain lossは、更新中のモデルと学習用augmentationによる記録である。固定前処理のtrain-subset評価や背景損失とは別の指標として扱う。
- train背景も実際の学習目的関数そのものではなく、validation背景も実checkpointの損失そのものではない。背景は選んだ平面上での評価であり、射影残差がある実checkpointの損失と一般には一致しない。

train背景は訓練データ側の地形を観察するための主表示、validation背景は学習に使わないデータ側の地形を観察するための補助表示とする。背景の優劣を一般化するための比較ではない。

---

# 10. Animation Requirements

動画は本実験の主成果物なので、後付けにしない。

## Required outputs

すべてのアニメーションはGIF形式とし、各ファイルを3 MB以下にする。

### Phase 1

| 比較 | train背景（主表示） | validation背景（補助版） |
| --- | --- | --- |
| seed 0 | `phase1_seed0_batch_compare.gif` | `phase1_seed0_batch_compare_val.gif` |
| seed 1 | `phase1_seed1_batch_compare.gif` | `phase1_seed1_batch_compare_val.gif` |
| seed 2 | `phase1_seed2_batch_compare.gif` | `phase1_seed2_batch_compare_val.gif` |
| 全9 runs | `phase1_all_runs_summary.gif` | `phase1_all_runs_summary_val.gif` |

最小版ではseed 0の2ファイル、全9 runsが揃った版では上記8ファイルを作成する。各ファイルにそれぞれ3 MB以下の制約を適用する。

### Phase 2

- `phase2_normal_swa_fge.gif`
- `phase2_swa_running_average.gif`

### Phase 3

- `phase3_multi_run_trajectories.gif`
- `phase3_soup_formation.gif`

## Frame design

1つの動画内で座標系を途中変更しない。

表示内容:

- loss contour
- past trajectory
- current point
- epoch（Phase 1の比較時間軸）とoptimizer step
- current LR
- val loss
- val accuracy

Phase 1ではtrain-subsetの実測loss / accuracyも表示し、validation実測値は全validationを対象とする旨を明記する。背景のラベルにはtrain subset / validation subset、件数、平面上の損失であることを示す。背景を切り替えるためにPCA・軌跡・実checkpoint指標を計算し直さず、保存済みの結果を共有する。

毎epochを1frame、5fps、最終frameに追加1000msの保持時間とする。寄与率と現在点の射影残差も表示する。GIFは960×640・128色から始め、必要なら64色、幅800、640へ順次調整する。3 MBは3,000,000 bytesとし、全epochと必須指標を保持したまま上限を満たせなければ未達として報告する。

Phase 2 では必要に応じて SWA running average point も表示。

---

# 11. Interpretation Rules

## Allowed claims

- “B64 showed a broader trajectory in this projection.”
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

- 最小版はseed 0の3 runs・2GIF、Phase 1完了時は全9 runs・8GIFとその全記録が揃う
- 共通PCA・両背景・実checkpoint指標を区別でき、寄与率・射影残差が記録される
- 保存済み成果物だけから、各3 MB以下のGIFを再描画できる
- 再開・保存復元が確認され、設定と再現手順が揃う

精度改善、batch-size間の差、高いPCA寄与率は完了の必須条件にしない。寄与率が低い場合は解釈の制限を表示する。checkpoint interpolationはPhase 2の準備として実装・検証し、Phase 1の終了条件には含めない。

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
