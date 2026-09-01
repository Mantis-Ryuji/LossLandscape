# 参考文献・参考実装

> **参照用・採用範囲は個別に明記** — ユーザー提供資料と、設計検討で補足した一次資料をまとめています。リンクの掲載だけで、手法・実装・依存ライブラリの採用を確定するものではありません。

**D-03の確定範囲:** R05の第7.2節を基礎に、軌跡からPCAの上位2方向を選ぶ方法を採用します。複数runの共通化・平均原点・射影残差の保存・分割処理は今回の設計判断です。詳しくは[実験計画の射影仕様](EXPERIMENT_PLAN.md#92-projection)を参照してください。

論文・公式実装を手法の確認に、解説記事を論点の整理に、可視化ツールを表示方法の参考に使います。GitHubのコードは参照時点の内容であり、利用する場合のcommit固定や動作検証は別途必要です。

[ドキュメント案内に戻る](README.md)

## ユーザー提供資料

番号は提供されたリンクの順序に対応します。補足資料には別のIDを付けます。

### 可視化・最適化軌跡

| ID | 資料 | 種別 | 参照する内容 |
| --- | --- | --- | --- |
| R01 | [logancyang/loss-landscape-anim](https://github.com/logancyang/loss-landscape-anim) | 参考実装 | 実モデルの学習軌跡を2D損失断面へ射影するアニメーション。射影した軌跡と背景損失の解釈上の注意。 |
| R02 | [lilipads/gradient_descent_viz](https://github.com/lilipads/gradient_descent_viz) | 可視化アプリ | optimizerの比較、軌跡、更新の説明、再生操作の参考。 |
| R03 | [mysimulator: Gradient Descent](https://www.mysimulator.uk/ai-ml/gradient-descent/) | 可視化サイト | ユーザー指定URL。指定ページは取得できなかったため、補足資料S06の同サイト解説を参照。操作自体は未検証。 |
| R04 | [Gradient Lab](https://gradientlab.ai/) | 可視化サイト | 同じ開始点からのoptimizer比較、1D/2D/3D表示という紹介内容を参照。JavaScriptによる操作自体は未検証。 |
| R05 | [Visualizing the Loss Landscape of Neural Nets — Li et al.](https://arxiv.org/pdf/1712.09913v3) | 原論文 | 第7.2節・Figure 9のPCAによる軌跡可視化と寄与率表示をD-03の基礎とする。filter normalizationと断面の解釈も参照するが、filter-normalized方向は初版の対象外。指定されたv3を参照。 |

### バッチサイズ・Sharp / Flat Minima

| ID | 資料 | 種別 | 参照する内容 |
| --- | --- | --- | --- |
| R06 | [大バッチ学習はなぜ汎化しにくいのか：Sharp Minima 論文から見る最適化と汎化の関係](https://zenn.dev/mantis_ryuji/articles/3a6c3b210ffdeb) | 解説記事 | R07の実験とsharp / flat minimaの論点整理。 |
| R07 | [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima — Keskar et al.](https://arxiv.org/pdf/1609.04836) | 原論文 | バッチサイズ、最終解、汎化の関係。原論文と今回のモデル・バッチ条件の違いにも注意。 |
| R08 | [Sharp Minima は本当に汎化を説明するのか：大バッチ学習と再パラメータ化からの反論](https://zenn.dev/mantis_ryuji/articles/0f7b2bb4dd8696) | 解説記事 | sharpnessの座標依存性と再パラメータ化による反論。対応する原論文はS01。 |

### Mode Connectivity・Fast Geometric Ensembling

| ID | 資料 | 種別 | 参照する内容 |
| --- | --- | --- | --- |
| R09 | [Fast Geometric Ensembling 論文解説：低損失経路から高速アンサンブルへ](https://zenn.dev/mantis_ryuji/articles/27dbe201808952) | 解説記事 | 低損失経路とFGEの関係を整理。 |
| R10 | [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs — Garipov et al.](https://arxiv.org/abs/1802.10026) | 原論文 | 曲線経路による接続と、短い周期の学習率を使うFGE。 |
| R11 | [timgaripov/dnn-mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity) | 著者実装 | FGEのLR schedule、snapshotの採取時点、予測ensembleの定義。 |

### Stochastic Weight Averaging

| ID | 資料 | 種別 | 参照する内容 |
| --- | --- | --- | --- |
| R12 | [Stochastic Weight Averaging (SWA) 論文解説：SGD 軌道の平均化と損失地形](https://zenn.dev/mantis_ryuji/articles/93caaeefe94919) | 解説記事 | 軌道上の重み平均、平均モデル、BatchNorm統計量の扱い。 |
| R13 | [Averaging Weights Leads to Wider Optima and Better Generalization — Izmailov et al.](https://arxiv.org/abs/1803.05407) | 原論文 | 固定・周期的LRで得た重みの等重み平均。§3.2・§4.1・Appendix A.1を確認し、固定LR版と周期LR版を区別する。 |
| R14 | [timgaripov/swa](https://github.com/timgaripov/swa) | 著者実装 | SWAの学習・平均・評価処理。 |
| R15 | [Stochastic Weight Averaging in PyTorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/) | 著者による公式解説 | SWAの使い方とoptimizerへの適用。記事中の旧torchcontrib APIをそのまま依存先にすることは想定しない。 |

### Model Soup

| ID | 資料 | 種別 | 参照する内容 |
| --- | --- | --- | --- |
| R16 | [Model soups 論文解説：Transformer 時代の重み平均](https://zenn.dev/mantis_ryuji/articles/4ebef9541758c1) | 解説記事 | 共通初期値、uniform / greedy soup、validationによる選択。 |
| R17 | [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time — Wortsman et al.](https://arxiv.org/abs/2203.05482) | 原論文 | fine-tunedモデルの重み平均、候補選択、logit ensembleとの比較。 |
| R18 | [mlfoundations/model-soups](https://github.com/mlfoundations/model-soups) | 著者実装 | 候補の順位づけ、等重み平均、Greedy Soupの採用判定。 |

### ConvNeXt V2・FCMAE

| ID | 資料 | 種別 | 参照する内容 |
| --- | --- | --- | --- |
| R19 | [ConvNeXt V2 論文解説：CNN のための Masked Autoencoder](https://zenn.dev/mantis_ryuji/articles/9628b8eef173d4) | 解説記事 | FCMAEとGRN、モデル構造と自己教師あり事前学習の関係を整理。技術的な採用判断はR20・S09と照合する。 |
| R20 | [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders — Woo et al.](https://arxiv.org/abs/2301.00808) | 原論文 | ConvNeXt V2の構造、GRN、FCMAEの設計。モデル構造の採用と事前学習済み重みの利用を区別する。 |

ConvNeXt V2公式repositoryは下表のS09に掲載する。

## 補足した一次資料・参照箇所

| ID | 資料 | 用途 |
| --- | --- | --- |
| S01 | [Sharp Minima Can Generalize For Deep Nets — Dinh et al.](https://arxiv.org/abs/1703.04933) | R08に対応する原論文。見かけのsharpnessと汎化を直結しないための根拠。 |
| S02 | [PyTorch 2.10: Weight Averaging (SWA and EMA)](https://docs.pytorch.org/docs/2.10/optim.html#weight-averaging-swa-and-ema) | 標準APIのAveragedModel、SWALR、update_bnを確認した版。プロジェクトの採用バージョンや実測環境のバージョンを意味しない。 |
| S03 | [FGE公式実装: fge.py](https://github.com/timgaripov/dnn-mode-connectivity/blob/master/fge.py) | 最低LRとなる周期中央でensembleへ追加する処理。 |
| S04 | [FGE公式実装: utils.py](https://github.com/timgaripov/dnn-mode-connectivity/blob/master/utils.py) | 三角形状のLR scheduleとsoftmax後の予測確率を確認。 |
| S05 | [Model Soup公式実装: main.py](https://github.com/mlfoundations/model-soups/blob/main/main.py) | Greedy Soupがvalidation accuracyの厳密な改善を条件に採用する処理。 |
| S06 | [mysimulator: Gradient Descent & Modern Optimisers](https://www.mysimulator.uk/articles/gradient-descent/) | R03が取得できなかった際の補助的な公式解説。 |
| S07 | [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html) | 元画像が32×32のカラー画像であること、件数、クラス構成。公式ページはcave.cs.toronto.eduへリダイレクト。 |
| S08 | [Li et al.著者実装: projection.py](https://github.com/tomgoldstein/loss-landscape/blob/master/projection.py) | 軌跡行列からPCAの2方向を求める処理、寄与率の保存、最終モデルを基準とする射影を確認。D-03の共通平均原点・複数run共通化とは区別する。 |
| S09 | [ConvNeXt V2公式repository](https://github.com/facebookresearch/ConvNeXt-V2) | 全フェーズ共通のモデル構造。Tinyと、FCMAEのみ／教師ありfine-tuning後の重みの区別を確認。 |
| S10 | [timm/convnextv2_tiny.fcmae](https://huggingface.co/timm/convnextv2_tiny.fcmae) | Phase 3のModel Soup用候補。分類headを持たない純粋な自己教師ありFCMAE checkpoint。Phase 0・1・2では使わない。 |
| S11 | [timm/convnextv2_tiny.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in1k) | FCMAE後にImageNet-1kで教師ありfine-tuningした重み。S10と取り違えないための参照で、純SSL初期値候補には採用しない。 |
| S12 | [ConvNeXt V2公式main_finetune.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/main_finetune.py) | 公式学習コードの既定optimizerはAdamW。CIFAR-10スクラッチ用のLR・epochが検証された資料ではない。 |
| S13 | [ConvNeXt V2公式TRAINING.md](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md) | FCMAEのImageNet事前学習・fine-tuning手順。Tinyの300epoch例も事前学習checkpointを使うため、CIFAR-10スクラッチレシピの検証根拠にはしない。 |

## 設計検討で区別すること

D-05のAPI確認: [PyTorch 2.6 Reproducibility](https://docs.pytorch.org/docs/2.6/notes/randomness.html)、[PyTorch 2.6 DataLoader](https://docs.pytorch.org/docs/2.6/data.html)、[NumPy 2.1 eigh](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eigh.html)。分割Gram行列方式・FP64・保存完了手順は今回の設計判断であり、これらの資料が実験結果や処理速度を保証するものではない。

- **初期化と実験の境界**: Phase 0・1・2はConvNeXt V2-Tinyをスクラッチ学習し、最後のModel SoupだけS10の共通SSL初期値からfine-tuningする。SWA/FGEやsharp minimaの着想を採用することと、著者実装のモデル・optimizer・条件を再現することは分ける。
- **Optimizer**: AdamWを使う。[SWA著者実装](https://github.com/timgaripov/swa/blob/master/train.py)と[mode-connectivity著者実装](https://github.com/timgaripov/dnn-mode-connectivity/blob/master/train.py)のmomentum SGDとは条件が異なる。重み平均・周期LRの着想をConvNeXt V2とAdamWへ適用する実験として扱い、著者実験の再現とは呼ばない。
- **固定LRの採用範囲**: Phase 0・1は100epoch・共通固定LR 1e-3、Model Soupは各runで固定LR 1e-4。どちらもwarmup・decayなし。R20・S12・S13がCIFAR-10での最適性や収束を保証しているわけではない。[実験計画4節](EXPERIMENT_PLAN.md#4-scratch-training-recipe)に採用理由と確認範囲を記録する。
- **SWA/FGEの原設定と共通比較条件**: 原論文の代表的なCIFAR設定と、本実験の80〜100epoch・共通4epoch三角周期・同じ5点での平均比較は[実験計画7.3節](EXPERIMENT_PLAN.md#73-branches)に分けて記録する。SWAの約75%開始、FGEの約80%開始という説明を区別し、原論文のSGD用LRをAdamWの確定値として流用しない。
- **元画像とモデル入力**: CIFAR-10の元画像は32×32、モデル入力は224×224。画素数は49倍だが、VRAMが単純に49倍になるという意味ではない。実測条件は[環境とPhase 0検証](IMPLEMENTATION_SPEC.md#21-environment-and-phase-0-verification)に記録する。
- **射影と実測**: PCA平面の背景損失と、平面外にも成分を持つ実モデルの損失を区別する。
- **直線と曲線**: 線形補間に障壁があっても、低損失の曲線経路が存在しないとは言えない。
- **重み平均と予測平均**: SWA / Soupの重み平均、FGEの確率平均、Model Soup論文のlogit ensembleを区別する。
- **Greedy Soupの同点**: 原論文には「精度が低下しない」という説明がある一方、公開main.pyは厳密な改善（`>`）で採用している。設計提案では後者を推奨しているが、確定済みの仕様とは扱わない。
- **紹介と再現実験**: 参考ツールの表示や原論文の着想を使うことと、原論文の実験条件・結果を再現することは区別する。

参考文献の掲載だけで実験条件を変更することはしません。D-03として承認された範囲は、実験計画・実装仕様・設定例に明記しています。
