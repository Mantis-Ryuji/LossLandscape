# 参考文献・参考実装

> **参照用・設計は検討中（Draft）** — ユーザー提供の18件と、設計検討で補足した一次資料をまとめています。リンクの掲載は、手法・実装・依存ライブラリの採用を確定するものではありません。

確認日: 2026-08-31。論文・公式実装を手法の確認に、解説記事を論点の整理に、可視化ツールを表示方法の参考に使います。GitHubのコードは参照時点の内容であり、利用する場合のcommit固定や動作検証は別途必要です。

[ドキュメント案内に戻る](README.md)

## ユーザー提供資料

番号は提供されたリンクの順序に対応します。

### 可視化・最適化軌跡

| ID | 資料 | 種別 | 参照する内容 |
| --- | --- | --- | --- |
| R01 | [logancyang/loss-landscape-anim](https://github.com/logancyang/loss-landscape-anim) | 参考実装 | 実モデルの学習軌跡を2D損失断面へ射影するアニメーション。射影した軌跡と背景損失の解釈上の注意。 |
| R02 | [lilipads/gradient_descent_viz](https://github.com/lilipads/gradient_descent_viz) | 可視化アプリ | optimizerの比較、軌跡、更新の説明、再生操作の参考。 |
| R03 | [mysimulator: Gradient Descent](https://www.mysimulator.uk/ai-ml/gradient-descent/) | 可視化サイト | ユーザー指定URL。指定ページは取得できなかったため、補足資料S06の同サイト解説を参照。操作自体は未検証。 |
| R04 | [Gradient Lab](https://gradientlab.ai/) | 可視化サイト | 同じ開始点からのoptimizer比較、1D/2D/3D表示という紹介内容を参照。JavaScriptによる操作自体は未検証。 |
| R05 | [Visualizing the Loss Landscape of Neural Nets — Li et al.](https://arxiv.org/pdf/1712.09913v3) | 原論文 | PCAによる軌跡の可視化、filter normalization、断面の解釈。指定されたv3を参照。 |

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
| R13 | [Averaging Weights Leads to Wider Optima and Better Generalization — Izmailov et al.](https://arxiv.org/abs/1803.05407) | 原論文 | 一定・周期的LRで得た重みの等重み平均と、その評価。 |
| R14 | [timgaripov/swa](https://github.com/timgaripov/swa) | 著者実装 | SWAの学習・平均・評価処理。 |
| R15 | [Stochastic Weight Averaging in PyTorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/) | 著者による公式解説 | SWAの使い方とoptimizerへの適用。記事中の旧torchcontrib APIをそのまま依存先にすることは想定しない。 |

### Model Soup

| ID | 資料 | 種別 | 参照する内容 |
| --- | --- | --- | --- |
| R16 | [Model soups 論文解説：Transformer 時代の重み平均](https://zenn.dev/mantis_ryuji/articles/4ebef9541758c1) | 解説記事 | 共通初期値、uniform / greedy soup、validationによる選択。 |
| R17 | [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time — Wortsman et al.](https://arxiv.org/abs/2203.05482) | 原論文 | fine-tunedモデルの重み平均、候補選択、logit ensembleとの比較。 |
| R18 | [mlfoundations/model-soups](https://github.com/mlfoundations/model-soups) | 著者実装 | 候補の順位づけ、等重み平均、Greedy Soupの採用判定。 |

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
| S08 | [timm/vit_small_patch16_224.dino — Model card](https://huggingface.co/timm/vit_small_patch16_224.dino) | DINOモデル候補の224×224入力、約21.7M parameters、事前学習情報、モデル別前処理。 |

## 設計検討で区別すること

- **元画像とモデル入力**: CIFAR-10の元画像は32×32だが、現在の草案は224×224へresizeする。画素数は49倍になるが、学習時のVRAM全体が単純に49倍になるという意味ではない。実機でのB256確認は [実装仕様案の環境確認記録](IMPLEMENTATION_SPEC.md#21-environment-check-2026-08-31) を参照。
- **射影と実測**: PCA平面の背景損失と、平面外にも成分を持つ実モデルの損失を区別する。
- **直線と曲線**: 線形補間に障壁があっても、低損失の曲線経路が存在しないとは言えない。
- **重み平均と予測平均**: SWA / Soupの重み平均、FGEの確率平均、Model Soup論文のlogit ensembleを区別する。
- **Greedy Soupの同点**: 原論文には「精度が低下しない」という説明がある一方、公開main.pyは厳密な改善（`>`）で採用している。設計提案では後者を推奨しているが、確定済みの仕様とは扱わない。
- **紹介と再現実験**: 参考ツールの表示や原論文の着想を使うことと、原論文の実験条件・結果を再現することは区別する。

この一覧への追記によって、実験計画や設定例の条件は変更していません。
