# Loss Landscape Dynamics on CIFAR-10

> **2026-09-01 検証状況更新** — V-04/V-05は完了しました。細線版の追加6件・全86件のCPU testがユーザー実行で成功し、Phase 0のtrain/validation GIFペアも完成。manifest、各3 MB以下、全epoch frame、固定軸・共通色尺度、保存済み成果物だけからの再描画、3本の識別性を確認しました。次はM-01でseed 0のB64/B256/B1024を各100epoch実行します。

作業順は[TODO](docs/TODO.md)、研究条件は[実験計画](docs/EXPERIMENT_PLAN.md)、保存・再開・射影の契約は[実装仕様](docs/IMPLEMENTATION_SPEC.md)を参照してください。[資料一覧](docs/README.md)と[参考文献](docs/REFERENCES.md)もあります。

## 目的と実験の境界

最適化軌跡が損失地形上をどう進むかを、アニメーションを中心に観察します。事前学習済み初期値の近傍に限らない学習過程をPhase 1で観察し、Phase 2でLR操作・SWA・FGE、Phase 3で共通のSSL初期値からのModel Soupを扱います。

| Phase | 初期状態と学習 | 主な比較 |
| --- | --- | --- |
| 0 | ConvNeXt V2-Tinyの全体をスクラッチ初期化 | 短い学習・評価・保存・再開の確認 |
| 1 | 同一のスクラッチ初期checkpointを全runで共有 | B64/B256/B1024、seed 0から開始してseed 1・2へ拡張 |
| 2 | Phase 1のスクラッチ学習途中から分岐 | Normal / SWA / FGE。外部の事前学習重みは読み込まない |
| 3 | 同一のConvNeXt V2-Tiny FCMAE checkpoint＋同一の10-class headから各runを開始 | 独立したfine-tuning群とUniform/Greedy Soup |

Phase 3にPhase 1・2の学習済みモデルを流用しません。全フェーズでモデル構造は共通でも、スクラッチ初期値とSSL初期値は別のものです。Phase 2・3の詳細条件・実装は、その段階で確定します。

2Dの近さだけでsame basin、sharp/flat minima、線形接続性を断定しません。寄与率・射影残差・実checkpointの指標を併記し、線形barrierや平均モデルの評価は別に行います。参考論文の着想を使う実験であり、異なるモデルやoptimizerの結果を原論文の再現と呼びません。

## 維持する共通条件

- CIFAR-10公式train 50,000件を、共有train 45,000件・validation 5,000件へ分割。公式testは最終評価以外に使用しません。
- 入力224×224、全parameterを学習。既存のcrop/flipと決定的な評価前処理を維持します。正規化・resize/cropの実値はConvNeXt V2のtimm設定から記録します。
- Phase 1は実効B64/B256/B1024、全条件microbatch 64・accum 1/4/16、seed 0/1/2。まずseed 0の3runを完成させます。run seedは学習中の乱数用で、初期重みは変えません。
- epoch 0と毎epoch終了時を記録。解析用はFP32、再開用は重み・optimizer・scheduler・乱数・DataLoader状態を保存します。
- 共通PCA、train背景の主GIFとvalidation背景の補助GIF。背景は各固定1,000件の21×21格子、実checkpointは同じtrain subsetとvalidation全体をFP32で評価します。
- GIFは各3,000,000 bytes以下、epochを間引きません。保存容量上限や自動削除は設けず、RAM・VRAM制約は守ります。

モデルを変更したため、DINOで測ったVRAM・時間・精度をConvNeXt V2の見積もりや検証結果には使いません。

## スクラッチ初期化を明示する設定

設定schemaはv3です。初期モデルの保存schemaはv2のままで、既存の共通初期重みを再利用します。

```yaml
model:
  name: convnextv2_tiny
  initialization: scratch
  num_classes: 10
  image_size: 224
  full_finetune: true
  init_seed: 0
```

`full_finetune: true`は既存API名を維持しており、「全parameterを学習する」という意味です。事前学習重みの利用を意味しません。

初回はCPU上でinit seed 0を設定し、`timm.create_model("convnextv2_tiny", pretrained=False, num_classes=10)`を一度呼びます。backbone・headともモデル既定の初期化を使い、headだけを後から再初期化しません。固定値で始まる正規化層等も含め、モデル全体の初期stateを保存します。

保存先は`artifacts/init/convnextv2_tiny_scratch/theta_0.pt`と同名JSONです。初期化方式・seed・`pretrained: false`を記録し、`pretrained_reference`はnullにします。timmが持つ前処理用の`pretrained_cfg`を、重みを読み込んだ証拠として記録しません。各runはこの同じ全stateを復元します。Phase 0・1で`initialization: pretrained`、FCMAE付きモデル名、旧DINOモデル名を指定すると拒否します。

Model Soup用の候補は、教師ありfine-tuning前の[`convnextv2_tiny.fcmae`](https://huggingface.co/timm/convnextv2_tiny.fcmae)です。`fcmae_ft_in1k`等とは区別します。Phase 3の初期化CLIや学習条件はまだ実装していません。

## スクラッチ用の学習レシピ

ユーザー指定により、Phase 0・1はAdamW・全期間100epoch・固定LR 1e-3を採用します。warmup・LR decayは行いません。Model Soupは各fine-tuning runで固定LR 1e-4とし、後段の設計条件として記録しています。

| 項目 | Phase 0・1の共通設定 |
| --- | --- |
| optimizer | AdamW、betas=(0.9, 0.999)、eps=1e-8 |
| 全学習期間 | 100epoch |
| learning rate | 1e-3、B64/B256/B1024で共通、全更新で固定 |
| scheduler / warmup | constant / 0。初回から最終更新までLR 1e-3を維持 |
| weight decay | 0.05、全学習parameterへ一様に適用 |
| 記録 | epoch 0〜100の101時点/run、間引きなし |

`training.batch_size`およびCLIの`--batch-size`は実効バッチサイズです。`training.microbatch_size: 64`を固定し、蓄積回数はその比から導出します。独立したaccum設定は持たず、B256を指定すると自動的に4回になります。

| 実効batch | microbatch | accum | AdamW更新/epoch |
| --- | --- | --- | --- |
| 64 | 64 | 1 | 704 |
| 256 | 64 | 4 | 176 |
| 1024 | 64 | 16 | 44 |

各microbatchの勾配を画像数で平均してから1回だけAdamWを更新します。epoch末の実効バッチ8/200/968件も実枚数で平均し、全45,000件を使用します。stepとgradient normはAdamW更新単位、オンラインlossは蓄積用の係数を掛ける前の画像数平均です。評価batchは64・FP32のままです。実効batch・microbatch・蓄積回数は設定確認出力と保存contract/environmentで追跡できます。

LRを時間変化させず、軌跡を動かす外部要因を1つ減らすための設定です。AdamWの適応的な更新や、バッチ間の更新回数の差は残るため、勾配ノイズだけを切り分けた実験とは呼びません。CIFAR-10での最適性・収束は未検証です。[採用理由と参考資料との違い](docs/EXPERIMENT_PLAN.md#4-scratch-training-recipe)を記録し、augmentation・評価・初期化・保存条件は維持します。

記録数の増加に合わせ、PCAの読込ブロック幅は65,536から16,384 parameterへ縮小します。RAM使用量を抑えるための処理単位の変更で、対象checkpoint・PCAの定義・FP64精度は変えません。PCA実装と資源の実測はこれからです。

Phase 0は100epochの予定を保ち、**最初から固定LR 1e-3で5epoch学習して停止**します。100epochの安定性・収束は、この短い確認だけでは保証しません。Phase 2のNormalも固定LR。SWA/FGEはepoch 80終了時から100まで、共通の1e-3 → 1e-5 → 1e-3、4epoch周期を5回とし、同じ5点で重み平均と予測確率平均を比較します。最低LRのepoch 82・86・90・94・98終了時を毎epochの記録から選び、半epochでの追加保存は行いません。詳細は[原論文との対応と共通条件](docs/EXPERIMENT_PLAN.md#73-branches)を参照します。Phase 2は未実装です。

## 現在の実装

- `config.py`：階層YAML・型検証・設定保存。v3で実効batchとmicrobatchを分離し、旧設定v1/v2は拒否。
- `data.py` / `seeds.py`：共有split/subset、前処理、DataLoaderと乱数の保存・復元。
- `models.py`：共通のスクラッチ初期stateを作成し、hash・layout・runtime・初期化方式を検証して復元。
- `train.py` / `evaluate.py`：AdamW、固定LR、勾配蓄積、FP32評価、再開。端数の画像数平均と更新単位のstepを維持。設定の100epochとPhase 0の5epoch停止を分離し、最終更新でもLRを0にしません。
- `checkpoints.py` / `logging_utils.py`：毎epoch保存、完了manifest、分岐の親子関係、JSON/CSV。
- `loss_surface.py`：完成PCAの厳密な読込、共通格子、train/validation固定subsetの同時評価、共通色尺度、immutable surface artifact。
- `animation.py`：完成PCA・両損失平面・実checkpoint指標の厳密な読込、固定表示、GIF圧縮fallback、immutable animation artifact。モデル・dataset・checkpointは読み込みません。
- `scripts/`：設定確認、split準備、初期値作成・確認、学習、共通PCA、損失平面、GIF生成のCLI。
- `tests/`：小さなCPU fixtureによる検証。細線版を含むV-04の6件と全86件がユーザー実行で成功しました。Agentは実行していません。
- Phase 0の共通PCA、両損失平面、細線版GIFペアまで実データで検証済みです。次はPhase 1のseed 0・3run、SWA/FGE/Soupの実装はその後です。

## 今回の検証コマンド

実行ディレクトリはリポジトリルート。ユーザーのターミナルは`conda activate torch_env`済みなので、`python`を使います。[AGENTS.md](AGENTS.md)に従い、Agentはプログラム・テスト・学習を実行しません。依存の追加・更新も行っていません。

以下の設定確認4件と64件のCPUテストは、2026-09-01にユーザー実行で成功済みです。再現用のコマンドとして残します。設定確認は成果物を作りません。S-01のsplitと共通初期重みは作り直し不要です。

```powershell
python -B scripts\check_config.py --config configs\phase0.yaml
python -B scripts\check_config.py --config configs\phase1.yaml
python -B scripts\check_config.py --config configs\phase1.yaml --batch-size 256
python -B scripts\check_config.py --config configs\phase1.yaml --batch-size 1024
python -B -m unittest discover -s tests -v
```

各設定の`configuration_valid`と64件の`OK`（11.626秒）を受領しました。全条件で`microbatch_size: 64`、実効B64/B256/B1024の`accumulation_steps: 1/4/16`・`optimizer_steps_per_epoch: 704/176/44`が一致。両phaseの予定100epoch、Phase 0の5epoch終了・6時点、Phase 1の100epoch終了・101時点も確認できています。有効設定のhashは[TODOのI-06](docs/TODO.md#2-学習保存の基盤を実装)へ記録しました。設定・CPUテストの成功はGPUの成立性を保証しません。

変更前の記録として、100epoch・固定LR・scheduler状態schema v2への移行後に、ユーザーから両設定の`configuration_valid`と58件の`OK`（11.231秒）を受領しました。旧設定のSHA-256は[TODOのI-05](docs/TODO.md#2-学習保存の基盤を実装)に残しています。今回の設定schema v3・勾配蓄積の検証結果とは区別します。

## S-01: 実データと共通初期重みの準備

CIFAR-10と共有splitは取得・作成済みで保持しています。2026-09-01、ユーザー実行でsplitの再検証と、新しい共通スクラッチ初期値の作成・再読込が成功しました。再ダウンロード・初期値の作り直しは不要です。

以下は完了した手順の記録です。再実行時は既存成果物を検証し、上書きしません。エラーが出たら止め、不完全な成果物を削除しないでください。

```powershell
python -B scripts\prepare_splits.py --config configs\phase0.yaml
python -B scripts\create_init_checkpoint.py --config configs\phase0.yaml
python -B scripts\create_init_checkpoint.py --config configs\phase0.yaml --verify-only
```

splitは`split_verified`、初期値は`initial_checkpoint_created`、再読込は`initial_checkpoint_verified`が期待値です。初期値の作成・再読込とも`initialization.mode: scratch`・`pretrained: false`・`pretrained_fetch_requested: false`を確認します。作成時と再読込時のSHA-256は一致する必要があります。GPU・学習・重みダウンロードは行いません。

今回、この3件の成功を受領しました。初期値は27,874,186 parameter・CPU FP32・init seed 0で、作成と再読込のSHA-256が一致しています。完全なhashと前処理・runtimeは[TODOのS-01実行記録](docs/TODO.md#3-phase-0-実データによる短い確認)に記載しています。以後の初期値CLIは既存ファイルを検証するため、作成コマンドでも`initial_checkpoint_verified`になります。

共有splitの保存先は`artifacts/splits/cifar10_train_val_seed20260831_subset20260831.npz`と同名JSON。件数はtrain 45,000・validation 5,000・両subset各1,000です。

## 学習・保存・再開の確認

**変更前B64の5epoch・再開記録の照合、変更後の64件CPUテスト、B64/B256/B1024の5epoch GPU確認が完了しています。以下のB64手順は再現用であり、確認済み成果物へ再開・上書きしません。**

```powershell
python -B scripts\run_train.py --config configs\phase0.yaml --name phase0_accum
```

この確認runは`artifacts/runs/phase0_accum/b64_seed0/`の新規segmentにepoch 0〜5を保存します。毎epochの`analysis.pt`・`resume.pt`・`metrics.json`・`metadata.json`・`complete.json`と、segmentの`metrics.csv`・実行環境を記録します。同名runを繰り返し上書きすることはできません。

本番は`cuda:0`を使い、CPU fallbackやOOM時のbatch変更はしません。bf16・FP32・決定性設定を記録し、実際のメモリ・時間は新モデルで測定します。保存には同一directory内のhard linkを使うため、対応filesystemが必要です。

再開は同じ設定・コードで作った完了epoch directoryを明示し、別segmentへ保存します。旧B64成果物は保持し、今回のコードから旧runへ再開しません。

```powershell
python -B scripts\run_train.py --config configs\phase0.yaml --name phase0_accum --resume-from "artifacts\runs\phase0_accum\b64_seed0\segments\<新しいsegment_id>\epochs\epoch_0002"
```

設定、初期値、split、runtime/GPU、ソース識別が異なる場合は停止します。S-03では再開一致・保存記録・B256/B1024のGPUメモリを確認済みです。

再開用の`scheduler_state`はschema v2で、`scheduler: constant`を保存します。旧schemaやcosineの状態を固定LRとして読み替えません。今回、設定のみv3へ変更し、初期モデルv2・epoch成果物の外側のv1とCSV列は維持します。蓄積ありのGPU動作はB256/B1024の5epoch probeで確認済みです。

## B256 / B1024 GPU probe

B64の変更後runと再開確認は完了しました。Phase 0のseed 0・100epoch予定・5epoch停止を維持したまま、実効batchだけを上書きしてaccum 4/16を実データで確認します。両runは`phase0_accum_probe`系列の別directoryへ保存し、Phase 1の本runには数えません。

まず設定とCPUテストを確認します。期待値はB256がmicrobatch 64・accum 4・176更新/epoch、B1024がmicrobatch 64・accum 16・44更新/epochです。

```powershell
python -B scripts\check_config.py --config configs\phase0.yaml --name phase0_accum_probe --batch-size 256
python -B scripts\check_config.py --config configs\phase0.yaml --name phase0_accum_probe --batch-size 1024
python -B -m unittest discover -s tests -v
```

全件成功後、GPU jobを同時に走らせず、次の順で各5epochを実行します。

```powershell
python -B scripts\run_train.py --config configs\phase0.yaml --name phase0_accum_probe --batch-size 256
python -B scripts\run_train.py --config configs\phase0.yaml --name phase0_accum_probe --batch-size 1024
```

成果物は`artifacts/runs/phase0_accum_probe/b256_seed0/`と`b1024_seed0/`へ分離します。各epochでtrain 45,000件、最終epochでそれぞれ880/220更新、固定LR 1e-3、有限なloss/gradient、完了manifest、GPU peakを確認します。microbatchは両方64なのでOOM時に条件を自動変更しません。設定・ソース変更後のため、確認済み`phase0_accum/b64_seed0`へはさらに再開しません。

## V-02: Phase 0共通PCA

V-02までのCPU unit test 75件と、Phase 0の実成果物による18点の共通PCAはユーザー実行で成功済みです。

```powershell
python -B -m unittest discover -s tests -v
```

全件成功後、確認済みB64/B256/B1024をこの順で明示し、epoch 0〜5の計18点から共通PCAを作成します。GPUは使いません。各epochの完成manifestと4ファイルをhash検証するため、checkpointの読み出しに時間がかかります。

```powershell
python -B scripts\compute_projection.py `
  --config configs\phase0.yaml `
  --comparison-scope phase0_probe_seed0 `
  --segment "artifacts\runs\phase0_accum\b64_seed0\segments\20260831T192447802206Z_4bc8003101f24d2abd30ed7c8223f7e8" `
  --segment "artifacts\runs\phase0_accum_probe\b256_seed0\segments\20260901T040810290356Z_33ba760ec54141f9a53a8d22f44234c4" `
  --segment "artifacts\runs\phase0_accum_probe\b1024_seed0\segments\20260901T042536616130Z_55bb9c27811f40b3a51b9731a2f55313"
```

作業用の18×27,874,186 FP32行列を`artifacts/work/<projection_id>/weights.npy`へ保存し、自動削除しません。平均・PC1・PC2・座標・残差・固有値・寄与率とmetadata/完了manifestは`artifacts/projections/<projection_id>/`へ新規保存します。成功時は最後に`status: projection_ready`、`sample_count: 18`、`parameter_count: 27874186`、2以上の`effective_rank`、有限な`explained_variance_ratio`を表示します。再実行は新しいprojection IDになり、既存結果を上書きしません。

実行結果のprojection IDは`phase0_phase0_probe_seed0_20260901T085202502163Z_12676e9d04b9477d85b01242b139ae65`です。18点・27,874,186 parameter・有効rank 15、PC1/PC2の寄与率69.2020%/10.7069%で、完成manifest・配列shape・有限値・中心化・記録sizeの照合も完了しています。

## V-03: Phase 0のtrain/validation損失平面

V-03追加5件を含む計80件のCPU unit testは、ユーザー実行で成功済みです（11.256秒）。

```powershell
python -B -m unittest discover -s tests -v
```

全件成功後、V-02で完成したprojectionを明示して、同じ21×21格子上でtrain/validationの固定subset各1,000件をFP32評価します。各格子点でparameterを一度だけ割り当て、両背景を順に評価します。評価前処理済みbatchをCPUに保持するため約1.2 GBに加え、PCAのmean/PCとモデルの作業メモリを使用します。CUDA必須で、AMP・TF32・CPU fallbackは使いません。

```powershell
python -B scripts\compute_loss_surfaces.py `
  --config configs\phase0.yaml `
  --projection "artifacts\projections\phase0_phase0_probe_seed0_20260901T085202502163Z_12676e9d04b9477d85b01242b139ae65"
```

各背景441点、計882,000画像評価を行います。成果物は`artifacts/surfaces/<projection_id>/`に保存し、両loss/accuracy格子、共通20区間の色尺度、評価条件、subsetのhash、元checkpointのtrain-subset/full-validation実測値を含みます。実測validationは5,000件の元checkpoint評価で、validation背景の1,000件とは区別します。既存または未完成の同名surface directoryは上書き・補修しません。

ユーザー実行で成果物が完成し、21×21の両格子は全点有限、lossの共通範囲は1.06553490018845〜2.17837490463257、train accuracy範囲は0.217〜0.628、validation accuracy範囲は0.233〜0.637でした。x/y軸と21個の色境界は単調増加し、色範囲は両lossを包含。18件の実checkpoint記録はtrain subset各1,000件・full validation各5,000件で、完成manifestに列挙された9ファイルのsize/hashも一致しました。

## V-04: 保存済み成果物からのGIF生成

V-04では[animation.py](src/landscape_exp/animation.py)と[render_animation.py](scripts/render_animation.py)を追加しました。完成projectionとsurfaceの全file hashを配列読込前に検証し、epoch 0〜最終epochを1frameずつ描画します。train背景版とvalidation背景版は軌跡、epoch、軸、共通色尺度、解像度、palette数を共有し、モデル評価・dataset読込・checkpoint deserializationを行いません。実GIFで既定色が背景と混ざることを確認したため、batchはB64=`#D55E00`、B256=`#56B4E9`、B1024=`#CC79A7`で固定し、各軌跡へ黒の外縁と白の内縁を付けます。これら5色を固定GIF paletteへ予約し、量子化後も保持します。seedは線種と現在点markerで表します。

```powershell
python -B -m unittest discover -s tests -v
python -B scripts\render_animation.py `
  --config configs\phase0.yaml `
  --projection "artifacts\projections\phase0_phase0_probe_seed0_20260901T085202502163Z_12676e9d04b9477d85b01242b139ae65" `
  --name phase0_seed0_batch_compare
```

描画はペア共通で960×640・128色から始め、960×640・64色、幅800・64色、幅640・64色の順にfallbackします。5 fps、最終frameの追加1,000 ms、全epoch、必須指標を維持し、両方がそれぞれ3,000,000 bytes以下になった場合だけ`artifacts/animations/<animation_id>/`へGIF 2本・metadata・完了manifestを新規公開します。`animation_id`は出力名・UTC時刻・UUIDから作り、sourceの`projection_id`はmetadataに固定します。このため、視認性を変えて再描画しても既存GIFを上書きしません。全設定で上限を超えた場合は未達として停止し、frameを間引きません。細線版`phase0_seed0_batch_compare_20260901T144613293970Z_dd07ca7cd8074514b4dc72724e365fae`は960×640・128色、train 79,284 bytes・validation 78,699 bytesで完成し、ユーザーが線幅と3本の識別性を承認しました。

## 方針転換時の削除

2026-09-01、ユーザーの明示的な指示で、旧DINOの`artifacts/init/theta_0.pt`・`theta_0.json`と`artifacts/runs/phase0/b64_seed0/`を削除しました。これは今回だけの手動整理で、保存コードの自動削除方針は変更していません。CIFAR-10、共有split、ソースコード、環境は保持しています。

## Phase 1の完了条件

まずseed 0の3runについて、共通のスクラッチ初期値、全epochのcheckpoint・実測ログ、共通PCAの座標・寄与率・残差、train背景とvalidation背景の2GIFを揃えます。その後seed 1・2を加え、9run・8GIFと再現手順を揃えます。

バッチ条件による差、精度改善、高いPCA寄与率を必須にはしません。毎epochの記録と各GIF 3 MB以下は維持します。
