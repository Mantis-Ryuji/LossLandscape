# Loss Landscape Dynamics on CIFAR-10

> **現在地** — M-01のseed 0・B64/B256/B1024各100epoch学習は完了しました。現在はrepo全体を`D:\LossLandscape`へ安全に移行し、D:側で共通PCA・両損失背景・比較GIFを作成する段階です。

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
- 入力224×224、全parameterを学習。RandomResizedCrop・HorizontalFlipと決定的な評価前処理を使い、正規化・resize/cropの実値はConvNeXt V2のtimm設定から記録します。
- Phase 1は実効B64/B256/B1024、全条件microbatch 64・accum 1/4/16、seed 0/1/2。まずseed 0の3runを完成させます。run seedは学習中の乱数用で、初期重みは変えません。
- epoch 0と毎epoch終了時を記録。解析用はFP32、再開用は重み・optimizer・scheduler・乱数・DataLoader状態を保存します。
- 共通PCA、train背景の主GIFとvalidation背景の補助GIF。背景は各固定1,000件の21×21格子、実checkpointは同じtrain subsetとvalidation全体をFP32で評価します。
- GIFは各3,000,000 bytes以下、epochを間引きません。保存容量上限や自動削除は設けず、RAM・VRAM制約は守ります。

## スクラッチ初期化を明示する設定

設定schemaはv3、初期モデルの保存schemaはv2です。全runで同じ共通初期重みを使います。

```yaml
model:
  name: convnextv2_tiny
  initialization: scratch
  num_classes: 10
  image_size: 224
  full_finetune: true
  init_seed: 0
```

`full_finetune: true`は「全parameterを学習する」という意味で、事前学習重みの利用を意味しません。

初回はCPU上でinit seed 0を設定し、`timm.create_model("convnextv2_tiny", pretrained=False, num_classes=10)`を一度呼びます。backbone・headともモデル既定の初期化を使い、headだけを後から再初期化しません。固定値で始まる正規化層等も含め、モデル全体の初期stateを保存します。

保存先は`artifacts/init/convnextv2_tiny_scratch/theta_0.pt`と同名JSONです。初期化方式・seed・`pretrained: false`を記録し、`pretrained_reference`はnullにします。timmが持つ前処理用の`pretrained_cfg`を、重みを読み込んだ証拠として記録しません。各runはこの同じ全stateを復元します。Phase 0・1のCLIはスクラッチ初期化以外を受け付けません。

Model Soup用の候補は、教師ありfine-tuning前の[`convnextv2_tiny.fcmae`](https://huggingface.co/timm/convnextv2_tiny.fcmae)です。`fcmae_ft_in1k`等とは区別します。Phase 3の初期化CLIや学習条件はまだ実装していません。

## スクラッチ用の学習レシピ

Phase 0・1はAdamW・全期間100epoch・固定LR 1e-3を採用します。warmup・LR decayは行いません。Model Soupは各fine-tuning runで固定LR 1e-4とし、後段の設計条件として記録しています。

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

PCAは16,384 parameter単位の分割処理とFP64のGram固有値分解を使います。全checkpointをRAMまたはGPUへ同時常駐させず、対象checkpointの間引きもしません。

Phase 0は100epochの予定を保ち、**最初から固定LR 1e-3で5epoch学習して停止**します。100epochの安定性・収束は、この短い確認だけでは保証しません。Phase 2のNormalも固定LR。SWA/FGEはepoch 80終了時から100まで、共通の1e-3 → 1e-5 → 1e-3、4epoch周期を5回とし、同じ5点で重み平均と予測確率平均を比較します。最低LRのepoch 82・86・90・94・98終了時を毎epochの記録から選び、半epochでの追加保存は行いません。詳細は[原論文との対応と共通条件](docs/EXPERIMENT_PLAN.md#73-branches)を参照します。Phase 2は未実装です。

## 現在の実装

- `config.py`：schema v3の階層YAML・型検証・設定保存。実効batchとmicrobatchを分離。
- `data.py` / `seeds.py`：共有split/subset、前処理、DataLoaderと乱数の保存・復元。
- `models.py`：共通のスクラッチ初期stateを作成し、hash・layout・runtime・初期化方式を検証して復元。
- `train.py` / `evaluate.py`：AdamW、固定LR、勾配蓄積、FP32評価、再開。端数の画像数平均と更新単位のstepを維持。設定の100epochとPhase 0の5epoch停止を分離し、最終更新でもLRを0にしません。
- `checkpoints.py` / `logging_utils.py`：毎epoch保存、完了manifest、分岐の親子関係、JSON/CSV。
- `loss_surface.py`：完成PCAの厳密な読込、共通格子、train/validation固定subsetの同時評価、共通色尺度、immutable surface artifact。
- `animation.py`：完成PCA・両損失平面・実checkpoint指標の厳密な読込、固定表示、GIF圧縮fallback、immutable animation artifact。モデル・dataset・checkpointは読み込みません。
- `scripts/`：設定確認、split準備、初期値作成・確認、学習、共通PCA、損失平面、GIF生成のCLI。
- `tests/`：小さなCPU fixtureによる検証。移設対応追加前の全86件はユーザー実行で成功済みです。今回、絶対配置だけを変えたrunの互換性testを1件追加しました。
- Phase 0の学習・再開・共通PCA・両損失平面・細線版GIFペアまで実データで検証済みです。大容量のPhase 0成果物は保持対象外であり、Phase 1の入力には使用しません。

## 検証済みの範囲

設定確認、CPU unit test、共通初期重みの作成・復元、Phase 0のB64/B256/B1024による5epoch GPU学習、B64のepoch境界再開、PCA、両損失平面、GIF生成までユーザー実行で確認済みです。B256/B1024でもmicrobatch 64を維持し、peak allocatedは約7.19 GiB、reservedは約7.61 GiBでした。

現在のCPU test suiteは87件です。追加した移設対応testを含む現在のsuiteは未実行です。Agentは[AGENTS.md](AGENTS.md)に従い、プロジェクトのプログラム・テスト・学習を実行しません。

```powershell
python -B scripts\check_config.py --config configs\phase1.yaml
python -B scripts\check_config.py --config configs\phase1.yaml --batch-size 256
python -B scripts\check_config.py --config configs\phase1.yaml --batch-size 1024
python -B -m unittest discover -s tests -v
```

設定確認の期待値は、B64/B256/B1024について`microbatch_size: 64`、`accumulation_steps: 1/4/16`、`optimizer_steps_per_epoch: 704/176/44`です。

## M-01: seed 0の3run

次の3runは完了済みです。

```powershell
python -B scripts\run_train.py --config configs\phase1.yaml --batch-size 64 --seed 0
python -B scripts\run_train.py --config configs\phase1.yaml --batch-size 256 --seed 0
python -B scripts\run_train.py --config configs\phase1.yaml --batch-size 1024 --seed 0
```

| Batch | Segment | Epoch | Final step | Validation accuracy |
| --- | --- | ---: | ---: | ---: |
| 64 | `20260901T145535527738Z_9267847eb7824937a315ec7d9b8aaa8e` | 0〜100 | 70,400 | 92.02% |
| 256 | `20260901T204835413455Z_720286048a0b4cb9b5d9365ad7f49978` | 0〜100 | 17,600 | 93.86% |
| 1024 | `20260902T062841348044Z_0432ed683bff4909bcd5f77fc4a1634e` | 0〜100 | 4,400 | 93.14% |

各runは101 epoch directory・CSV 101行を持ち、必須ファイルの欠損、manifest記載サイズ、epoch/step identityの不一致がないことをread-onlyで確認済みです。3run合計は約125.32 GiBです。

### repo全体をD:へ完全移行

コード、`.git`、設定、dataset、共通初期checkpoint、既存3 runを含むrepo全体を`D:\LossLandscape`へコピーします。準備時点でrepo内に`.venv`はないため、移設対象外の仮想環境はありません。学習・解析processが動いていない状態で、次を1回実行します。

まずC:側で、移設対応を含むCPU test 87件を確認します。

```powershell
python -B -m unittest discover -s tests -v
```

全件`OK`ならrepo全体をコピーします。

```powershell
robocopy "C:\Users\PC_User\Python\LossLandscape" "D:\LossLandscape" /E /COPY:DAT /DCOPY:DAT /R:2 /W:5 /J /XJ
$copyExitCode = $LASTEXITCODE
$copyExitCode
```

`robocopy`の終了code 0〜7は成功、8以上は失敗です。コピー元はまだ削除しません。コピー後、relative path・ファイル数・各ファイルのsizeをrepo全体で照合します。`SourceFiles`と`DestinationFiles`、`SourceGiB`と`DestinationGiB`がそれぞれ一致し、`Mismatches=0`になることが完了条件です。

```powershell
function Compare-CopiedTree {
  param([string]$Source, [string]$Destination)

  $sourceRoot = (Resolve-Path -LiteralPath $Source).Path.TrimEnd('\')
  $destinationRoot = (Resolve-Path -LiteralPath $Destination).Path.TrimEnd('\')
  $sourceFiles = @(Get-ChildItem -LiteralPath $Source -Recurse -File)
  $destinationFiles = @(Get-ChildItem -LiteralPath $Destination -Recurse -File)
  $sourceIndex = @{}
  $destinationIndex = @{}

  foreach ($file in $sourceFiles) {
    $relative = $file.FullName.Substring($sourceRoot.Length).TrimStart('\')
    $sourceIndex[$relative] = $file.Length
  }
  foreach ($file in $destinationFiles) {
    $relative = $file.FullName.Substring($destinationRoot.Length).TrimStart('\')
    $destinationIndex[$relative] = $file.Length
  }

  $mismatches = @(
    foreach ($relative in $sourceIndex.Keys) {
      if (-not $destinationIndex.ContainsKey($relative) -or
          $destinationIndex[$relative] -ne $sourceIndex[$relative]) {
        $relative
      }
    }
    foreach ($relative in $destinationIndex.Keys) {
      if (-not $sourceIndex.ContainsKey($relative)) {
        $relative
      }
    }
  )
  [pscustomobject]@{
    Source = $Source
    Destination = $Destination
    SourceFiles = $sourceFiles.Count
    DestinationFiles = $destinationFiles.Count
    SourceGiB = [math]::Round(($sourceFiles | Measure-Object Length -Sum).Sum / 1GB, 2)
    DestinationGiB = [math]::Round(($destinationFiles | Measure-Object Length -Sum).Sum / 1GB, 2)
    Mismatches = $mismatches.Count
  }
}

Compare-CopiedTree `
  "C:\Users\PC_User\Python\LossLandscape" `
  "D:\LossLandscape"
```

照合後はD:側を新しい作業repoとして開き、以後のコマンドを`D:\LossLandscape`で実行します。通常の相対パス設定がすべてD:配下へ解決されるため、専用設定は使いません。

```powershell
Set-Location "D:\LossLandscape"
git status --short
python -B scripts\check_config.py --config configs\phase1.yaml
python -B scripts\check_config.py --config configs\phase1.yaml --batch-size 256
python -B scripts\check_config.py --config configs\phase1.yaml --batch-size 1024
```

既存runの`config.json`に記録済みのC:絶対パスは、移設後も来歴として書き換えません。射影の互換性判定では絶対配置を実験identityから除外し、学習条件・runtime・前処理・source・checkpoint contractは従来どおり照合します。さらにD:側の共通初期checkpoint本体と各epochが宣言する初期checkpointのSHA-256を一致確認します。

このCodexタスクはC:側workspaceに紐づいているため、照合後は`D:\LossLandscape`をCodexで開いて作業を続けます。D:側でPCAが`projection_ready`になるまでは、C:側のrepoを削除しません。

### seed 0の共通PCA

D:上の3つのsegmentを明示し、epoch 0〜100の計303点から共通PCAを作成します。作業用FP32行列は約31.46 GiBで`D:\LossLandscape\artifacts\work\<projection_id>\weights.npy`へ保存されます。

```powershell
python -B scripts\compute_projection.py `
  --config configs\phase1.yaml `
  --comparison-scope phase1_seed0 `
  --segment "D:\LossLandscape\artifacts\runs\phase1\b64_seed0\segments\20260901T145535527738Z_9267847eb7824937a315ec7d9b8aaa8e" `
  --segment "D:\LossLandscape\artifacts\runs\phase1\b256_seed0\segments\20260901T204835413455Z_720286048a0b4cb9b5d9365ad7f49978" `
  --segment "D:\LossLandscape\artifacts\runs\phase1\b1024_seed0\segments\20260902T062841348044Z_0432ed683bff4909bcd5f77fc4a1634e"
```

成功時の`projection_id`を、同じ設定で損失平面とGIF生成へ渡します。

```powershell
python -B scripts\compute_loss_surfaces.py `
  --config configs\phase1.yaml `
  --projection "D:\LossLandscape\artifacts\projections\<projection_id>"

python -B scripts\render_animation.py `
  --config configs\phase1.yaml `
  --projection "D:\LossLandscape\artifacts\projections\<projection_id>" `
  --name phase1_seed0_batch_compare
```

PCAは全303 epochのmanifestと4ファイルをhash検証し、`analysis.pt`を復元してから計算します。PCAが`projection_ready`で完了するまではC:側の3runを削除しません。既存または未完成の成果物を上書き・補修せず、再実行時は新しいIDへ保存します。

## Phase 1の完了条件

まずseed 0の3runについて、共通のスクラッチ初期値、全epochのcheckpoint・実測ログ、共通PCAの座標・寄与率・残差、train背景とvalidation背景の2GIFを揃えます。その後seed 1・2を加え、9run・8GIFと再現手順を揃えます。

バッチ条件による差、精度改善、高いPCA寄与率を必須にはしません。毎epochの記録と各GIF 3 MB以下は維持します。

seed 1・2を追加するときもD:側の`configs/phase1.yaml`を使い、今後の重みを`D:\LossLandscape\artifacts`へ保存します。
