# ToDo List

> **2026-09-01 方針転換** — 全フェーズをConvNeXt V2-Tinyに統一。Phase 0・1・2はスクラッチ、Phase 3のModel Soupだけ同一のSSL checkpointからfine-tuningします。旧DINO成果物は削除済み。AdamW維持を確定し、Phase 0・1の初期レシピを100epoch・固定LR 1e-3（warmupなし）へ改訂。実装済みと実行検証済みを区別します。

**確定したアニメーション要件:** GIF形式で、各ファイル3 MB以下とします。

**確定した保存容量方針（D-05の一部）:** 保存容量に設計上の上限を設けず、容量節約のために記録を間引いたり既存成果物を自動削除したりしません。RAM・VRAMの制約とGIFの各ファイル3 MB以下は引き続き守ります。

**確定した保存・再開方針（D-05の一部）:** 解析用はFP32の`.pt`、再開用は毎epochの重み・optimizer・scheduler・乱数・データ順序などの状態を保存します。最後に保存を完了したepochの次から再開し、中断したepochは先頭からやり直します。

更新日: 2026-09-01

**現在地:** 設定schema v3・勾配蓄積への変更後、ユーザー実行で設定確認4件とCPU unit test 64件の成功（11.626秒）を受領し、I-06を完了しました。実効B64/B256/B1024、microbatch 64、accum 1/4/16、704/176/44更新/epochが設定上一致。**次はS-03で変更後コードのGPU動作を確認します。** 変更前B64の5epoch・再開記録、共通初期値とsplitは保持。今回のCPU成功をGPU検証済みとは扱わず、SWA/FGE・Soupの条件・実装範囲は変更しません。

[ドキュメント案内](README.md) / [実験計画](EXPERIMENT_PLAN.md) / [実装仕様案](IMPLEMENTATION_SPEC.md) / [参考文献](REFERENCES.md)

## 進め方

- 上から順に進めます。各段階の完了条件を満たしてから、依存する次の段階へ進みます。
- ユーザーが仕様・優先順位を決定し、Agentがその範囲の文書・コードを作成、レビューします。実装の完了と、実行による検証の完了は分けて記録します。
- プログラム・テスト・学習・解析の実行は [AGENTS.md](../AGENTS.md) に従ってユーザーが担当します。Agentによる実行が個別に許可された場合は、その許可範囲で進めます。環境・B256確認に対する既存の許可を、依存パッケージの変更や本学習全体への許可には拡張しません。
- 完了時は `[ ]` を `[x]` にし、成果物または実行結果の所在を添えます。差が出なかった実験や、精度が改善しなかった実験も、手順と記録が成立していれば完了として扱います。
- 数値、データ分割、モデル、評価方法の変更は黙って行いません。決定後に実験計画・実装仕様・設定を揃えます。

## 0. 完了済みの準備

- [x] P-01: 概要をルート、詳細資料を`docs/`、設定例を`docs/examples/`へ整理し、Draftであることを明記。[資料案内](README.md)
- [x] P-02: ユーザー提供資料と補足資料を、用途とともに記録。2026-09-01にConvNeXt V2の解説・原論文をR19・R20へ追加し、既存の公式repo S10もユーザー指定資料として更新。[参考文献一覧](REFERENCES.md)
- [x] P-03: GPU・RAM・Python環境を確認し、224×224・B256・bf16・AdamWで合成入力による3回の更新を実施。ピークallocatedは約8.65 GiB。[測定条件と限界](IMPLEMENTATION_SPEC.md#21-environment-check-2026-08-31)
- [x] P-04: 既存草案と会話中の設計提案の差分・未決定事項を、このリストに整理。

P-03は旧DINO構造での参考記録です。ConvNeXt V2のメモリ・時間・学習成立性を保証しません。新モデルで再測定します。

## 1. Phase 1に必要な設計調整

担当: ユーザーが選択・決定、Agentが整合確認と文書化。

- [x] D-01: ユーザー確定で最小版をseed 0の実効B64/B256/B1024、microbatch 64・accum 1/4/16へ変更。ConvNeXt V2-Tinyのスクラッチ学習、AdamW・100epoch・共通固定LR 1e-3・weight decay 0.05は維持。[実験計画](EXPERIMENT_PLAN.md#62-conditions)
- [x] D-02: checkpointの保存間隔、時間軸、観察できる粒度を確定する。初期状態（epoch 0）と毎epoch終了時に記録し、epoch基準で比較する。保存時のoptimizer stepも併記し、更新単位の揺らぎは主張しない。[記録仕様](EXPERIMENT_PLAN.md#63-checkpoint-frequency)
- [x] D-03: PCAの原点・方向・対象checkpoint・メモリ方針を確定する。比較対象runの毎epochのcheckpointをまとめ、共通平均を原点、上位2主成分を直交基底として使う。寄与率・射影残差を保存し、全重みのRAM/GPU常駐を避けて分割処理する。[射影仕様と出典](EXPERIMENT_PLAN.md#92-projection)
- [x] D-04: 損失平面の評価データ・前処理・精度と、実測損失の表示方法を確定する。train・validationから各クラス100件ずつの固定subsetを作り、共通の21×21格子・評価用前処理・FP32で評価する。train背景を主表示、validation背景を別GIFの補助版とし、実checkpointの指標とは区別する。[評価仕様](EXPERIMENT_PLAN.md#93-loss-surface-evaluation)
- [x] D-05: 設定schema v3で実効batchとmicrobatchを分離。初期モデルv2は再利用し、100epochの更新数を70,400/17,600/4,400へ変更。端数の画像数平均、更新単位のstep/norm、旧runとの再開非互換を明記。記録数・PCA・GIF要件は維持。[実装仕様22節](IMPLEMENTATION_SPEC.md#22-phase-0--1-operational-contract-d-05--d-06)
- [x] D-06: 決定したバッチ条件を[実験計画](EXPERIMENT_PLAN.md)・[実装仕様](IMPLEMENTATION_SPEC.md)・[設定例](examples/config_example.yaml)へ反映。変更前の58件成功・B64実データの確認結果と、変更後の64件CPU成功・GPU未確認を区別。

| 対象 | 現状・決定状態 | 補足・検討方向（未完了項目は提案） |
| --- | --- | --- |
| D-01: 実験範囲 | **改訂確定:** CIFAR-10、ConvNeXt V2-Tiny、224×224、Phase 0・1・2は全parameterのスクラッチ学習。最小版はseed 0のB64/B256/B1024、その後seed 1・2を追加。 | 全条件microbatch 64・accum 1/4/16。同一のスクラッチ初期値を共有。Model SoupのPhase 3のみ同一FCMAE初期値から独立したfine-tuning群を作る。 |
| D-01: 学習条件 | **採用:** AdamW・100epoch・共通固定LR 1e-3・weight decay 0.05。 | Phase 0は5epochで停止し、最初から固定LRでの初期動作を確認。base LRでの安定性や収束は未検証。run別のLR調整はしない。 |
| D-02: 保存間隔 | **確定:** 初期状態（epoch 0）を起点とし、解析用checkpointと指標を全期間1 epoch単位、各epoch終了時に記録する。Phase 0も同じ間隔とする。 | 動画はepoch基準で比較し、実際のepoch・optimizer stepを記録する。checkpoint間を結んだ線から更新単位の揺らぎを主張しない。 |
| D-03: PCA | **確定:** 同一比較の全runについて、epoch 0と毎epoch終了時のcheckpointをまとめた中心化PCAを使う。共通平均を原点とし、上位2主成分を直交単位基底とする。 | 寄与率・各checkpointの射影残差を保存する。Li et al.第7.2節を基礎とし、複数runの共通化・平均原点は今回の比較向けの設計と区別する。 |
| D-03: メモリ | **確定:** 全checkpointの重みをRAM/GPUに常駐させず、順次読み込み・分割処理する。checkpoint数を100以下に間引く案は採用しない。 | D-05でディスク上FP32行列、16,384 parameter/block、FP64のGram固有値分解を採用。100epochへの増加に合わせてブロック幅だけを縮小し、PCAの定義は維持。ピークRAM・時間はユーザーが実測する。 |
| D-04: 背景損失 | **確定:** train主表示とvalidation補助版の2種類を用意する。各splitから固定1,000件（各クラス100件）を抽出し、共通のPCA座標・21×21格子・色尺度で別GIFにする。 | モデル指定の評価用前処理とFP32を使い、AMP・TF32は無効。実checkpointは同じtrain subsetとvalidation全体で評価する。背景のsubset・平面上の損失と、実モデルの指標を表示で区別する。 |
| D-05: 保存・再開 | **確定:** 解析用はFP32の`.pt`。毎epochの学習・乱数・データ順序を保存し、完了manifestのあるepochの次から再開する。 | workerをepochごとに再作成し、評価の乱数を隔離。変更前B64の再開記録は照合済み。今回の蓄積実装は新しい設定・ソース識別で確認し、旧runへ継続しない。 |
| D-05: 設定・パス | **改訂:** 階層型YAMLとfrozen dataclass、設定schema v3。初期値は既存の`artifacts/init/convnextv2_tiny_scratch/theta_0.pt`を再利用。 | 実効batchとmicrobatchを明示。loss/accuracyは画像数平均、gradient normは蓄積後のAdamW更新回数平均。 |
| D-05: 予算 | **確定:** 保存容量上限・自動削除・壁時計時間の自動打ち切りを設けない。RAM・VRAM制約は維持する。 | Phase 0は6時点。Phase 1は101時点/run、最小3runで303・全9runで909時点。新モデルで学習・評価・保存・PCA・格子・描画を分けて測り、旧モデルの値は流用しない。 |
| D-05: 完了条件 | **確定:** 学習・記録・共通座標・両背景の動画の成立で判定。 | batch間の差・精度改善・高い寄与率は必須にしない。補間評価はPhase 2準備に分離。低い寄与率は表示と解釈の制限として残す。 |

完了条件: Phase 1の実装に必要な判断が文書上で揃い、実装時にデータや数値の意味を推測する必要がない。

## 2. 学習・保存の基盤を実装

依存: D-01〜D-06。担当: Agentが実装・レビュー、ユーザーが検証コマンドを実行。

- [x] I-01: [config.py](../src/landscape_exp/config.py)をschema v3へ改訂し、実効B64/B256/B1024とmicrobatch 64を固定。ConvNeXt V2・scratch・init seed 0は維持。旧設定schema・DINO・pretrained指定は拒否。今回の検証はI-06に分離。
- [x] I-02: [data.py](../src/landscape_exp/data.py)・[seeds.py](../src/landscape_exp/seeds.py)・[split準備CLI](../scripts/prepare_splits.py)に、CIFAR-10の読込、共有split/subset、前処理、独立したDataLoader乱数とepoch間の復元を実装。公式test・自動ダウンロードを対象外にした。2026-08-31、ユーザー実行で既存8件＋追加17件の計25件成功（7.728秒）。実モデル・学習との接続は別途検証する。
- [x] I-03: [models.py](../src/landscape_exp/models.py)・[初期重みCLI](../scripts/create_init_checkpoint.py)を、backboneとheadの一括スクラッチ初期化・共通state保存・厳密な復元へ変更。pretrained=Falseを固定し、初期化の出所をmetadataへ記録。tiny fixtureでの検証は受領した57件に含む。実モデルでのCPU作成・復元の成功もS-01で受領済み。
- [x] I-04: [train.py](../src/landscape_exp/train.py)・[evaluate.py](../src/landscape_exp/evaluate.py)・[checkpoints.py](../src/landscape_exp/checkpoints.py)・[logging_utils.py](../src/landscape_exp/logging_utils.py)と[学習CLI](../scripts/run_train.py)に、AdamW・固定LR・FP32評価・実測値・毎epochの解析/再開状態・新segmentへの再開を実装。2026-09-01に新初期化契約の57件成功（12.478秒）を受領。その後、100epochの固定LR・5epoch停止・scheduler状態schema v2へ変更し、旧cosineとの混同を拒否する検証と最終更新までの固定LRテストを反映。
- [x] I-05: 2026-09-01、固定LRに合わせた設定・再開形式・テスト変更後に、ユーザー実行でPhase 0・1の`configuration_valid`と58件の`OK`（11.231秒）を受領。最終更新までの固定LR、CPU fixtureでのepoch再開一致を含む。[検証コマンド](../README.md#今回の検証コマンド)
- [x] I-06: 2026-09-01、ユーザー実行でPhase 0とPhase 1の実効B64/B256/B1024設定確認4件の`configuration_valid`、CPU unit test 64件の`OK`（11.626秒）を受領。端数200/968件を含む小モデルでの物理batchとのgradient/AdamW近似一致、蓄積時のepoch再開一致、非有限値での更新停止を含む。Agentは実行していない。[検証コマンド](../README.md#今回の検証コマンド)

I-06の設定schema v3の実行記録（全件`created_artifacts: false`）:

| run_id | 実効batch | microbatch | accum | 更新/epoch | effective_sha256 |
| --- | --- | --- | --- | --- | --- |
| phase0/b64_seed0 | 64 | 64 | 1 | 704 | `eac561e752bec5bcb7de8fe877392e35d87ad70688c6037c34dbfbac2c33b1c4` |
| phase1/b64_seed0 | 64 | 64 | 1 | 704 | `b8d10a4b31acbd3bf7435932328ae856b8b9221ac5b011d393834b81d3eb2afd` |
| phase1/b256_seed0 | 256 | 64 | 4 | 176 | `f154001c9c384c73e143f4defa5da435f1e35ac23adc66b5de0eac95ac65fc35` |
| phase1/b1024_seed0 | 1024 | 64 | 16 | 44 | `65ac76ccb2ce0b7f4272fd0125a02a902f8e4de5320ac2494b74e5fc5f2d0873` |

全条件の予定期間は100epoch。Phase 0は5epoch終了・6時点、Phase 1は100epoch終了・101時点。設定確認はデータ・モデル・GPUを実行せず、CPU fixtureの成功も実モデルのbf16学習やGPUメモリを保証しない。

I-05で受領した旧設定schema v2の確認は両方とも`created_artifacts: false`。終了epochと記録数は現在も同じだが、以下のhashは今回のv3設定のhashではない。

| 設定 | schedule_epochs | stop_after_epoch | analysis_points_per_run | effective_sha256 |
| --- | --- | --- | --- | --- |
| Phase 0 | 100 | 5 | 6 | `487d624603e24f64a261215aa5a4ecbfccd7da7715e7d8fc0b223d8198a991ff` |
| Phase 1 | 100 | 100 | 101 | `b0a9674a39550624aaea4a654b656701a3238b3f2644ea9ec4e279bcefc5d1f3` |

以前受領した57件成功（12.478秒）とeffective SHA-256 `8284c252f5692264c7f59ef37a261b9509cd9b5bbb6ab38f195229791f700b89`は旧50epoch・LR 1e-4・warmup 5epoch時点の記録。今回の結果と混同しない。Agentはプログラム・テストを実行していない。設定確認とCPU検証は、ConvNeXt V2実モデルの作成・実データ・GPU・再開一致を保証しない。

完了条件: 決定済みの設定で学習・保存・評価を呼び出せるコードと実行手順が揃い、未検証箇所が明記されている。

## 3. Phase 0: 実データによる短い確認

依存: I-01〜I-06。実行担当: ユーザー（Agent実行の明示的な許可がある場合はその範囲）。

- [x] S-01: 2026-09-01、ユーザー実行で`split_verified`、`initial_checkpoint_created`、`initial_checkpoint_verified`を受領。既存CIFAR-10・共有splitを再利用し、ConvNeXt V2-Tinyの共通スクラッチ初期値をCPUで作成・再読込確認。作成時と検証時のSHA-256が一致。[手順](../README.md#s-01-実データと共通初期重みの準備)
- [x] S-02: 変更前のConvNeXt V2-Tiny・B64・seed 0・固定LR 1e-3で、ユーザー実行のepoch 0〜5成果物を確認。5epochで3,520更新、最終validation accuracy 62.82%、元runのpeak allocated約7.10GiB・reserved約7.60GiB。今回の勾配蓄積版の実機確認とは区別。
- [ ] S-03: 変更前B64のepoch 2からの再開を照合済み。今回の勾配蓄積版のGPU動作・再開・B256/B1024のメモリ確認は未完了。新コードの確認runは別名にし、既存成果物を上書きしない。失敗時は原因を修正して再確認する。

S-02・S-03の変更前B64記録（2026-09-01に照合）:

- 元segment: `artifacts/runs/phase0/b64_seed0/segments/20260831T175509863426Z_d155e9fa7d954d018ea80df8851a6565`。
- 再開segment: `20260831T182119866267Z_04cffcb5663244deb807390096e0ab72`。元epoch 2を親とし、epoch 3〜5のloss/accuracy/gradient norm/displacement/LR/stepと、analysis.pt・resume.ptの保存済みSHA-256が一致。時間・VRAM等の計測値は別。
- Agentは小さなJSON/CSV記録をread-onlyで照合した。checkpoint本体の再ハッシュ・tensor読込は行っていない。この結果を今回の勾配蓄積版の検証済み根拠にはしない。

S-01の実行記録:

- split: train 45,000、validation 5,000、各subset 1,000。labels SHA-256は`f5cfe00b0f00968c0cc5ff3b1d2de51b10e33efa277c2986f4e0fa63e58c9f4f`で既存と一致。公式test不使用・ダウンロードなし。
- 初期値: `artifacts/init/convnextv2_tiny_scratch/theta_0.pt`。SHA-256は`8398a420b5503b214505d5d8c9d4f507580ecc15a6e21f452458a780bb2bc8d5`。parameter数27,874,186、CPU・FP32、`initialization.mode=scratch`・seed 0・`pretrained=false`。作成・再読込とも`pretrained_fetch_requested=false`・`training_started=false`。
- 保存metadataの前処理: 入力3×224×224、center crop、crop_pct 0.875、bicubic、mean [0.485, 0.456, 0.406]、std [0.229, 0.224, 0.225]。runtimeはPython 3.13.2、torch 2.6.0+cu126、torchvision 0.21.0+cu126、timm 1.0.24、numpy 2.1.3、Pillow 11.1.0。
- Agentは小さなmetadataとファイルの存在・サイズをread-onlyで確認。checkpointの再読込・hash照合はユーザー実行の結果であり、Agentが再実行したものではない。学習・GPUメモリ・実データでの再開結果はS-02・S-03に分離して記録する。

完了条件: 実データで学習・評価・保存が成立し、再開と資源見積もりが確認できる。可視化の短い確認は次のV-05で行う。

## 4. 射影・損失平面・アニメーションを実装

依存: 学習・保存基盤とPhase 0の成果物。担当: Agentが実装・レビュー、ユーザーが実行。

- [ ] V-01: 重みのベクトル化と復元、parameterの順序・shape・dtype・bufferの扱いを実装し、roundtripを検証。
- [ ] V-02: 共通PCA、座標、寄与率、射影残差を生成・保存。比較対象を追加してPCAを作り直す場合は別の識別子で保存し、異なる座標系を混在させない。
- [ ] V-03: train・validationそれぞれの固定subsetで、共通の21×21格子による損失平面を事前計算・保存。評価設定と対象subsetを追跡可能にする。
- [ ] V-04: 固定した背景・軸・色尺度に、履歴・現在点・epoch・step・LR・実測指標を表示する動画生成を実装。train背景の主版とvalidation背景の補助版を同じ軌跡から生成し、再描画時はモデル評価を行わない。
- [ ] V-05: Phase 0のcheckpointからtrain背景・validation背景の短い動画を生成し、時間軸、実測値、座標、保存済み成果物からの再描画を確認。各ファイル3 MB以下のGIF出力を確認。

完了条件: 保存済みの座標・格子・指標だけで動画を再生成でき、背景損失と実測損失の意味が表示から区別できる。

## 5. Phase 1: 最初の成果物を完成

依存: S-03、V-01〜V-05。

- [ ] M-01: 共通`theta_0`からseed 0のB64/B256/B1024（microbatch 64・accum 1/4/16）を実行し、同じ時間軸・共通座標で比較するtrain背景の主動画とvalidation背景の補助動画を完成させる。
- [ ] M-02: ログ、保存間隔、学習条件、射影残差、動画表示を点検し、最小版の完了を確認。
- [ ] M-03: seed 1・2を追加して全9 runsを揃え、共通PCAを再計算。seedごとの動画と全runのsummary動画をtrain背景・validation背景の両方で作成し、条件別の評価表とともに保存。

最小版の完了条件: seed 0の3バッチ比較動画（train背景の主版・validation背景の補助版）と、その元となる設定・checkpoint・実測ログが揃う。

Phase 1の完了条件: 予定した全9 runsの記録と動画が揃い、再現手順が明記される。バッチ条件による差が出ること自体は必須にしない。

## 6. Phase 2: 補間・SWA・FGE

依存: Phase 1の完了。未確定の条件は、この段階の着手前に確定する。

- [ ] A-01: SWA/FGEはepoch 80終了時の同じ重み・AdamW・乱数状態から、1e-3 → 1e-5 → 1e-3の4epoch三角周期を5回、epoch 100終了まで行う条件を採用済み。最低LRの同じ5点を毎epochの記録から採取する。update単位のLR適用・端点、評価・射影の残る詳細を確定。Normalは固定LR 1e-3で同じ期間を比較し、B64・seed 0を既定候補とする。[原論文との対応・共通条件](EXPERIMENT_PLAN.md#73-branches)
- [ ] A-02: 高次元の重みを直接線形補間する評価を実装し、端点・同一モデル・有限な評価点でのbarrier定義を検証。初期案は21点。2D表示からbarrierを推定しない。
- [ ] A-03: SWA/FGE共通のraw trajectoryと5点のsnapshot、SWAの採取時running average、FGEの確率平均ensembleを実装。同じ5点で単体・重み平均・予測平均を評価する。FGE weight averageとSWAの一致も確認し、学習・checkpointを重複させない。
- [ ] A-04: 共通分岐点から実行し、同じ座標系でのNormal/SWA/FGE動画、SWA平均点の動画、補間結果、性能比較を保存。

SWA固定LR・毎epoch平均案、75開始・5epoch周期案は撤回。ユーザー指示により、SWA/FGEは80〜100epochの同じ周期学習と同じ5点を共有し、重み平均と予測確率平均の違いを比較する。最低LR地点はepoch 82・86・90・94・98終了時であり、毎epochの記録から選ぶ。半epochでの追加保存は行わず、再開とGIFも1epoch単位を維持する。Phase 1の100epochも維持する。原実験の再現とは呼ばず、採用済み部分も未実装・未検証として扱う。スクラッチ学習、1 seedの結果をseed間で一般化しない方針は維持する。

完了条件: 学習率操作と平均化を区別して比較でき、動画・評価値・使用checkpointを追跡できる。精度改善は必須にしない。

## 7. Phase 3: 複数runとModel Soup

依存: Phase 2と補間・平均化の検証完了。

- [ ] U-01: ここで初めて同一のSSL checkpoint（純粋な`convnextv2_tiny.fcmae`候補）と共通headから独立したfine-tuning群を作る。AdamW・固定LR 1e-4・warmup/decayなしはユーザー指定で確定。取得・初期化・候補集合・epoch・weight decay・Greedy Soupの順位・同点・採用条件は残る設計事項。Phase 1・2のスクラッチ学習済み重みは再利用しない。
- [ ] U-02: Uniform/Greedy Soupとlogit ensembleの評価を実装し、同一モデルの平均、候補の互換性、整数buffer、test非使用の選択処理を検証。
- [ ] U-03: 候補間のbarrier、単体モデル、Soup、ensembleを評価し、複数軌跡・Soup形成の動画と評価表を保存。Soup自体も実モデルとして評価する。

現時点の候補: Greedy Soupはvalidation accuracyの厳密な改善のみ採用し、同率の候補はrun ID順。学習率違いの追加runは、この最小版とは分けて判断する。

完了条件: 候補・採否・平均方法・評価結果が記録され、重み平均と予測平均が混同されていない。

## 8. 最終評価・成果物の整理

依存: 対象実験の候補選択と設定が固定されていること。

- [ ] F-01: 固定済みのモデル・手法を公式test setで最終評価。testの結果を見て候補選択やハイパーパラメータをやり直さない。
- [ ] F-02: 動画、補間図、barrier matrix、性能表、設定、実行環境、再現手順を整理。コードの版はGitを利用する場合にcommitを記録し、未管理ならその事実を明記。
- [ ] F-03: 観察と解釈を分けて記述し、2D射影・有限点補間・seed数・subset・数値精度の限界を明記。参照した手法の出典を付ける。

## 初版の対象外・後で判断すること

- 更新単位の高密度な軌跡記録と、勾配ノイズの因果効果を調べる対照実験。
- 曲線を学習するmode connectivity、Hessianやsharpnessの詳細な評価。
- filter-normalized方向による追加可視化、独立した3D/対話UI。
- 追加モデル・追加データセット・追加ハイパーパラメータ群・分散学習・FCMAE等の自己教師あり事前学習そのもの。

これらは自動で着手せず、必要性と予算を確認してからリストへ追加します。
