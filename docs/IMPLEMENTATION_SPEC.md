# Implementation Specification for Codex

> **2026-09-01 検証状況更新** — V-04/V-05は完了しました。細線版の追加6件・全86件のCPU testがユーザー実行で成功し、Phase 0実GIFペアも6 frame・960×640・128色・各3 MB以下で完成。manifest、保存済み成果物だけからの再描画、線幅と3本の識別性を確認しました。次はM-01でseed 0のB64/B256/B1024を各100epoch実行します。3条件の5epoch GPU動作とB64再開一致、ConvNeXt V2-Tiny、Phase 0・1・2のスクラッチ、SoupだけSSL初期値、AdamW・100epoch・固定LR 1e-3は維持します。

[ドキュメント案内に戻る](README.md)

## 1. Scope

Phase 0・1についてはスクラッチ初期化・記録・評価・固定LR 1e-3の契約を定める。Phase 2のSWA/FGEはepoch 80終了時から100まで、共通の4epoch三角周期を5回、同じ最低LRの5点での平均比較を採用。Phase 3のModel Soupは固定LR 1e-4を採用済み。Phase 2・3の残る詳細・実装はDraftとして残す。保存容量の上限は設けず、解析用はFP32の`.pt`、再開用は毎epochの学習状態を保持する。詳細な設定・再開手順・PCA・完了条件は22節を正とする。設計の確定は実装・実行の完了を意味しない。

Phase 1は、まずseed 0のB64/B256/B1024比較動画を完成させ、その後seed 1・2を追加する。共通条件とPhase 0で検証する学習条件は[実験計画のD-01](EXPERIMENT_PLAN.md#62-conditions)に従う。SWA/FGE/Model SoupはPhase 1の実装対象に含めない。

目的はCIFAR-10とConvNeXt V2-Tinyを用いて、スクラッチの最適化軌跡、SWA/FGE、最後にSSL初期値からのModel Soupを観察するpipelineを構築すること。Phase 1・2のcheckpointをPhase 3のSoup候補へ流用しない。

過剰な抽象化より、再現性・型安全性・実験追加の容易さを優先する。

Python 3.10+。

---

# 2. Coding Requirements

- 全関数・引数・戻り値に型ヒント
- `Any` は極力避ける
- `pathlib.Path`
- dataclass / TypedDict 等で設定契約を明示
- fail fast validation
- NumPy/scikit-learn style docstring
- I/O と pure logic を分離
- device / dtype / seed を明示
- checkpoint format を固定
- config は YAML
- matplotlib を使用
- seaborn は使用しない
- 1 chart = 1 figure
- 色は matplotlib default を優先し、ハードコードしない

---

# 3. Suggested Repository Layout

以下は全フェーズを見通した配置案であり、一括作成しない。I-01で設定基盤、I-02でデータ・seed、I-03で初期モデルを追加した。I-04では`train.py`・`evaluate.py`・`checkpoints.py`・`logging_utils.py`・`scripts/run_train.py`と対応するCPUテストを追加した。Phase 1は単一設定とbatch・seedの明示的な上書きで実行し、条件ごとの重複設定や設定継承は設けない。現在の配置は[ドキュメント案内](README.md)を参照する。

```text
LossLandscape/
├── AGENTS.md
├── README.md
├── docs/
│   ├── README.md
│   ├── TODO.md
│   ├── EXPERIMENT_PLAN.md
│   ├── IMPLEMENTATION_SPEC.md
│   ├── REFERENCES.md
│   └── examples/
│       └── config_example.yaml
├── configs/
│   ├── phase0.yaml
│   ├── phase1.yaml
│   ├── phase2_swa.yaml
│   ├── phase2_fge.yaml
│   └── phase3_soup.yaml
├── src/
│   └── landscape_exp/
│       ├── __init__.py
│       ├── config.py
│       ├── seeds.py
│       ├── data.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── checkpoints.py
│       ├── projection.py
│       ├── landscape.py
│       ├── loss_surface.py
│       ├── interpolation.py
│       ├── averaging.py
│       ├── animation.py
│       └── logging_utils.py
├── scripts/
│   ├── check_config.py
│   ├── prepare_splits.py
│   ├── create_init_checkpoint.py
│   ├── run_train.py
│   ├── run_phase2_branch.py
│   ├── compute_projection.py
│   ├── compute_loss_surfaces.py
│   ├── compute_interpolations.py
│   ├── evaluate_soups.py
│   └── render_animation.py
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_seeds.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_landscape.py
│   ├── test_loss_surface.py
│   ├── test_interpolation.py
│   ├── test_averaging.py
│   ├── test_checkpoint_roundtrip.py
│   └── test_projection.py
└── artifacts/
    ├── splits/
    ├── init/
    ├── runs/
    ├── projections/
    ├── surfaces/
    ├── interpolations/
    ├── soups/
    └── animations/
```

---

# 4. Configuration Contract

YAMLの各sectionと同じ階層のfrozen dataclassを`ExperimentConfig`にまとめる。schema v3の全項目は[設定例](examples/config_example.yaml)、型と入力検証は`src/landscape_exp/config.py`で管理する。未使用のPhase 2・3項目はPhase 1のschemaへ含めない。

未知・重複・欠落キー、型違い、非有限値、確定条件との不整合を入力時に拒否する。YAMLはSafeLoaderで読み、任意のPython object・設定継承・環境変数展開は認めない。パスはプロジェクトルート基準で絶対パスに解決し、元のYAMLと解決済み設定を分けて保存する。

`training.checkpoint_interval_epochs`は整数`1`。`training.epochs`はscheduleの全期間、`stop_after_epoch=5`はPhase 0の停止地点であり、scheduleを5epochへ短縮しない。Phase 1は`stop_after_epoch=null`で全期間実行する。

---

# 5. Reproducibility

## Seeds

以下を統一して設定:

- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA

validation transform は stochastic にしない。

DataLoader generator も seed 固定。

## Split

CIFAR-10 train 50,000 から stratified 45k/5k split を一度生成し、index を

`artifacts/splits/cifar10_train_val_seed<split_seed>_subset<subset_seed>.npz`

へ保存する。現在の設定では両seedは`20260831`。train・validation・各1,000件のsubsetの4配列をint64で保存し、同名の`.json`にschema、抽出条件、ラベル順序のSHA-256、NPZのSHA-256、NumPy versionを保存する。JSONを最後に書き、両ファイルの存在・整合が確認できる記録だけを再利用する。不完全な記録を補修・上書き・削除しない。

全 run で同じ split を使う。

I-02の`prepare_splits.py`は取得済みの公式trainだけを読み、自動ダウンロードはしない。`build_preprocessing`はI-03で取得する`timm.data.resolve_model_data_config(model)`の結果を受け、学習用augmentationと決定的な評価用transformを分ける。画像本体は1つのsourceで共有し、公式testを受け取るviewは設けない。

`LoaderGenerators`はtrain順序用と4種類のworker用を分離する。`preserve_random_state`はPython・NumPy・PyTorchと指定されたloader乱数を例外時も復元する。GPUを使う呼出側は先にdeviceを初期化し、その後にseed設定・snapshot取得を行う。CPU処理からCUDAを初期化しない。実際の評価モード・FP32設定・再開checkpointへの接続はI-04の責務とする。

---

# 6. Model Initialization

**2026-09-01改訂:** Phase 0・1では `model.name=convnextv2_tiny`、`model.initialization=scratch`、`model.init_seed=0`を固定する。旧 `head_seed` は全体の `init_seed` に置換し、設定・初期モデルのschemaをv2へ更新する。Phase 2はPhase 1のスクラッチ学習途中から分岐する。

```python
with preserve_random_state(), torch.device("cpu"):
    seed_global(0)
    model = timm.create_model(
        "convnextv2_tiny", pretrained=False, num_classes=10,
    )
```

backbone・headを一度の構築でモデル既定の方式により初期化し、headだけを再初期化しない。正規化やGRN等の固定初期値も含めた全stateを共有する。`full_finetune=True`は既存API名を維持し、「全parameterが更新対象」という意味で使う。事前学習重み利用の意味は持たない。

各runでは同じ構造を`pretrained=False`で作り、保存済み全stateを`strict=True`で復元する。復元前にキー・順序・shape・dtypeを照合し、暗黙の型変換を許さない。仮に構築した乱数重みを学習へ渡さない。初期モデルはCPU FP32・eval、呼出元のPython・NumPy・torch乱数を保存・復元する。

前処理は`timm.data.resolve_model_data_config`から取得して記録する。モデルに`pretrained_cfg`が付属していても、重みの読込を意味しない。初期値CLIは重みのダウンロード・CIFAR-10読込・GPU使用を行わない。

保存先は`artifacts/init/convnextv2_tiny_scratch/theta_0.pt`と同名JSON。`.pt`はtensorとprimitive containerで表すmodel_stateとmetadataを含む。metadataにはschema v2・kind・epoch 0・global_step 0、model設定、`initialization.mode=scratch`・init seed・モデル既定初期化という方式・`pretrained=false`、前処理、parameter/buffer/stateのlayout、作成時設定とhash、runtimeを記録する。`pretrained_reference=null`とし、timmの既定download URLを読込元と誤記しない。

JSONは重みの保存・close後、サイズ・SHA-256を付けて最後に保存する。全runは同じ完成した初期値を復元する。既存・部分保存・破損は上書き・修復せず、初期化方式・model・runtime・layout・前処理・hashの不一致も拒否する。旧schema v1のDINO記録は新契約として受け付けない。ロードは信頼済みの自プロジェクト成果物へ`map_location="cpu", weights_only=True`を使用する。

`create_init_checkpoint.py --verify-only`は生成を行わず、保存済み記録を再読込する。作成・復元とも`pretrained_fetch_requested=false`を表示する。unit testにはbackbone/headの既定初期化、全batch/seedの初期値共有、事前学習指定の拒否、runtime・破損拒否を含める。変更後の実行検証はユーザーが行う。

**S-01のユーザー検証結果（2026-09-01）:** 実モデルは27,874,186 parameter、CPU FP32、init seed 0のscratchとして作成・再読込が成功し、SHA-256も一致。保存された評価前処理は224×224・center crop・crop_pct 0.875・bicubic。完全なhash・runtime・normalizationは[TODOのS-01実行記録](TODO.md#3-phase-0-実データによる短い確認)を参照する。これはGPU学習や再開の成立性を確認した結果ではない。

**Phase 3のみ:** [純粋なFCMAE checkpoint](https://huggingface.co/timm/convnextv2_tiny.fcmae)を共通SSL backboneとして使用する案。教師ありfine-tuning済みの`fcmae_ft_in1k`等とは区別する。別の共通head・初期値・run系列を作り、Phase 1の学習済み重みは使わない。Phase 3の設定・取得・初期化・学習はU-01で詳細を確定してから実装する。現在のPhase 0・1 CLIにpretrainedモードは設けない。

---

# 7. Checkpoint Format

**保存・再開の確定事項（D-05の一部、2026-08-31）:** training resume 用checkpointとanalysis 用checkpointを分ける。以下の保存内容・頻度・再開位置をPhase 0とPhase 1に適用する。

**保存容量の方針（D-05の一部確定）:** 初期状態と毎epochの解析用checkpointをすべて保持し、容量節約を目的とした間引き・精度低下・既存成果物の自動削除は行わない。再開用checkpointも既存の保存済みファイルを上書き・自動削除しない。設計上の容量上限を設けないことは、保存先の空き容量が無限にあることを意味しない。

**記録タイミング（D-02・D-05確定）:** 共通初期状態をepoch 0・`global_step = 0`として解析用に記録し、以降の解析用checkpoint・指標・再開用checkpointは全期間1 epoch単位、毎epoch終了時に記録する。Phase 0も同じ間隔とする。epochは完了したepoch数、`global_step`は累積optimizer更新回数を表す。

**Phase 2の平均用採取:** SWA/FGE共通の4epoch三角周期で最低LRとなるepoch 82・86・90・94・98終了時のcheckpointを、毎epochの記録から選ぶ。半epochでの追加保存は行わない。

## Resume checkpoint

毎epoch終了時に、同じepoch境界に対応する以下の学習状態を保存する。

- model_state（重みとbuffer）
- optimizer_state
- scheduler_state
- scaler_state（使用している場合）
- epoch（完了したepoch数）
- global_step（累積optimizer更新回数）
- config snapshot
- 乱数状態（Python random、NumPy、PyTorch CPU、使用するCUDA device）
- データ順序の状態（DataLoader / samplerのgenerator状態など、次epochの順序を復元するために必要な情報）

再開時はこれらの状態を復元し、最後に保存を完了したepochが`e`なら次のepoch `e + 1`から学習する。epochと`global_step`は継続値とし、0へ戻さない。epoch途中のbatchからは再開せず、中断したepochは先頭からやり直す。解析用checkpointだけではoptimizerや乱数状態が不足するため、学習再開には再開用checkpointを使う。

保存するのはseed値だけでなく、その時点の乱数・データ順序の状態とする。workerはepochごとに作り直し、評価の乱数を隔離する。保存完了manifestを最後に記録し、再開は新しいsegmentに分離する。詳細は22節。中断しない実行との一致は未検証であり、Phase 0で確認する。

## Analysis checkpoint

- model_state（parameterはFP32、bufferは元のdtypeと値を保持）
- epoch
- global_step
- train metrics（学習中の指標と固定train subsetでの実checkpoint評価を区別）
- validation metrics（validation全体での実checkpoint評価）
- LR
- gradient norm

解析用checkpointはCPU tensorへ移し、FP32のparameterを`.pt`形式で保存する。共通初期状態（epoch 0）も同じ形式とする。学習時のbf16 AMPとは分け、保存のためにparameterをbf16 / FP16へ変換しない。

解析用にはoptimizer・scheduler・乱数・データ順序の状態を含めない。これらは再開用で保持する。`safetensors`を選択肢とする草案はPhase 1では採用しない。checkpointの保存dtypeがFP32であることは、PCA内部の計算dtypeを確定するものではない。

---

# 8. Parameter Vector Utilities

loss landscape 処理の基盤。

実装必須:

```python
def flatten_parameters(
    state_dict: Mapping[str, torch.Tensor],
    parameter_names: Sequence[str],
) -> torch.Tensor:
    ...
```

```python
def assign_parameter_vector(
    model: nn.Module,
    vector: torch.Tensor,
    parameter_spec: ParameterSpec,
) -> None:
    ...
```

`ParameterSpec` に

- name
- shape
- numel
- dtype

を保持する。

BatchNorm running stats 等の buffer は parameter vector へ含めない。

ConvNeXt V2の標準構造にはBatchNormを使わないが、bufferの扱いと実モデルのlayoutは明示的に記録・検証する。

---

# 9. Projection

## PCA

**D-03確定:** 対象checkpoint、共通平均 \(\mu\)、直交単位基底 \(v_1,v_2\)、射影残差の定義は[実験計画のPhase 1射影仕様](EXPERIMENT_PLAN.md#92-projection)に従う。元論文を参照する部分と今回の設計判断は同節に区別して記録する。

比較対象は3 runs版と9 runs版を分け、各runのepoch 0から最終epochまでの全記録点を使う。全runに同じ原点・基底を適用する。対象を追加して再計算するときは、別のprojection識別子で保存する。

### メモリ方針

- checkpointを順次読み込み、分割処理で共通平均・PCA・座標・残差を計算する。
- 全checkpointのparameter vectorをRAMまたはGPUに同時常駐させない。PCA計算時の行列コピーや作業配列も見積もりに含める。
- メモリ節約のために対象checkpointを100件以下へ間引く案は採用しない。
- 保存容量の上限は設けないが、RAM・VRAMの制約は維持する。22節の分割Gram行列方式、FP64、16,384 parameter/blockを採用する。IncrementalPCA・ランダム近似・checkpointの間引きは使わない。

容量は新しい初期モデルのparameter数Pと、比較run数R・全期間Eから見積もる。FP32行列本体は4×P×R×(E+1) bytesで、checkpoint・作業配列は別に必要。旧DINOの21.7M parameter・50epochの見積もりは流用しない。行列をRAMへ一括ロードせず、必要容量・ピークRAM・所要時間はConvNeXt V2のPhase 0で実測する。保存容量上限や自動削除は設けない。

### 保存する内容

`ProjectionArtifact`

- projection識別子
- mean parameter vector（共通原点 \(\mu\)）
- pc1
- pc2
- explained variance ratio
- 対象run・epoch・checkpointの対応と使用順序
- coordinates for all checkpoints
- 各checkpointの射影残差 \(\|w_k-\mu-Vz_k\|_2\)
- parameter specification hash
- PCA計算のdtype・solverなど、再現に必要な設定

損失平面には、このartifactの共通平均・基底をそのまま渡す。保存済みの原点・基底を別のcheckpoint由来のものに置き換えない。

---

# 10. Loss Landscape Evaluation

**D-04確定（Phase 0・Phase 1）:** train由来の主背景とvalidation由来の補助背景を、同じPCA artifact・21×21格子で評価する。各splitから各クラス100件ずつ、計1,000件の固定subsetを使い、元split・index・抽出条件を保存する。testは使わない。

背景・実checkpointともにモデル指定の評価用前処理を使い、224×224入力で解決した具体値を保存する。学習用augmentationは適用しない。評価時のparameterと入力はFP32とし、評価モード・勾配計算なしで実行する。AMP・TF32は無効にし、学習時のbf16設定とは分離する。

## Core API

```python
def evaluate_loss_surface(
    model: nn.Module,
    reference_vector: torch.Tensor,
    direction_1: torch.Tensor,
    direction_2: torch.Tensor,
    x_values: np.ndarray,
    y_values: np.ndarray,
    dataloader: DataLoader,
    device: torch.device,
) -> LossSurface:
    ...
```

`LossSurface`:

- x grid
- y grid
- mean loss grid
- optional accuracy grid
- coordinate metadata
- dataset subset hash
- 背景の元split（train / validation）・subsetの件数とindexへの参照
- 解決済みの評価用前処理・FP32・AMP/TF32無効の設定

## Efficiency

- landscape subsetはtrain・validationそれぞれ固定1,000 samples
- 各格子点のlossは、各画像のCross Entropyを件数で平均する
- model.eval()とinference modeを使い、AMP・TF32は無効
- model allocation を grid point ごとにやり直さない
- parameter vector の assign のみ行う
- Phase 1では両背景とも21×21点を使い、各441点・計882点を事前計算する。31×31への拡大は行わない
- train・validationの損失格子を区別して保存し、projection識別子・x/y座標を共有する
- 追加の学習runは行わず、背景データの違いでPCAや軌跡を計算し直さない

**V-03実装・検証済み:** `loss_surface.py`は完成projectionのmanifestに列挙された全fileのsize/hashを、NumPy配列を開く前に照合する。現在のtheta_0・parameter layout・buffer・epoch 0・評価前処理・設定hashもprojectionと照合する。評価用前処理を一度だけ適用したtrain/validation batchをCPUに保持し、各格子点でFP64保存値から平面上のvectorを組み立ててFP32へ変換する。modelは一度だけGPUへ置き、各点でparameterを一度割り当てた後に両subsetを順に評価する。終了・例外のどちらでも呼出前のparameterとmodule modeを復元する。初回GPU実行でpin-memoryが乱数snapshot後にCUDAを初期化する順序不整合を検出し、CUDA初期化をcache前へ移した後、ユーザー実行の80件と実成果物で確認した。

両背景は`artifacts/surfaces/<projection_id>/`にまとめ、`x_values.npy`、`y_values.npy`、両splitのloss/accuracy、20区間用の`color_levels.npy`、`checkpoint_metrics.json`、`metadata.json`、最後に`complete.json`を保存する。`checkpoint_metrics.json`のvalidationは元の高次元checkpointで測った全5,000件、背景の`validation_loss.npy`は固定1,000件であり、scopeを分けて記録する。同名directoryは完成・未完成を問わず上書きや補修を行わない。再計算には別projection IDを使う。

## Actual checkpoint evaluation

- 初期状態（epoch 0）と毎epoch終了時の実checkpointを、同じtrain subsetおよびvalidation全体で評価し、loss / accuracyを保存する。
- 実測値は平面への射影・復元を行っていない元checkpointから求める。格子上の損失や、その補間値を使わない。
- validation背景の固定1,000件と、validation実測値の対象であるvalidation全体を区別する。
- 学習中の指標は、更新中のモデル・学習用augmentationによる記録として別に扱う。再開用に保存する状態は7節、評価による学習乱数への干渉防止は22節に従う。

---

# 11. Linear Interpolation

```python
def interpolate_vectors(
    vector_a: torch.Tensor,
    vector_b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    ...
```

```python
def evaluate_linear_path(...) -> InterpolationResult:
    ...
```

alpha は default 21 points。

結果に

- alpha
- loss
- accuracy
- endpoint losses
- barrier

を保存する。

barrier 定義の案は [実験計画の Connectivity evaluation](EXPERIMENT_PLAN.md#75-connectivity-evaluation) を参照する。

---

# 12. SWA

**採用済み・未実装:** FGEと同じ学習記録を使う。epoch 80終了時の共通重み・AdamW・乱数状態を保持し、1e-3 → 1e-5 → 1e-3の4epoch三角周期を5回、epoch 100終了まで行う。update単位のLR定義や評価・射影の残る詳細はA-01で確定する。[共通条件](EXPERIMENT_PLAN.md#73-branches)

両方式とも各周期中央の最低LR地点、epoch 82・86・90・94・98終了時の5点を平均対象とする。SWAは等重みの重み平均であり、平均モデルを学習側へ戻さず、optimizer状態も平均・リセットしない。running averageは採取時だけ更新する。採取間は前の平均を保持し、最初の採取前は未定義とする。通常記録・平均用checkpointの採取・再開はいずれも整数epoch境界で扱う。

PyTorch の SWA utility を利用してよい。

ただし analysis 用に

- raw checkpoint vectors
- running average vector

を毎回保存または再構成できるようにする。

Animation では raw trajectory と running-average trajectory を別系列として描く。

---

# 13. FGE

FGEはSWAと共通のcyclic LR系列から同じ5点を参照する。SWA用とFGE用に学習を二重実行しない。

最低限 metadata:

- cycle index
- position within cycle
- current LR
- snapshot flag

prediction ensemble evaluatorとweight average evaluatorを別関数にする。同じcheckpoint ID・順序・等重みを使い、前者はsoftmax後の予測確率平均、後者はSWAと同一の重み平均として評価する。FGE weight averageを独立した別手法として数えない。予測ensembleには単一の重み座標を割り当てない。

---

# 14. Model Soup

実装:

```python
def average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    ...
```

整数 buffer を誤って平均しない。

ConvNeXt V2の実際のparameter/buffer layoutに従い、浮動小数点stateと整数bufferを区別する。

Greedy Soup:

1. candidate を validation accuracy descending で sort
2. best model で開始
3. candidate を平均に追加
4. validation metric が改善した場合のみ accept
5. test set は selection に使用しない

Tie-breaking rule を固定する。

---

# 15. Animation

`matplotlib.animation.FuncAnimation` を基本とする。

## Requirements

- contour は固定背景
- axis limits は動画中固定
- trajectory history を残す
- current point を明示
- phase ごとに同じ coordinate system
- Phase 1は共通のepoch時間軸で比較し、各保存点のoptimizer stepも併記する
- 保存点間の線を、epoch内の実際の経路や更新単位の揺らぎとして扱わない
- frame ごとに重い model evaluation をしない
- animation は事前計算済み coordinate / metrics のみ参照
- Phase 1はtrain背景の主版とvalidation背景の補助版を別GIFにし、原点・基底・格子範囲・軌跡・epochの進行を共有する
- 対になるGIFは両格子の損失範囲を覆う共通の色尺度を使い、背景のsplit・件数・平面上の損失であることを表示する
- 実checkpointのtrain-subset指標とvalidation全体の指標を表示し、背景損失や学習中のtrain指標と混同しない

## Output

- GIF形式で出力する
- 各アニメーションファイルを3 MB以下にする
- Phase 1は既存の出力名をtrain背景の主版に使い、`_val.gif`をvalidation背景の補助版に使う。[出力一覧](EXPERIMENT_PLAN.md#10-animation-requirements)に従い、最小版で2ファイル、全9 runs版で8ファイルを生成する

## Animation data structure

```python
@dataclass(frozen=True)
class TrajectoryFrame:
    step: int
    epoch: float
    x: float
    y: float
    learning_rate: float | None
    train_subset_loss: float
    train_subset_accuracy: float
    validation_loss: float
    validation_accuracy: float
    gradient_norm: float | None
```

---

# 16. Logging

各segmentに`metrics.csv`を保存する。各epochの`metrics.json`と保存完了manifestを正本とし、CSVは完了epochだけの一覧とする。再開時に既存CSVを修正・切り詰めせず、新segmentで継続する。

D-02に従い、指標は毎epoch終了時に、対応する解析用checkpointと同じepoch・`global_step`で記録する。更新ごとのログ保存は行わない。D-04の実checkpoint評価値はepoch 0でも保存する。

`train_loss` / `train_accuracy`は学習中のepoch単位の指標、`train_subset_loss` / `train_subset_accuracy`は固定1,000件での実checkpoint評価、`val_loss` / `val_accuracy`はvalidation全体での実checkpoint評価として区別する。lossは画像数で重み付けした平均、accuracyは正解数/画像数（0〜1）。gradient normは各更新の全parameter勾配L2 normの更新回数平均。epoch 0の学習中の指標・gradient normはJSONでnull、CSVで空欄、動画でN/Aとし、0では埋めない。詳細は22節。

必須列:

- global_step
- epoch
- train_loss
- train_accuracy
- train_subset_loss
- train_subset_accuracy
- val_loss
- val_accuracy
- learning_rate
- gradient_norm
- batch_size
- seed

Phase 1ではTensorBoard・Parquetを追加しない。数値は丸めずに保存し、表示時だけ丸める。必須列にrun_id、segment_id、parameter_displacement、learning_rate_next、epoch_secondsを追加する。

---

# 17. Tests

Phase 1で用意する検証は、設定の不正値・重複キー・パス解決・既存成果物の保護、初期状態の共有、checkpoint復元、決定的な評価、epoch境界からの再開、parameter roundtrip、既知の低rank行列でのPCAとする。Agentは実行せず、ユーザーが実行する。以下の補間・平均の検証はPhase 2・3用のDraftとして残す。

## Parameter vector roundtrip

`state_dict -> vector -> model` 後に parameter が一致。

## Linear interpolation endpoints

- alpha=0 → model A と一致
- alpha=1 → model B と一致

## Average identity

同一 model を K 個平均したら元 model と一致。

## Checkpoint roundtrip

save/load 後に evaluation metric が一致。

## Projection reproducibility

同じ artifact から同じ coordinate が得られる。

---

# 18. Execution Order for Codex

実装順序を守ること。

## Milestone 1

- config
- CIFAR-10 split
- shared scratch initial-model loader（Phase 3のSSL loaderは後段）
- theta_0 creation
- training
- checkpointing
- evaluation

## Milestone 2

- parameter vectorization
- checkpoint trajectory extraction
- PCA projection

## Milestone 3

- 2D loss surface
- Phase 1 animation

**ここで Phase 1 を実際に完走させる。**

## Milestone 4

- interpolation evaluator
- SWA
- FGE
- Phase 2 animation

## Milestone 5

- multi-run comparison
- uniform soup
- greedy soup
- barrier matrix
- Phase 3 animation

---

# 19. Do Not Do Initially

初期実装では不要:

- Hessian 全計算
- mode connectivity curve optimization
- filter-normalized random directions
- FCMAE等の自己教師あり事前学習そのもの
- distributed training
- wandb dependency
- overly generic framework abstraction
- multiple datasets
- multiple architectures

必要になったら後から追加する。

---

# 20. Definition of Done

## Phase 1 done

- 最小版はseed 0の3 runs、Phase 1完了時は3 batch sizes x 3 seedsの実記録が揃う
- 共通 theta_0 を使用
- trajectory coordinates が生成される
- train・validationそれぞれのloss surfaceが共通座標・格子で生成される
- seedごとのbatch comparisonと全runのsummaryを、train背景の主版・validation背景の補助版で生成し、各GIFが3 MB以下である
- metrics が保存される
- 保存済みartifactだけで同じ動画を再描画でき、寄与率・射影残差と再現手順が記録される
- 精度改善・batch間の軌跡差・寄与率の高さは合否条件にしない。低い寄与率は解釈の制限として表示する
- 補間評価はPhase 2の準備に分離し、Phase 1完了の条件にはしない

## Phase 2 done

- スクラッチ学習の確定したbranch pointからNormal/SWA/FGEを再現可能
- SWA running-average trajectory が可視化される
- FGE cycles が可視化される
- interpolation と performance comparison が保存される

## Phase 3 done

- multiple fine-tuning trajectories が可視化される
- pairwise barrier matrix が生成される
- uniform/greedy soup が評価される
- soup formation animation が生成される

全実験について config と git commit hash を artifact metadata に残す。

---

# 21. Environment Check (2026-08-31)

**旧モデルの参考記録:** 以下はDINO構造での過去の環境確認であり、ConvNeXt V2の学習・メモリ保証ではない。2026-09-01の方針転換後は新モデルで再測定する。DINO初期重み・run成果物はユーザー指示で削除済み。

この節は設計案の確定事項ではなく、ユーザーがコマンド実行を許可したうえで行った環境確認の記録である。実データによる学習・性能評価・収束確認は行っていない。

## Hardware and environment

| 項目 | 確認結果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| VRAM | 16,376 MiB（約16 GiB） |
| NVIDIA driver | 591.86 |
| RAM | 合計31.85 GiB、確認時の利用可能量18.93 GiB |
| Cドライブ空き容量 | 確認時約149.6 GiB |
| 実測に使用したPython | `C:\Users\PC_User\anaconda3\envs\torch_env\python.exe`（Python 3.13.2） |
| PyTorch / CUDA runtime | `2.6.0+cu126` / `12.6` |
| torchvision / timm | `0.21.0+cu126` / `1.0.24` |

既定の`python`は別のPython 3.14.3を指していた。メモリ確認には既存のAnaconda `torch_env` を明示して使用し、パッケージの追加・更新は行っていない。空き容量は観測時点の値であり、実行時には変動する。

## B256 memory probe

- モデル: `vit_small_patch16_224.dino`、`num_classes=10`、`pretrained=False`。
- パラメータ数: 21,669,514。全パラメータを更新対象とした。
- 入力: GPU上のランダム画像 `[256, 3, 224, 224]` とランダムな10クラスのラベル。
- 精度: パラメータはfloat32、forwardはbf16 autocast。
- Attention: timmの`fused_attention=True`。実際のCUDAカーネル種別までは測定していない。
- Optimizer: AdamW、LR `1e-4`、weight decay `0.05`。
- 処理: `train()` モードでforward、Cross Entropy、backward、`optimizer.step()`を3回。各回でCUDA同期を行い、AdamWの状態作成を含めてピークを測定。
- gradient accumulation / activation checkpointing / torch.compileは使用していない。

| 指標 | 結果 |
| --- | --- |
| 3回の更新 | 完了。各回のlossが有限であることを確認 |
| PyTorch peak allocated | 8.65 GiB |
| PyTorch peak reserved | 8.88 GiB |
| 3回目の更新後、プロセス終了前の空きVRAM | 約5.80 GiB |
| 更新1回目の時間 | 約0.914秒 |
| 更新2・3回目の時間 | それぞれ約0.200秒 |

**この測定条件では、224×224入力のB256がVRAM内に収まった。** PyTorchのallocated / reservedはGPU全体の使用量とは異なる。時間はGPUに配置済みの同じ合成入力による短い測定であり、DataLoader、画像変換、validation、checkpoint保存などを含む実験全体の所要時間へそのまま換算しない。

事前学習重みとCIFAR-10データはダウンロードしておらず、モデルやcheckpointの保存も行っていない。重みの値が異なる合成データでの確認であり、実際のfine-tuning pipeline全体の動作やメモリ上限を保証するものではない。モデル、入力解像度、精度、Attentionの実装、同時保持するモデル数を変更した場合は再測定が必要になる。

元画像の32×32とモデル候補の224×224については [参考文献S07・S08](REFERENCES.md) を参照する。実験計画・設定例は今回の確認では変更していない。

---

# 22. Phase 0 / 1 Operational Contract (D-05 / D-06)

**改訂（2026-09-01）:** 記録・射影・両背景・GIFの契約を維持し、モデルと初期化はユーザー訂正に従ってConvNeXt V2のスクラッチへ変更する。Phase 0・1はユーザー指定のAdamW・全期間100epoch・固定LR 1e-3、warmupなし。以下は実装契約であり、実行済みの報告ではない。

## Configuration, identity and paths

- 設定schema v3はYAMLの階層とfrozen dataclassの階層を一致させる。model.initializationと全体のinit_seed、training.microbatch_sizeを明示し、旧設定schema v1/v2は拒否する。全項目を必須とし、暗黙の設定継承は使わない。Phase 2・3はこのschemaへ混ぜない。初期モデルのschema v2と保存済みtheta_0は変更しない。
- `training.batch_size`と`--batch-size`は実効バッチ64/256/1024、`training.microbatch_size`は64固定。`Training.accumulation_steps`は両者の比から導出し、独立したYAML項目やCLI引数を設けない。設定確認の出力とrun/segmentのenvironment、再開contractの`batching`へ実効batch・microbatch・蓄積回数・epoch更新数を記録する。
- `configs/phase0.yaml`はB64・seed 0、`phase1.yaml`は共通条件のテンプレート。batch・seed・実験名だけCLIで上書きでき、上書き後の全設定も保存する。異なるrunのLR等をCLIから個別変更する仕組みは設けない。
- Phase 0はseed 0・5epoch停止を固定したまま、実効B64/B256/B1024を許可する。B64はpipeline全体の基本sanity、B256/B1024はaccum 4/16のGPU probeである。probeには`phase0_accum_probe`という別の実験名を使い、Phase 1の100epoch本比較と混同しない。その他のbatch、seed 1/2、停止epochの変更は拒否する。
- `experiment.name`は実験系列名。run_idは`<name>/b<batch>_seed<seed>`。名前には英小文字・数字・`_`・`-`のみを許可し、パス区切り・Windows予約名を拒否する。
- 相対パスは実行時cwdやYAMLの親ではなく、明示したproject root基準。省略時はコードのあるリポジトリルート。`~`・環境変数の展開はしない。成果物をraw data配下へ配置しない。
- I-01の`check_config.py`は既定で読み取りだけ。`--prepare-run`を明示したときだけ新規runの`source.yaml`・`config.json`・`prepared.json`を保存する。既存runには書き込まずエラー。途中失敗のファイルは自動削除しない。
- `config.json`には解決済み絶対パスを含む全設定、`prepared.json`には元YAMLと有効設定のSHA-256、schema version、source pathを記録する。設定の確認はデータ・モデル・GPU・依存ライブラリの動作確認を意味しない。
- 学習開始時に新規`environment.json`へPython・依存version・device・dtype・解決済み前処理・Git commitとdirty状態・ソースの識別情報を記録する。未コミットならcommit hashだけで再現可能とは扱わない。
- split・subsetのindexは`.npz`とJSON metadata、射影の平均・基底・座標・残差はFP64の`.npy` / `.npz`とJSON、格子は`.npz`とJSON、学習状態は`.pt`に分離する。object配列は使わない。
- projection_idは`<実験名>_<比較範囲>_<UTC時刻>_<UUID>`とし、対象run・epoch・checkpointの順序をmanifestに固定する。loss surfaceは同じprojection_id配下で管理する。GIFは再描画をimmutableに残せるよう`<出力名>_<UTC時刻>_<UUID>`のanimation_id配下へ保存し、source projection_idとそのmanifest hashをmetadataに固定する。別projectionの座標を混ぜない。

## Training and the Phase 0 budget

- 2026-09-01、ユーザー指定でAdamW・全期間100epoch・固定LR 1e-3を確定し、既存weight decay 0.05を維持する。`scheduler=constant`、`warmup_epochs=0`を必須とし、Phase 0・1ではcosine・非ゼロwarmup・別のLR値を拒否する。`epochs`は予定期間、`stop_after_epoch`は終了地点。Phase 0は本比較用の100epochを保って5epoch終了で停止する。採用理由と実証範囲は[実験計画4節](EXPERIMENT_PLAN.md#4-scratch-training-recipe)を参照する。
- AdamWは全学習parameterに一様にweight decayを適用し、betas=(0.9, 0.999)、eps=1e-8を維持。層別LR・weight decayの除外groupを追加しない。勾配蓄積1/4/16回を使い、clipping・compile・分散学習は使わない。モデルparameterと蓄積gradientはFP32で維持し、forwardのみbf16 autocast。bf16ではscalerを使わない。
- train DataLoaderは全条件で64件ずつ返す。同一seedではsampler・workerへの割当単位も共通。各実効バッチの先頭だけzero_gradとLR設定を行い、microbatchごとの平均lossに「microbatch画像数/当該実効バッチの実画像数」を掛けてbackwardする。全microbatch後にgradient normを計測してAdamWを1回更新する。activation graphはmicrobatch間で保持せず、全体をGPUへ載せたりretain_graphを使ったりしない。
- 最後の実効バッチはB64で8件、B256で200件（64+64+64+8）、B1024で968件（64×15+8）。端数を名目batch数や固定の蓄積回数で割らず、実画像数で平均する。epoch内の全画像と更新数を検証し、途中の蓄積状態は保存・次epochへ持越ししない。非有限loss/gradientは当該optimizer更新前に停止し、完了epochを公開しない。
- `drop_last=false`でtrain全45,000件を各epochで使う。1epochの更新数をS=ceil(45000 / batch_size)、総更新数T=epochs×Sとする。
- 1epochの更新数はB64: 704、B256: 176、B1024: 44。100epochでは70,400/17,600/4,400更新。共通epochで比較するため更新数は揃わない。Phase 0のB64・5epochは初回からLR 1e-3で3,520更新を行う。
- 1始まりの更新番号j=1〜Tについて常にLR=base_lr=1e-3。optimizer更新前に同じLRを設定し、最後の更新も0にしない。保存は最後に使用したLRと、次回使用するLRを区別する。epoch 0の使用済みLRはnull、次は1e-3。最終epochの使用済みLRは1e-3、次はnull。Phase 0の5epoch停止後は100epochの予定が残るため次は1e-3。
- 学習のRandomResizedCropはscale=(0.08,1.0)、ratio=(3/4,4/3)、HorizontalFlipはp=0.5。補間・normalizationはモデルの評価設定から解決して保存し、全runで固定する。Mixup/CutMix/RandAugmentは無効。
- split_seedとsubset_seedは20260831、モデル全体のinit_seedは0。クラスごとに独立したNumPy Generatorでindexをshuffleし、各クラスの先頭500件をvalidation、残り4500件をtrainへ割り当てる。subsetは各split・クラスの別乱数系列で100件ずつ取り、保存indexは原dataset index順に整列する。分割と抽出は一度だけ行う。
- seed派生はSHA-256(`losslandscape-v1|seed|namespace`)の先頭8byteをbig-endian整数にし、2**63で剰余を取る。namespaceに用途・split・classを含め、Pythonのhash()は使わない。NumPy legacy RNGへ渡す場合だけ2**32で剰余を取る。
- 学習・評価・解析を同時に複数走らせず、GPU jobは1つ。評価はB64・4workers・FP32を起点とする。実データでのOOMを隠して条件を変えず、原因と適用設定を報告する。
- 壁時計時間による自動打ち切り・容量上限は設けない。Phase 0で学習、毎epoch評価、保存、PCA、両格子、描画の時間を分けて測り、本比較前にユーザーへ見積もりを提示する。途中のlossや精度でearly stoppingはしない。NaN/Inf・保存失敗は停止して未完了として残す。
- Phase 0のloss/accuracyは挙動確認に使い、5epochでの精度向上を必須の合否条件にはしない。finiteな更新、保存復元、再開、評価、メモリ、動画の成立を確認する。

## RNG, evaluation and resumption

- Python・NumPy・torch CPU/CUDAの乱数を保存する。train samplerとworker seed用のtorch Generatorは分離し、同じseedのbatch比較では同じsamplerの乱数系列を使う。各Generatorの状態も保存する。
- `persistent_workers=false`、epochごとにworkerを再作成。worker_init_fnでtorch.initial_seed()を元にPython・NumPyのworker乱数を初期化する。epoch境界でworker内部状態を持ち越さない。0workers時は親の保存済み乱数を使う。
- train shuffleの乱数をworker生成や評価に使わない。評価loaderにも独立Generatorを渡し、評価前後で学習のPython・NumPy・torch CPU/CUDA状態を退避・復元する。modelのtrain/eval状態と精度設定もfinallyで戻す。
- deterministic algorithmsを有効、cuDNN benchmarkと学習・評価のTF32を無効にする。学習entry pointはtorch/CUDA初期化前に、その子プロセス内で`CUBLAS_WORKSPACE_CONFIG=:4096:8`を設定する。既存の異なる値は上書きせず不整合を通知する。OSの永続環境設定は変更しない。
- API上の再現性制御は[PyTorch 2.6の再現性資料](https://docs.pytorch.org/docs/2.6/notes/randomness.html)と[DataLoader資料](https://docs.pytorch.org/docs/2.6/data.html)に基づく。再現性の対象は同じ環境・同じ設定。異なるversion・GPU間のbitwise一致は保証しない。
- 新規学習は`segments/<UTC時刻_UUID>/`を作り、各epochを`epochs/epoch_0001/`のような新規directoryへ保存する。解析・再開状態・metrics・metadataの書き込みとcloseを完了してから、ファイルサイズとSHA-256を含む`complete.json`を最後に保存する。manifest自体は一時ファイルから公開し、既存の完成品は置換しない。
- 完了manifestのないepochは解析・再開の対象外。欠損やhash不一致を完了扱いにせず、エラーとして報告する。失敗した書き込みや古いsegmentは上書き・自動削除しない。
- 再開は明示した完了checkpointから新segmentを作り、親segment・epochを記録する。同じepochまでの正本は親、以降は子segmentとする。別branchや失敗途中のCSV行を混ぜない。
- モデル等を構築・ロードした後、次のloader iteratorを作る直前にRNGとGenerator状態を復元する。epochとglobal_stepを引き継ぐ。config・split・初期重み・parameter構造・runtimeの不整合は黙認しない。
- 共通epoch 0の解析重みは`theta_0.pt`と同一にする。I-04では保存容量を制限しない方針に従い、他epochと同じ`analysis.pt`形式で複製保存し、元の`theta_0.pt`のpath・hashも記録する。epoch 0の指標と、各run固有の初期RNG/optimizer/schedulerを持つ再開状態は初期segmentに記録する。これにより最初のepoch中の中断も再開可能にする。
- `.pt`にはtensorとprimitive containerのみを保存する。NumPyの状態はtensorまたはprimitiveへ変換し、任意objectのpickleを避ける。ロードはweights_onlyで、自プロジェクトが生成した信頼済みcheckpointに限定する。

## Metrics and visual meaning

- オンラインtrain lossは各microbatchのスケーリング前の平均loss×実画像数の総和/epoch画像数、accuracyは正解数/画像数。学習中のモデルの記録であり、epoch末モデルのtrain lossとは呼ばない。metricsのbatch_sizeは実効バッチ、global_stepはAdamWの更新回数を表す。CSV/JSONの既存列は維持し、microbatch条件はconfigとenvironmentへ記録する。
- gradient_normは実効バッチ全体の勾配蓄積後・optimizer.step前に全学習parameterの勾配L2 normを計算し、epoch内のAdamW更新回数で平均する。microbatchごとのnorm平均ではない。clipはせず「epoch mean grad L2」として、checkpointで再計算した勾配と区別する。
- epoch 0のオンラインtrain指標・gradient_normはnull / 空欄 / N/A。train-subsetとfull-validationの実測はepoch 0にもある。parameter_displacementは元のFP32 checkpointとtheta_0との差をFP64で集計し、epoch 0は0。
- learning_rateは最後に実際に使った値、epoch 0はnull。learning_rate_nextは次更新予定値、最終完了時はnull。CSV・JSONには丸め前の値を残す。epoch_secondsは学習・評価・保存の内訳をmetadataへ併記する。
- 動画はepoch 0〜最終epochの各点を1frameずつ順番に表示し、5fps、最後のframeを追加1000ms保持する。間の架空checkpointや平滑化は加えない。
- 左に共通座標・固定contour・全履歴・現在点、右に各runのtrain-subset/full-validation lossとaccuracy、epoch平均gradient norm、LR、step。寄与率・現在点の射影残差も表示する。全9run版も各runの値を省略しない。loss推移など詳細は保存CSVから確認できる。
- バッチ条件はOkabe–Ito配色を基礎とし、最小版ではB64=`#D55E00`（vermillion）、B256=`#56B4E9`（sky blue）、B1024=`#CC79A7`（reddish purple）に固定する。各軌跡へ黒の外縁と白の内縁を付け、損失背景の暗部・明部の両方で輪郭を保つ。960px時は線本体1.5 pt、白halo 2.5 pt、黒halo 3.5 pt、現在点7 ptとし、縮小profileでは比率に応じて縮小しつつ視認性のための下限を設ける。3本の軌跡色と黒・白は固定GIF paletteへ予約し、量子化後も保持する。seedは線種・現在点markerで区別する。train/validationの対では同じ色・軌跡・軸・色尺度・時刻を使い、背景のsplit・各1,000件・FP32・plane lossを明記する。
- GIFは960×640・128色を起点とし、必要時は64色、次に幅800、640へ縦横比を維持して再描画する。固定paletteと差分frameを使い、epoch点や必須指標は落とさない。文字に加えて各軌跡が背景の明暗全域で識別できることをV-05で確認し、3,000,000 bytes以下であることを実測する。最小設定でも超過したら未達として報告し、学習や記録の間引きで対処しない。

## PCA and the common grid

- 全checkpointのparameter名・shape・順序をtheta_0のnamed_parameters順に照合し、parameter spec hashを保存する。bufferはPCAから除外し、平面評価はtheta_0のbufferを共有する。Phase 1でbufferの値が変わっていたら、この前提の不成立として停止する。
- N行P列のFP32重み行列を`paths.scratch_root/<projection_id>/`へ置き、parameter方向に16,384要素ずつ読む。全checkpointをRAM/GPUに常駐させない。ディスクの作業領域は自動削除しない。
- ブロックをFP64へ変換して列平均を引き、X_bを作る。G=Σ_b X_b X_b^TをFP64で累積し、対称化したGを`numpy.linalg.eigh`で分解する。固有値の降順で上位2組を取り、v_i=X^T u_i/sqrt(lambda_i)をブロック再読で求める。全データの中心化PCAと同じ代数的定義であり、丸め誤差まで無くなるという意味ではない。
- 固有値と固有vectorのAPIは[NumPy eigh](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eigh.html)を参照。FP64を使うのは近いcheckpointの差とGram計算の数値誤差を抑えるため。rank判定の閾値は1e-10×最大固有値。実質rankが2未満なら架空の軸を作らず、2D表示不能として報告する。
- 最大絶対係数を正にして軸の符号を揃え、同率時はparameterの先頭を採る。V^T Vと復元誤差を検証する。寄与率はlambda_i/trace(G)、座標はXV、残差はブロックごとの(X-XVV^T)の二乗和から求める。平均・基底・座標・残差をFP64で保存する。
- Phase 1は100epoch＋epoch 0で101時点/run。最小3runのN=303、全9runのN=909を全て使う。旧50epochの459行を前提とした見積もりは使わない。
- 909行×16,384要素のFP64ブロックは約114MiB、FP64のGram行列本体は約6.3MiB。旧50epoch時より行数が増えるため、ブロック幅を縮小して作業領域に余裕を持たせる。複数の作業配列と平均・基底を含め、明示的な配列は約2GiB以内を目標に逐次処理する。OS cache・BLAS・Pythonを含むピークRAMの保証値ではなく、実際の対象行数に応じて測定する。Phase 0の6行での測定だけで全9runのピークを保証しない。ブロック幅を小さくする場合も記録点・PCAの定義は変えない。
- x/y範囲は全比較対象の座標min/maxに、各軸の幅の10%ずつ余白を加える。幅が0の軸は中心±1e-6とする。格子は各軸21点の等間隔。色尺度は両背景の全有限lossのmin/maxを覆う20段階の線形尺度とし、外れ値を切り捨てない。一定loss時だけ描画用の微小幅を加え、元値を保存する。
- 3run版と9run版は別projection。subsetは共通。格子範囲を再計算するときは両背景を同じ新IDに保存する。モデル評価は一度で、GIFの再描画では実行しない。

## Completion and verification boundaries

- D-05・D-06の完了は設計と設定契約の整合まで。I-01は設定基盤、I-02〜I-05はデータ・初期重み・学習・評価・保存・検証。まだ実装していない機能を、設定があるだけで実装済みとは扱わない。
- I-01は標準ライブラリと既存PyYAMLを使用。後続は既存torch/torchvision/timm/NumPy/matplotlib/Pillowを起点とし、依存更新はしない。2026-08-31のread-only調査でNumPyの2.1.3と2.4.2のdist-infoが併存していた。実際のimport versionと動作はユーザー実行で確認し、自動修復しない。
- 最小版はseed 0の3 runs・2GIF、Phase 1完了は全9 runs・8GIF・全epochの設定/重み/実測/座標/残差と再現手順。差や精度改善が無くても、記録・評価・表示が成立すれば完了。
- 寄与率が低い場合は図に明記し、3D化・別projection・条件追加を黙って実施しない。checkpoint補間はPhase 2準備に分離する。
- 検証はユーザーが実行。コードの静的確認、unit testの結果、実データでの学習・再開結果、GIFの視認性・容量確認を別々に記録する。

---

# 23. Training and Checkpoint Interfaces (I-04)

**実装済み・勾配蓄積変更後のCPU/GPU成功を受領（2026-09-01）:** 設定確認4件とCPU unit test 64件が成功（11.626秒）。変更後コードのB64・5epochはepoch 0〜5、3,520更新で完了し、旧B64と実測指標が一致。epoch 2からの再開でもepoch 3〜5の実測指標とanalysis/resume/metadataの記録済みSHA-256が元segmentと一致した。B256/B1024もmicrobatch 64・accum 4/16でepoch 0〜5を完了し、peak allocated約7.19 GiB、reserved約7.61 GiB。Agentは小さなJSON/CSV/manifestとfile sizeのみread-onlyで照合し、checkpoint本体のhash再計算やtensor読込はしていない。コード変更後はソース識別が変わるため、確認済みrunへさらに再開しない。

## Entry point and responsibility boundaries

- `run_train.py --config ...`は、既存のCIFAR-10・共有split/JSON・初期重み/JSONを読み込む。自動ダウンロード、splitや初期重みの生成、CPUへのfallback、OOM後のbatch変更はしない。
- `--batch-size`・`--seed`・`--name`はI-01と同じ上書き範囲。`--resume-from`は完了epochのdirectoryを指定する。相対パスはproject root基準。latestの推測や自動再開は設けない。
- 本番deviceは`cuda:0`。torchをimportする前にプロセス内だけで`CUBLAS_WORKSPACE_CONFIG=:4096:8`を設定し、既存の異なる値は拒否する。native bf16が必要で、parameterはFP32、scalerなし。環境version・GPU識別・数値設定を記録する。
- `EpochSchedule`は設定したepochの全期間を表し、Phase 0の停止条件とは分離する。CLIは設定の`scheduler`を明示して渡す。`constant`ではwarmup=0だけを許可し、`apply_next`で毎回同じbase LRを設定する。更新完了時だけ`completed_updates`を進め、最終更新もbase LRを保持し、全期間終了後のnext LRはnull。
- 既存のcosine計算は低水準APIとそのunit testに残すが、Phase 0・1の設定では拒否する。`scheduler_state`はschema v2で`scheduler`種別も保存する。旧schema v1・種別欠落・別種別を推測で補わず、復元前に不一致として拒否する。
- `train_one_epoch`と`run_segment`は`accumulation_steps`を受け取り、本番entry pointが設定の比1/4/16を渡す。低水準CPU fixtureの既定値は1。scheduleのsteps_per_epochはmicrobatch数ではなくceil(train件数/実効batch)。全サンプルを一度ずつ使い、蓄積後の未clip勾配L2 normをFP64で集計して更新回数平均を取る。lossは画像数平均。非有限loss/gradient、欠損gradient、途中で終わるiteratorは完了epochとして保存しない。
- `evaluate`は学習指標と別に実checkpointをFP32で評価する。全moduleのtrain/eval状態、乱数、autocast、TF32、matmul精度、deterministic flagsを例外時も復元する。CLIへの進捗通知も乱数を退避・復元し、通知の有無で学習列を変えない。
- CPUテストは下位の`run_segment`へ小さなモデル・データ・3epoch schedule・FP32を明示して渡す。これは本番設定や比較条件を変更するCLI機能ではない。

## Storage and publication

```text
artifacts/runs/<name>/b<batch>_seed<seed>/
├── source.yaml
├── config.json
├── prepared.json
├── environment.json
└── segments/<UTC時刻_UUID>/
    ├── segment.json
    ├── environment.json
    ├── metrics.csv
    └── epochs/epoch_0000/  # 以降も毎epoch同じ形式
        ├── analysis.pt
        ├── resume.pt
        ├── metrics.json
        ├── metadata.json
        └── complete.json
```

- 未使用の新規runはI-01で設定を保存する。事前に`check_config.py --prepare-run`だけを実行した完全なrunも、その記録を照合して利用できる。既にsegmentがあれば、明示的な再開なしには書き込まない。不完全な準備記録は修復しない。
- `analysis.pt`はschema/kind/epoch/global_step/contract hashとFP32のmodel state。指標は同じ完了manifestに結び付く`metrics.json`を正本とする。bufferは元のdtypeを保ち、初期状態から値が変われば共有bufferを使う平面評価の前提が崩れるため停止する。
- `resume.pt`は同じmodel stateに加え、全config・初期値・split・runtime・ソース識別を含むcontract、AdamW、全期間schedule、RNG、loader generator、scaler=nullを保存する。torchのpickle対象はtensorとprimitiveのみで、復元は信頼済み成果物へ`weights_only=True`を使用する。
- 全ファイルを新規作成し、重みを保存・flush/fsyncした後にサイズ/hashを計算する。JSONを保存してから、一時manifestを同一directory内のhard linkで`complete.json`として公開する。既存の完成品を置換できるrename/replaceへfallbackしない。NTFSなどhard link対応filesystemが必要。成功時の一時manifest名だけを除去し、失敗時の不完全ファイル・既存成果物は保持する。
- `metrics.csv`は完了manifest公開後に追記する補助一覧。CSV追記だけに失敗した場合も、公開済みのepochは完了記録として残る。解析・再開はCSVだけで判断せず、manifestとJSONを使う。既存CSVの切詰め・修復はしない。
- 各epochの実数値は丸めず、accuracyは0〜1。epoch 0のオンラインloss/accuracy/gradient/使用済みLRはnull・CSV空欄。sample数・train/評価/保存時間・CUDA peak allocated/reservedも保存する。
- `checkpoint_seconds`はCPU snapshot・displacement・optimizer複製・重みの保存/fsync/hashを含む。`epoch_seconds`はepoch開始から重みhash完了までで、最後の小さなJSON/manifest公開/CSV追記を除く。測定範囲をmetadataに記録する。CPUテストではVRAM欄を0とする。ピークRAMはこの実装では自動測定せず、Phase 0で別途確認する。

## Resume identity and branch selection

- 再開前に完了manifest、4ファイルのサイズ/hash、schema・epoch・step・metric identityを照合する。全config、split、初期重み、runtime/GPU条件、`src/`・`scripts/`のPythonソースhashが異なれば停止し、黙って継続しない。Git commit/dirty状態と個別ソースhashはenvironmentに記録する。
- 今回の設定schema v3・勾配蓄積への変更は旧B64 runと互換の再開ではない。旧成果物は保持し、新しい確認runには`--name phase0_accum`等の別名を使う。初期モデルv2と共有splitは再利用する。scheduler状態v2、epoch成果物の外側のschema v1は維持する。新コード同士の再開でもmicrobatch/実効batch/蓄積条件を含むcontractが異なれば拒否する。
- 再開先は必ず新segment。親のepoch directory・epoch・完了manifest hashを`segment.json`に保存する。同じrun内の既存checkpointだけを親にできる。
- 新しく作成したmodel/optimizerへ、modelのキー・順序・dtype・shape、AdamWのgroup・LR・moment・step、scheduleの全期間を確認して復元する。最後にloader generatorとPython・NumPy・CPU/CUDA RNGを復元してから、次epochのiteratorを作る。
- `completed_lineage`は指定したsegmentの親をたどり、親の指定epochまでと子の後続だけを返す。完了manifestのないepochを除き、完了列の欠番・重複・循環・親の変化は拒否する。別branch・親の指定epoch以降の成果物は混ぜない。
- CPUテストでは、固定LR・dropoutとPython/NumPy/torchを使うaugmentationを含めて、蓄積あり/なしの各設定で連続実行とepoch 0/途中epochからの復元後の重み・AdamW・LR・乱数・実測値を比較する。別の確率的処理を持たない小モデルでは物理batchとのgradient・AdamW状態・loss/normの近似一致、端数200/968件、後続microbatchの非有限値も検証する。GPUでのbitwise一致を確認したという意味にはしない。
