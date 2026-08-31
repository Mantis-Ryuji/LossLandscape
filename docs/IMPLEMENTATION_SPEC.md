# Implementation Specification for Codex

> **検討中（Draft）** — 実装仕様の草案です。本文中の契約、必須要件、実行順序、完了条件はすべて提案であり、現時点で実装を開始する指示ではありません。

[ドキュメント案内に戻る](README.md)

## 1. Scope

この文書は検討用の実装案であり、確定した実装契約ではない。

目的は CIFAR-10 + SSL pretrained ViT を用いて、Phase 1 → Phase 2 → Phase 3 の loss-landscape animation pipeline を構築すること。

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

以下は実装に進む場合の将来案であり、現在のフォルダ構成ではない。現在の配置は [ドキュメント案内](README.md) を参照する。`configs/` 以下の実行設定は、設定案を見直したうえで作成する想定とし、現段階では作成しない。

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
│   ├── base.yaml
│   ├── phase1_b16.yaml
│   ├── phase1_b64.yaml
│   ├── phase1_b256.yaml
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
│       ├── parameters.py
│       ├── projection.py
│       ├── landscape.py
│       ├── interpolation.py
│       ├── averaging.py
│       ├── animation.py
│       └── logging_utils.py
├── scripts/
│   ├── create_init_checkpoint.py
│   ├── run_train.py
│   ├── run_phase2_branch.py
│   ├── compute_projection.py
│   ├── compute_landscape.py
│   ├── compute_interpolations.py
│   ├── evaluate_soups.py
│   └── render_animation.py
├── tests/
│   ├── test_parameters.py
│   ├── test_interpolation.py
│   ├── test_averaging.py
│   ├── test_checkpoint_roundtrip.py
│   └── test_projection.py
└── artifacts/
    ├── splits/
    ├── init/
    ├── runs/
    ├── projections/
    ├── landscapes/
    ├── interpolations/
    ├── soups/
    └── animations/
```

---

# 4. Configuration Contract

`ExperimentConfig` を dataclass として実装する。

最低限:

```python
@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    dataset_root: Path
    output_root: Path
    model_name: str
    init_checkpoint: Path
    image_size: int
    num_classes: int
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    amp_dtype: str
    checkpoint_interval_steps: int
```

Phase 2 用には別 dataclass を追加してよい。

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

`artifacts/splits/cifar10_train_val_seedXXXX.npz`

へ保存する。

全 run で同じ split を使う。

---

# 6. Model Initialization

既定モデル識別子は `vit_small_patch16_224.dino` とする。現行 timm では、例えば以下でロードできる。

```python
model = timm.create_model(
    "vit_small_patch16_224.dino",
    pretrained=True,
    num_classes=10,
)
```

モデル固有の normalization / resize 設定はハードコードせず、可能なら `timm.data.resolve_model_data_config` を用いて取得する。

`create_init_checkpoint.py` は以下のみ行う。

1. SSL pretrained backbone をロード
2. CIFAR-10 用 10-class head を構築
3. head を一度だけ初期化
4. model state を保存
5. metadata を保存

metadata:

```json
{
  "model_name": "...",
  "num_classes": 10,
  "head_seed": 0,
  "pretrained_source": "..."
}
```

すべての training run はこの checkpoint から始める。

---

# 7. Checkpoint Format

training resume 用 checkpoint と analysis 用 lightweight checkpoint を分ける。

## Resume checkpoint

- model_state
- optimizer_state
- scheduler_state
- scaler_state if needed
- epoch
- global_step
- config snapshot

## Analysis checkpoint

- model_state only
- epoch
- global_step
- train metrics
- validation metrics
- LR
- gradient norm

analysis checkpoint は CPU tensor へ移して保存。

可能なら `safetensors` でもよいが、最初は `.pt` でよい。

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

ViT 系を使うため BN 再推定は原則不要だが、buffer の扱いは明示する。

---

# 9. Projection

## PCA

全 checkpoint vector を GPU に同時ロードしない。

推奨手順:

1. checkpoint を1つずつロード
2. `theta - theta_ref` を CPU float32 に変換
3. 必要なら memmap / incremental accumulation
4. PCA を計算

モデルが約 20M parameters の場合、checkpoint 数が多いと行列が大きくなる。

実装初期は checkpoint 数を 100 程度以下に制限して通常 PCA でよい。

必要になれば IncrementalPCA を追加する。

保存:

`ProjectionArtifact`

- reference checkpoint
- pc1
- pc2
- explained variance ratio
- coordinates for all checkpoints
- parameter specification hash

---

# 10. Loss Landscape Evaluation

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

## Efficiency

- landscape subset は固定 1,000 samples
- inference mode
- AMP 使用可
- model allocation を grid point ごとにやり直さない
- parameter vector の assign のみ行う
- coarse 21x21 を先に生成
- 必要な場合のみ 31x31

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

PyTorch の SWA utility を利用してよい。

ただし analysis 用に

- raw checkpoint vectors
- running average vector

を毎回保存または再構成できるようにする。

Animation では raw trajectory と running-average trajectory を別系列として描く。

---

# 13. FGE

FGE branch は cyclic LR を実装する。

最低限 metadata:

- cycle index
- position within cycle
- current LR
- snapshot flag

prediction ensemble evaluator と weight average evaluator を別関数にする。

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

ViT 系を想定し、基本は floating parameter/buffer のみ。

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
- frame ごとに重い model evaluation をしない
- animation は事前計算済み coordinate / metrics のみ参照

## Output

優先:

- MP4 (H.264 via ffmpeg)

fallback:

- GIF

## Animation data structure

```python
@dataclass(frozen=True)
class TrajectoryFrame:
    step: int
    epoch: float
    x: float
    y: float
    learning_rate: float
    validation_loss: float
    validation_accuracy: float
    gradient_norm: float | None
```

---

# 16. Logging

各 run に `metrics.csv` または parquet を保存する。

必須列:

- global_step
- epoch
- train_loss
- train_accuracy
- val_loss
- val_accuracy
- learning_rate
- gradient_norm
- batch_size
- seed

必要なら TensorBoard を追加してよいが、CSV/Parquet を正本とする。

---

# 17. Tests

最低限の unit tests:

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
- pretrained model loader
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
- DINO pretraining itself
- distributed training
- wandb dependency
- overly generic framework abstraction
- multiple datasets
- multiple architectures

必要になったら後から追加する。

---

# 20. Definition of Done

## Phase 1 done

- 3 batch sizes x 3 seeds が実行可能
- 共通 theta_0 を使用
- trajectory coordinates が生成される
- loss surface が生成される
- seed ごとの batch comparison MP4 が生成される
- metrics が保存される

## Phase 2 done

- epoch 30 branch から Normal/SWA/FGE が再現可能
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
