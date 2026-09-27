# PFAS bioaccumulation code and data

Reproducible molecular features, transport-property learning and BAF/BCF prediction. This repository contains runnable source code and input data. Training outputs are generated locally under `outputs/`.

## Install

Use Linux (or WSL) with **Python 3.8.20**. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py check
```

The reference GPU environment uses PyTorch **1.13.1+cu117**. For the matching CUDA wheel, install it before the requirements:

```bash
python -m pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

CPU is supported with `--device cpu`; GPU is selected automatically when available. Training uses Linux file locking. The pinned environment is the reproducibility target, not a claim of compatibility with every Python or PyTorch release.

## Run one complete example

```bash
python run.py example --output outputs/example
```

This starts from input data without fitted checkpoints. It fits one fixed chemical holdout with all nine models, using **40 pretraining, 400 auxiliary and 120 biological epochs**. Expected: **324 predictions**, a zero chemical overlap check, upstream identity-exclusion checks and successful neural-checkpoint reload. Results are in `outputs/example/results_final/`; checks are in `outputs/example/audit/example_validation.json`. This is actual training, not recalculation from saved predictions. Runtime depends strongly on the device.

Choose a new output directory for an independent run. Existing outputs are rejected unless `--resume` is explicitly supplied. Resuming skips completed tasks; it does not establish a fresh replication.

## Run the complete analysis

```bash
python run.py all --output outputs/full
```

This runs, in order: 15 chemical holdouts and six study holdouts; seven similarity holdouts; external source-transfer fits; external chemical-held-out predictions; grouped calibration; metrics and paired comparisons; checkpoint validation. It is substantially more expensive than the example. Use `--resume` with the same directory after an interruption. Numeric results are written to `outputs/full/summary/`. Models and outputs are not committed to this repository.

To regenerate **all 63,816 molecular feature rows** from the supplied SMILES and compare them with the packaged arrays:

```bash
python run.py features
```

The regenerated arrays are written to `outputs/rebuilt_features/`. The routine preserves row order and checks every value, including graph-derived fluorine counts. `check` performs faster sampled feature checks plus full admission and split checks.

## What is included

| Path | Contents |
|---|---|
| `run.py` | Single command-line entry point |
| `code/` | Training, prediction, evaluation, feature regeneration and checks |
| `code/src/` | Model, feature definitions, standardization and training-only preprocessing |
| `data/` | Model inputs, feature arrays, fixed partitions and auxiliary source labels |
| `data/raw/` | Original transport and biological compilation workbooks and ITRC aquatic workbook |
| `data/sources/` | Record qualification ledger, screening counts and exact literature queries |
| `data/README.md` | Data schema, source links, qualification and endpoint definitions |

## Features and fixed settings

The 1,057-dimensional input consists of 14 continuous RDKit descriptors, 1,024 Morgan bits (radius 2), 14 PFAS SMARTS counts, and five counts computed directly from the molecular graph (`n_F`, `n_CF1`–`n_CF4`). **No MACCS keys** are used. Exact descriptor names, SMARTS expressions and feature order are in `code/src/featurize.py`.

Structure grouping uses RDKit `MolFromSmiles` with default sanitization, `GetMolFrags(sanitizeFrags=True)`, the largest carbon-containing fragment, `Uncharger`, and non-isomeric canonical SMILES. Exact isomeric SMILES are retained separately. Archived grouping keys are preserved; current feature regeneration uses RDKit 2024.03.2. Grouping prevents related forms straddling a chemical split; it does not equate their measured properties.

The network has hidden layers 512/256/128 and a 64-unit endpoint head. Adam settings are: pretraining learning rate 0.001, batch 512, dropout 0.30; auxiliary learning rate 0.0003, weight decay 0.001; biological learning rate 0.0005, weight decay 0.0001, dropout 0.25. Auxiliary and biological fits use full training batches and a masked endpoint-averaged squared-error loss. There is **no grid search or test-selected early stopping**. Context vocabularies, median imputation and scaling are fitted only on training inputs. MC-dropout activates Dropout while keeping BatchNorm in evaluation mode.

Chemical holdouts exclude test identities from biological fitting, transport pretraining and auxiliary fitting. Study holdouts test a different question and can retain the same compound from another study. Similarity groups are connected components at Morgan Tanimoto similarity ≥0.70. Explicit task files retain endpoint-specific observations rather than implying identical test populations.

ExtraTrees uses 128 trees, minimum leaf size 3 and max_features 0.5. Histogram gradient boosting uses 100 iterations, 15 leaves and L2 regularization 1. Ridge uses alpha 100. The remaining comparisons are structure-only context, transport-only context, single-task, shuffled auxiliary labels, and a nearest-chemical/context predictor.

## 中文使用说明

先安装依赖，运行 `python run.py check` 检查数据，再运行 `python run.py example --output outputs/example` 从头训练一个完整示例。全部分析使用 `python run.py all --output outputs/full`。代码和数据均使用相对路径，不需要原服务器账号或目录。`data/raw/` 保留原始汇编工作簿；实际纳入训练的 216 条观测及筛选依据见 `data/README.md`。所有预测、模型和核查结果在本地生成于 `outputs/`。
