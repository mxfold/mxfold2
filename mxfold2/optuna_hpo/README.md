# MXFold2 Optuna Hyperparameter Optimization

A module for optimizing mxfold2 train hyperparameters using Optuna.

## Basic Usage

```bash
uv run python -m mxfold2.optuna_hpo.cli \
    --train-data data/TrainSetA.lst \
    --val-data data/TestSetA.lst \
    --shape-data data/TrainSetB-SHAPE-wu.lst \
    --n-trials 100 \
    --epochs 30 \
    --gpu 0 \
    --storage sqlite:///hpo_results/study.db \
    --study-name mxfold2-hpo \
    --output-dir hpo_results
```

## Main Options

| Option | Default | Description |
|--------|---------|-------------|
| `--train-data` | (required) | Training data list file |
| `--val-data` | (required) | Validation data list file |
| `--shape-data` | None | SHAPE data list file(s) (multiple allowed) |
| `--n-trials` | 100 | Number of trials |
| `--epochs` | 30 | Number of epochs per trial |
| `--gpu` | 0 | GPU ID (-1 for CPU) |
| `--storage` | None | Optuna storage URL (for persistence) |
| `--study-name` | mxfold2-hpo | Study name |
| `--output-dir` | ./hpo_results | Output directory |
| `--search-space-config` | None | Search space YAML configuration file |
| `--fixed-params` | None | Fixed parameters JSON file or string |
| `--no-pruning` | False | Disable pruning |
| `--patience` | 50 | Number of trials without improvement before stopping |

## How to Fix Parameters

### Method 1: Limit the range to a single value in YAML config

Like in `configs/hpo/learning_only.yaml`, set `min` and `max` to the same value for parameters you want to fix.

```yaml
# Fixed parameter
num_lstm_layers:
  min: 2
  max: 2

# Optimized parameter
lr:
  min: 0.0001
  max: 0.01
  log: true
```

### Method 2: Specify JSON with `--fixed-params` option

```bash
# Specify a JSON file
--fixed-params configs/hpo/fixed_network.json

# Or specify JSON string directly
--fixed-params '{"num_lstm_layers": 2, "embed_size": 64}'
```

## Fixing Network Layer Structure

To fix hierarchical structure like in `train-all-with-test.sh`, specify in list format:

```json
{
    "num_filters_list": [64, 64, 64, 64, 64, 64, 64, 64],
    "filter_size_list": [5, 3, 5, 3, 5, 3, 5, 3],
    "num_paired_filters_list": [64, 64, 64, 64, 64, 64, 64, 64],
    "paired_filter_size_list": [5, 3, 5, 3, 5, 3, 5, 3]
}
```

This is equivalent to the following shell arguments:
```bash
--num-filters 64 --filter-size 5 --num-filters 64 --filter-size 3 \
--num-filters 64 --filter-size 5 --num-filters 64 --filter-size 3 \
--num-filters 64 --filter-size 5 --num-filters 64 --filter-size 3 \
--num-filters 64 --filter-size 5 --num-filters 64 --filter-size 3
```

## Configuration Files

| File | Description |
|------|-------------|
| `configs/hpo/default.yaml` | Search space for all parameters |
| `configs/hpo/learning_only.yaml` | Fixed network structure, optimize learning + Loss |
| `configs/hpo/optimizer_only.yaml` | Optimize optimizer-related parameters only |
| `configs/hpo/fixed_network.json` | Fixed values for network layer structure |
| `configs/hpo/fixed_all_best.json` | All current best settings fixed |

## Usage Examples

### 1. Optimize all parameters
```bash
uv run python -m mxfold2.optuna_hpo.cli \
    --search-space-config configs/hpo/default.yaml \
    --train-data data/TrainSetA.lst \
    --val-data data/TestSetA.lst \
    --n-trials 100 --epochs 30 --gpu 0
```

### 2. Fix network structure and optimize learning parameters
```bash
uv run python -m mxfold2.optuna_hpo.cli \
    --search-space-config configs/hpo/learning_only.yaml \
    --train-data data/TrainSetA.lst \
    --val-data data/TestSetA.lst \
    --n-trials 100 --epochs 30 --gpu 0
```

### 3. Optimize only some parameters based on current best settings
```bash
uv run python -m mxfold2.optuna_hpo.cli \
    --search-space-config configs/hpo/default.yaml \
    --fixed-params configs/hpo/fixed_all_best.json \
    --train-data data/TrainSetA.lst \
    --val-data data/TestSetA.lst \
    --n-trials 100 --epochs 30 --gpu 0
```

### 4. Run with script
```bash
# Customizable with environment variables
N_TRIALS=200 EPOCHS=50 ./scripts/run_hpo.sh

# Or submit to job scheduler
qsub scripts/run_hpo.sh
```

## Output Files

```
hpo_results/
├── best_params.json           # Best parameters
├── optimization_results.json  # Results summary
├── optimization_history.html  # Optimization history visualization
├── param_importances.html     # Parameter importances
├── parallel_coordinate.html   # Parallel coordinate plot
├── trials.csv                 # All trial results
└── study.db                   # Optuna database
```

## Optimizable Parameters

### Learning Parameters
- `optimizer`: Adam, AdamW, AdaBelief, Lion, SGD, RMSprop
- `lr`: Learning rate (1e-4 ~ 1e-2)
- `scheduler`: None, CyclicLR, CosineAnnealingLR
- `dropout_rate`: Dropout rate (0.0 ~ 0.5)
- `fc_dropout_rate`: Fully-connected layer dropout rate (0.0 ~ 0.5)
- `l2_weight`: L2 regularization (1e-5 ~ 0.1)
- `clip_grad_norm`: Gradient clipping (0.0 ~ 5.0)

### Model Structure
- `embed_size`: Embedding dimension (0, 32, 64, 128)
- `num_filters`: CNN filter count (32 ~ 128)
- `filter_size`: CNN filter size (3, 5, 7)
- `num_lstm_layers`: Number of LSTM layers (0 ~ 3)
- `num_lstm_units`: Number of LSTM units (16 ~ 128)
- `num_att`: Number of attention heads (0, 4, 8)
- `pair_join`: Pair vector joining method (cat, add, mul)

### Loss Functions
- `loss_func`: hinge, fy, f1
- `perturb`: Perturbation (0.1 ~ 1.0)
- `nu`: For F1 loss (0.01 ~ 1.0)
- `score_loss_weight`: Score loss weight (0.0 ~ 0.5)

### SHAPE Loss
- `shape_loss_func`: shape_nll, shape_fy, shape_rank
- `shape_perturb`: SHAPE perturbation (0.1 ~ 1.0)
- `shape_nu`: SHAPE nu (0.1 ~ 10.0)
- `shape_margin`: Margin for shape_rank (0.0 ~ 1.0)

## Notes

- Objective function is **maximization of validation F1 score**
- MedianPruner terminates unpromising trials early
- GPU memory is automatically released after each trial
- Using SQLite storage allows resuming after interruption
