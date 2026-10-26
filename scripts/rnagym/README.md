# RNAgym weak-supervision experiment

## Full coverage and sequence generalization (2026-10-03)

`full_coverage_experiments.py` uses every official RNAgym train sequence and
every attached observation in its final refits. The source contains 117,773
normalized sequences and 198,411 observations across 2A3, DMS, 1M7, NMIA and
CMCT. Raw profiles and experimental metadata are retained in a compact Parquet
table; reactivity and confidence arrays use memory-mapped NumPy files.

One coverage cycle visits all sequence groups exactly once, including all
observations within each group. The optimizer, random states, replay position,
cycle and permutation cursor survive job boundaries. The native Torch LSTM
dropout implementation is used, and its random state can be
checkpointed; cuDNN convolution remains enabled. The same LSTM implementation
is used for the starting reference and all evaluations. Four chemical sequences
are interleaved with one TrainSetA structural example; the control uses five
TrainSetA examples for the same five sample slots and optimizer step. It has
more structural exposures. A second control uses the same mixed schedule and
IMLE forward/RNG consumption as rank, with its chemical coefficient set to zero;
it matches structural exposure counts and optimizer steps. Each loss averages replicates,
conditions and probes within its sequence; bounded inverse-square-root cluster
weights reduce domination by large sequence groups without dropping members.

The development arms are TrainSetA-only, the matched chemical-loss-zero control,
probe-aware IMLE-rank, and the same
rank loss with a soft chemical shape-FY term. DMS uses A/C and CMCT G/U, with
rank comparisons only within the same base for these two probes. SHAPE probes
use all four bases. Finite negative background-subtracted reactivities remain
observed. Missing positions are masked. SNR weights are bounded; copied error
profiles and negative/nonfinite error entries cannot become inverse-variance
weights. All original experimental conditions are retained, and different
conditions get separate FY teachers. The FY energy is a bounded positive tilt
against pairing at high reactivity; low reactivity never forces pairing.

Official RNAgym 40% identity clusters define the development split. Clusters
with a TrainSetA homology hit are assigned to training. A deterministic 5% of
the remaining clusters are held out in their entirety. The chemical validation
panel has at most 1,024 sequence groups, selected across these held-out clusters;
the panel limit does not limit the training data. A screened TrainSetB subset
has no detected TrainSetA or development-training RNAgym homology hit. MMseqs2
uses 40% identity, 80% coverage of both sequences, and forward nucleotide search
on the probed cores, excluding common adapters. This heuristic screen does not
establish RNA-family independence.

Development uses seed1234 and two complete coverage cycles. Cycle1 or2 is
selected independently per arm on cluster-macro structural F1 when the screened
TrainSetB subset has at least30 sequences. Otherwise the fallback is chemical
cluster-macro Spearman with a TrainSetB F1 guardrail. Cycle zero and partial
coverage checkpoints are ineligible. Development fixes the cycle count before
each arm is refit on **all** RNAgym train with seeds1234/1235/1236. TestSetB is a
previously inspected diagnostic benchmark and never enters selection. A separate
official RNAgym test panel is screened against all RNAgym train and TrainSetA
before scores are computed; it assesses chemical consistency, not ground-truth
base-pair accuracy. Both metrics and full coverage counts are reported.

```bash
.venv/bin/python scripts/rnagym/full_coverage_experiments.py prepare --root WORKFLOW_ROOT
.venv/bin/python scripts/rnagym/full_coverage_experiments.py freeze --root WORKFLOW_ROOT
.venv/bin/python scripts/rnagym/full_coverage_experiments.py submit-preflight --root WORKFLOW_ROOT
# Wait for preflight.json success, then:
.venv/bin/python scripts/rnagym/full_coverage_experiments.py submit --root WORKFLOW_ROOT
```

The downloaded official cluster table is
`data/rnagym/secondary_structure_prediction/cluster_40.parquet`. A static
MMseqs2 executable is needed for preparation (`--mmseqs PATH`). Source hashes,
split sizes, error-field audit and homology commands are saved under `data/`.
The frozen `plan.json` records hyperparameters and implementation hashes;
`locked_refit_plan.json` records development selection before final evaluation.
GPU preflight checks real backward passes and estimates chunk runtime. Workers
require its success marker as well as predecessor chunk completion. Arrays use
at most two concurrent GPUs. Job IDs are recorded in `submissions.jsonl`.
Failed dependency markers stop subsequent training/evaluation rather than
treating a scheduler hold as evidence of success. Existing experiments are
left untouched.

## Short adaptation improvement experiment (2026-10-03)

`rank_improvement_experiments.py` compares the promising short IMLE-rank
adaptation with explicit continuation controls. It starts every run from the
same TrainSetA epoch-100 checkpoint and preserves the existing v2 results.
Stage 1 uses the existing 512 unique RNAgym-train 2A3 sequences and the same
128 separate RNAgym-train validation sequences:

| condition | chemical target | TrainSetA replay | draws / epoch | accumulation |
|---|---|---:|---:|---:|
| rank delta 0.002 / 0.05 / 0.10 | IMLE-rank, percentile teacher | 50% | 1,024 | 4 |
| TrainSetA continuation | BPSEQ only | all draws | 1,024 | 4 |
| matched mixed schedule, chemical weight zero | chemical supervision disabled | 50% | 1,024 | 4 |
| existing shape-FY positive control | SHAPE-guided reference | none | 512 | 8 |
| partial-label shape-FY | reactivity >= 0.8 implies unpaired; other bases free | 50% | 1,024 | 8 |

All seven conditions use seeds 1234/1235/1236, AdamW lr=1e-4,
weight decay=1e-4, gradient clipping=1, structural loss weight=1,
chemical loss weight=0.5 (zero for the matched null condition), and
score-loss weight=0. Each run retains five one-epoch checkpoints. Stopping
is selected from epochs 0/1/2/5 on **TrainSetB macro F1**, with earliest
epoch winning ties. TestSetB is evaluated only after that selection.
The pure TrainSetA control has more structural exposures; the null mixed
control matches the chemical runs' sampling, structural exposure counts,
and total optimizer updates while supplying no chemical gradient.

Stage 2 increases the RNAgym sequence pool to 4,096 while keeping the same
draw and update budgets. Its rank delta is chosen from stage-1 **TrainSetB**
results; TestSetB scores are not used to select it. Chemical validation uses
the corrected symmetric, full-precision BPP evaluator. This pilot does not
submit whole-RNAgym-test evaluation arrays.

```bash
.venv/bin/python scripts/rnagym/rank_improvement_experiments.py prepare --root WORKFLOW_ROOT
.venv/bin/python scripts/rnagym/rank_improvement_experiments.py submit --root WORKFLOW_ROOT
# Continue an interrupted submission without duplicating recorded arrays:
.venv/bin/python scripts/rnagym/rank_improvement_experiments.py resume-submit --root WORKFLOW_ROOT
```

The workflow records its protocol and data hashes in `plan.json`, scheduler
IDs in `submissions.tsv`, and per-run training/validation/selection results
under its own root. Summary intervals are paired sequence bootstrap
intervals conditional on the fixed starting checkpoint, not independent
baseline-training or RNA-family intervals. Family disjointness is not
claimed. Every worker checks predecessor completion markers; a scheduler
dependency alone is not treated as successful training.

## REPORT-based full benchmark

The current benchmark is submitted with `submit_rnagym_benchmark.sh`.  It
uses the `TrainA_only_new_parameter` configuration and epoch-100 checkpoint
specified in `scripts/implicit_mle/REPORT.md`, with the REPORT best settings
for IMLE-NLL, IMLE-rank, and shape-FY.

There are nine pseudo-label conditions: RNAfold-only; assisted RNAfold with
2A3 (called SHAPE here) and DMS; and IMLE-NLL, IMLE-rank, and shape-FY with
2A3 and DMS.  Direction A trains from the model initialization for five
RNAgym epochs and then fine-tunes on TrainSetA for 100 epochs.  Direction B
starts from the existing TrainSetA epoch-100 checkpoint and trains on the
corresponding RNAgym condition for five epochs.

Each RNAgym test evaluation is split into eight array shards.  BPP and BPSEQ
files are created under the job-local temporary directory, converted to
compact NumPy shard archives, and removed before the aggregation job.  The
aggregator reports the RNAgym chemical metrics and TestSetB sequence-macro
and micro Sen/PPV/F1.  Every training/evaluation job requests one `gpu_h` and
`h_rt=23:50:00`.

Submit with the known project group:

```bash
QSUB_GROUP=tga-satolab-gtex bash scripts/rnagym/submit_rnagym_benchmark.sh
```

The workflow root and all dependency job IDs are printed after submission.
The number of test shards can be changed with `RNAGYM_EVAL_SHARDS`; eight is
the default chosen to leave a large margin under the 24-hour limit.

The six-branch scripts and launcher below are retained as the earlier
benchmark workflow for provenance; they do not use the REPORT-based nine
condition setup above.

This directory contains the reproducible RNAgym S40 workflow requested for
mxfold2.  The source data are already under
`data/rnagym/secondary_structure_prediction/`:

* `RMDB_s40_train.parquet` is used for all pretraining branches;
* `RMDB_s40_test.parquet` is held out for chemical-probing evaluation; and
* `/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst` and
  `/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst` are used only for
  structure fine-tuning/evaluation.

Only DMS and 2A3 rows are used.  The six parallel branches are:

| task id | modifier | pretraining target |
|---:|---|---|
| 1 | DMS | RNAfold-only BPSEQ pseudo label |
| 2 | DMS | RNAfold assisted by DMS reactivity |
| 3 | 2A3 | RNAfold-only BPSEQ pseudo label |
| 4 | 2A3 | RNAfold assisted by 2A3 reactivity |
| 5 | DMS | chemical probing only (mxfold2 implicit MLE) |
| 6 | 2A3 | chemical probing only (mxfold2 implicit MLE) |

The assisted-folding conversion is ViennaRNA Deigan `Dm1.8b-0.6`, matching the
local Yamauchi RNAfold setup.  It can be changed with `SHAPE_SLOPE` and
`SHAPE_INTERCEPT` for the preparation job.  The mxfold2 model/network,
optimizer, learning rate, L2 weight, score-loss weight, perturbation, `Mix`,
`Zuker`, and other settings come from
`/gs/bs/tga-satolab-gtex/yamauchi/Pretrain/logs/BestTrainSetA/BestTrainSetA.conf`
and the corresponding BestTrainSetA training script.  Fine-tuning starts from
the latest pretraining checkpoint and uses TrainSetA for 100 epochs by default.

## Submit on TSUBAME

From the repository root:

```bash
QSUB_GROUP=gXXXX PRETRAIN_TOTAL_EPOCHS=20 PRETRAIN_EPOCHS_PER_JOB=1 \
  FT_TOTAL_EPOCHS=100 FT_EPOCHS_PER_JOB=20 \
  bash scripts/rnagym/submit_rnagym_workflow.sh
```

`QSUB_GROUP` must be the TSUBAME project/account group authorized for the
user; the launcher refuses to submit without it so that a production run is
not accidentally sent to the trial queue.

The launcher submits one preparation job, then sequential epoch-segment arrays
for the six pretraining branches, sequential fine-tuning segment arrays, and
one final evaluation array.  Within every array, task IDs 1--6 are independent
and run concurrently when the scheduler permits.  Every job requests exactly
one `gpu_h` and `h_rt=23:50:00`, leaving a small margin below TSUBAME's
24-hour maximum.  Pretraining is split into `PRETRAIN_EPOCHS_PER_JOB` epochs
per job (1 by default), and fine-tuning into `FT_EPOCHS_PER_JOB` epochs per job
(20 by default).  Each segment resumes the previous segment's checkpoint, so
no single job needs to run the full 20/100 epochs.

`loss.log` records seconds per epoch.  The first three pretraining segment
arrays provide the actual GPU throughput; after that measurement,
`PRETRAIN_EPOCHS_PER_JOB` can be increased in a new workflow if several
epochs safely fit under the 24-hour limit.

Useful monitoring commands, replacing `PRETRAIN_JOB_ID` with the ID printed by
the launcher:

```bash
qstat -t -j PRETRAIN_JOB_ID
qacct -j PRETRAIN_JOB_ID
find qsub_logs/rnagym/WORKFLOW_ID/pretrain -name loss.log -print
tail -f qsub_logs/rnagym/WORKFLOW_ID/pretrain/segment-1/1-DMS_rnafold/train/loss.log
```

The job scripts also accept `PROJECT_ROOT`, `PREP_ROOT`, `THREADS`,
`PRETRAIN_TOTAL_EPOCHS`, `PRETRAIN_EPOCHS_PER_JOB`, `FT_TOTAL_EPOCHS`, and
`FT_EPOCHS_PER_JOB` as environment overrides.  `PRETRAIN_EPOCHS` and
`FT_EPOCHS` remain accepted as aliases for the total epoch counts.  The source
parquets and generated per-row files are shared through the repository path,
so all jobs must see that same filesystem.

## Metrics

`evaluate_rnagym.py` reproduces the official RNAgym chemical score convention:
reactivity and predicted unpaired probabilities are clipped to [0, 1], then
MAE, ROC-AUC, MCC, and macro-F1 are reported after independent median
thresholding.  Predictions are obtained from mxfold2 sparse BPP output as
`1 - sum_j P(i,j)`, matching the RNAgym MXfold2 scorer.  Results are written to
`metrics.json` and `chemical_metrics.csv`, with separate DMS, 2A3, and All
rows.

For TestSetB, mxfold2's native BPSEQ comparison additionally writes
sequence-macro and micro `PPV`, `SEN`, and `F1` to the JSON result under
`structure`.  Thus the final evaluation directory for each branch contains
both the RNAgym chemical metrics and TestSetB structure metrics.

## Compare early pretraining checkpoints

To compare fine-tuning from pretraining epochs 1, 2, and 5, use the source
workflow's completed checkpoints:

```bash
QSUB_GROUP=tga-satolab-gtex \
  SOURCE_WORKFLOW_ID=rnagym-20260914-002946 \
  FT_TOTAL_EPOCHS=100 FT_EPOCHS_PER_JOB=20 \
  bash scripts/rnagym/submit_rnagym_checkpoint_comparison.sh
```

The comparison launcher defaults to the deterministic 1/100 RNAgym test
subset in `data/rnagym/prepared/subsets/`.  It contains 4,336 of the 433,539
unique test sequences, selected by the lowest SHA-256 hashes of normalized
sequences.  A different subset can be supplied with `RNAGYM_TEST_FASTA` and
`RNAGYM_TEST_MAP`.

This submits three independent chains, each containing a six-task evaluation
array before fine-tuning, five six-task TrainSetA fine-tuning arrays, and a
six-task evaluation array after fine-tuning.  Each job requests one `gpu_h`
and `h_rt=23:50:00`.  Results are stored under
`qsub_logs/rnagym/<comparison-workflow>/checkpoint-comparison/epoch-{1,2,5}/`.

For a quick local smoke test without generating the full label set:

```bash
python scripts/rnagym/prepare_rnagym.py \
  --train-parquet data/rnagym/secondary_structure_prediction/RMDB_s40_train.parquet \
  --test-parquet data/rnagym/secondary_structure_prediction/RMDB_s40_test.parquet \
  --out /tmp/rnagym-smoke --max-train-rows 4 --max-test-sequences 5 --workers 2
```
