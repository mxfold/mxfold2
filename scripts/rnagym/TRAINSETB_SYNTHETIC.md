# TrainSetB synthetic probing experiment (2026-10-20)

Experiment root: `qsub_logs/rnagym/trainsetb-synthetic-20261020`.

The full 1,094-sequence TrainSetB is assigned one synthetic profile per probe.
There are 112,398 nucleotide positions; 81 ambiguous bases remain missing.
Calibration uses only the **official train split** of the packed RNAgym data:
95,906 2A3 and 83,887 DMS profiles. Validation and test values do not enter
the calibration. Valid negative reactivities are retained; missing values are
`-999`. DMS values are assigned only to A/C.

The Wu paired GEV / unpaired gamma model from `mxfold2/utils/sim_shape.py`
provides a noisy structure-dependent ordering. An empirical quantile mapping
matches RNAgym's measured distribution separately for each probe and base.
This conditional structure model is an explicit simulation assumption: it is
not a fitted estimate of true paired/unpaired RNAgym distributions. The
within-base paired/unpaired AUC is approximately 0.77. The reference and
generated distributions, quantiles, coverage, and KS statistics are recorded
in `data/simulation_report.json`.

Missingness is uniform within base and independent of structure, with base
rates estimated from RNAgym and rescaled to the desired global coverage.
This approximates the proportion of observed bases, not the positional
patterns, experimental error, or profile-to-profile variation in RNAgym.
Default coverage is 59.976% for 2A3 and 30.461% for DMS, expressed as a
fraction of **all nucleotides**, including probe-inapplicable positions.
Every TrainSetB RNA receives a profile in both synthetic datasets; the
proportion of RNAs that have each probe in the original RNAgym is not copied.

Generate a new dataset with RNAgym coverage:

```bash
.venv/bin/python scripts/rnagym/simulate_trainsetb_reactivity.py \
  --output /absolute/path/to/new-data
```

Set custom coverage (here 50% of all bases for 2A3 and 25% for DMS):

```bash
.venv/bin/python scripts/rnagym/simulate_trainsetb_reactivity.py \
  --coverage-2a3 0.50 --coverage-dms 0.25 --seed 20261020 \
  --output /absolute/path/to/new-data
```

`--coverage` sets a common default for both probes; a probe-specific option
overrides it. Infeasible coverage (e.g. DMS beyond the available A/C fraction)
is rejected. Existing output folders are never overwritten. The output
contains individual SHAPE files, SHAPE file lists, and packed arrays compatible
with `PackedRNAGym`. Structure labels are used by the generator; chemical
training reads only the generated profiles.

The 16 conditions cross IMLE-Rank / SHAPE-FY, 2A3 / DMS, and replay fractions
5%, 10%, 20%, and 50%. Replay is the fraction of total sample slots, so each
epoch contains 1,094 chemical samples plus respectively 58, 122, 274, or
1,094 TrainSetA FY samples. The small upward rounding is recorded in each
condition plan. Replay cycles through deterministic shuffled TrainSetA
coverage orders. Increasing replay also increases total optimizer updates
per epoch; comparisons at equal chemical epochs are not equal-update studies.

All conditions start from the same TrainSetA epoch-100 checkpoint and restore
the exact AdaBelief moments and step 316,600. Learning rate and decoupled
weight decay are both 1e-4; gradient accumulation is 4. Turner parameters are
frozen, and Turner score regularization is 0.125 on both chemical and replay
losses. Chemical loss has outer weight 0.5, replay 1.0. Rank uses perturb=0.1,
nu=0.1, aligned/RMS/central estimation with one sample, and base-stratified
DMS comparisons. SHAPE-FY uses perturb=0.1, slope=2.6, intercept=-0.8. Replay
FY uses perturb=0.75. These loss implementations and optimizer restoration
are imported directly from the preceding full-RNAgym comparison scripts.

Training runs for 100 additional epochs with seed 0. Checkpoints are evaluated
at epochs 0, 10, ..., 100, with no test-based checkpoint selection:

- TestSetB: per-RNA and mean PPV, sensitivity, F1.
- Chemical probing: the same fixed 1,024-RNA RNAgym official test panel,
  reporting per-probe sequence/cluster macro Spearman and the probe macro mean.
- TrainSetB: training-fit metrics, not independent generalization accuracy.

The exact-sequence overlap audit is in `plan.json`. TrainSetA and TrainSetB
share 40 exact sequences. The existing chemical panel was homology-screened
against TrainSetA and RNAgym training sequences, but has not been screened
against TrainSetB homologs. Thus that panel preserves comparability with the
earlier experiment; it does not establish family-independent transfer from
TrainSetB. This is a synthetic positive-control experiment, not evidence by
itself for the effectiveness of training on real chemical measurements.

Source files, compiled folding extension, input structures, datasets,
baseline, and test panel are frozen or hashed. Each condition has a GPU
preflight which checks restored moments, finite nonzero gradients, active
regularization, and unchanged Turner parameters. The source snapshot is used
for every scheduled job.

Each of the 16 array tasks requests `gpu_h=1` with group `tga-satolab-gtex`;
the array concurrency is 16. Ten consecutive 10-epoch jobs use per-task
dependencies, so conditions progress independently in parallel. All 100
epochs and their evaluations are submitted together. A failed dependency
cannot silently skip a segment: the next job requires its predecessor's
completion marker. `submissions.tsv` records the jobs. Scheduler availability
determines actual start times.

Each epoch and every 250 optimizer updates save an atomic resume checkpoint,
including model, optimizer, cursor, and Python/NumPy/Torch/CUDA RNG state.
The checkpoint for each evaluated epoch is retained. To retry a failed task,
run the frozen script with the original root, task ID, and segment number;
the saved cursor is restored. Retries must finish before dependent segments
are resubmitted. The launcher refuses duplicate full-grid submission.

`conditions/<condition>/evaluation_curve.tsv` contains fixed-epoch results;
`runs/progress.json` contains progress. The final dependency job produces
`all_evaluation_curves.tsv` and fails explicitly if any condition is incomplete.
For an interim aggregate, run:

```bash
.venv/bin/python \
  qsub_logs/rnagym/trainsetb-synthetic-20261020/source/scripts/rnagym/trainsetb_synthetic_experiment.py \
  aggregate --root qsub_logs/rnagym/trainsetb-synthetic-20261020
```
