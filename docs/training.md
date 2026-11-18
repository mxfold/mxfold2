# Training and evaluating MXfold2 models

This guide covers training from reference structures, fine-tuning from
chemical probing measurements, and evaluating the resulting model. The
[README](../README.md) covers installation and a short prediction example.
Run the commands from the source checkout, with MXfold2 installed in your active
environment. The examples select CUDA device 0 with `--gpu 0`; replace that
with `--gpu -1` to try the small example without CUDA.

## Prepare your inputs

### Reference structures in BPSEQ format

A BPSEQ file contains one RNA. Each row has the nucleotide position, base, and
pairing partner. Number positions from 1 in sequence order. If position `i`
pairs with `j`, give `j` as the partner on row `i` and `i` on row `j`. Use
`0` for unpaired positions. The example below encodes the reference structure
`(((...)))`:

```sh
mkdir -p data/examples
cat > data/examples/hairpin.bpseq <<'EOF'
1 G 9
2 G 8
3 G 7
4 A 0
5 A 0
6 A 0
7 C 3
8 C 2
9 C 1
EOF
printf '%s\n' data/examples/hairpin.bpseq > data/train.lst
```

### Chemical probing profiles

Each profile file contains three whitespace-separated columns per row:
1-based position, nucleotide, and measured reactivity. Keep positions in
sequence order and include rows for missing measurements with reactivity
`-999`. For example:

```sh
mkdir -p data/examples
cat > data/examples/hairpin.shape <<'EOF'
1 G 0.07
2 G 0.12
3 G 0.09
4 A 0.92
5 A 1.15
6 A -999
7 C 0.16
8 C 0.11
9 C 0.05
EOF
printf '%s\n' data/examples/hairpin.shape > data/profiles.lst
```

Use the same three-column layout for other compatible reactivity measurements,
such as DMS. Prepare values so that higher reactivity corresponds to greater
accessibility for the assay, and mask positions where the assay did not report
a usable measurement.

For both kinds of data, write one path per line in the list file. Use absolute
paths or paths relative to the working directory where you invoke MXfold2.
Avoid spaces in the paths, blank lines, and list headers. The toy files above
are suitable for checking the commands. For a real validation split, keep
related sequences and replicates together to reduce overlap across splits.

## Train on reference structures

The [structure training command in the README](../README.md#train-on-known-structures)
uses the `Mix` model with Zuker folding and the Fenchel–Young structural loss
(`--loss-func fy`). Reuse the bundled `.conf` for its architecture. Pass a new
`--param` path after `@config.conf` to save weights in your run directory.

- Add `--test-input data/validation.lst` to evaluate a separate BPSEQ list
  during training.
- Use `--epochs`, `--optimizer`, and `--lr` to set the training length and
  optimizer. `--grad-accum-steps` sets how many examples contribute to an update.
- Use `--checkpoint-every-n-epochs N` to save fewer checkpoints. The final
  epoch is always saved when you set an output directory.

The `.pth` written by `--param` contains the trained model weights.
`--save-config` writes the architecture and weights path for later
prediction. A checkpoint in `--output-dir` additionally contains the epoch
and optimizer state.

## Fine-tune with IMLE-Rank

Use `--task Implicit_MLE --shape data/profiles.lst --shape-loss-func shape_rank`
to train from reactivities. The Mann–Whitney rank objective
(`--shape-rank-objective mann_whitney`) uses the relative order of measured
reactivities within each profile to guide predicted pairedness. The README
example uses central differences and RMS normalization for the implicit
gradient estimate. Add `--init-param` with compatible weights to start from
an existing structure model with a new optimizer.

The positional argument is a BPSEQ list. Give it an empty file for
profile-only training, or a populated list to train on both kinds of
examples. For a separate pool of reference structures with a controlled
sampling fraction, put that list in `--input-2` and set
`--replay-fraction`. The following command initializes from the structure
run in the README and allocates an expected 20% of the draws to structure
examples:

```sh
mkdir -p runs/joint
: > runs/joint/empty.bpseq.lst
mxfold2 train @runs/structure/model.conf runs/joint/empty.bpseq.lst \
  --init-param runs/structure/model.pth \
  --task Implicit_MLE --shape data/profiles.lst \
  --input-2 data/train.lst --replay-fraction 0.2 \
  --loss-func fy --shape-loss-func shape_rank --shape-rank-objective mann_whitney \
  --shape-imle-difference central --shape-imle-normalization rms \
  --shape-loss-weight 0.5 --second-loss-weight 1.0 \
  --gpu 0 --epochs 3 --optimizer AdaBelief --lr 0.0001 --grad-accum-steps 4 \
  --param runs/joint/model.pth --save-config runs/joint/model.conf \
  --output-dir runs/joint/checkpoints
```

Replay samples with replacement. Set `--samples-per-epoch N` to control the
total number of draws per epoch. The two loss weights scale the profile and
replay contributions independently of the sampling fraction.

For sequential fine-tuning on two profile sets, use the first stage's saved
configuration and weights as the second stage's `@config.conf` and
`--init-param`. Use a new output directory for each stage.

### Resume a run

Use `--resume` with a training checkpoint to continue with its stored
optimizer state. Keep the original data and architecture and set `--epochs`
to the new total. For example, this continues the README structure run
from epoch 10 through epoch 15:

```sh
mxfold2 train @runs/structure/model.conf data/train.lst \
  --resume runs/structure/checkpoints/epoch-10 \
  --gpu 0 --epochs 15 --optimizer AdaBelief --loss-func fy \
  --param runs/structure/model.pth --save-config runs/structure/model.conf \
  --output-dir runs/structure/checkpoints
```

## Evaluate and use a trained model

To predict with a trained model, pass its saved configuration to
`mxfold2 predict`. If a model has been moved, you can point directly to its
weights with `--param` after `@config.conf`:

```sh
mxfold2 predict @runs/joint/model.conf --param runs/joint/model.pth example.fa
```

To compare predicted base pairs against reference structures, pass a held-out
BPSEQ list and request a CSV. The `F` column is per-sequence base-pair F1
(the harmonic mean of precision and recall).

```sh
mxfold2 predict @runs/joint/model.conf data/validation.lst \
  --result runs/joint/validation.csv
```

If you have SHAPE measurements for the input sequence at prediction time,
you can add pseudoenergy restraints using `--shape-file`. For multiple
sequences, use `--shape` with one reactivity file path per line in FASTA order.
Keep the reactivity positions aligned with the full input sequence.

```sh
mxfold2 predict example.fa --shape-file data/examples/hairpin.shape
```

## Bundled model provenance

The 0.3.0 default weights (`nr80_imle_rank.pth`) were pretrained for 10 epochs
on 67,681 structure-labeled RNAs of up to 1,000 nt, clustered at 80% sequence
identity. The collection included curated structures and RNAcentral
pseudolabels, with curated representatives preferred when available. The
model then underwent three epochs of IMLE-Rank fine-tuning on Ribonanza
(45,729 sequences) and three on pri-miRNA and human mRNA (2,471 sequences
combined). The release training run carried optimizer state across phases
and mixed in structure examples at a fixed quota of approximately 20% in each
fine-tuning stage.

The archived datasets, exact settings, and training launcher are in the
[mxfold2-data default model recipe](https://github.com/mxfold/mxfold2-data/tree/main/default-model).
Access to the repository is required.

The previous `all_cdhit.pth` weights come from the
`with_ipknot` branch at commit `40d654f8e`. The accompanying `all_cdhit.conf`
uses the current CLI options for the same architecture. The `TrainSetAB`
model is also available. Pass the matching `.conf` file to select either
model. To find the bundled models in an installed package:

```sh
python -c 'from pathlib import Path; import mxfold2; print(Path(mxfold2.__file__).resolve().parent / "models")'
```
