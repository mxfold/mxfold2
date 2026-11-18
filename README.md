# MXfold2

MXfold2 predicts RNA secondary structure from sequence. It combines scores from
a neural network with thermodynamic parameters to choose which bases pair.
You can use the bundled model for prediction or train a model with known
structures, chemical probing profiles, or both.

## Installation

MXfold2 requires Python 3.10 or newer and PyTorch 2.4.1 or newer. Building from
source also requires CMake 3.15 or newer and a C++17 compiler.

From a source checkout, create an environment and install the package:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
mxfold2 --help
```

For an editable installation, use `python -m pip install -e .`. If you prefer
a prebuilt wheel, pick one that matches your platform and Python version from
the [GitHub releases](https://github.com/mxfold/mxfold2/releases) and install it
with `python -m pip install /path/to/downloaded.whl`.

## Predict an RNA structure

Create a FASTA file with one or more RNA sequences, then run `predict`:

```sh
cat > example.fa <<'EOF'
>hairpin
GGGAAACCC
EOF

mxfold2 predict example.fa
```

For each sequence, MXfold2 prints the FASTA header, the sequence, and the
predicted structure in dot-bracket notation with its score in parentheses. A
dot represents an unpaired base, and matching parentheses indicate a base
pair. To write a BPSEQ file for each sequence, create an output directory and
run `mxfold2 predict example.fa --bpseq predictions`. Use `--gpu 0` to run
the neural network on the first visible CUDA GPU.

### Bundled models

In version 0.3.0, plain `mxfold2 predict` uses the bundled
`nr80_imle_rank` model. It was pretrained for 10 epochs on RNA secondary
structures clustered at 80% sequence identity, then fine-tuned with IMLE-Rank
for three epochs on Ribonanza and three on pri-miRNA and human mRNA probing
profiles. It accepts FASTA input directly.

To use the previous `all_cdhit` model or the original TrainSetA/B model,
select its configuration explicitly. These paths are relative to the source
checkout:

```sh
mxfold2 predict @mxfold2/models/all_cdhit.conf example.fa
mxfold2 predict @mxfold2/models/TrainSetAB.conf example.fa
```

An `@config.conf` argument loads the model architecture and the path to its
weights. Keep the configuration with the matching `.pth` file.

### Reproduce the default model training

The [mxfold2-data repository](https://github.com/mxfold/mxfold2-data) contains
the prepared datasets, training settings, and a GPU launcher for the 0.3.0
default model. Repository access is required. See the
[default model recipe](https://github.com/mxfold/mxfold2-data/tree/main/default-model)
for the training schedule and how to run or resume it.

Study workflows, individual experiment scripts, and their regression tests
are maintained in the same repository.

## Train on known structures

Put each reference structure in a BPSEQ file and write one file path per line
in `data/train.lst`. BPSEQ records give the 1-based position, the nucleotide,
and the position of its pairing partner. Use `0` for an unpaired base. The
[training guide](docs/training.md#prepare-your-inputs) includes sample files.

This command uses the bundled architecture and trains the neural weights from
scratch. Run the training examples from the source checkout:

```sh
mkdir -p runs/structure
mxfold2 train @mxfold2/models/nr80_imle_rank.conf data/train.lst \
  --gpu 0 --epochs 10 --optimizer AdaBelief --loss-func fy \
  --param runs/structure/model.pth --save-config runs/structure/model.conf \
  --output-dir runs/structure/checkpoints
```

`--param` is the output weights file for `train`. The command also writes the
matching prediction configuration and saves a checkpoint after each epoch.
Use `--gpu -1` for a small example on a machine without CUDA. To predict with the
trained model, run `mxfold2 predict @runs/structure/model.conf example.fa` from
the same working directory.

## Fine-tune on chemical probing profiles

Save each profile as three columns: position, nucleotide, and reactivity.
Include every position, using `-999` for measurements that are missing or
masked. Put one profile path per line in `data/profiles.lst`. The
`--shape` flag reads this format for SHAPE or other compatible reactivity
profiles, such as DMS.

This example initializes from the structure training run above and fine-tunes
with IMLE-Rank. The empty BPSEQ list selects a run that uses only the profiles.

```sh
mkdir -p runs/probing
: > runs/probing/empty.bpseq.lst
mxfold2 train @runs/structure/model.conf runs/probing/empty.bpseq.lst \
  --init-param runs/structure/model.pth \
  --task Implicit_MLE --shape data/profiles.lst \
  --shape-loss-func shape_rank --shape-rank-objective mann_whitney \
  --shape-imle-difference central --shape-imle-normalization rms \
  --gpu 0 --epochs 3 --optimizer AdaBelief --lr 0.0001 --grad-accum-steps 4 \
  --param runs/probing/model.pth --save-config runs/probing/model.conf \
  --output-dir runs/probing/checkpoints
```

Use `mxfold2 predict @runs/probing/model.conf example.fa` to run the fine-tuned
model. See the [training guide](docs/training.md) for combining structures and
profiles, resuming checkpoints, evaluating a model, and using SHAPE
reactivities at prediction time. Run `mxfold2 train --help` or
`mxfold2 predict --help` for the full CLI options.

## Web interface and citation

Try the [MXfold2 web interface](https://ws.sato-lab.org/mxfold2/) for browser-based
prediction. Training data and earlier models are available in the
[release archives](https://github.com/mxfold/mxfold2/releases/tag/v0.2.0).

Sato, K., Akiyama, M. & Sakakibara, Y. RNA secondary structure prediction using
deep learning with thermodynamic integration. *Nature Communications* **12**,
941 (2021). [DOI: 10.1038/s41467-021-21194-4](https://doi.org/10.1038/s41467-021-21194-4)
