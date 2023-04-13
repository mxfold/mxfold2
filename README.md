# MXfold2
RNA secondary structure prediction using deep learning with thermodynamic integration

## Installation

### System requirements
* python (>=3.7)
* pytorch (>=1.4)
* C++17 compatible compiler (tested on Apple clang version 12.0.0 and GCC version 7.4.0) (optional)

### Install from wheel

We provide the wheel python packages for several platforms at [the release](https://github.com/mxfold/mxfold2/releases). You can download an appropriate package and install it as follows:

    % pip3 install mxfold2-0.1.2-cp310-cp310-manylinux_2_17_x86_64.whl

### Install from sdist

You can build and install from the source distribution downloaded from [the release](https://github.com/mxfold/mxfold2/releases) as follows:

    % pip3 install mxfold2-0.1.2.tar.gz

To build MXfold2 from the source distribution, you need a C++17 compatible compiler.

## Prediction

You can predict RNA secondary structures of given FASTA-formatted RNA sequences like:

    % mxfold2 predict test.fa
    >DS4440
    GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG
    (((((((........(((((..((((.....))))...)))))...................(((((.......)))))))))))). (24.8)

By default, MXfold2 employs the parameters trained from TrainSetA and TrainSetB (see our paper).

We provide other pre-trained models used in our paper. You can download [``models-0.1.0.tar.gz``](https://github.com/mxfold/mxfold2/releases/download/v0.1.0/models-0.1.0.tar.gz) and extract the pre-trained models from it as follows:

    % tar -zxvf models-0.1.0.tar.gz

Then, you can predict RNA secondary structures of given FASTA-formatted RNA sequences like:

    % mxfold2 predict @./models/TrainSetA.conf test.fa
    >DS4440
    GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG
    (((((((.((....))...........(((((.......))))).(((((......))))).(((((.......)))))))))))). (24.3)

Here, ``./models/TrainSetA.conf`` specifies a lot of parameters including hyper-parameters of DNN models.

## Training

MXfold2 can train its parameters from BPSEQ-formatted RNA sequences. You can also download the datasets used in our paper at [the release](https://github.com/mxfold/mxfold2/releases/tag/v0.1.0). 

    % mxfold2 train --model MixC --param model.pth --save-config model.conf data/TrainSetA.lst

You can specify a lot of model's hyper-parameters. See ``mxfold2 train --help``. In this example, the model's hyper-parameters and the trained parameters are saved in ``model.conf`` and ``model.pth``, respectively.

## Web server

A web server is working at http://www.dna.bio.keio.ac.jp/mxfold2/.


## References

* Sato, K., Akiyama, M., Sakakibara, Y.: RNA secondary structure prediction using deep learning with thermodynamic integration. *Nat Commun* **12**, 941 (2021). https://doi.org/10.1038/s41467-021-21194-4
