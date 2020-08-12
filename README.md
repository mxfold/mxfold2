# MXfold2
RNA secondary structure prediction using deep learning with thermodynamic integrations  

## Installation

## System requirements
* python (>=3.6)
* pytorch (>=1.3)
* C++17 compatible compiler (tested on Apple clang version 12.0.0 and GCC version 7.4.0) (optional)
* cmake (>=3.10) (optional)

### Install from wheel

We provide the wheel python packages for several platforms at [the release](https://github.com/keio-bioinformatics/mxfold2/releases). You can download an appropriate package and install it as follows:

    % pip3 install mxfold2-0.1.0-cp38-cp38-macosx_10_15_x86_64.whl

### Install from sdist

You can build and install from the source distribution downloaded from [the release](https://github.com/keio-bioinformatics/mxfold2/releases) as follows:

    % pip3 install mxfold2-0.1.0.tar.gz

TO build MXfold2 from the source distribution, you need a C++17 compatible compiler and cmake.

## Prediction

We provide the pre-trained models at [the release](https://github.com/keio-bioinformatics/mxfold2/releases). You can download ``models-0.1.0.tar.gz`` and extract the pre-trained models from it as follows:

    % tar -zxvf models-0.1.0.tar.gz

Then, you can predict RNA secondary structures of given FASTA-formatted RNA sequences like:

    % mxfold2 predict @./models/TrainSetA.conf test.fa
    >DS4440
    GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG
    (((((((.((....))...........(((((.......))))).(((((......))))).(((((.......)))))))))))). (24.3)

Here, ``./models/TrainSetA.conf`` specifies a lot of parameters including hyper-parameters of DNN models.

## Training

MXfold2 can train its parameters from BPSEQ-formatted RNA sequences. You can also download the datasets used in our manuscript at [the release](https://github.com/keio-bioinformatics/mxfold2/releases). 

    % mxfold2 train --model MixC --param model.pth --save-config model.conf data/TrainSetA.lst

You can specify a lot of model's hyper-parameters. See ``mxfold2 train --help``. In this example, the model's hyper-parameters and the trained parameters are saved in ``model.conf`` and ``model.pth``, respectively.

## Web server

Comming soon.


## References

* Sato, K., Akiyama, M., Sakakibara, Y.: RNA secondary structure prediction using deep learning with thermodynamic integrations,  [preprint](https://www.biorxiv.org/content/10.1101/2020.08.10.244442v1).