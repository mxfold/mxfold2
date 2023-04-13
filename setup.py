# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['mxfold2', 'mxfold2.fold', 'mxfold2.utils']

package_data = \
{'': ['*'], 'mxfold2': ['models/*', 'src/*', 'src/fold/*', 'src/param/*']}

install_requires = \
['numpy>=1.18,<2.0',
 'pybind11>=2.6.2,<3.0.0',
 'torch>=1.4,<2.0',
 'torchvision>=0,<1',
 'tqdm>=4.40,<5.0',
 'wheel>=0.38.0,<0.39.0']

entry_points = \
{'console_scripts': ['mxfold2 = mxfold2.__main__:main']}

setup_kwargs = {
    'name': 'mxfold2',
    'version': '0.1.2',
    'description': 'RNA secondary structure prediction using deep neural networks with thermodynamic integration',
    'long_description': "# MXfold2\nRNA secondary structure prediction using deep learning with thermodynamic integration\n\n## Installation\n\n### System requirements\n* python (>=3.7)\n* pytorch (>=1.4)\n* C++17 compatible compiler (tested on Apple clang version 12.0.0 and GCC version 7.4.0) (optional)\n\n### Install from wheel\n\nWe provide the wheel python packages for several platforms at [the release](https://github.com/keio-bioinformatics/mxfold2/releases). You can download an appropriate package and install it as follows:\n\n    % pip3 install mxfold2-0.1.1-cp38-cp38-macosx_10_15_x86_64.whl\n\n### Install from sdist\n\nYou can build and install from the source distribution downloaded from [the release](https://github.com/keio-bioinformatics/mxfold2/releases) as follows:\n\n    % pip3 install mxfold2-0.1.1.tar.gz\n\nTo build MXfold2 from the source distribution, you need a C++17 compatible compiler.\n\n## Prediction\n\nYou can predict RNA secondary structures of given FASTA-formatted RNA sequences like:\n\n    % mxfold2 predict test.fa\n    >DS4440\n    GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG\n    (((((((........(((((..((((.....))))...)))))...................(((((.......)))))))))))). (24.8)\n\nBy default, MXfold2 employs the parameters trained from TrainSetA and TrainSetB (see our paper).\n\nWe provide other pre-trained models used in our paper. You can download [``models-0.1.0.tar.gz``](https://github.com/keio-bioinformatics/mxfold2/releases/download/v0.1.0/models-0.1.0.tar.gz) and extract the pre-trained models from it as follows:\n\n    % tar -zxvf models-0.1.0.tar.gz\n\nThen, you can predict RNA secondary structures of given FASTA-formatted RNA sequences like:\n\n    % mxfold2 predict @./models/TrainSetA.conf test.fa\n    >DS4440\n    GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG\n    (((((((.((....))...........(((((.......))))).(((((......))))).(((((.......)))))))))))). (24.3)\n\nHere, ``./models/TrainSetA.conf`` specifies a lot of parameters including hyper-parameters of DNN models.\n\n## Training\n\nMXfold2 can train its parameters from BPSEQ-formatted RNA sequences. You can also download the datasets used in our paper at [the release](https://github.com/keio-bioinformatics/mxfold2/releases/tag/v0.1.0). \n\n    % mxfold2 train --model MixC --param model.pth --save-config model.conf data/TrainSetA.lst\n\nYou can specify a lot of model's hyper-parameters. See ``mxfold2 train --help``. In this example, the model's hyper-parameters and the trained parameters are saved in ``model.conf`` and ``model.pth``, respectively.\n\n## Web server\n\nA web server is working at http://www.dna.bio.keio.ac.jp/mxfold2/.\n\n\n## References\n\n* Sato, K., Akiyama, M., Sakakibara, Y.: RNA secondary structure prediction using deep learning with thermodynamic integration. *Nat Commun* **12**, 941 (2021). https://doi.org/10.1038/s41467-021-21194-4\n",
    'author': 'Kengo Sato',
    'author_email': 'satoken@mail.dendai.ac.jp',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/mxfold/mxfold2',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.7,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
