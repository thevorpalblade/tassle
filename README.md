# searchSimulation

Code to simulate the acquisition of an axion signal in CASPEr wind

## Installation

### Installation using conda

Create a new environment called 'casper' with the required packages.

    conda create -n casper python=3.7 numba numpy scipy astropy matplotlib
    source activate casper

### Installation using pip

    pip install -r requirements.txt

### Build C extension

    python setup.py install
