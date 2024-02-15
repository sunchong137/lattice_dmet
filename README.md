Density Matrix Embedding Theory (DMET) for lattice models
=========================================================
A Python code for running DMET simulations for lattice models. 

## Authors: 
Chong Sun <sunchong137@gmail.com>\\
Ushnish Ray

# Features
1. Ground state DMET for Lattice models
2. Finite-temperature DMET simulations
	    
# Installation

## Dependencies
1. PySCF 
2. Block2 for DMET solver and finite T solver.

First clone the repo from GitHub
```bash
git clone git@github.com:sunchong137/latdmet.git
```
Then install via pip
```bash
cd latdmet
pip install -e .
```
Try to import latdmet in python.

# Citing latdmet
Please cite the following [paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.075131):
C. Sun et. al., Phys. Rev. B 101, 075131 (2020)

### Bibtex Format
```
@article{PhysRevB.101.075131,
  title = {Finite-temperature density matrix embedding theory},
  author = {Sun, Chong and Ray, Ushnish and Cui, Zhi-Hao and Stoudenmire, Miles and Ferrero, Michel and Chan, Garnet Kin-Lic},
  journal = {Phys. Rev. B},
  volume = {101},
  issue = {7},
  pages = {075131},
  numpages = {8},
  year = {2020},
  month = {Feb},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.101.075131},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.101.075131}
}
```

