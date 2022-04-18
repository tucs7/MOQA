# MOQA

MOQA is a framework for multiobjective optimization of peptide properties using QUBO samplers.

It consists from the binary autencoder that is used for the binary descrete latent space construction. It provides binary encoding of peptide sequences.
MOQA employs following implementation:

* Ruslan Baynazarov and Irina Piontkovskaya. Binary autoencoder for text modeling. Communications in Computer and Information Science, 2019

For extraction of peptide sequences with desired properies from the latent space FMQA based approach is used. For details please see 

* Koki Kitai, Jiang Guo, Shenghong Ju, Shu Tanaka, Koji Tsuda, Junichiro Shiomi, and Ryo Tamura. Designing metamaterials with quantum annealing and factorization machines. Phys. Rev. Research, 2:013319, Mar 2020

MOQA is extention of FMQA to simultaneous optimization of multiple properties.

# Usage

* In order make a demo run execute

```
python sampler.py
```
