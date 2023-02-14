# MOQA

MOQA is a framework for multiobjective optimization of peptide properties using QUBO samplers.

It consists from the binary autencoder that is used for the binary discrete latent space construction. It provides binary encoding for peptide sequences.
MOQA employs the implementation from

* R. Baynazarov and I. Piontkovskaya. Binary autoencoder for text modeling. Communications in Computer and Information Science, 2019

For extraction of peptide sequences with desired properties from the latent space approach similar to FMQA is used. For details please see 

* K. Kitai, J. Guo, S. Ju, S. Tanaka, K. Tsuda, J. Shiomi, and R. Tamura. Designing metamaterials with quantum annealing and factorization machines. Phys. Rev. Research, 2:013319, Mar 2020

MOQA is extension of FMQA for simultaneous optimization of multiple properties.

# Usage

* To train binary autoencoder on the own set of sequences run

```
python train.py experiment_configs/binary.json
```

Note, that the trained binary autoencoder on the peptide sequences from

* A. Tucs, D. P. Tran, A. Yumoto, Y. Ito, T. Uzawa and K. Tsuda, Generating ampicillin-level antimicrobial peptides with activity-aware generative adversarial networks, ACS Omega, 2020, 5(36), 22847–22851.

is already provided here. So you can directly proceed to the sampling.

* For sampling run

```
python sampler.py
```

In the particular example three peptide properties are optimized. These include charge density, instability index and Boman index. Charge density is maximized, while instability and Boman indexes are minimized. Properties are determined using modlAMP package. Simulated annealing QUBO sampler is exploited in this example.
