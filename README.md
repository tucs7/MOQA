# MOQA

MOQA is a framework for multiobjective optimization of peptide properties using QUBO samplers.

It consists from the binary autencoder that is used for the binary descrete latent space construction. It provides binary encoding of peptide sequences.
MOQA employs the implementation from

* R. Baynazarov and I. Piontkovskaya. Binary autoencoder for text modeling. Communications in Computer and Information Science, 2019

For extraction of peptide sequences with desired properies from the latent space FMQA based approach is used. For details please see 

* K. Kitai, J. Guo, S. Ju, S. Tanaka, K. Tsuda, J. Shiomi, and R. Tamura. Designing metamaterials with quantum annealing and factorization machines. Phys. Rev. Research, 2:013319, Mar 2020

MOQA is extention of FMQA for simultaneous optimization of multiple properties.

# Usage

* For the latent space construction for the own set of sequences run

```
python train.py experiment_configs/binary.json
```

Note, that trained binary autoencoder on the sequnces from

* A. Tucs, D. P. Tran, A. Yumoto, Y. Ito, T. Uzawa and K. Tsuda, Generating ampicillin-level antimicrobial peptides with activity-aware generative adversarial networks, ACS Omega, 2020, 5(36), 22847–22851.

is already provided. So you can proceed directly to the sampling.

* For sampling run

```
python sampler.py
```

In the particular example three peptide properties are optimized. These are charge density, instability index and Boman index. Charge density is maximized, while instability and Boman indexes are minimized. Simulated annealing sampler is exploted for this.
