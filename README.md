# CP4SBI: Local Conformal Calibration of Credible Sets in SBI

[![arXiv](https://img.shields.io/badge/arXiv-2508.17077-b31b1b.svg)](https://arxiv.org/abs/2508.17077)
[![HAL](https://img.shields.io/badge/HAL-05433498-0073a8.svg)](https://dumas.ccsd.cnrs.fr/LJK-PS-STATIFY/hal-05433498v1)

CP4SBI is a post-hoc, model-agnostic framework that integrates local conformal prediction into simulation-based inference (SBI) pipelines. It wraps any trained posterior approximator and produces credible sets with **finite-sample marginal coverage** and **asymptotic conditional coverage** — no retraining required.

---

## The problem

Standard SBI methods (normalizing flows, diffusion models, ABC) often produce miscalibrated posteriors: credible regions fail to achieve their intended coverage levels. CP4SBI fixes this using only a small calibration set simulated from the prior and model.

## Two variants

| Variant | Mechanism | Guarantee |
|---|---|---|
| **LoCart CP4SBI** | Regression tree partition of data space | Finite-sample local + asymptotic conditional coverage |
| **CDF CP4SBI** | Conditional CDF recalibration of conformity scores | Asymptotic conditional coverage as posterior improves |

---

## Features

- Works on top of any trained SBI posterior estimator (NPE, NLE, diffusion models, flow matching)
- Supports HPD regions, symmetric intervals, quantile-based sets, and custom scoring functions
- Handles nuisance parameters and parameter transformations
- Plug-and-play with [`sbi`](https://github.com/sbi-dev/sbi) and other popular SBI libraries

---

## Installation

**pip**
```bash
git clone https://github.com/Monoxido45/CP4SBI.git
cd CP4SBI
pip install .
```

**conda**
```bash
conda create -n cp4sbi_env python=3.9
conda activate cp4sbi_env
git clone https://github.com/Monoxido45/CP4SBI.git
cd CP4SBI
pip install .
```

---

## Citation

```bibtex
@article{cabezas2025cp4sbi,
  title     = {CP4SBI: Local Conformal Calibration of Credible Sets in Simulation-Based Inference},
  author    = {Cabezas, Luben M. C. and Santos, Vagner S. and Ramos, Thiago R. and Rodrigues, Pedro L. C. and Izbicki, Rafael},
  journal   = {arXiv preprint arXiv:2508.17077},
  year      = {2025}
}
```






