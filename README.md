# CFG-MP & CFG-MP+: Improving Classifier-Free Guidance of Flow Matching via Manifold Projection

<p align="center">
  <a href="https://arxiv.org/abs/2601.21892">
    <img src="https://img.shields.io/badge/arXiv-2601.21892-b31b1b.svg">
  </a>
  <a href="https://icml.cc/">
    <img src="https://img.shields.io/badge/ICML-2026-blue.svg">
  </a>
</p>

This repository provides the official implementation of **CFG-MP** and **CFG-MP+** for the paper:

> **Improving Classifier-Free Guidance of Flow Matching via Manifold Projection**  
> Jian-Feng Cai, Haixia Liu, Zhengyi Su, Chao Wang  
> Accepted to **ICML 2026**

**Paper**: [arXiv:2601.21892](https://arxiv.org/abs/2601.21892)

---

## Overview

Classifier-Free Guidance (CFG) is a widely used technique for improving conditional generation in diffusion and flow-based generative models. However, the standard linear extrapolation form of CFG may introduce a prediction gap, especially under large guidance scales.

We propose **CFG-MP (Manifold Projection)**, a training-free refinement method that formulates the correction of CFG as an **incremental gradient descent scheme** and solves it through a **fixed-point iteration**. To further improve stability and efficiency, we introduce **CFG-MP+**, which incorporates **Anderson Acceleration** into the fixed-point iteration.

CFG-MP/CFG-MP+ can improve generation fidelity and convergence efficiency, and are compatible with large-scale generative models, including:

- **DiT-XL-2-256** for class-to-image generation;
- **Stable Diffusion 3.5** for text-to-image generation.

For an accessible summary of the paper, we also recommend this third-party review on Moonlight:  
[[Literature Review] Improving Classifier-Free Guidance of Flow Matching via Manifold Projection](https://www.themoonlight.io/en/review/improving-classifier-free-guidance-of-flow-matching-via-manifold-projection)

---


## 📂 Repository Structure

The code is organized into two main projects:

* **[`CFG-MP_DiT/`](./CFG-MP_DiT/)**: Implementation for DiT-XL-2-256 (Large-scale Class-to-Image).
* **[`CFG-MP_SD/`](./CFG-MP_SD/)**: Implementation for Stable Diffusion 3.5 (Large-scale Text-to-Image).

---

## 🛠️ Installation & Setup

We recommend using **Miniconda** for environment management. Each project has its own specific dependencies.

### For DiT-XL-2-256 Generation
```bash
cd CFG-MP_DiT
conda create -n cfgmp-dit python=3.10 -y
conda activate cfgmp-dit
pip install -r requirements.txt
```

### For Stable Diffusion 3.5 Generation
```bash
cd CFG-MP_SD
conda create -n cfgmp-sd python=3.10 -y
conda activate cfgmp-sd
pip install -r requirements.txt
```

## 🚀 Usage Guide
### 1. DiT-XL-2-256 (Distributed Generation)
Optimized for multi-GPU throughput via `torchrun`:
```bash
cd CFG-MP_DiT
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 demo_dit.py
```
### 2. Stable Diffusion 3.5 (Single Image Generation)
```bash
cd CFG-MP_SD
python demo_SD.py
```

## ⚙️ Configuration & Hyperparameters
While the core logic is shared, the parameter names differ slightly across the two implementations to align with their respective codebases:

### DiT-XL-2-256 Hyperparameters (`demo_dit.py`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| **enable_anderson** | True | Whether to use Anderson Acceleration (CFG-MP+) or Picard iteration (CFG-MP). |
| **aa_correction_steps** | 2 | Maximum fixed-point iterations per sampling step. |
| **aa_window_size** | 1 | History window size $m$. |
| **aa_damping** | 1.0 | Damping factor $\beta$. |
| **time_threshold** | 0.6 | Threshold to disable correction in late sampling stages to save compute. |
| **guidance_scale** | 2 | Guidance scale for DiT-XL-2-256. |

### SD 3.5 Hyperparameters (`demo_SD.py`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| **use_aa** | True | Whether to use Anderson Acceleration (CFG-MP+) or Picard iteration (CFG-MP). |
| **max_aa_iter** | 3 | Maximum fixed-point iterations per sampling step. |
| **aa_window_size** | 1 | History window size $m$ . |
| **aa_damping** | 1.0 | Damping factor $\beta$ . |
| **time_threshold** | 0.6 | Threshold to disable correction in late sampling stages to save compute. |
| **guidance_scale** | 4 | Guidance scale for SD3.5. |


## 🔬 Implementation Details
### Core Logic
The implementation of Manifold Projection and Anderson Acceleration is contained in the respective utils files:

- CFG-MP_DiT/utils_dit.py: Includes CFGMPScheduler and D2F (Diffusion-to-Flow) Alignment.

- CFG-MP_SD/utils_SD.py: Includes CFGMPSD3Pipeline and the switching logic for refinement.

### Two-Phase Sampling Scheme
Each sampling step is divided into:

- Manifold Projection Phase: Refines the latent position through fixed-point iterations to minimize the prediction gap.

- CFG Sampling Phase: Performs the standard CFG update to advance to the next timestep.

## Citation

If you find this work helpful for your research, please consider citing:

```bibtex
@inproceedings{cai2026improvingclassifierfreeguidanceflow,
  title     = {Improving Classifier-Free Guidance of Flow Matching via Manifold Projection},
  author    = {Cai, Jian-Feng and Liu, Haixia and Su, Zhengyi and Wang, Chao},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  note      = {Accepted at ICML 2026},
  eprint    = {2601.21892},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2601.21892}
}
