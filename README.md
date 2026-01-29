# CFG-MP & CFG-MP+: Manifold Projection for Diffusion Models

This repository contains the official implementation of **CFG-MP** (Manifold Projection) and **CFG-MP+** (Anderson-Accelerated Manifold Projection) for both **Diffusion Transformers (DiT)** and **Stable Diffusion 3.5 (SD3.5)**.

Our method introduces an **incremental gradient descent scheme**, formulated as a **fixed-point iteration** (FPI), to minimize the prediction gap during Classifier-Free Guidance (CFG), and we further utilize **Anderson Acceleration** (AA) to speed up and stabilize this FPI. CFG-MP/MP+ can significantly enhancing generation fidelity and convergence efficiency.

---

## 📂 Repository Structure

The code is organized into two main projects:

* **[`CFG-MP_DiT/`](./CFG-MP_DiT/)**: Implementation for DiT-XL-2-256 (ImageNet generation).
* **[`CFG-MP_SD/`](./CFG-MP_SD/)**: Implementation for Stable Diffusion 3.5 (Large-scale Text-to-Image).

---

## 🛠️ Installation & Setup

We recommend using **Miniconda** for environment management. Each project has its own specific dependencies.

### For DiT Experiments
```bash
cd CFG-MP_DiT
conda create -n cfgmp-dit python=3.10 -y
conda activate cfgmp-dit
pip install -r requirements.txt
```

### For Stable Diffusion 3.5 Experiments
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

### DiT Hyperparameters (`demo_dit.py`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| **enable_anderson** | True | Whether to use Anderson Acceleration (CFG-MP+) or Picard iteration (CFG-MP). |
| **aa_correction_steps** | 2 | Number of fixed-point iterations performed per sampling step. |
| **aa_window_size** | 1 | History window size $m$. |
| **aa_damping** | 1.0 | Damping factor $\beta$. |
| **time_threshold** | 0.6 | Threshold to disable correction in late sampling stages to save compute. |
| **guidance_scale** | 2 | Classifier-Free Guidance scale for DiT. |

### SD 3.5 Hyperparameters (`demo_SD.py`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| **use_aa** | True | Whether to enable Anderson Acceleration (CFG-MP or MP+). |
| **max_aa_iter** | 3 | Maximum fixed-point iterations per sampling step. |
| **aa_window_size** | 1 | History window size $m$ . |
| **aa_damping** | 0.7 | Damping factor $\beta$ . |
| **time_threshold** | 0.6 | Threshold to disable correction in late sampling stages to save compute. |
| **guidance_scale** | 4 | Guidance scale for Classifier-Free Guidance. |


## 🔬 Implementation Details
### Core Logic
The implementation of Manifold Projection and Anderson Acceleration is contained in the respective utils files:

- CFG-MP_DiT/utils_dit.py: Includes CFGMPScheduler and D2F (Diffusion-to-Flow) Alignment.

- CFG-MP_SD/utils_SD.py: Includes CFGMPSD3Pipeline and the switching logic for refinement.

### Two-Phase Sampling Scheme
Each denoising step is divided into:

- Manifold Projection Phase: Refines the latent position through fixed-point iterations to minimize the prediction gap.

- ODE Stepping Phase: Performs the standard CFG update to advance to the next timestep.

## 📄 License
This project is licensed under the MIT License.
