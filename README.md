<<<<<<< HEAD
<<<<<<< HEAD
# CFG-MP Combined Repository
=======
# CFG-MP and CFG-MP+ for Diffusion Transformers (DiT)

This repository contains the official implementation of **CFG-MP** (Manifold Projection) and **CFG-MP+** (Anderson-Accelerated Manifold Projection) for **DiT-XL-2-256**, as described in our ICML 2026 submission. 

Our method introduces an **incremental gradient descent scheme**, formulated as a **fixed-point iteration**, to minimize the prediction gap during Classifier-Free Guidance (CFG). This ensures latents remain aligned with the data manifold, significantly enhancing generation fidelity and convergence efficiency.
=======
# CFG-MP and CFG-MP+ for Stable Diffusion 3.5

This repository contains the official implementation of **CFG-MP** (Manifold Projection) and **CFG-MP+** (Anderson-Accelerated Manifold Projection), as described in our ICML 2026 submission. 

Our method introduces an **incremental gradient descent scheme**, which is also a **fixed-point iteration**, to ensure latents to have small prediction gap during Classifier-Free Guidance (CFG), significantly improving generation fidelity and convergence efficiency.
>>>>>>> origin-sd/main

---

## 🛠️ Installation & Setup

### 1. Environment Setup
We recommend using **Miniconda** for environment management. You can set up the environment using the following commands:

```bash
# Create and activate environment
<<<<<<< HEAD
conda create -n cfgmp-dit python=3.10 -y
conda activate cfgmp-dit
=======
conda create -n cfgmp python=3.10 -y
conda activate cfgmp
>>>>>>> origin-sd/main

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Access
<<<<<<< HEAD
This code utilizes the DiT-XL-2-256 model.

- Local Path: Update the model_path variable in demo_dit.py to point to your local DiT checkpoint.

- Automatic Download: By default, the script will attempt to download the weights from Hugging Face if the local path is not found.

## 🚀 Usage Guide
### Distributed Generation

This implementation is optimized for multi-GPU throughput via `torchrun`. To generate a batch of images (e.g., using 1 GPU):

```Bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 demo_dit.py
```

## ⚙️ Configuration & Hyperparameters
The following parameters in `demo_dit.py` control the refinement phase and Anderson Acceleration logic:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **enable_anderson** | True | Whether to use Anderson Acceleration (CFG-MP+) or Picard iteration (CFG-MP). |
| **aa_correction_steps** | 2 | Number of fixed-point iterations performed per sampling step. |
| **aa_window_size** | 1 | History window size $m$. |
| **aa_damping** | 1.0 | Damping factor $\beta$. |
| **time_threshold** | 0.6 | Threshold to disable correction in late sampling stages to save compute. |
| **guidance_scale** | 2 | Classifier-Free Guidance scale for DiT. |

## 🔬 Implementation Details
### Core Logic in `utils_dit.py`

- `CFGMPScheduler`: Implements D2F (Diffusion-to-Flow) Alignment, which synchronizes Flow Matching inference timesteps with the discrete training schedules of diffusion models.

- `solve_anderson_weights`: Performs a regularized least-squares optimization to find optimal history coefficients.

- Two-Phase Sampling Scheme: Each sampling step is divided into two phases:
    - Phase 1: Manifold Projection: Uses `step_anderson_correction` to reduce the prediction gap and thus refine the latent position through fixed-point iterations.
    - Phase 2: ODE Stepping: Uses `step_cfg_flow `to perform the standard CFG update and advance to the next timestep.

## 📄 License
This project is licensed under the MIT License.

## 🤫 Double-Blind Submission Note
This repository is prepared for the ICML 2026 review process. In accordance with the double-blind policy, all author identities and institutional affiliations have been removed from the source code and documentation.
>>>>>>> origin-dit/main
=======
This code utilizes the official Stable Diffusion 3.5 weights from Hugging Face.

- Local Path: Update the MODEL_DIR variable in demo_SD.py to point to your local SD3.5 checkpoint.

- Hugging Face: Ensure you have logged in via huggingface-cli login if you are accessing the weights directly from the hub.

## 🚀 Usage Guide
Single Image Generation
To generate a sample using the CFG-MP+ (Anderson Acceleration enabled) pipeline with default settings:

```Bash
python demo_SD.py
```

## ⚙️ Configuration & Hyperparameters
The following parameters in demo_SD.py control the fixed-point iteration and acceleration logic:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **use_aa** | True | Whether to enable Anderson Acceleration (CFG-MP or MP+). |
| **max_aa_iter** | 3 | Maximum fixed-point iterations per sampling step. |
| **aa_window_size** | 1 | History window size $m$ . |
| **aa_damping** | 0.7 | Damping factor $\beta$ . |
| **guidance_scale** | 4 | Guidance scale for Classifier-Free Guidance. |

## 🔬 Implementation Details
### Core Logic in `utils_SD.py`
- `CFGMPScheduler`: Extends the `SchedulerMixin` to implement a custom noise schedule and the Anderson mixing solver. It solves the constrained least-squares problem to find the optimal extrapolation of previous latent states.

- `CFGMPSD3Pipeline`: A specialized pipeline for SD3.5 that overrides the sampling loop.
    - It implements the `iterate_fixed_point` method, which reduces the prediction gap and improve the velocity estimation accuracy in the CFG sampling process.
    - It applies a switching logic: for timesteps above the `switching_threshold`, it performs fixed-point iterations using "G"  to project latents onto the manifold.

## 📄 License
This project is licensed under the **MIT License**.

## 🤫 Double-Blind Submission Note
This repository is prepared for the **ICML 2026** review process. In accordance with the double-blind policy, all author identities and institutional affiliations have been removed from the source code and documentation.
>>>>>>> origin-sd/main
