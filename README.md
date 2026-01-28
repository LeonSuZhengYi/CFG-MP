# CFG-MP and CFG-MP+ for Stable Diffusion 3.5

This repository contains the official implementation of **CFG-MP** (Manifold Projection) and **CFG-MP+** (Anderson-Accelerated Manifold Projection), as described in our ICML 2026 submission. 

Our method introduces an **incremental gradient descent scheme**, which is also a **fixed-point iteration**, to ensure latents to have small prediction gap during Classifier-Free Guidance (CFG), significantly improving generation fidelity and convergence efficiency.

---

## 🛠️ Installation & Setup

### 1. Environment Setup
We recommend using **Miniconda** for environment management. You can set up the environment using the following commands:

```bash
# Create and activate environment
conda create -n cfgmp python=3.10 -y
conda activate cfgmp

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Access
This code utilizes the official Stable Diffusion 3.5 weights from Hugging Face.

- Local Path: Update the MODEL_DIR variable in demo_SD.py to point to your local SD3.5 checkpoint.

- Hugging Face: Ensure you have logged in via huggingface-cli login if you are accessing the weights directly from the hub.

## 🚀 Usage Guide
Single Image Generation
To generate a sample using the CFG-MP+ (Anderson Acceleration enabled) pipeline with default settings:

```Bash
python demo_SD.py
```
Reproducibility: The script uses a fixed `random seed (42)` to ensure the reproducibility of results presented in the paper.


Output: The final output will be saved as `output.png.`

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
