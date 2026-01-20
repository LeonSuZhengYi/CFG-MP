# CFG-MP and CFG-MP+ for DiT

This project implements CFG-MP and CFG-MP+  based on **Diffusion Transformers (DiT)** and the D2F technique.

## Key Features
- **D2F Alignment**: Synchronizes flow matching inference timesteps with diffusion discrete training schedules.
- **Fixed-Point Refinement**: Supports both Picard iteration (adopted by CFG-MP) and Anderson Acceleration (adopted by CFG-MP+)for latent position correction.
- **Distributed Ready**: Native support for multi-GPU generation via `torchrun`.

## Usage
Run distributed generation across 1 GPU:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 demo_dit.py
```
