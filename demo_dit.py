import torch
import numpy as np
import os
import gc
from tqdm.auto import tqdm
import torch.distributed as dist
from diffusers import DiTPipeline
from PIL import Image
from utils_dit import CFGMPScheduler

def setup_distributed():
    """Initializes the distributed process group and sets the CUDA device."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def main():
    #  1. Generation Config 
    num_inference_steps = 10
    guidance_scale = 1.75
    
    # CFG-MP / CFG-MP+ Control
    aa_correction_steps = 2  # Number of FPI iterations per step
    aa_window_size = 1       # m: History window for Anderson Acceleration
    aa_damping = 1.0         # beta: 1.0 means no damping, < 1.0 for more stability
    enable_anderson = True   # True for CFG-MP+, False for CFG-MP (Picard)
    time_threshold = 0.6     # Turn off correction in late sampling stages
    
    batch_size = 40
    output_dir = "demo_output"
    model_path = "DiT-XL-2-256" # Use local path if necessary
    
    local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"
    is_main_process = local_rank == 0

    #  2. Setup Model and Scheduler 
    pipe = DiTPipeline.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    scheduler = CFGMPSchedulerScheduler(num_inference_steps=num_inference_steps)
    latent_c = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size
    
    #  3. Task Allocation 
    all_labels = np.repeat(np.arange(1000), 1)
    all_seeds = np.arange(42, 42 + len(all_labels))
    my_labels = np.array_split(all_labels, world_size)[local_rank]
    my_seeds = np.array_split(all_seeds, world_size)[local_rank]
    
    if is_main_process and not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    dist.barrier()

    generator = torch.Generator(device="cpu")

    #  4. Generation Loop 
    with torch.no_grad():
        pbar = tqdm(total=len(my_labels), desc=f"Rank {local_rank}", disable=not is_main_process)
        for i in range(0, len(my_labels), batch_size):
            b_labels, b_seeds = my_labels[i : i+batch_size], my_seeds[i : i+batch_size]
            curr_bs = len(b_labels)
            
            # Initialize random noise latents
            latents_list = [torch.randn((1, latent_c, latent_size, latent_size), 
                            generator=generator.manual_seed(int(s))) for s in b_seeds]
            latents = torch.cat(latents_list, dim=0).to(device)
            class_labels = torch.from_numpy(b_labels).to(device).long()
            null_labels = torch.full((curr_bs,), 1000, device=device).long()

            for step in range(num_inference_steps):
                p = scheduler.get_step_params(step, device)
                
                #  Phase 1: Manifold Projection (Refinement) 
                if aa_correction_steps > 0 and p['t_curr'] < time_threshold:
                    latents = scheduler.step_anderson_correction(
                        model=pipe.transformer, 
                        latents=latents, 
                        params=p, 
                        labels=class_labels, 
                        null_labels=null_labels, 
                        aa_steps=aa_correction_steps, 
                        latent_c=latent_c, 
                        m=aa_window_size, 
                        use_aa=enable_anderson,
                        damping_beta=aa_damping 
                    )
                    
                #  Phase 2: CFG Sampling (ODE Stepping) 
                latents = scheduler.step_cfg_flow(
                    model=pipe.transformer, 
                    latents=latents, 
                    params=p, 
                    labels=class_labels, 
                    null_labels=null_labels, 
                    guidance_scale=guidance_scale, 
                    latent_c=latent_c, 
                    step_idx=step
                )

            #  5. Final Decode and Save 
            latents_to_decode = latents / pipe.vae.config.scaling_factor
            decoded = pipe.vae.decode(latents_to_decode, return_dict=False)[0]
            imgs = (decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            
            for j, img_arr in enumerate(imgs):
                img_save_path = os.path.join(output_dir, f"img_{b_seeds[j]}.png")
                Image.fromarray((img_arr * 255).astype(np.uint8)).save(img_save_path)

            # Cleanup memory
            gc.collect()
            torch.cuda.empty_cache()
            if is_main_process: 
                pbar.update(curr_bs)

    if is_main_process:
        print(f"Generation complete. Images saved to {output_dir}")

if __name__ == "__main__":
    main()