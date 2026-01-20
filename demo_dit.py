import torch
import numpy as np
import os
import gc
from tqdm.auto import tqdm
import torch.distributed as dist
from diffusers import DiTPipeline
from PIL import Image
from utils_dit import AndersonFlowScheduler

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def main():
    # Config 
    num_inference_steps = 10
    guidance_scale = 1.75
    aa_correction_steps = 0 # to control the numbers of FPI
    aa_window_size = 1  # to control the AA windowsize
    enable_anderson = False  # to turn on or off the anderson acceleration for the FPI
    time_threshold = 0.6 # to turn off the FPI correction in the last several steps

    batch_size = 40
    output_dir = "demo_output"
    model_path = "DiT-XL-2-256"
    
    local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"
    is_main_process = local_rank == 0

    # Setup Model and Scheduler 
    pipe = DiTPipeline.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    scheduler = AndersonFlowScheduler(num_inference_steps=num_inference_steps)
    latent_c = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size
    
    # labels and seeds 
    all_labels = np.repeat(np.arange(1000), 1)
    all_seeds = np.arange(42, 42 + len(all_labels))
    my_labels = np.array_split(all_labels, world_size)[local_rank]
    my_seeds = np.array_split(all_seeds, world_size)[local_rank]
    
    if is_main_process and not os.path.exists(output_dir): os.makedirs(output_dir)
    dist.barrier()

    generator = torch.Generator(device="cpu")

    with torch.no_grad():
        pbar = tqdm(total=len(my_labels), disable=not is_main_process)
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
                
                # 1. manifold projection phase
                if aa_correction_steps > 0 and p['t_curr'] < time_threshold:
                    
                    latents = scheduler.step_anderson_correction(
                        pipe.transformer, latents, p, class_labels, null_labels, 
                        aa_correction_steps, latent_c, aa_window_size, enable_anderson
                    )
                    
                # 2. CFG sampling phase
                latents = scheduler.step_cfg_flow(
                    pipe.transformer, latents, p, class_labels, null_labels, 
                    guidance_scale, latent_c, step
                )

            # Final Decode 
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            imgs = (decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            for j, img_arr in enumerate(imgs):
                Image.fromarray((img_arr * 255).astype(np.uint8)).save(os.path.join(output_dir, f"img_{b_seeds[j]}.png"))

            gc.collect()
            torch.cuda.empty_cache()
            if is_main_process: pbar.update(curr_bs)

if __name__ == "__main__":
    main()