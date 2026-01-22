import torch
from PIL import Image
from utils_SD import CFGMPSD3Pipeline, CFGMPScheduler

def generate_sample(
    prompt: str,
    model_path: str,
    use_aa: bool = True,
    seed: int = 42
) -> Image.Image:
    """
    Generates a single image using the Anderson-accelerated SD3.5 pipeline.

    Args:
        prompt (str): The text prompt for image generation.
        model_path (str): Local path to the pretrained SD3.5 model weights.
        use_aa (bool): Whether to enable Anderson Acceleration (AA).
        seed (int): Random seed for reproducible generation.

    Returns:
        PIL.Image.Image: The final generated and post-processed image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # 1. Initialize custom components
    scheduler = CFGMPScheduler(num_train_timesteps=1000, shift=3.0)
    pipe = CFGMPSD3Pipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        torch_dtype=dtype
    )
    pipe.to(device)

    # 2. Perform Inference
    print(f"[*] Generating image (Anderson Acceleration: {use_aa})...")
    generator = torch.Generator(device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5, 
        num_inference_steps=10,
        use_aa=use_aa,
        max_aa_iter=3,
        aa_window_size=1,      
        aa_damping=0.7,        
        generator=generator
    )

    return result

if __name__ == "__main__":
    MODEL_DIR = "sd3.5" # please change to the local model path.
    PROMPT_TEXT = "A high-tech cyberpunk city street, neon lights, rainy weather, ultra-realistic."
    
    output = generate_sample(PROMPT_TEXT, MODEL_DIR, use_aa=True)
    
    if hasattr(output, "images"):
        img = output.images[0]
    else:
        img = output[0]
        
    img.save("CFG-MP_output.png")
    print("Successfully saved image as CFG-MP_output.png")