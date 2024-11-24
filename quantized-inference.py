import torch
import gc
from diffusers import AllegroPipeline
from diffusers import AutoencoderKLAllegro
from diffusers.utils import export_to_video
import torch.nn as nn

torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)

def clear_cuda_cache():
    """Clear CUDA cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def quantize_model(model):
    """Quantize model to INT8"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(module, inplace=True)
            torch.quantization.convert(module, inplace=True)
    return model

# Clear initial cache
clear_cuda_cache()

# Load models with reduced precision
vae = AutoencoderKLAllegro.from_pretrained(
    "rhymes-ai/Allegro",
    subfolder="vae",
    torch_dtype=torch.float16
)

pipe = AllegroPipeline.from_pretrained(
    "rhymes-ai/Allegro",
    vae=vae,
    torch_dtype=torch.float16
)

# Enable memory optimizations
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# Enable CPU offloading
pipe.enable_sequential_cpu_offload()

# Additional memory optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Define generation parameters
prompt = "A bustling city street during a light rain shower, with reflections of neon signs on the wet pavement. People walk with colorful umbrellas, and cars pass by, their headlights creating dynamic light streaks. The camera pans slowly, capturing the vibrant energy of the scene with detailed textures and realistic movement of raindrops."

positive_prompt = """
(masterpiece), (best quality), (ultra-detailed), (unwatermarked),
{}
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo,
sharp focus, high budget, cinemascope, moody, epic, gorgeous
"""

negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality,
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

prompt = positive_prompt.format(prompt.lower().strip())

print("prompt", prompt)

# Clear cache before inference
clear_cuda_cache()

try:
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        max_sequence_length=512,
        num_inference_steps=30,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    
    # Process output
    video = output.frames[0]
    export_to_video(video, "output.mp4", fps=15)
    print("Generation complete!!!!!")
except Exception as e:
    print(f"Error during generation: {str(e)}")
    raise

finally:
    # Cleanup
    del pipe
    del vae
    clear_cuda_cache()
