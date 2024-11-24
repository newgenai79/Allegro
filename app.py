import torch
import gc
from diffusers import AllegroPipeline
from diffusers import AutoencoderKLAllegro
from diffusers.utils import export_to_video
import torch.nn as nn
import gradio as gr
import os
from datetime import datetime

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

class VideoGenerator:
    def __init__(self):
        self.pipe = None
        self.initialize_models()

    def clear_cuda_cache(self):
        """Clear CUDA cache and garbage collect"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def initialize_models(self):
        """Load and initialize models"""
        self.clear_cuda_cache()
        
        print("Loading models... This might take a few minutes...")
        
        vae = AutoencoderKLAllegro.from_pretrained(
            "rhymes-ai/Allegro",
            subfolder="vae",
            torch_dtype=torch.float16
        )

        self.pipe = AllegroPipeline.from_pretrained(
            "rhymes-ai/Allegro",
            vae=vae,
            torch_dtype=torch.float16
        )

        # Enable memory optimizations
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        self.pipe.enable_sequential_cpu_offload()

        # Additional memory optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        print("Models loaded successfully!")

    def generate_video(self, prompt, positive_prompt, negative_prompt, guidance_scale, 
                      max_sequence_length, num_inference_steps, seed):
        try:
            # Format prompt
            full_prompt = positive_prompt.format(prompt.lower().strip())
            
            # Set seed
            if seed == -1:
                seed = int(torch.randint(0, 1000000, (1,)).item())
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            # Generate output
            output = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=float(guidance_scale),
                max_sequence_length=max_sequence_length,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("output", f"video_{timestamp}.mp4")
            
            # Save video
            video = output.frames[0]
            export_to_video(video, output_path, fps=15)
            
            self.clear_cuda_cache()
            
            return output_path, f"Generation complete! Seed used: {seed}"
            
        except Exception as e:
            return None, f"Error during generation: {str(e)}"

    def __del__(self):
        if self.pipe is not None:
            del self.pipe
            self.clear_cuda_cache()

# Create a global instance of VideoGenerator
generator = VideoGenerator()

# Define the Gradio interface
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Video Generation UI")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Enter your main prompt here..."
                )
                
                positive_prompt = gr.Textbox(
                    label="Positive Prompt Template",
                    lines=3,
                    value="""(masterpiece), (best quality), (ultra-detailed), (unwatermarked),
{}
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo,
sharp focus, high budget, cinemascope, moody, epic, gorgeous"""
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    value="""nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality,
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."""
                )
                
                guidance_scale = gr.Textbox(
                    label="Guidance Scale",
                    value="7.5"
                )
                
                max_sequence_length = gr.Slider(
                    label="Max Sequence Length",
                    minimum=128,
                    maximum=512,
                    step=128,
                    value=512
                )
                
                num_inference_steps = gr.Slider(
                    label="Number of Inference Steps",
                    minimum=10,
                    maximum=100,
                    step=1,
                    value=100
                )
                
                seed = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    precision=0
                )
                
                generate_btn = gr.Button("Generate Video")
            
            with gr.Column():
                output_video = gr.Video(label="Generated Video")
                output_text = gr.Textbox(label="Status")
        
        generate_btn.click(
            fn=generator.generate_video,
            inputs=[
                prompt,
                positive_prompt,
                negative_prompt,
                guidance_scale,
                max_sequence_length,
                num_inference_steps,
                seed
            ],
            outputs=[output_video, output_text]
        )
    
    return demo

# Launch the UI
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True)