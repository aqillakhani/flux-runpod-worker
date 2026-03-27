"""RunPod Serverless Handler for FLUX.1-dev — quality-optimized for 48GB GPU."""

import runpod
import torch
import base64
import io
import os
import gc

# HuggingFace auth — set HF_TOKEN as env var on RunPod endpoint
# Required for gated models like FLUX.1-dev

# Cache model to network volume if available, otherwise /tmp
CACHE_DIR = "/runpod-volume/huggingface" if os.path.exists("/runpod-volume") else "/tmp/huggingface"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PIPE = None


def get_pipe():
    """Lazy-load FLUX.1-dev with memory optimizations for 48GB GPU."""
    global PIPE
    if PIPE is None:
        from diffusers import FluxPipeline
        print("Loading FLUX.1-dev (quality-optimized)...", flush=True)
        PIPE = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        # Memory optimizations (still useful on 48GB for multi-image batches)
        PIPE.enable_model_cpu_offload()
        PIPE.vae.enable_slicing()
        PIPE.vae.enable_tiling()
        print("FLUX.1-dev ready!", flush=True)
    return PIPE


def handler(job):
    """Generate images from prompt."""
    inp = job["input"]
    prompt = inp.get("prompt", "")
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    steps = inp.get("num_inference_steps", 25)
    guidance = inp.get("guidance_scale", 5.0)
    num_images = inp.get("num_images", 1)
    seed = inp.get("seed")

    if not prompt:
        return {"error": "No prompt provided"}

    pipe = get_pipe()
    gen = torch.Generator("cpu").manual_seed(seed) if seed else None

    images = []
    for i in range(num_images):
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        ).images[0]

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        images.append({"image_base64": base64.b64encode(buf.getvalue()).decode(), "index": i})

        # Free VRAM between images
        gc.collect()
        torch.cuda.empty_cache()

    return {"images": images}


runpod.serverless.start({"handler": handler})
