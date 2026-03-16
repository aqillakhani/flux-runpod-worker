"""RunPod Serverless Handler for FLUX.1-schnell — simple version."""

import runpod
import torch
import base64
import io
import os

# Cache model to network volume if available, otherwise /tmp
CACHE_DIR = "/runpod-volume/huggingface" if os.path.exists("/runpod-volume") else "/tmp/huggingface"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

PIPE = None


def get_pipe():
    """Lazy-load model on first request."""
    global PIPE
    if PIPE is None:
        from diffusers import FluxPipeline
        print(f"Loading FLUX.1-schnell to {CACHE_DIR}...", flush=True)
        PIPE = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        PIPE.to("cuda")
        PIPE.enable_attention_slicing()
        print("Model ready!", flush=True)
    return PIPE


def handler(job):
    """Generate images from prompt."""
    inp = job["input"]
    prompt = inp.get("prompt", "")
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    steps = inp.get("num_inference_steps", 4)
    guidance = inp.get("guidance_scale", 0.0)
    num_images = inp.get("num_images", 1)
    seed = inp.get("seed")

    if not prompt:
        return {"error": "No prompt provided"}

    pipe = get_pipe()
    gen = torch.Generator("cuda").manual_seed(seed) if seed else None

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

    return {"images": images}


runpod.serverless.start({"handler": handler})
