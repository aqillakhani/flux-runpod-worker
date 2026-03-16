"""RunPod Serverless Handler for FLUX.1-schnell image generation."""

import runpod
import torch
import base64
import io
import os
from diffusers import FluxPipeline


def load_model():
    """Load FLUX.1-schnell model at startup."""
    print("Loading FLUX.1-schnell model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("Model loaded successfully!")
    return pipe


# Load model once at cold start
PIPE = load_model()


def handler(job):
    """Handle incoming image generation requests."""
    job_input = job["input"]

    prompt = job_input.get("prompt", "")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    num_inference_steps = job_input.get("num_inference_steps", 4)
    guidance_scale = job_input.get("guidance_scale", 0.0)
    num_images = job_input.get("num_images", 1)
    seed = job_input.get("seed", None)

    if not prompt:
        return {"error": "No prompt provided"}

    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)

    results = []
    for i in range(num_images):
        image = PIPE(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        results.append({"image_base64": img_b64, "index": i})

    return {"images": results, "prompt": prompt}


runpod.serverless.start({"handler": handler})
