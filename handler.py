"""RunPod Serverless Handler for FLUX.1-dev + LoRA — coloring book optimized."""

import runpod
import torch
import base64
import io
import os
import gc
import shutil

# Cache model to network volume if available, otherwise /tmp
CACHE_DIR = "/runpod-volume/huggingface" if os.path.exists("/runpod-volume") else "/tmp/huggingface"
LORA_DIR = "/runpod-volume/loras" if os.path.exists("/runpod-volume") else "/tmp/loras"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clean up old schnell model to free disk space
_OLD_MODEL = os.path.join(CACHE_DIR, "models--black-forest-labs--FLUX.1-schnell")
if os.path.exists(_OLD_MODEL):
    print("Removing old FLUX.1-schnell cache...", flush=True)
    shutil.rmtree(_OLD_MODEL, ignore_errors=True)
    print("Old model removed.", flush=True)

PIPE = None
LOADED_LORA = None

# Default LoRA — change this to switch between LoRAs
DEFAULT_LORA = os.environ.get("LORA_REPO", "renderartist/coloringbookflux")
DEFAULT_LORA_TRIGGER = os.environ.get("LORA_TRIGGER", "c0l0ringb00k")


def get_pipe():
    """Lazy-load FLUX.1-dev with memory optimizations for 48GB GPU."""
    global PIPE
    if PIPE is None:
        from diffusers import FluxPipeline
        print("Loading FLUX.1-dev...", flush=True)
        PIPE = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        PIPE.enable_model_cpu_offload()
        PIPE.vae.enable_slicing()
        PIPE.vae.enable_tiling()
        print("FLUX.1-dev ready!", flush=True)
    return PIPE


def load_lora(pipe, lora_repo, lora_weight_name=None):
    """Load a LoRA from HuggingFace hub. Caches to network volume."""
    global LOADED_LORA

    if LOADED_LORA == lora_repo:
        return  # Already loaded

    # Unload previous LoRA if any
    if LOADED_LORA is not None:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

    print(f"Loading LoRA: {lora_repo}...", flush=True)

    kwargs = {"cache_dir": LORA_DIR}
    if lora_weight_name:
        kwargs["weight_name"] = lora_weight_name

    pipe.load_lora_weights(lora_repo, **kwargs)
    LOADED_LORA = lora_repo
    print(f"LoRA loaded: {lora_repo}", flush=True)


def handler(job):
    """Generate images from prompt with optional LoRA."""
    inp = job["input"]
    prompt = inp.get("prompt", "")
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    steps = inp.get("num_inference_steps", 28)
    guidance = inp.get("guidance_scale", 3.5)
    num_images = inp.get("num_images", 1)
    seed = inp.get("seed")
    lora_scale = inp.get("lora_scale", 0.85)

    # LoRA configuration — can be overridden per request
    lora_repo = inp.get("lora_repo", DEFAULT_LORA)
    lora_weight_name = inp.get("lora_weight_name")
    use_lora = inp.get("use_lora", True)

    if not prompt:
        return {"error": "No prompt provided"}

    pipe = get_pipe()

    # Load LoRA if requested
    if use_lora and lora_repo:
        try:
            load_lora(pipe, lora_repo, lora_weight_name)
        except Exception as e:
            print(f"LoRA load failed: {e}", flush=True)
            return {"error": f"Failed to load LoRA {lora_repo}: {str(e)[:200]}"}

    gen = torch.Generator("cpu").manual_seed(seed) if seed else None

    images = []
    for i in range(num_images):
        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": gen,
        }

        # Apply LoRA scale via joint_attention_kwargs
        if use_lora and LOADED_LORA:
            kwargs["joint_attention_kwargs"] = {"scale": lora_scale}

        result = pipe(**kwargs).images[0]

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        images.append({"image_base64": base64.b64encode(buf.getvalue()).decode(), "index": i})

        gc.collect()
        torch.cuda.empty_cache()

    return {"images": images, "lora_used": LOADED_LORA if use_lora else None}


runpod.serverless.start({"handler": handler})
