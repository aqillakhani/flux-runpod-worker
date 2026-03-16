# FLUX.1-schnell RunPod Serverless Worker

Runs FLUX.1-schnell for image generation as a RunPod serverless endpoint.

## API Usage

```python
import requests, base64

resp = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_RUNPOD_API_KEY"},
    json={"input": {
        "prompt": "Black and white coloring page, bold outlines...",
        "width": 1024,
        "height": 1024,
        "num_images": 2,
        "num_inference_steps": 4,
    }}
)
data = resp.json()["output"]
for img in data["images"]:
    with open(f"page_{img['index']}.png", "wb") as f:
        f.write(base64.b64decode(img["image_base64"]))
```

## GPU Requirements
- 24GB VRAM (RTX 3090, A5000, RTX 4090)
- FLUX.1-schnell uses ~12GB VRAM at bfloat16
