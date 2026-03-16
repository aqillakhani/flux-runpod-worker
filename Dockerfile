FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    diffusers>=0.30.0 \
    transformers>=4.44.0 \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf

# Pre-download the FLUX.1-schnell model during build
RUN python -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell', torch_dtype='auto')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
