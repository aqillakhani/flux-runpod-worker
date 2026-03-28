FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.31.0 \
    transformers==4.46.0 \
    accelerate==1.1.0 \
    peft>=0.13.0 \
    safetensors \
    sentencepiece \
    protobuf \
    huggingface_hub

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
