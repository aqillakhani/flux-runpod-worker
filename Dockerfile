FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    diffusers>=0.30.0 \
    transformers>=4.44.0 \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
