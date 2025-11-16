FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

# If the host is 24.0 and CUDA should be upgraded to 12.6+.

RUN apt-get update && apt-get install -y python3.10 python3-pip git

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

COPY src/ src/

CMD ["bash"]