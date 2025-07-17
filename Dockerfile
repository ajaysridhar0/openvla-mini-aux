# ===
# OpenVLA Dockerfile
#   => Base Image :: Python 3.10
# ===
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Sane Defaults
RUN apt-get update
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    docker.io \
    ffmpeg \
    git \
    htop \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    libgl1 \
    libopenexr-dev \
    mesa-utils \
    freeglut3-dev \
    libsdl2-2.0-0 \
    python-pygame


# IMPORTANT :: Uninstall & Reinstall Torch
RUN pip install --upgrade pip
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install Prismatic + VLA Python Dependencies (`pip`)
RUN pip install \
    accelerate>=0.25.0 \
    draccus@git+https://github.com/dlwh/draccus@55e456a3047a97135e47bbeefe06873d2727dd41 \
    einops \
    huggingface_hub \
    jsonlines \
    matplotlib \
    pyyaml-include==1.4.1 \
    rich \
    sentencepiece \
    timm==0.9.10 \
    tokenizers==0.21.0 \
    transformers==4.49.0 \
    wandb \
    tensorflow==2.15.0 \
    tensorflow_datasets==4.9.3 \
    tensorflow_graphics==2021.12.3 \
    dlimp@git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a

# Flash Attention 2 Installation
RUN pip install packaging ninja
RUN pip install flash-attn==2.5.5 --no-build-isolation

# Fixes some error
RUN pip uninstall transformer-engine -y

# Enables webp in Pillow
RUN pip install --upgrade --force Pillow

# Install VQ
ADD /vq_bet_official /vq_bet_official
RUN pip install -e ../vq_bet_official/

# Install peft for LoRA
RUN pip install peft

ENV PYTHONPATH=/workspace/openvla-mini-aux/
ENV HF_HOME=/workspace/.cache/huggingface

RUN wandb login 613699f8b80f68f5c666ed96334c9bd97a651d2c

CMD ["/bin/bash"]