FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    pkg-config \
    software-properties-common \
    g++ \
    sudo \
    ccache \
    cmake \
    libjpeg-dev \
    libpng-dev \
    emacs

RUN apt-get install -y \
    rsync \
    swig \
    curl \
    git \
    wget \
    unzip \
    zsh \
    screen \
    tmux \
    openssh-server \
    openssh-client \
    liblzma-dev \
    lzma

RUN apt-get install -y libopenmpi-dev

RUN apt-get install -y \
    python3.10 \
    python3-pip

RUN apt-get update && \
    apt-get install -y pciutils net-tools iputils-ping && \
    apt-get install -y htop && \
    rm -rf /var/lib/apt/lists/*

RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

# configure environments.
RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/*
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install soundfile==0.12.1
RUN pip install librosa==0.10.2
RUN pip install deepspeed==0.12.6 
RUN pip install accelerate==0.27.0 
RUN pip install ninja==1.11.1.1
# RUN pip install flash-attn --no-build-isolation
RUN pip install mpi4py

# transformers
RUN pip install --no-cache-dir nltk==3.6.7 transformers==4.37.2 peft==0.8.2 tokenizers==0.15.1 datasets==2.18.0
RUN pip install huggingface-hub==0.23.4
RUN pip install tensorboardX

RUN pip install requests==2.32.3
RUN pip install numpy==1.23.4
RUN pip install protobuf==3.19.5

# miscs
RUN pip install shortuuid
RUN pip install sentencepiece==0.1.99
RUN pip install bitsandbytes pydantic markdown2 svgwrite wavedrom
RUN pip install scikit-learn==1.2.2
RUN pip install gradio==4.16.0 gradio_client==0.8.1
RUN pip install httpx==0.24.0 uvicorn fastapi
RUN pip install einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 numpy==1.23.4


# You may want to install flash-attn for speeding up
# Please c.f. https://github.com/Dao-AILab/flash-attention/ for building from source

# For tool using models, you would need to install madmom:
# https://madmom.readthedocs.io/en/latest/
# But you may need to manually resolve some package conflicts due to its legacy dependencies.