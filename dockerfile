FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git build-essential curl ca-certificates \
    graphviz libgraphviz-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools

WORKDIR /workspace/dowhy-deep
ENV PYTHONPATH=/workspace/dowhy-deep

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /root/.cache/tabpfn
COPY tabpfn_ckpt/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt \
     /root/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt

CMD ["/bin/bash"]