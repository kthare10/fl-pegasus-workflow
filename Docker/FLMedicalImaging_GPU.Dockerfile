FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pegasus-wms==5.1.2 \
    PyYAML==6.0.2 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    Pillow==10.4.0 \
    torchvision==0.20.1 \
    flwr==1.14.0 \
    psutil==6.1.0 \
    pynvml==11.5.3 \
    scikit-learn==1.5.2 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    pydicom==3.0.1 \
    SimpleITK==2.4.0 \
    nibabel==5.3.2

WORKDIR /workflow
CMD ["/bin/bash"]
