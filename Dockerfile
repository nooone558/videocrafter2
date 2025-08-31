FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \ 
    git wget curl  vim ffmpeg libsm6 libxext6 python3 python3-pip python3-dev \ 
    && rm -rf /var/lib/apt/lists/* 
RUN pip3 install --upgrade pip 
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install fastapi uvicorn python-multipart pydantic \
    && pip3 install xformers==0.0.22.post7 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /workspace/ 
RUN pip3 install --upgrade pip && pip3 install -r /workspace/requirements.txt
WORKDIR /workspace

COPY inference.py /workspace/inference.py
COPY checkpoints/ /workspace/checkpoints/


EXPOSE 8000
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
