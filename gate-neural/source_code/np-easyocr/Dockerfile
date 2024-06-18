FROM docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG gh_username=JaidedAI
ARG service_home="/home/EasyOCR"
ARG DEBIAN_FRONTEND=noninteractive


# Configure apt and install packages
RUN apt-get update -y && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-dev git \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists


# Clone EasyOCR repo
RUN mkdir "$service_home" \
    && git clone "https://github.com/$gh_username/EasyOCR.git" "$service_home" \
    && cd "$service_home" \
    && git remote add upstream "https://github.com/JaidedAI/EasyOCR.git" \
    && git pull upstream master


# Build
RUN cd "$service_home" \
    && python setup.py build_ext --inplace -j 4 \
    && python -m pip install -e .


# ImageZMQ for image frame transfer
RUN pip install --no-cache-dir "imagezmq==1.1.1"


WORKDIR /workspace/app
COPY . .

CMD ["python3", "-u", "main.py"]
