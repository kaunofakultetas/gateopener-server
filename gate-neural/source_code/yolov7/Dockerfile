FROM docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y    \
    libgl1-mesa-glx libglib2.0-0        &&  \
    rm -rf /var/lib/apt/lists/*


# Base
RUN pip install --no-cache-dir          \
    "opencv-python==4.10.0.84"          \
    "scipy==1.13.1"


# Plotting
RUN pip install --no-cache-dir          \
    "pandas==2.2.2"                     \
    "seaborn==0.13.2"

# ImageZMQ for image frame transfer
RUN pip install --no-cache-dir          \
    "imagezmq==1.1.1"


COPY . .

CMD ["python3", "-u", "detect.py"]
