FROM python:3.14.4

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update &&                       \
    apt-get install -y libgl1 &&            \
    rm -rf /var/lib/apt/lists/*

# Base
RUN pip install --no-cache-dir              \
    "opencv-python==4.13.0.92"              \
    "requests==2.33.1"

# Flask Streamer
RUN pip install --no-cache-dir              \
    "flask_opencv_streamer==1.4"

# ImageZMQ for image frame transfer
RUN pip install --no-cache-dir              \
    "imagezmq==1.2.0"

# Copy the main python script
COPY main.py /app/


EXPOSE 3030
CMD ["python3", "-u", "main.py"]

