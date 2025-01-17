FROM python:3.12.1

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update &&                       \
    apt-get install -y libgl1-mesa-glx &&   \
    rm -rf /var/lib/apt/lists/*

# Base
RUN pip install --no-cache-dir              \
    "opencv-python==4.10.0.84"              \
    "requests==2.32.3"                      \
    "mysql-connector-python==8.4.0"

# ImageZMQ for image frame transfer
RUN pip install --no-cache-dir              \
    "imagezmq==1.1.1"

# Copy the main python script
COPY importer.py /app/


EXPOSE 3030
CMD ["python3", "-u", "importer.py"]

