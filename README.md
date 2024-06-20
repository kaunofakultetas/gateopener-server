# Gate opener using YOLOv7 and EasyOCR

This repository contains the setup and deployment of gate opener with ANPR capabilities. The system which this project is deployed on Ubuntu 22.04 server and utilizes NVIDIA 3060Ti graphics card.

<br/>
<div align="center">
  <img  src="https://github.com/KaunoFakultetas/gateopener-server/blob/main/docs/media/IncomingCars.gif?raw=true "width="480" height="320" alt="demo1">
  <img  src="https://github.com/KaunoFakultetas/gateopener-server/blob/main/docs/media/ExitingCars.gif?raw=true" width="480" height="320" alt="demo2">
</div>

<br>

## Getting Started

### Prerequisites

Before starting this project you need to install into your system:
- Nvidia Drivers 
- Docker
- Docker Compose

<br>

### Docker and Docker Compose:
```sh
sudo curl -L https://github.com/docker/compose/releases/download/v2.26.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo apt install -y docker.io
```

<br>

### NVIDIA drivers:
```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo apt install nvidia-driver-535
sudo reboot
```

<br>

### Building and Running the Container Stacks

1. **Clone the repository**:
    ```sh
    git clone https://github.com/kaunofakultetas/gateopener-server.git
    cd gateopener-server
    ```


2. **Start the neural network containers**:
    ```sh
    cd gate-neural
    ./runUpdateThisStack.sh
    cd ..
    ```

3. **Edit the processing stack environment variables and start containers**:
    ```sh
    cd gate-processor
    cp docker-compose.yml.sample docker-compose.yml
    nano docker-compose.yml
    ./runUpdateThisStack.sh
    cd ..
    ```

<br>

## Contributing

Feel free to submit issues and pull requests.

<br>

## Acknowledgements

- [YOLOv7 creators and orginal implementation developer](https://github.com/WongKinYiu/yolov7)
- [YOLOv7 modified model creator to detect numberplates](https://github.com/mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN)
- The open-source community for their continuous contributions.


