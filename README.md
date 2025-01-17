# Gate opener using Ultralytics YOLO11

This repository contains the setup and deployment of gate opener with ANPR capabilities. The system which this project is deployed on is Ubuntu 24.04.1 LTS server and running on CPU.

<br/>
<div align="center">
  <img  src="https://github.com/KaunoFakultetas/gateopener-server/blob/main/docs/media/IncomingCars.gif?raw=true "width="360" height="240" alt="demo1">
  <img  src="https://github.com/KaunoFakultetas/gateopener-server/blob/main/docs/media/ExitingCars.gif?raw=true" width="360" height="240" alt="demo2">
</div>

<br>

## Overall structure
<div align="center">
  <img  src="https://github.com/KaunoFakultetas/gateopener-server/blob/main/docs/media/Structure.png?raw=true" alt="structure">
</div>

<br>

## 1. Install dependencies

### Prerequisites

Before starting this project you need to install into your system:
- Docker and Docker Compose
- Nvidia Drivers

<br>

### 1.1. Docker and Docker Compose:
```sh
sudo curl -L https://github.com/docker/compose/releases/download/v2.32.4/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo apt install -y docker.io
```

<br>

### 1.2. NVIDIA drivers (ONLY IF YOU PLAN USING GPU):
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

### 1.2.1. Test the NVIDIA GPU and if everything is ok - proceed
```sh
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

<br>

## 2. Building and Running the Container Stacks

### 2.1. **Clone the repository**: 
```sh
git clone https://github.com/kaunofakultetas/gateopener-server.git
cd gateopener-server
```


### 2.2. **Start the neural network containers**:
```sh
cd gate-neural
./runUpdateThisStack.sh
cd ..
```

### 2.3. **Edit the processing stack environment variables and start containers**:
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
