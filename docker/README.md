# Anakin 2.0 And Docker
---

## Requirement

> 1. You should install docker in you local os.
> 2. Please use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)  run all GPU docker images.


## Usage

### GPU Docker
#### Build Image
```bash
$docker build --network=host -t your_docker_image_name:tag_info /path/to/Dockerfile -f Dockerfile
```

#### Run docker
```bash
$systemctl start nvidia-docker
$nvidia-docker run --network=host -it your_docker_image_name:tag_info /bin/bash
```

### CPU Docker

> Not support yet

### ARM Docer

> Not support yet
