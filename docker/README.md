# Anakin 2.0 And Docker
---

## Requirement

> 1. You should install docker in you local os.
> 2. Please use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)  build and run all GPU docker images.


## Usage

### GPU Docker
#### Build Image
```bash
$nvidia-docker build --network=host -t your_docker_image_name:tag_info /path/to/Dockerfile -f Dockerfile
```

#### Run docker
```bash
$systemctl start nvidia-docker
$export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
$export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
$export BINS=$(\ls /usr/bin/nvidia* | xargs -I{} echo '-v {}:{}')
$nvidia-docker run ${CUDA_SO} ${DEVICES} ${BINS} --network=host -it your_docker_image_name:tag_info /bin/bash
```

### X86 Docker

> Not support yet

### ARM Docer

> Not support yet
