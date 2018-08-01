# Anakin 2.0 And Docker
---

## Requirement

+ You should install docker in you local os.
+ Please use [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))  build and run all `NVIDIA GPU` docker images.

## Usage

You are recommended to use `anakin_docker_build_and_run.sh` script to build and run anakin docker.

```bash
Usage: anakin_docker_build_and_run.sh -p <place> -o <os> -m <Optional>

Options:

   -p     Hardware Place where docker will running [ NVIDIA-GPU / AMD_GPU / X86-ONLY / ARM ]
   -o     Operating system docker will reside on [ Centos / Ubuntu ]
   -m     Script exe mode [ Build / Run / All] default mode is build and run
```

### GPU Docker
#### Build Image
```bash
$/usr/bash anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Build
or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Build
```

#### Run docker
```bash
$/usr/bash anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Run
or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Run
```

### AMD Docker
#### Build Image
```bash
$/usr/bash anakin_docker_build_and_run.sh  -p AMD-GPU -o Centos -m Build
or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p AMD-GPU -o Centos -m Build
```

#### Run docker
```bash
$/usr/bash anakin_docker_build_and_run.sh  -p AMD-GPU -o Centos -m Run
or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p AMD-GPU -o Centos -m Run
```
### X86 Docker

> Not support yet

### ARM Docer

> Not support yet
