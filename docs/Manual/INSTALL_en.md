## Building Anakin from Source ##

We've built and tested Anakin on CentOS 7.3 and Ubuntu 16.04. The other operating systems installation are coming soon.

### Installation overview ###

* [Installing on CentOS](#0001)
* [Installing on Ubuntu](#0002)
* [Installing on ARM](#0003)
* [Verifying installation](#0004)


### <span id = '0001'> Installing on CentOS </span> ###

#### 1. System requirements ####

*  make 3.82+
*  cmake 2.8.12+
*  gcc 4.8.2+
*  g++ 4.8.2+
*  protobuf 3.4.0

1.1 install protobuf  
    Download source from https://github.com/google/protobuf/releases/tag/v3.4.0  
  >tar -xzf protobuf-3.4.0.tar.gz  
    >$ cd protobuf-3.4.0   
    >$ ./autogen.sh  
    >$ ./configure    
    >$ make  
    >$ make check   
    >$ make install  

   Check  
  >$ protoc --version  

  Any problems for protobuf installation, Please see [here](https://github.com/google/protobuf/blob/master/src/README.md)
#### 2. Building Anakin for CPU-only ####

Please make sure that all pre-requirements have been installed on your system.

2.1 Clone Anakin and build

```bash
$ git clone https://github.com/PaddlePaddle/Anakin.git  
$ cd Anakin 
$ git checkout developing  
# tools directory contains multi-platform build scripts.
# 1. using script to build.
$ ./tools/x86_build.sh
# 2. or you can build directly. This way you can build x86 and gpu  
# at the same time by set corresponding variables(USE_GPU_PLACE, 
# USE_X86_PLACE, USE_ARM_PLACE) in CMakeList.txt. In this situation, 
# set USE_X86_PLACE with YES, others with NO.
$ mkdir build  
$ cd build  
$ cmake ..  
$ make   
```

#### 3. Building Anakin with NVIDIA GPU Support ####

- 3.1. Install dependences
- 3.1.1 CUDA Tookit
  - [CUDA 8.0](https://developer.nvidia.com/cuda-zone) or higher. For details, see [NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
  - [cuDNN v7](https://developer.nvidia.com/cudnn). For details, see [NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).  
  When you have installed cuDNN v7, please set the following variable to your `bashrc` 
     ```bash
      # CUDNN
      export CUDNN_ROOT=/usr/local/cudnn_v7 # the path where your cuDNN installed.
      export LD_LIBRARY_PATH=${CUDNN_ROOT}/lib64:$LD_LIBRARY_PATH
      export CPLUS_INCLUDE_PATH=${CUDNN_ROOT}/include:$CPLUS_INCLUDE_PATH
     ```
          Don't forget to source your bashrc

- 3.2. Clone Anakin and build
```bash
$ git clone https://github.com/PaddlePaddle/Anakin.git  
$ cd Anakin 
$ git checkout developing  
# tools directory contains multi-platform build scripts.
# 1. using script to build.
$ ./tools/gpu_build.sh
# 2. or you can build directly. This way you can build x86 and gpu  
# at the same time by set corresponding variables(USE_GPU_PLACE, 
# USE_X86_PLACE, USE_ARM_PLACE) in CMakeList.txt. In this situation, 
# set USE_GPU_PLACE with YES, others with NO.
$ mkdir build  
$ cd build  
$ cmake ..  
$ make   
```

#### 4. Building Anakin with AMD GPU Support ####

Coming soon..

### <span id = '0002'> Installing on Ubuntu </span> ###
We only support [docker installation](../../docker/README.md) on ubuntu for now.

### <span id = '0003'> Installing on ARM </span> ###

Please refer to [run on arm](run_on_arm_en.md)

### <span id = '0004'> Verifying installation </span> ###

If build successfully, the libs will be in the directory `output/`, and you can run unit test in `output/unit_test` to verify your installation.


