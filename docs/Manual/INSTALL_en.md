## Building Anakin from Source ##

We've built and tested Anakin on CentOS 7.3. The other operating systems installation are coming soon.

### Installation overview ###

* [Installing on CentOS](#0001)
* [Installing on Ubuntu](#0002)
* [Installing on ARM](run_on_arm_en.md)
* [Verifying installation](#0004)


### <span id = '0001'> Installing on CentOS </span> ###

#### 1. System requirements ####

*  make 3.82+
*  cmake 2.8.12+
*  gcc 4.8.2+
*  g++ 4.8.2+

#### 2. Building Anakin for CPU-only ####

Not support yet.

#### 3. Building Anakin with NVIDIA GPU Support ####

- 3.1. Install dependences
 - 3.1.1 protobuf 3.4.0  
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

  - 3.1.2 CUDA Tookit
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

   


- 3.2. Compile Anakin
  >$ git clone https:/xxxxx  
  >$ cd anakin  
  >$ mkdir build  
  >$ cd build  
  >$ cmake ..  
  >$ make   

#### 4. Building Anakin with AMD GPU Support ####

Coming soon..


### <span id = '0002'> Installing on Ubuntu </span> ###

Coming soon..


### <span id = '0003'> Installing on ARM </span> ###

Coming soon..

### <span id = '0004'> Verifying installation </span> ###


