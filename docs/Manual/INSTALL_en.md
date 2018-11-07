## Building Anakin from Source ##

We've built and tested Anakin on CentOS 7.3. The other operating systems installation are coming soon.

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

 For more detials of ROCm please see [RadeonOpenCompute/ROCm](https://github.com/RadeonOpenCompute/ROCm) 
 
- 4.1. Setup Environment   

 - 4.1.1 Update OS (Option, if your OS is able to be updated)     
    >$sudo yum update
    
 - 4.1.2 Add ROCM repo   
    Create a /etc/yum.repos.d/rocm.repo file with the following contents:
    ```bash
    [ROCm]
    name=ROCm
    baseurl=http://repo.radeon.com/rocm/yum/rpm
    enabled=1
    gpgcheck=0
    ```
    
 - 4.1.3 Install ROCK-DKMS     
    Please check your kernel version before installing ROCk-DKMS and make sure the result is same as your installed kernel related packages, such as kernel-headers and kerenl-devel
    >$ uname -r 
  
  - 4.1.3.1 For kernel ver 3.10.0-`693` (Option 1)     
     Download kernel-devel-3.10.0-693.el7.x86_64.rpm and kernel-headers-3.10.0-693.el7.x86_64.rpm    
     >$sudo yum install kernel-devel-3.10.0-693.el7.x86_64.rpm  kernel-headers-3.10.0-693.el7.x86_64.rpm    
     
  - 4.1.3.2 For kernel ver 3.10.0-`862`  (Option 2)
     >$ sudo yum install kernel-devel kernel-headers    
     
  - 4.1.3.3 Install ROCk-DKMS   
     >$ sudo yum install epel-release   
     >$ sudo yum install dkms   
     >$ sudo yum install rock-dkms  
      
     Use below command to check amdgpu is installed successful or not.    
     >$ sudo dkms status    
     >$ 'amdgpu, 1.8-151.el7, ..., x86_64: installed (original_module exists)'    
      
  - 4.1.3.4    
     Reboot your device.
     
 ** If you are using docker than step 4.1.4 to 4.1.9 are not required **
 
 - 4.1.4 Install ROCm-OpenCL
    >$sudo yum install rocm-opencl rocm-opencl-devel rocm-smi rocminfo
 
 - 4.1.5 Install MIOpenGemm and necessary pckage
    >$sudo yum install miopengemm rocm-cmake openssl-devel

 - 4.1.6 Add user to the video (or wheel) group 
    >$sudo usermod -a -G video $LOGNAME 
    
 - 4.1.7 Setting Environment variables
    ```bash
    echo 'export PATH=/opt/rocm/bin:/opt/rocm/opencl/bin/x86_64:$PATH' >> $HOME/.bashrc
    echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> $HOME/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/lib64' >>$HOME/.bashrc
    source ~/.bashrc
    ```
   Check 
    >$ clinfo
 
 - 4.1.8 protobuf 3.4.0  
    Download source from https://github.com/google/protobuf/releases/tag/v3.4.0
    >tar -zxvf protobuf-cpp-3.4.0.tar.gz  
    >$ cd protobuf-3.4.0    
    >$ ./configure    
    >$ make  
    >$ make install  

   Check  
    >$ protoc --version

    Any problems for protobuf installation, Please see [here](https://github.com/google/protobuf/blob/master/src/README.md)
    
 - 4.1.9 cmake 3.2.0
    Download source from https://cmake.org/files/v3.2/cmake-3.2.0.tar.gz
    >tar -zxvf cmake-3.2.0.tar.gz  
    >$ cd cmake-3.2.0
    >$ ./bootstrap   
    >$ make -j4    
    >$ make install  
    
- 4.2. Compile Anakin
  >$ git clone xxx  
  >$ cd anakin  
  >$ ./tools/amd_gpu_build.sh
 
- 4.3. Run Benchmark
  >$ cd output/unit_test    
  >$ benchmark ../../benchmark/CNN/models/ vgg16.anakin.bin 1 2 100

### <span id = '0002'> Installing on Ubuntu </span> ###

#### 1. Building Anakin with AMD GPU Support ####

For more detials of ROCm please see [RadeonOpenCompute/ROCm](https://github.com/RadeonOpenCompute/ROCm) 

- 1.1. Setup Environment

 - 1.1.1 Update OS (Option, if your OS is able to be updated)
    >$sudo apt-get update

 - 1.1.2 Add ROCM apt repo
    For Debian based systems, like Ubuntu, configure the Debian ROCm repository as follows:
    >$wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
    >$echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

 - 1.1.3 Install ROCK-DKMS
    Please check your kernel version before installing ROCk-DKMS and make sure the result is same as your installed kernel related packages, such as kernel-headers and kerenl-devel
    >$ uname -r

  - 1.1.3.1 Removing pre-release packages (Option)
     Before proceeding, make sure to completely [uninstall any previous ROCm package](https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages)
     >$ sudo apt purge hsakmt-roct
     >$ sudo apt purge hsakmt-roct-dev
     >$ sudo apt purge compute-firmware
     >$ sudo apt purge $(dpkg -l | grep 'kfd\|rocm' | grep linux | grep -v libc | awk '{print $2}')

  - 1.1.3.2 Install ROCk-DKMS
     >$ sudo apt-get update
     >$ sudo apt-get install rock-dkms

     Use below command to check amdgpu is installed successful or not.
     >$ sudo dkms status
     >$ 'amdgpu, 1.8-151.el7, ..., x86_64: installed (original_module exists)'

  - 1.1.3.3
     Reboot your device.

 ** If you are using docker than step 1.1.4 to 1.1.9 are not required **

 - 1.1.4 Install ROCm-OpenCL
    >$ sudo apt-get install rocm-opencl rocm-opencl-devel rocm-smi rocminfo

 - 1.1.5 Install MIOpenGemm and necessary pckage
    >$ sudo apt-get install miopengemm rocm-cmake libssl-dev libnuma-dev

 - 1.1.6 Add user to the video (or wheel) group
    >$ sudo usermod -a -G video $LOGNAME

 - 1.1.7 Setting Environment variables
    >$ echo 'export LD_LIBRARY_PATH=/opt/rocm/opencl/lib/x86_64:/opt/rocm/hsa/lib:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/rocm.sh
    >$ echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh
    Check
    >$ clinfo

 - 1.1.8 protobuf 3.4.0
    Download source from https://github.com/google/protobuf/releases/tag/v3.4.0
    >tar -zxvf protobuf-cpp-3.4.0.tar.gz
    >$ cd protobuf-3.4.0
    >$ ./configure
    >$ make
    >$ make install

   Check
    >$ protoc --version

    Any problems for protobuf installation, Please see [here](https://github.com/google/protobuf/blob/master/src/README.md)

 - 1.1.9 cmake 3.2.0
    Download source from https://cmake.org/files/v3.2/cmake-3.2.0.tar.gz
    >$ tar -zxvf cmake-3.2.0.tar.gz
    >$ cd cmake-3.2.0
    >$ ./bootstrap
    >$ make -j4
    >$ make install

- 1.2. Compile Anakin
  >$ git clone xxx
  >$ cd anakin
  >$ ./tools/amd_gpu_build.sh

- 1.3. Run Benchmark
  >$ cd output/unit_test
  >$ benchmark ../../benchmark/CNN/models/ vgg16.anakin.bin 1 2 100

### <span id = '0003'> Installing on ARM </span> ###

Please refer to [run on arm](run_on_arm_en.md)

### <span id = '0004'> Verifying installation </span> ###

If build successfully, the libs will be in the directory `output/`, and you can run unit test in `output/unit_test` to verify your installation.


