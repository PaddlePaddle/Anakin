# **Paddle Anakin**

## 目录:

+ User Manual
  + Instal and Compile
    - Docker
    - Builde from source
  + Run on ARM
    - Android
    - Linux
    - IOS
    - External Converter
  - Examples
  + Benchmark
    - NV GPU
    - ARM
    - More devices
+ Developing Guide
  + C++ APIs
    - Anakin working principle
    - APIs 
    - Code example
  - How to contribute
  - How to add custom operators
  - How to add new device


## User Manual
---

本节主要包含以下五个方面内容：
  
+ [Instal and Compile](#10001)
+ [Run on ARM](#10002)
+ [External Converter](#10003)
+ [Examples](#10004)
+ [Benchmark](#10005)


## <span id = '10001'> Instal and Compile </span>
---
### Docker

#### Requirement

+ 本地系统需要安装docker工具
+ 请使用[nvidia-docker2](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))工具进行编译运行`NVIDIA GPU` docker

##### Usage

请使用`anakin_docker_build_and_run.sh`脚本文件进行编译运行anakin代码

```bash
Usage: anakin_docker_build_and_run.sh -p <place> -o <os> -m <Optional>

Options:

   -p     Hardware Place where docker will running [ NVIDIA-GPU / AMD_GPU / X86-ONLY / ARM ]
   -o     Operating system docker will reside on [ Centos / Ubuntu ]
   -m     Script exe mode [ Build / Run / All] default mode is build and run
```

#### GPU Docker

- Build Image

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Build
or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Build
```

- Run docker

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Run
or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Run
```

#### X86 Docker

> 暂时不支持

#### ARM Docer

> 暂时不支持

### Builde from source

我们已经在CentOS 7.3上成功的安装和测试了Anakin，对于其他操作系统，我们将很快支持

安装概览:

+ [在CentOS上安装 Anakin](#12001)
+ [在Ubuntu上安装 Anakin](#12002)
+ [在ARM上安装 Anakin](#12003)
+ [验证安装](#12004)

 
#### <span id = '12001'> 在CentOS上安装 Anakin </span>

<span id = ''> 1. 系统要求 </span>

*  make 3.82+
*  cmake 2.8.12+
*  gcc 4.8.2+
*  g++ 4.8.2+
*  其他需要补充的。。。

<span id = ''> 2. 编译CPU版Anakin </span>

> 暂时不支持

<span id = ''> 3. 编译支持NVIDIA GPU的Anakin </span>

  3.1. 安装依赖

    3.1.1 protobuf  

    >$ git clone https://github.com/google/protobuf  
    >$ cd protobuf  
    >$ git submodule update --init --recursive  
    >$ ./autogen.sh  
    >$ ./configure --prefix=/path/to/your/insall_dir  
    >$ make  
    >$ make check  
    >$ make install  
    >$ sudo ldconfig


    如安装protobuf遇到任何问题，请访问[这里](https://github.com/google/protobuf/blob/master/src/README.md)

  3.2 CUDA Toolkit

  - [CUDA 8.0](https://developer.nvidia.com/cuda-zone) or higher. 具体信息参见[NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
  - [cuDNN v7](https://developer.nvidia.com/cudnn). 具体信息参见[NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

  3.3  编译Anakin

  >$ git clone https:/xxxxx  
  >$ cd anakin  
  >$ mkdir build  
  >$ camke ..  
  >$ make

<span id = ''> 4. 编译支持AMD GPU的Anakin </span>

  > 暂时不支持

#### <span id = '12002'> 在Ubuntu上安装 Anakin </span>

> 暂时不支持

#### <span id = '12003'> 在ARM上安装 Anakin </span>

详情请参考[Run on ARM](#10002)

#### <span id = '12004'> 验证安装 </span>

> we are coming soon...


## <span id = '10002'> Run on ARM </span>
---

目前Anakin支持ARM Android平台，采用Android NDK交叉编译工具链，已在mac os和centos上编译和测试通过

安装概览

+ [系统需求](#13001)
+ [安装第三方依赖](#13002)
+ [Anakin源码编译](#13003)
+ [验证安装](#13004)

<span id = '13001'> 1. 系统需求 </span>

*  宿主机: linux, mac    
*  cmake 3.8.2+    
*  Android NDK r14, Linux 版本[从这里下载](https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip)

<span id = '13002'> 2. 安装第三方依赖 </span>

  2.1 protobuf3.4.0  

    源码从这里[下载](https://github.com/google/protobuf/releases/tag/v3.4.0)   

    2.1.1 为宿主机编译protobuf 

```bash
   $ tar -xzf protobuf-3.4.0.tar.gz  
   $ cd protobuf-3.4.0   
   $ ./autogen.sh  
   $ ./configure    
   $ make  
   $ make check   
   $ make install
```

   上述 $make install 执行后，可在 /usr/local/include/google 找到 libprotobuf 所需的头文件,将整个google文件夹拷贝至Anakin/third-party/arm-android/protobuf/下，
   如有问题，请点[这里](https://github.com/google/protobuf/blob/v3.4.0/src/README.md)。
   然后将已经生成文件清除。

```bash
   $ make distclean
   ```

    2.1.2 交叉编译Android`armeabi-v7a`的protobuf，注意设置ANDROID_NDK的路径，以及ARCH_ABI、HOSTOSN的值

```bash
   $ export ANDROID_NDK=your_ndk_path 
   $ ARCH_ABI="arm-linux-androideabi-4.9"
   $ HOSTOSN="darwin-x86_64"
   $ export SYSROOT=$ANDROID_NDK/platforms/android-9/arch-arm  
   $ export PREBUILT=$ANDROID_NDK/toolchains/$ARCH_ABI
   $ export LDFLAGS="--sysroot=$SYSROOT"
   $ export LD="$ANDROID_NDK/toolchains/$ARCH_ABI/prebuilt/$HOSTOSN/arm-linux-androideabi/bin/ld $LDFLAGS"
   $ export LIBS="-llog $ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/libgnustl_static.a"
   $ export CPPFLAGS=""
   $ export INCLUDES="-I$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/include/ -I$ANDROID_NDK/platforms/android-9/arch-arm/usr/include/ -I$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include/"
   $ export CXXFLAGS="-march=armv7-a -mfloat-abi=softfp -DGOOGLE_PROTOBUF_NO_RTTI --sysroot=$SYSROOT"
   $ export CCFLAGS="$CXXFLAGS"
   $ export CXX="$PREBUILT/prebuilt/$HOSTOSN/bin/arm-linux-androideabi-g++ $CXXFLAGS"
   $ export CC="$CXX"
   $ export RANLIB="$ANDROID_NDK/toolchains/$ARCH_ABI/prebuilt/$HOSTOSN/bin/arm-linux-androideabi-ranlib"  
   $ ./autogen.sh  
   $ ./configure --host=arm-linux-androideabi --with-sysroot=$SYSROOT --enable-cross-compile --with-protoc=protoc --disable-shared CXX="$CXX" CC="$CC" LD="$LD"  
   $ make
```
  
    > 编译生成 *.a 静态库，若希望编译*.so 动态链接库 ，请在./configure参数中改--disable-shared为--disable-static --enable-shared

    > 生成文件在src/.libs/下，将生成的文件拷贝至Anakin/third-party/arm-android/protobuf/lib下

    > 在[cmake](../cmake/find_modules.cmake)中更新`ARM_RPOTO_ROOT`的路径

```cmake
  set(ARM_RPOTO_ROOT "${CMAKE_SOURCE_DIR}/third-party/arm-android/protobuf")
```

  2.2 opencv 2.4.3+(optional)    

    * Anakin只在examples示例中使用opencv   
    * Android系统的opencv从[这里下载](https://opencv.org/releases.html)    
    * 解压后将 `3rdparty/libs/armeabi-v7a`中的库文件拷贝到`libs/armeabi-v7a`    
    在[cmake](../../cmake/find_modules.cmake)中搜索`anakin_find_opencv`, 
    并设置 `include_directories` 和 `LINK_DIRECTORIES`为自己安装的库的路径

    ```cmake
    include_directories(${CMAKE_SOURCE_DIR}/third-party/arm-android/opencv/sdk/native/jni/include/)
    LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third-party/arm-android/opencv/sdk/native/libs/armeabi-v7a/)
    ```

<span id = '13003'> 3. Anakin源码编译 </span>

  详情请见[Android](#0000)

<span id = '13004'> 4. 验证安装 </span> 

  * 编译好的库会放在目录`${Anakin_root}/output`下；    
  *  编译好的单测文件会放在`${Anakin_root}/output/unit_test`目录下     
  *  编译好的示例文件会放在`${Anakin_root}/output/examples`目录下 
  
  对于Android系统，打开设备的调试模式，通过ADB可以访问的目录是`data/local/tmp`，通过ADB push将测试文件、模型和数据发送到设备目录， 运行测试文件 

### <span id = '0000'> Android </span>

<span id = ''> 1. 克隆[源码](https://github.com/PaddlePaddle/Anakin/tree/arm) </span>

```bash
    cd your_dir
    git clone https://github.com/PaddlePaddle/Anakin.git
    cd Anakin
    git fetch origin arm
    git checkout arm
```

<span id = ''> 2. 修改`android_build.sh` </span>   

  2.1. 修改NDK路径

  ```bash
    #modify "your_ndk_path" to your NDK path
    export ANDROID_NDK=your_ndk_path
  ```

  2.2. 修改ARM 处理器架构 

  * 对于32位ARM处理器, 将ANDROID_ABI 设置为 `armeabi-v7a with NEON`， 
  * 对于64位ARM处理器, 可以将ANDROID_ABI 设置为 `armeabi-v7a with NEON`或者`arm64-v8a`。        
  * 目前我们只支持 `armeabi-v7a with NEON`；`arm64-v8a` 还在开发中。 

  ```bash
      -DANDROID_ABI="armeabi-v7a with NEON"
  ```

  2.3. 设置Android API    

  根据Android系统的版本设置API level， 例如API Level 21 -> Android 5.0.1   

  ```bash
      -DANDROID_NATIVE_API_LEVEL=21
  ```

  2.4. 选择编译静态库或动态库  

  设置`BUILD_SHARED=NO`编译静态库    
  设置`BUILD_SHARED=YES`编译动态库 

  ```bash
      -DBUILD_SHARED=NO
  ```
  2.5. OpenMP多线程支持    

  设置`USE_OPENMP=YES`开启OpenMP多线程   

  ```bash
      -DUSE_OPENMP=YES
  ```
  
  2.6. 编译单测文件  

  设置`BUILD_WITH_UNIT_TEST=YES`将会编译单测文件   

    ```bash
        -DBUILD_WITH_UNIT_TEST=YES
    ```

  2.7. 编译示例文件   

  设置`BUILD_EXAMPLES=YES`将会编译示例文件

    ```bash
        -DBUILD_EXAMPLES=YES
    ```

  2.8. 开启opencv   

  如果使用opencv，设置`USE_OPENCV=YES`    

    ```bash
        -DUSE_OPENCV=YES
    ```
    
  2.9. 开始编译   

  运行脚本 `android_build.sh` 将自动编译Anakin   

  ```bash
      ./android_build.sh
  ```

### Linux

> 暂时不支持

### IOS

> 暂时不支持


## <span id = '10003'> External Converter </span>
---
本节将介绍如何将其他 model转换为Anakin model

<span id = ''> 1. 系统要求 </span>

- python 2.7+
- pyyaml
- flask

<span id = ''> 2. 下载转换代码 </span>

```bash
git clone https://xxxxxxxxx
``` 

<span id = ''> 3. 使用 </span>

  3.1. 配置

    对工程目录中*config.yaml* 文件进行相关配置，具体配置流程如下：

```bash
OPTIONS:
    Framework: CAFFE # select a target dl-framework you want parsing
    SavePath: ./output
    ResultName: googlenet # the name you want when saving the parsed model
    Config:
        LaunchBoard: ON  # should be on if you want to launch graph board
        Server:
            ip: 0.0.0.0
            port: 8888
        OptimizedGraph:  # only enable(set enable(ON) and path) when you have optimized graph model.
            enable: ON
            path: /path/to/anakin_optimized_anakin_model/googlenet.anakin.bin.saved
    LOGGER:
        LogToPath: ./log/ # the path where log
        WithColor: ON  # colorful log message

TARGET:
    CAFFE:
        # path to proto files
        ProtoPaths:
            - /path/to/caffe/src/caffe/proto/caffe.proto
        PrototxtPath: /path/to/your/googlenet.prototxt
        ModelPath: /path/to/your/googlenet.caffemodel
   
  # not support yet 
    PADDLE:
        # path to proto files   
        ProtoPath:
            - /path/to/proto_0
            - /path/to/proto_1
            - /path/to/proto_n
        PrototxtPath: /path/to/prototxt
        ModelPath: /path/to/model
  # ... 
```

  3.2. 转换 

    完成相关配置后，就可以运行```python converter.py```脚本进行转换

  3.3. Launching dash board

    在转换完成后，转换后的Anakin model可在 http://0.0.0.0:8888 （可配置） 进行下载。

    > 如果你在配置文件中将远程服务器ip地址设置为 0.0.0.0, 则当打开本地的浏览器时，相应的地址栏中需要输入服务器的真实ip地址，而不是 0.0.0.0

  3.4. Note

    > 我们目前只支持caffe model的转换


## <span id = '10004'> Examples </span>
---

**Anakin**目前只支持NCHW的格式，示例文件在test/framework/net下

### 在NV的GPU上运行CNN模型

示例文件为example_nv_cnn_net.cpp，整体流程如下：

  - 将模型的的path设置为Anakin模型的路径，初始化NV平台的图对象。 anakin模型可以通过[转换工具](#10003)进行转换获得  
  - 根据模型设置网络图的输入尺寸，进行图优化
  - 根据优化后的网络图初始化网络执行器
  - 取出网络的输入tensor，将数据拷贝到输入tensor
  - 运行预测
  - 取出网络的输出tensor

> 以NV平台为例演示Anakin框架的使用方法，注意编译时需要打开GPU编译开关 

### 在X86上运行RNN模型

示例文件为example_x86_rnn_net.cpp

整体流程与在NV的GPU上运行CNN模型相似，不同之处如下：

  - 使用X86标识初始化图对象和网络执行器对象
  - rnn模型的输入尺寸是可变的，初始化图时的输入维度是维度的最大值，输入维度N代表总的词的个数
  - 设置输入tensor的seq_offset来标示这些词是如何划分为句子的,如{0,5,12}表示共有12个词，其中第0到第4个词是第一句话，第5到第11个词是第二句话

> 以X86平台为例演示Anakin框架的使用方法，注意编译时需要打开X86编译开关 

### 在NV的GPU上使用Anakin的线程池运行CNN模型 

示例文件为example_nv_cnn_net_multi_thread.cpp ，示例使用worker的同步预测接口

整体流程与在NV的GPU上运行CNN模型相似，不同之处如下：

  - 用模型地址和线程池大小初始化worker对象
  - 将输入tensor注入任务队列,获得输出tensor


## <span id = '10005'> Benchmark </span>
---

### NV GPU

1. Machine

  >  CPU: `12-core Intel(R) Xeon(R) CPU E5-2620 v2 @2.10GHz`
  >  GPU: `Tesla P4`  
  >  cuDNN: `v7`  

2. Anakin

  **`NVIDIA TensorRT`** 是公认的高性能前向预测引擎，故在BenchMark中本文将使用**`NVIDIA TensorRT 3`**与**`Anakin`**进行性能对比分析

3. Benchmark Model  

  本节主要列举了CNN model分别在 `Anakin` 和 `TenorRT3`框架上的前向预测性能数据
  你可以使用预训练好的caffe model或你自己训练好的caffe model进行性能测试

  > 注意在性能测试之前，请先将测试model通过[External Converter](#10003)转换为Anakin model
  > 对这些model，本文在单卡上进行单线程的不同batch size测试

- [Vgg16](#1)   *caffe model 可以在[这儿](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)下载*
- [Yolo](#2)  *caffe model 可以在[这儿](https://github.com/hojel/caffe-yolo-model)下载*
- [Resnet50](#3)  *caffe model 可以在[这儿](https://github.com/KaimingHe/deep-residual-networks#models)下载*
- [Resnet101](#4)  *caffe model 可以在[这儿](https://github.com/KaimingHe/deep-residual-networks#models)下载*
- [Mobilenet v1](#5)  *caffe model 可以在[这儿](https://github.com/shicai/MobileNet-Caffe)下载*
- [Mobilenet v2](#6)  *caffe model 可以在[这儿](https://github.com/shicai/MobileNet-Caffe)下载*
- [RNN](#7)  *暂不支持t*

  <span id = '1'> 3.1. VGG16 </span>  

  - Latency (`ms`) of different batch

  BatchSize | TensorRT | Anakin
    :---: | :---: | :---: |
    1 | 8.8690 | 8.2815
    2 | 15.5344 | 13.9116
    4 | 26.6000 | 21.8747 
    8 | 49.8279 | 40.4076 
    32 | 188.6270 | 163.7660 

  - GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 963 | 997
    2 | 965 | 1039
    4 | 991 | 1115
    8 | 1067 | 1269
    32 | 1715 | 2193

    
  <span id = '2'> 3.2. Yolo </span>  

  - Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 16.4596| 15.2124
    2 | 26.6347| 25.0442 
    4 | 43.3695| 43.5017
    8 | 80.9139 | 80.9880
    32 | 293.8080| 310.8810

  - GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1569 | 1775
    2 | 1649 | 1815
    4 | 1709 | 1887
    8 | 1731 | 2031
    32 | 2253 | 2907

  <span id = '3'> 3.3. Resnet50 </span> 

  - Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 4.2459   |  4.1061 
    2 |  6.2627  |  6.5159 
    4 | 10.1277  | 11.3327
    8 | 17.8209 |   20.6680 
    32 | 65.8582 | 77.8858

  - GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 531  | 503
    2 | 543  | 517
    4 | 583 | 541
    8 | 611 | 589
    32 |  809 | 879

  <span id = '4'> 3.4. Resnet101 </span> 

  - Latency (`ms`) of different batch 

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 7.5562 | 7.0837  
    2 | 11.6023 | 11.4079
    4 | 18.3650 | 20.0493 
    8 | 32.7632 | 36.0648
    32 | 123.2550 | 135.4880

  - GPU Memory Used (`MB)`

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 701  | 683
    2 | 713  | 697
    4 | 793 | 721
    8 | 819 | 769
    32 | 1043 | 1059
 

  <span id = '5'> 3.5. MobileNet V1 </span> 

  - Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 45.5156  |  1.3947
    2 |  46.5585  |  2.5483
    4 | 48.4242  | 4.3404
    8 |  52.7957 |  8.1513
    32 | 83.2519 | 31.3178

  - GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 329  | 283
    2 | 345   | 289
    4 | 371 | 299
    8 | 393 | 319
    32 |  531 | 433

  <span id = '6'> 3.6. MobileNet V2</span> 

  - Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 65.6861 | 2.9842
    2 | 66.6814 | 4.7472
    4 | 69.7114 | 7.4163
    8 | 76.1092 | 12.8779
    32 | 124.9810 | 47.2142

  - GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 341 | 293
    2 | 353 | 301
    4 | 385 | 319
    8 | 421 | 351
    32 | 637 | 551

4. 怎么运行 Benchmark models?

  > 首先, 使用[External Converter](#10003)对caffe model 进行转换  

  > 然后进入 *source_root/benchmark/CNN* 目录， 使用 'mkdir ./models' 创建 ./models 目录，并把转换好的Anakin models 放在该目录下 

  > 接着运行 'sh run.sh'命令  

  > 最后，终端显示器上将会打印该模型的运行时间

  > 如果你想看到model中每个OP的运行时间，只用修改 CMakeLists.txt 文件，将 `ENABLE_OP_TIMER` 修改为 `YES`，然后进行编译运行，最后终端显示器上会打印出每个OP的运行时间

### ARM

1. Machine

  + 测试模型Mobilenetv1, mobilenetv2, mobilenet-ssd
  + 采用android ndk交叉编译，gcc 4.9，enable neon， ABI： armveabi-v7a with neon -mfloat-abi=softfp
  + 测试平台
  - 荣耀v9(root): 处理器:麒麟960, 4 big cores in 2.36GHz, 4 little cores in 1.8GHz
  - ubia z17:处理器:高通835, 4 big cores in 2.36GHz, 4 little cores in 1.9GHz
  - 360 N5:处理器:高通653, 4 big cores in 1.8GHz, 4 little cores in 1.4GHz
  + 多线程：openmp
  + 时间：warmup10次，运行10次取均值

2. Anakin

  在BenchMark中本文将使用**`ncnn`**、**`TFlite`**和**`Anakin`**进行性能对比分析

3. BenchMark model

  > 注意在性能测试之前，请先将测试model通过[External Converter](#10003)转换为Anakin model

  > 对这些model，本文在ARM上进行多线程的单batch size测试。

  - [Mobilenet v1](#11)  *caffe model 可以在[这儿](https://github.com/shicai/MobileNet-Caffe)下载*
  - [Mobilenet v2](#22)  *caffe model 可以在[这儿](https://github.com/shicai/MobileNet-Caffe)下载*
  - [mobilenet-ssd](#33)  *caffe model 可以在[这儿](https://github.com/chuanqi305/MobileNet-SSD)下载*

    <span id = '11'> 3.1. mobilenetv1 </span>

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|107.7ms|61.1ms|38.2ms|152.8ms|85.2ms|51.9ms|152.6ms|nan|nan|
   |高通835|105.7ms|63.1ms|~~46.8ms~~|152.7ms|87.0ms|~~92.7ms~~|146.9ms|nan|nan|
   |高通653|120.3ms|64.2ms|46.6ms|202.5ms|117.6ms|84.8ms|158.6ms|nan|nan| 

    <span id = '22'> 3.2. mobilenetv2 </span>

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|93.1ms|53.9ms|34.8ms|144.4ms|84.3ms|55.3ms|100.6ms|nan|nan|
   |高通835|93.0ms|55.6ms|41.1ms|139.1ms|88.4ms|58.1ms|95.2ms|nan|nan|
   |高通653|106.6ms|64.2ms|48.0ms|199.9ms|125.1ms|98.9ms|108.5ms|nan|nan|

    <span id = '33'> 3.3. mobilenet-ssd </span>

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|213.9ms|120.5ms|74.5ms|307.9ms|166.5ms|104.2ms|nan|nan|nan|
   |高通835|213.0ms|125.7ms|~~98.4ms~~|292.9ms|177.9ms|~~167.8ms~~|nan|nan|nan|
   |高通653|236.0ms|129.6ms|96.0ms|377.7ms|228.9ms|165.0ms|nan|nan|nan|

4. 怎么运行 Benchmark models?

  > 首先, 使用[External Converter](#10003)对caffe model 进行转换 

  > 然后将转换后的Anakin model和编译好的benchmark_arm 二进制文件通过'adb push'命令上传至测试机  

  > 接着在测试机含有Anakin model的目录中运行'./benchmark_arm ./ anakin_model.anakin.bin 1 10 10 1' 命令  

  > 最后，终端显示器上将会打印该模型的运行时间  

  > 其中运行命令的参数个数和含义可以通过运行'./benchmark_arm'看到   

### More devices

#### CPU

1. Machine

  > CPU: `12-core Intel(R) Xeon(R) CPU E5-2620 v2 @2.10GHz`
  > Docker 1.13.1
  > CentOS Linux release 7.5.1804

2. Anakin

  > 在BenchMark中本文将使用**`Tensorflow 1.8.0`**和**`Anakin`**进行性能对比分析, `Tensorflow 1.8.0` 是通过 Anaconda 4.5.4 进行安装，python版本号是Python 3.6

  > Tensorflow 是使用 python api 进行运行, tensorfow 的线程数是实际处理的进程数目，你可以通过运行 ` sh benchmark_tensorflow.sh` 脚本进行测试

  > Anakin 是利用 api 进行运行, 并设置 openmp thread pool = 1, mkl thread pool=1. 你可以通过运行 `sh benchmark_anakin.sh` 脚本进行测试

3. Benchmark Model

  本节主要列举了CNN model分别在 `Anakin` 和 `Tensorflow`框架上的前向预测性能数据。

  你可以使用预训练好的caffe model或你自己训练好的caffe model进行性能测试。

  > 注意在性能测试之前，请先将测试model通过[External Converter](#10003)转换为Anakin model
  > 本文在单CPU上进行多线程的单batch size测试。

  - [Language model](#111)   *fluid model 可以在[这儿](https://github.com/PaddlePaddle/models/tree/develop/fluid/language_model)下载*

  测试结果：

  * [language model in i7-7700](#111)
  * [language model in E5-2620 v4](#222)
  * [language model in E5-2650 v4](#333)

  > 注意：对于language model， 本文使用'ptb_valid_txt'作测试数据集

  <span id = '111'> 3.1. language model in i7-7700 </span>

  - Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 5.64    | 2.44
    2 | 8.29    | 4.44
    4 | 14.23   | 9.91
    6 | 19.83   | 15.51

  - Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 3459 | 8536
    2 | 4772 | 9399
    4 | 5498 | 8418
    6 | 5764 | 8070

  <span id = '222'> 3.2. language model in E5-2620 v4 </span>

  - Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 6.31    | 2.84
    2 | 7.94    | 2.678
    4 | 8.66    | 4.32
    6 | 12.33   | 7.12

  - Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 2890 | 7257
    2 | 4726 | 15439
    4 | 8659 | 18351
    6 | 9414 | 17461

  <span id = '333'> 3.3. language model in E5-2650 v4 </span>

  - Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 3.69    | 2.84
    2 | 4.62    | 2.85
    4 | 7.78    | 3.48
    6 | 13.54   | 4.79

  - Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 4456 | 7300
    2 | 7522 | 14556
    4 | 9580 | 22086
    6 | 8664 | 23938


<span id = ''> 4. 怎么运行 Benchmark models? </span>

 这儿有两种方式运行：

  > 1.运行 `sh benchmark_tensorflow.sh` 和 `sh benchmark_anakin.sh` 脚本

  > 2.获取 caffe 或 fluid model, 将这些 model 转换为 anakin model,然后使用 net_test_*** 进行测试



# Developing Guide
---

本节主要包含以下四个方面内容：
  
+ [C++ APIs](#20001)
+ [How to contribute](#20002)
+ [How to add custom operators](#20003)
+ [How to add new device](#20004)


## <span id = '20001'> C++ APIs </span>
---

本节主要内容包含以下几个方面：

- [Anakin的工作原理](#principle)
- [Anakin APIs](#api)
- [示例代码](#example)

### <span id = 'principle'> Anakin的工作原理</span> ###

![Anakin_principle](pics/anakin_fm_ch.png)

用Anakin来进行前向计算主要分为三个步骤：

- 将外部模型通过[External Converter](#10003)转换为Anakin模型  
  在使用Anakin之前，用户必须将所有其他模型转换成Anakin模型，我们提供了转换脚本，用户可通过[Anakin Parser](./Converter_ch.md)进行模型转换
- 生成Anakin计算图
  加载Anakin模型生成原始计算图，然后需要对原始计算图进行优化。你只需要调用相应的API优化即可
- 执行计算图  
  Anakin会选择不同硬件平台执行计算图


### <span id ='api'>Anakin APIs </span> ###
#### Tensor ####

1. <span id =' '> Tensor结构 </span>

  `Tensor`提供基础的数据操作和管理，为ops提供统一的数据接口。`Tensor`包含以下几个属性：   

  - Buffer  
    数据存储区
  - Shape  
    数据的维度信息
  - Event  
    用于异步计算的同步

  > `Tensor` 类包含三个`Shape`对象， 分别是`_shape`, `_valid_shape`和 `offset`  
  > `_shape`为`tensor`真正空间信息  
  > `_valid_shape`表示当前`tensor`使用的空间信息  
  > `_offset`表示当前`tensor`数据指针相对于真正数据空间的信息  

  `Tensor`不同维度与分别与数学中的向量、矩阵等相对应如下表所示 

  Dimentions | Math entity |
  :----: | :----:
  1 | vector
  2 | matrix
  3 | 3-tensor
  n | n-tensor


2. <span id =' '> 声明tensor对象 </span>

  `Tensor`接受三个模板参数:


```c++
 template<typename TargetType, DataType datatype, typename LayOutType = NCHW>
 class Tensor .../* Inherit other class */{
  //some implements
  ...
 };
```

  > TargetType是平台类型，如X86，GPU等等，在Anakin内部有相应的标识与之对应  
  > datatype是普通的数据类型，在Anakin内部也有相应的标志与之对应   
  > [LayOutType](#layout)是数据分布类型，如batch x channel x height x width [NxCxHxW], 在Anakin内部用一个struct来标识  

  Anakin中数据类型与基本数据类型的对应如下:

  2.1. <span id='target'> TargetType </sapn>

  Anakin TargetType | platform
  :----: | :----:|
  NV | NVIDIA GPU
  ARM | ARM
  AMD | AMD GPU
  X86 | X86
  NVHX86 | NVIDIA GPU with Pinned Memory


  2.2. <sapn id='datatype'> DataType </span>

  Anakin DataType | C++ | Description 
  :---: | :---: | :---: |
  AK_HALF | short | fp16
  AK_FLOAT | float | fp32
  AK_DOUBLE | double | fp64
  AK_INT8 | char | int8
  AK_INT16 | short | int16
  AK_INT32 | int | int32
  AK_INT64 | long | int64
  AK_UINT8 | unsigned char | uint8
  AK_UINT16 | unsigned short | uint8
  AK_UINT32 | unsigned int | uint32
  AK_STRING | std::string | /
  AK_BOOL | bool | /
  AK_SHAPE | / | Anakin Shape 
  AK_TENSOR | / | Anakin Tensor 

  2.3. <span id = 'layout'> LayOutType </span>

  Anakin LayOutType ( Tensor LayOut ) | Tensor Dimention | Tensor Support | Op Support
  :---: | :---: | :---: | :---: |
  W | 1-D | YES | NO
  HW | 2-D | YES | NO
  WH | 2-D | YES | NO
  NW | 2-D | YES | YES
  NHW | 3-D | YES |YES
  NCHW ( default ) | 4-D | YES | YES
  NHWC | 4-D | YES | NO
  NCHW_C4 | 5-D | YES | YES

    理论上，Anakin支持申明1维以上的tensor。但是对于Anakin中的OP来说，只支持NW、NHW、NCHW、NCHW_C4这四种LayOut，
    其中NCHW是默认的LayOutType，NCHW_C4是专门针对于int8这种数据类型的。

3. 例子

  > 下面的代码将展示如何使用tensor， 建议先看看这些示例。
  > 要想获得更多关于tensor的信息， 请参考 *soure_path/core/tensor.h*

  + 使用shape对象初始化tensor

```c++  
  //create a null tensor. A null tensor holds for nothing.
  //tensor's buffer  is resident at CPU and its datatype is AK_FLOAT.
  //tensor's Layout is NCHW(default)
   Tensor<X86, AK_FLOAT> mytensor;

   //1. using shape object to create a tensor.
   Shape shape1(NUM); //1-D shape. NUM is the number of dimention.
   Tensor<X86, AK_FLOAT, W> mytensor1(shape1); //1-D tensor.

  // A 4-D shape
   Shape shape2(N, C, H, W); // batch x channel x height x width
```

    >`注意：Shape的维度必须和tensor的`[LayoutType](#layout)`相同，比如Shape(N,C,H,W), 那么Tensor的 LayoutType必须是NCHW，否则会出错。如下列代码所示`  

```c++
   // A 4-D tensor.
   Tensor<X86, AK_FLOAT> mytensor2(shape2);  //right

   //A 4-D tensor which is resident at GPU and its datatype is AK_INT8
   Tensor<NV, AK_INT8> mytensor3(shape2);   //right
   
   Tensor<X86, AK_FLOAT, NHW> mytensor4(shape2); //wrong!! shape's dimetion must be equal to tensor's Layout.
   Tensor<NV, AK_FLOAT, NCHW_C4> mytensor5(shape2); //wrong!!!!

```

  + 使用现有的数据和shape初始化tensor

```c++

   /**
   *  A construtor of Tensor.
   *  data_ptr is a pointer to any data type of data
   *  TargetType is type of a platform [Anakin TargetType]
   *  id : device id
   *  shape: a Anakin shape
   */
   Tensor(Dtype* data_ptr, TargetType_t target, int id, Shape shape);

   //using existing data feed to a tensor
   Tensor<X86, AK_FLOAT> mytensor(data_ptr, TargetType, device_id, shape); //shape must has dimention (N, C, H, W).

```

  + 使用tensor初始化tensor

```c++
   Tensor<NV, AK_FLOAT> tensor(exist_tensor);
```

    > 提示： 你可以用` typedef Tensor<X86, AK_FLOAT> Tensor4d_X86 `方便定义tensor


4. <span id =' '> 填充tensor数据区 </span>


  填充数据区得看你申明tensor的方式， 下面展示了如何填充tensor的数据区。

```c++
  首先来看看tensor的四种声明方式：

  1. Tensor<X86, AK_FLOAT> mytensor;
  2. Tensor<X86, AK_FLOAT, W> mytensor1(shape1);
  3. Tensor<X86, AK_FLOAT> mytensor(data_ptr, TargetType, device_id, shape);
  4. Tensor<NV, AK_FLOAT> tensor(exist_tensor);


  相关的声明方式的数据填充方法如下：

  1：声明一个空的tensor，此时没有为其分配内存，所以，我们需要手动的为其分配内存。
            
            //parama shape
            mytensor.re_alloc(Shape shape); 

            //Get writable pointer to mytensor.
            //parama index (int): where you start to write.
            //Dtype is your data type such int, float or double.
            Dtype *p = mytensor.mutable_data(index/*=0*/);
            //write data to mytensor
            for(int i = 0; i < mytensor.size(); i++){
              p[i] = 1.0f;
            }
            //do something ...

  2: 这种声明方式会自动分配内存 

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor1.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...

 
  3：在该种声明方式中，我们仍不需要手动为其分配内存。但在构造函数内部是否为其分配内存，得依情况而定。如果data_ptr和申明的
  tensor都在都一个目标平台上，那么该tensor就会与data_ptr共享内存空间，相反，如果他们不在同一个平台上（如data_ptr在X86上，而
  tensor在GPU上），那么此时tensor就会开辟一个新的内存空间，并将data_ptr所指向的数据拷贝到tensor的buffer中。

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...

  4：该种方式仍不需要手动分配内存

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...


  另外，你还可以获取一个tensor的可读指针，示例如下：
        //Get read-only pointer to mytensor.
        //parama index (int): where you start to read.
        //Dtype is your data type such int, float or double.
         Dtype *p = mytensor.data(index/*=0*/);
        //do something ...
```

  如果想更详细的了解tensor，请查阅*soure_path/saber/core/tensor.h*

5. 获取tensor的shape

```c++
//some declarations
// ...
Shape shape = mytensor.shape();

//Get a first dimetion size of tesor, if it has.
int d1 = shape[0];

//Get a second dimention size of tensor, if it has.
int d2 = shape[1];

...

//Get a n-th dimention size of tensor, if it has.
int dn = shape[n-1];


//Get a tensor's dimention
int dims = mytensor.dims();

//Get the size of tensor.
//size = d1 x d2 x ... x dn.
int size = mytensor.size();

//Get the size of tensor at interval [Di, Dj)
// form i-th dimention to j-th dimention, but not including the j-th dimention.
// which means di x (di+1) x ... x (dj -1)
int size = mytensor.count(start, end);
```

6. 设置tensor的shape

  我们可以用tensor的成员函数set_shape来设置tensor的shape。 下面是set_shape的定义


```c++
/**
 * \brief set a tensor's shape
 * \param valid_shape [a Shape object]
 * \param shape [a Shape object]
 * \param offset [a Shape object]
 * \return the status of this operation, that means whether it success * or not.
 */
SaberStatus set_shape(Shape valid_shape, Shape shape = Shape::zero(TensorAPI::layout_dims::value), Shape offset = Shape::minusone(TensorAPI::layout_dims::value)); 
```

  这个成员函数只设置tensor的shape。这些shape对象(valid_shape, shape, offset)的[LayOutType](#layout)必须和当前的tensor的相应三个shape对象的LayOutType相同，如果不同就会出错，返回SaberInvalidValue。 如果相同，那么将成功设置tensor的shape。

```c++

// some declarations
// ...
//valid_shape, shape , offset are Shape object;
//All these Shape object's LayOutType must be equal to mytensor's.
mytensor.set_shape(valid_shape, shape, offset);

```

7. 重置 tensor的shape

```c++
//some declarations
Shape shape, valid_shape, offset;

//do some initializations
... 
mytensor.reshape(valid_shape, shape, offset);
```

  注意： Reshape操作仍然需要shape的[LayOutType](#layout) 与tensor的相同


#### Graph ####

`Graph`类负责加载Anakin模型生成计算图、对图进行优化、存储模型等操作。

1. <span id =' '> 图的声明 </span>

  与`Tensor`一样，graph也接受三个模板参数。

```c++

template<typename TargetType, DataType Dtype, Precision Ptype>
class Graph ... /* inherit other class*/{
  
  //some implements
  ...

};
```

  前面已经介绍过[TargetType](#target)和[DataType](#datatype)是Anakin内部自定义数据类型。[TargetType](#target)表示平台类型 (如NV、X86), [DataType](#datatype)是Anakin基本数据类型与C++/C中的基本数据类型相对应。 [Precision](#precision)为op所支持的精度类型, 稍后我们在介绍它。


```c++

//Create a empty graph object.
Graph graph = Graph<NV, AK_FLOAT, Precision::FP32> tmp();

//Create a pointer to a empty graph.
Graph *graph = new Graph<NV, AK_FLOAT, Precision::FP32>();

//Create a pointer to a empty graph.
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();

```

2. 加载 Anakin 模型

```c++
//some declarations
...
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
std::string model_path = "the/path/to/where/your/models/are";
const char *model_path1 = "the/path/to/where/your/models/are";

//Loading Anakin model to generate a compute graph.
auto status = graph->load(model_path);

//Or this way.
auto status = graph->load(model_path1);
//Check whether load operation success.
if(!status){
  std::cout << "error" << endl;
  //do something...
}

```

3. 优化计算图

```c++
//some declarations
...
//Load graph.
...
//According to the ops of loaded graph, optimize compute graph.
graph->Optimize();

```

  > 注意： 第一次加载原始图，必须要优化。

4. 保存模型

  你可以在任何时候保存模型， 特别的， 你可以保存一个优化的模型，这样，下次再加载模型时，就不必进行优化操作。


```c++
//some declarations
...
//Load graph.
...
// save a model
//save_model_path: the path to where your model is.
auto status = graph->save(save_model_path);

//Checking
if(!status){
  cout << "error" << endl;
  //do somethin...
}
```

5. 重新设置计算图里的tensor的shape

```c++
//some declarations
...
//Load graph.
...
vector<int> shape{10, 256, 256, 10};
//input_name : std::string.
//Reshape a tensor named input_name.
graph->Reshape(input_name, shape);//Note: shape is a vector, not a Shape object.
```

6. 设置 batch size

`Graph` 支持重新设置batch size的大小。

```c++
//some declarations
...
//Load graph.
...
//input_name : std::string.
//Reset a tensor named input_name.
int new_batch_size = 4;
graph->ResetBatchSize(input_name, new_batch_size);
```

####  Net ####

`Net` 是计算图的执行器。你可以通过Net对象获得输入和输出

1. Creating a graph executor

  `Net`接受四个模板参数。  


```c++
template<typename TargetType, DataType Dtype, Precision PType OpRunType RunType = OpRunType::ASYNC>
class Net{
  //some implements
  ...

};
```
  由于有些Op可能支持多种精度，我们可以通过Precision来指定 

  > OpRunType表示同步或异步类型，异步是默认类型  
  > OpRunType::SYNC表示同步，在GPU上只有单个流  
  > OpRunType::ASYNC表示异步，在GPU上有多个流并以异步方式执行  

  实际上，Precision和OpRunType都是enum class, 详细设计请参考*source_root/framework/core/types.h*.


  + <span id = 'precision'> Precision </span>

  Precision | Op support
  :---: | :---:
  Precision::INT4 | NO
  Precision::INT8 | NO
  Precision::FP16 | NO
  Precision::FP32 | YES
  Precision::FP64 | NO

  > 现在Op的精度只支持FP32， 但在将来我们会支持剩下的Precision.

  + OpRunType

  OpRunType | Sync/Aync |Description
  :---: | :---: | :---:
  OpRunType::SYNC | Synchronization | single-stream on GPU
  OpRunType::ASYNC | Asynchronization | multi-stream on GPU

  用graph对象创建一个执行器。

```c++
//some declarations
...
//Create a pointer to a graph.
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
//do something...
...

//create a executor
Net<NV, AK_FLOAT, Precision::FP32> executor(*graph);

```

2. 获取输入输出tensor

  > 获取输入输出tensor，并填充输入tensor的buffer  
  > 如果想要获取输入和输出tensor，那么必须指定输入的名字，如"input_0", "input_1", "input_2", ..., 必须传入如上字符串才能够获得输入tensor  
  > 另外，如果想知道input_i对应哪个输入，你需要去dash board查看，如何使用dash board请看[Anakin Parser](./Converter_ch.md)  

  请看如下示例代码:

```c++
//some declaratinos
...

//create a executor
//TargetType is NV [NVIDIA GPU]
Net<NV, AK_FLOAT, Precision::FP32> executor(*graph);

//Get the first input tensor.
//The following tensors(tensor_in0, tensor_in2 ...) are resident at GPU.
//Note: Member function get_in returns an pointer to tensor.
Tensor<NV, AK_FLOAT>* tensor_in0 = executor.get_in("input_0");

//If you have multiple input tensors
//You just type this code below.
Tensor<NV, AK_FLOAT>* tensor_in1 = executor.get_in("input_1");
...
auto tensor_inn = executor.get_in("input_n");
```

  当得到输入tensor之后，就可以填充它的数据区了。

```c++
//This tensor is resident at GPU.
auto tensor_d_in = executor.get_in("input_0");

//If we want to feed above tensor, we must feed the tensor which is resident at host. And then copy the host tensor to the device's one.

//using Tensor4d = Tensor<Ttype, Dtype>;
Tensor4d<X86, AK_FLOAT> tensor_h_in; //host tensor;
//Tensor<X86, AK_FLOAT> tensor_h_in; 

//Allocate memory for host tensor.
tensor_h_in.re_alloc(tensor_d_in->valid_shape());
//Get a writable pointer to tensor.
float *h_data = tensor_h_in.mutable_data();

//Feed your tensor.
/** example
for(int i = 0; i < tensor_h_in.size(); i++){
  h_data[i] = 1.0f;
}
*/
//Copy host tensor's data to device tensor.
tensor_d_in->copy_from(tensor_h_in);

// And then
```

  > 类似的，我们可以利用成员函数get_out来获得输出tensor
  > 但与获得输入tensor不同的是， 我们需要指定输入tensor结点的名字，这个可以从dash board中看到，请从[Anakin Parser](./Converter_ch.md)中查看dash board的使用方法

  假如有个输出结点叫pred_out, 那么我们可以通过如下代码获得相应的输出tensor：

```c++
//Note: this tensor are resident at GPU.
Tensor<NV, AK_FLOAT>* tensor_out_d = executor.get_out("pred_out");

```

3. Executing graph

  当一切准备就绪后，我们就可以执行真正的计算了！
```c++
executor.prediction();
```
 
### <span id='example'> 示例代码 </span> ###

下面的例子展示了如何调用Anakin

在这儿之前， 请确保你已经有了Anakin模型。如果还没有，那么请使用[Anakin Parser](./Converter_ch.md)转换你的模型。

#### 单线程

  单线程例子在 *source_root/test/framework/net/net_exec_test.cpp*

```c++

std::string model_path = "your_Anakin_models/xxxxx.anakin.bin";
// Create an empty graph object.
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
// Load Anakin model.
auto status = graph->load(model_path);
if(!status ) {
    LOG(FATAL) << " [ERROR] " << status.info();
}
// Reshape
graph->Reshape("input_0", {10, 384, 960, 10});
// You must optimize graph for the first time.
graph->Optimize();
// Create a executer.
Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph);

//Get your input tensors through some specific string such as "input_0", "input_1", and 
//so on. 
//And then, feed the input tensor.
//If you don't know Which input do these specific string ("input_0", "input_1") correspond with, you can launch dash board to find out.
auto d_tensor_in_p = net_executer.get_in("input_0");
Tensor4d<X86, AK_FLOAT> h_tensor_in;
auto valid_shape_in = d_tensor_in_p->valid_shape();
for (int i=0; i<valid_shape_in.size(); i++) {
    LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i]; //see tensor's dimentions
}
h_tensor_in.re_alloc(valid_shape_in);
float* h_data = h_tensor_in.mutable_data();
for (int i=0; i<h_tensor_in.size(); i++) {
    h_data[i] = 1.0f;
}
d_tensor_in_p->copy_from(h_tensor_in);

//Do inference.
net_executer.prediction();

//Get result tensor through the name of output node.
//And also, you need to see the dash board again to find out how many output nodes are and remember their name.

//For example, you've got a output node named obj_pre_out
//Then, you can get an output tensor.
auto d_tensor_out_0_p = net_executer.get_out("obj_pred_out"); //get_out returns a pointer to output tensor.
auto d_tensor_out_1_p = net_executer.get_out("lc_pred_out"); //get_out returns a pointer to output tensor.
//......
// do something else ...
//...
//save model.
//You might not optimize the graph when you load the saved model again.
std::string save_model_path = model_path + std::string(".saved");
auto status = graph->save(save_model_path);
if (!status ) {
    LOG(FATAL) << " [ERROR] " << status.info();
}

```


## <span id = '20002'> How to contribute </span>
---
我们真诚地感谢您的贡献，欢迎通过 GitHub 的 fork 和 pull request 流程来提交代码。

***代码要求:***

- 代码注释请遵守[Doxygen](http://www.stack.nl/~dimitri/doxygen/)的样式
- 所有代码必须具有单元测试
- 通过所有单元测试
- 请遵守提交代码的一些约定

以下教程将指导您提交代码

<span id = ''> 1. Fork </span>

  首先跳转到[Anakin](https://github.com/PaddlePaddle/Anakin)的github首页，然后点击`Fork`, 生成自己目录下的仓库

<span id = ''> 2. 克隆（clone）</span>

  将远程仓库clone到本地：

```bash
git clone YOUR_REPOSITORY_URL
cd Anakin
```

<span id = ''> 3. 创建本地分支 </span>

  > Anakin目前使用[Git流分支模型](https://nvie.com/posts/a-successful-git-branching-model/)进行开发, 测试和维护

  > 所有的feature和bug fix的开发工作都应该在一个新的分支上完成，根据需要从现有分支上创建新分支

  > 使用`git checkout -b`创建并切换到新分支

```bash
git checkout -b YOUR_NEW_BRANCH
```

<span id = ''> 4. 开发 </span>

  4.1. 编写代码

  4.2. 构建和测试

    详细请参考 [Instal and Compile](#10001)

  4.3. 提交(commit)

    提交代码时，请认真写好提交说明，这样其他人就可以清楚的知道这次提交做了哪些改变：

  ```bash
  git commit -m 'description'
  ```

<span id = ''> 5. 保持本地仓库最新 </span>

  在发起Pull Request之前，需要与原始仓库同步。

  如果还没添加原仓库，请先添加源，可通过`git remote -v`查看是否添加源：

```bash
git remote -v
origin .... (fetch)
origin .... (push)
```
  如果只出现origin，说明还未添加源，可通过如下命令添加源：

```bash
git remote add upstream ORIGIN_REPOSITORY_URL
```
  获取 upstream 的最新代码并更新当前分支

```bash
git fetch upstream
git pull upstream BRANCH_NAME
```

6. Push到远程仓库

  将本地的修改push到远程仓库上

```bash
git push origin BRANCH_NAME
```

7. 提交Pull Request

  切换到所建分支，然后点击`New pull request`

  ![contri1](./contri1.JPG)

  选择目标分支：

  ![contri2](./contri2.JPG)

  接下来等待review

8. 删除远程分支

  当PR被合进主仓库后，可以在PR的界面删除远程仓库的分支

  也可以通过以下命令删除远程分支：

```bash
git push origin :YOUR_NEW_BRANCH
```

9. 删除本地分支

  可以通过以下命令删除本地分支:

```bash
#切换到其他分支
git checkout OTHER_BRANCH

#删除YOUR_NEW_BRANCH分支
git branch -D YOUR_NEW_BRANCH
```

  至此，我们就完成了一次代码贡献的过程

  ***提交代码的一些约定***

  为了使评审人在评审代码时更好地专注于代码本身，请您每次提交代码时，遵守以下约定：

+ 提交Pull Request前：  
- 注意commit的数量

  - 原因：如果仅仅修改一个文件但提交了十几个commit，每个commit只做了少量的修改，这会给评审人带来很大困扰。评审人需要逐一查看每个commit才能知道做了哪些修改，且不排除commit之间的修改存在相互覆盖的情况。

  - 建议：每次提交时，保持尽量少的commit，可以通过`git commit --amend`补充上次的commit。对已经Push到远程仓库的多个commit，可以参考[squash commits after push](https://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed)
  
- 注意每个commit的名称：应能反映当前commit的内容，不能太随意。

+ 如果解决了某个Issue的问题，请在该Pull Request的第一个评论框中加上：`fix #issue_number`，这样当该Pull Request被合并后，会自动关闭对应的Issue。关键词包括：close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved，请选择合适的词汇。详细可参考[Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages)。

  在回复评审人意见时，请您遵守以下约定：  
+ 评审人的每个意见都必须回复
   - 对评审意见同意且按其修改完的，给个简单的Done即可
   - 对评审意见不同意的，请给出您自己的反驳理由 
+ 如果评审意见比较多
   - 请给出总体的修改情况 
   - 请采用[start a review](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/)进行回复，而非直接回复的方式 


## <span id = '20003'> How to add custom operators </span>
---

<span id = ''> 1. 基本概念 </span>

  1.1. 与Operator相关的基本概念

    简单介绍下几个与Operator相关的基本概念，详情请参考设计文档。

    + ```framework```: 上层的逻辑代码，负责从parser中获取参数及weights，添加op时主要修改framework/operator目录下的内容。

    + ```saber```: 底层的实现代码，Anakin通过saber封装了不同的backends，不同的实现(impl)分别特化出自己的实现，外层framework通过不同的template进入各自的impl完成调用。各个op的parameter放在saber/saber_funcs_param.h文件中，增加op主要修改saber/funcs下的内容。

    + saber的文件结构：
      - saber/funcs下的是各个funcs的外部接口，这一层的op与具体的设备实现无关，只与各op完成的功能有关。由于跟实现(impl)无关，本层文件明均不带impl。
      - saber/funcs/impl下是各个op的impl声明，特定设备需要完成该层声明的特化版本，如saber/funcs/impl/x86实现了上一层impl声明的x86特化版本，saber/funcs/impl/cuda实现了上一层impl声明的NV特化版本。当增加新的backends时需要特化出新的实现。本层代码同实现相关，均带有```impl_```前缀。
      - saber/funcs/impl/cuda/base/cuda_c内有cuda```.cu```扩展名的文件，添加cuda的kernel需要在该文件目录下添加。
      - saber/funcs/impl/cuda/base/sass 内有不同架构的汇编代码编译的静态库。

  2.2. 涉及到的基类及各个类之前的关系

    简单介绍相关的基类

    + ```anakin::Operator```: framework的operator基类，位于framework/core/operator/operator.h

    + ```anakin::saber::BaseFunc```: saber对外的op接口基类，提供统一的对外接口，位于saber/funcs/base.h。BaseFunc的```compute_output_shape```接口只根据input的shape和param的参数计算输出的shape，并通过```tensor```的```set_shape```接口(只设置shape，不分配空间)设置到output中。```operator()```接口为各个op的计算接口。

    + ```ankain::saber::ImplBase```: saber设备实现的op的接口，所有设备相关实现的基类。位于saber/funcs/impl/impl_base.h。实现版本中这里分为两类，一类以```vender_```为前缀，带有```vender_```代码意为使用第三方库来实现该op，如cudnn的conv，或mkl的conv等等，这类op的性能我们难以调优，因此单独列为一类。另一类是带有源码的saber实现，这些实现都带有```saber_```为前缀，此类实现带有源码，能够通过后续优化不断提升性能，实现起名时需要注意这一点。

<span id = ''> 2. 添加operator </span>

  添加一个新的op需要以下几步：

  - 添加saber的param
  - 定义saber的Operator类
  - 定义新的impl声明
  - 完成新的impl实现
  - 增加framework的实现或特化

  接下来就针对这几步，以一个简单例子为例介绍实现。

  例如我们要添加新的Mul op，给出计算公式如下：$$Out = alpha \dot X * Y$$

  2.1. 为operator增加param

    涉及到的文件：```saber/saber_funcs_param.h```。如果之前已经存在需要添加的op的param，这一步可以跳过

    这里```XXXParam```是一个```struct```。包含一个无参数的构造函数，含参数的构造函数，复制构造函数，```operator=()```及```operator==()```

```bash
template <typename opTensor> // 能够获得target, datatype, layout
struct MulParam{
  MulParam()
    : alpha(0)
  {}
  MulParam(float alpha_in)
    : alpha(alpha_in)
  {}
  MulParam(const MulParam& right)
    : alpha(right.alpha)
  {}
  MulParam &operator=(const MulParam &right) {
    alpha = right.alpha;
  }
  bool operator==(const MulParam &right) {
    return alpha == right.alpha;
  }
  float alpha;
};
```

  2.2. 定义Operator类

    涉及到的文件:```saber/funcs/mul.h```。如果之前定义过该op的类，这里需要修改输入的impl定义头文件

    下面给出一个相对完整的定义结构供参考:

```bash
//不同的设备需要包含对应的operator实现
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_mul.h"
#include "saber/funcs/impl/cuda/vender_mul.h"
#endif
//如果一个设备现在还没有对应的operator实现，需要包含声明
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_mul.h"
#endif
namespace anakin {
namespace saber {
template<typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_FLOAT,
        DataType outDtype = AK_FLOAT,
        typename LayOutType_op = NCHW,
        typename LayOutType_in = NCHW,
        typename LayOutType_out = NCHW>
class Mul : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase, MulParam> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase, MulParam>::BaseFunc;
    Mul() = default;
    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef MulParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        //计算输出的shape，
        Shape output_shape = (input[0]->valid_shape());
        /* code */
        return output[0]->set_shape(output_shape);
    }
    virtual SaberStatus init_impl(ImplEnum implenum) override {
      // 不同设备均使用此init_impl, 此接口创建对应impl的实现。
      switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderMul <TargetType,
                OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            case SABER_IMPL:
                this->_impl.push_back(new SaberMul <TargetType,
                OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            default:
                return SaberUnImplError;
        }
    }
private:
    virtual void pick_best_static() override {
        if (true) // some condition?
            this->_best_impl = this->_impl[0];
    }
    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
};
} // namespace saber
} // namespace anakin
```

  2.3. 为operator增加新的impl声明

    涉及的文件:```saber/funcs/impl/impl_mul.h```。不同的设备都特化同一个声明，特化版本放在对应的文件夹下，这里的声明就是给出所有设备的统一声明。

    下面给出一个参考:

```bash
#include "saber/funcs/impl/impl_macro.h"
namespace anakin{
namespace saber{
DEFINE_OP_CLASS(Mul, MulParam); // 第一个参数是op的名字，第二个是对应param的名字
}
}
```

  2.4. 完成新的operator特定后端实现

    涉及的文件:```saber/funcs/impl/xxx/vender_mul.h```或```saber/funcs/impl/xxx/saber_mul.h```

- ```xxx```指代特定的一种设备
- ```vender```是指的使用第三方库实现的op
- ```saber```指的源码实现的op

    这里以cuda的vender实现为例，简单介绍一下特化出的函数的几个基本接口:

```bash
// include 对应的声明
#include "saber/funcs/impl/impl_mul.h"

namespace anakin{
namespace saber{
template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderMul<NV, //偏特化出需要的后端。
    OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out> :
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>,
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        MulParam<Tensor<NV, OpDtype, LayOutType_op> > >
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;
    VenderMul(){}
    ~VenderMul() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            MulParam<OpTensor>& param, Context<NV>& ctx) {
        this->_ctx = ctx;
        create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            MulParam<OpTensor>& param, Context<NV>& ctx) {
        // set内部参数
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                        MulParam<OpTensor>& param) {
        // dispatch kernel.
    }

private:
};
}
}
```

    > 注意：
    ```init```和```create```的区别：```init```接口是第一次初始化op的时候进入的接口，此函数只在第一次初始化op时调用，这个接口一般放一些只需要执行一次的代码，如malloc或者create之类的函数。```create```函数除了第一次init执行外，在输入发生变化或者param发生变化时会再次触发，create一般放置set函数，设置内部变量，当input发生变化时这里执行一些同input或weights直接相关的代码。但create因为触发位置在网络内，如果```create```函数执行了一些严重耗时的操作，这里会拖慢整个op的执行时间，需要慎重选择操作放置的位置。

  2.5. 添加framework的特化

    涉及的文件:```framework/operators/mul.h```和```framework/operators/mul.cpp```

    这里简单介绍下如果添加或修改framework内的operator:

```bash
#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/mul.h" // 需要包对应的saber头文件
namespace anakin {
namespace ops {
template<typename Ttype, DataType Dtype, Precision Ptype>
class MulHelper;

template<typename Ttype, DataType Dtype, Precision Ptype>
class Mul : public Operator<Ttype, Dtype, Ptype> {
public:
    Mul() {}
    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx,
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }
    friend class MulHelper<Ttype, Dtype, Ptype>;
};
template<typename Ttype, DataType Dtype, Precision Ptype>
class MulHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    MulHelper() = default;
    ~MulHelper();
    Status InitParam() override;

    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

public:
    saber::MulParam<Tensor4d<Ttype, Dtype>> _param_mul;
    saber::Mul<Ttype, Dtype> _funcs_mul;
};
}
} /* namespace anakin */
```

    对应的```.cpp```文件如下：

```bash
#include "framework/operators/mul.h"

namespace anakin {
namespace ops {

#ifdef USE_CUDA
template<>
void Mul<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<MulHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<MulHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_mul;
    impl->_funcs_mul(ins, outs, param, ctx);
}
#endif

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::InitParam() {
    auto alpha = GET_PARAMETER(float, alpha);
    MulParam<Tensor4d<Ttype, Dtype>> param_mul(alpha);
    _param_mul = param_mul;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {

    SABER_CHECK(_funcs_mul.init(ins, outs, _param_mul, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_mul.compute_output_shape(ins, outs, _param_mul));
    return Status::OK();
}

#ifdef USE_CUDA
template class MulHelper<NV, AK_FLOAT, Precision::FP32>;
#endif
#ifdef USE_ARM_PLACE
template class MulHelper<ARM, AK_FLOAT, Precision::FP32>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Mul, MulHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Mul, MulHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Mul)
.Doc("Mul operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("mul")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("mul")
#endif
.num_in(1)
.num_out(1)
.Args<float>("alpha", " alpha of Mul "); //注册

} /* namespace ops */

} /* namespace anakin */
```

  2.6. 实现单元测试

    涉及的文件:```test/saber/xxx/test_saber_funcs_mul_xxx.cpp```

    在对应的test下需要添加新的单元测试如下所示:

```bash
TEST(TestSaberFuncNV, test_depthwise_conv) {

    // init tensors and some param.

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // create param
    MulParam<Tensor<NV, AK_FLOAT, NCHW> > param(alpha);

    std::vector<Tensor<NV, AK_FLOAT, NCHW>*> input;
    std::vector<Tensor<NV, AK_FLOAT, NCHW>*> output;

    // create saber op
    Mul<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> mul;

    // compute output shape
    mul.compute_output_shape(input, output, param);

    // re_alloc output tensors memory based on output shape
    output[0]->re_alloc(output[0]->shape());

    // init saber op(calling init and create)
    mul.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    // call operator()
    mul(input, output, param, ctx1);

    // cuda specified, record events
    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    
    // param changed 
    param.alpha = 2.0;
    // auto calling saber op(create and dispatch)
    mul(input, output, param, ctx1);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv){
    anakin::saber::Env<NV>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

```

<span id = ''> 3. 调试及注意事项 </span>

  一个op需要有对外的op接口和内部实现，由于存在saber/funcs/impl的非特化版本声明，当有op在某种设备下没有对应实现时，也能够编译，但此时是没有任何实现的空实现


## <span id = '20004'> How to add new device </span>
---

