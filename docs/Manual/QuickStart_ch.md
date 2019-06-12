# 快速上手Anakin
本文档用于快速验证和使用Anakin，用户也可以使用对应的脚本完成快速验证，[传送门](../../examples/anakin/demo_link_lib)。也可以依照文档逐步使用Anakin。
## 获取Anakin预测库
### 拉产出获取预测库
```bash
```

使用wget命令拉产出获取如下目录，每个目录中对应的tar.gz压缩包就是对应平台的Anakin预测库（如nv_x86表示该库同时支持nv和avx2的x86平台）。由于所有库都使用了相同的命名，用户需要根据应用平台解压对应的库到对应的目录。
```
|-- arm
|   |-- arm
|   |   `-- anakin_release_arm.tar.gz
|   |-- libanakin_static.a
|-- cpuv4
|   `-- cpuv4
|       `-- anakin_release_native_x86_v4.tar.gz
|-- cuda10
|   `-- gpucpuv2_cuda10
|       `-- anakin_release_nv_cuda10.tar.gz
|-- cuda8
|   `-- gpucpuv2
|       `-- anakin_release_nv.tar.gz
|-- cuda9
|   `-- gpucpuv2_cuda9
|       `-- anakin_release_nv_cuda9.tar.gz
|-- mac
|   `-- anakin_ios_release.tar.gz
|-- mlu100
|   `-- mlu100
|       `-- anakin_release_native_mlu100.tar.gz
`-- nv_x86
    `-- gpucpu
        `-- anakin_release_nv_x86.tar.gz
```
## 源码编译预测库
参考[中文安装手册](./INSTALL_ch.md)

## 使用example快速上手

### 获取示例代码
Anakin的example提供各个平台的示例代码,[X86_Example](../../examples/anakin/demo_link_lib/x86/demo_test_x86.cpp)，[NV_Example](../../examples/anakin/demo_link_lib/nv/demo_test_nv.cpp)。代码的注释提供调用流程说明。
### 编译示例代码
编译命令中需要用户补全以下变量
GXX_COMPILER : 编译器位置,如/opt/compiler/gcc-4.8.2/bin/g++
OUTPUT_DIR : 预测库编译完后的output路径，也是产出库默认是output路径，该路径下包含libanakin.so和include
SCRIPT_PATH : demo文件的父路径
CUDNN_ROOT : CUDNN的安装路径，该路径下包含CUDNN的include和lib64目录
CUDA_ROOT : CUDA的安装路径，该路径下包含include和bin  
#### x86编译指令
```bash
${GXX_COMPILER} "${SCRIPT_PATH}/demo_test_x86.cpp" -std=c++11 -I"${OUTPUT_DIR}/" -I"${OUTPUT_DIR}/include/" -I"${OUTPUT_DIR}/mklml_include/include/" -L"${OUTPUT_DIR}/" -L/opt/compiler/gcc-4.8.2/lib -L/usr/lib64/ -L /lib64/  -ldl -liomp5 -lmkldnn -lmklml_intel -lanakin_saber_common -lanakin  -o demo_test_x86
```
#### NV编译指令
```bash
${GXX_COMPILER} "${SCRIPT_PATH}/demo_test_nv.cpp" -std=c++11 -I"${OUTPUT_DIR}/" -I"${OUTPUT_DIR}/include/" -I"${OUTPUT_DIR}/mklml_include/include/" -I"${CUDA_ROOT}/include/" -I"${CUDNN_ROOT}/include" -L"${OUTPUT_DIR}/" -L"${CUDNN_ROOT}/lib64/" -L"${CUDA_ROOT}/lib64/" -L/opt/compiler/gcc-4.8.2/lib -L/usr/lib64/ -L /lib64/  -ldl -lcudart -lcublas -lcurand -lcudnn -lanakin_saber_common -lanakin  -o demo_test_nv
```
### 获取示例模型
Anakin预测库只识别Anakin的自有模型，用户可以通过Parser把Caffe,Paddle,Tensorflow,Onnx的模型解析成Anakin模型。参考[Parser文档](./Converter_ch.md)。我们提供[转换好的mobilenet_v2模型](https://github.com/qq332982511/Anakin/releases/download/0.1.1/mobilenet-v2.anakin2.bin)作为验证用模型
### 运行示例代码
```bash
#x86 test
./demo_test_x86 mobilenet-v2.anakin2.bin 1 4
#nv test
./demo_test_nv mobilenet-v2.anakin2.bin 1
```



