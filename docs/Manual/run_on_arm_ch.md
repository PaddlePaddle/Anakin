## 从源码编译安装Anakin ##

我们目前支持Android平台，采用NDK交叉编译工具链

### 安装概览 ###

* [系统需求](#0001)
* [安装依赖](#0002)
* [Anakin 源码编译](#0003)
* [验证安装](#0004)


### <span id = '0001'> 1. 系统要求 </span> ###

*  宿主机 linux, mac
*  cmake 3.8.2+
*  Android NDK r14

### <span id = '0002'> 2. 安装依赖 </span> ###

- 2.1 protobuf3.4.0
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
    
- 2.2 opencv 2.4.3+(optional)


### <span id = '0003'> 3. Anakin 源码编译 </span> ###
    
    

### <span id = '0004'> 4. 验证安装 </span> ###
    
    运行



