# CUDA汇编

本文件目录下保存有我们实现的cuda汇编编译的库，不同架构需要不同的实现，分别放在对应的文件夹下，目前支持sm50和sm61架构，未来会支持更多架构。sass打包成了.a的形式，入口为上层的```sass_funcs.h```。

## 为什么用CUDA汇编代码

CUDA真正的汇编代码是sass代码。NV为了封装不同架构的机器码，在真正机器码之上做了一层ptx伪汇编的抽象，这层抽象能比cuda更接近最底层的sass汇编，能够封装不同架构带来的机器码的差异，需要ptxas编译后才能编成cubin。cubin就能被driver api载入和发射kernel。为了获得更好的性能，这里直接使用sass实现高性能kernel，避免编译器做的一些调整，更好的实现想要的对SM的控制。

## 如何反汇编
NV对于汇编也有官方的[文档](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)。
这里使用NV提供的工具```cuobjdump```就可以看到实现的sass代码，安装cuda会自动装上这个工具，默认路径在/usr/local/cuda/bin/cuobjdump。我这里使用了cuda 8.0.61这个版本。这里以sm_61为例。

1. 解开.a的静态包
```
ar -x libanakin_saber_sass_sm_61.a
```

2. 解开后会有若干.o的文件，这里任选一个为例，其它的都是类似方法，使用下面一句命令可以看到.o里包含的cubin有哪些。这里使用了sm_61的代码，会显示一个.sm_61.cubin结尾的文件
```
cuobjdump -lelf tmpxft_00002f40_00000000-13_ker_fp32_deconv_implicit_gemm_K4_S2_P1_16x64.o
```

3. 解出单独的cubin使用下面一句命令, xelf 就是extract-elf，最后需要加上架构信息
```
cuobjdump tmpxft_00002f40_00000000-13_ker_fp32_deconv_implicit_gemm_K4_S2_P1_16x64.o -xelf sm_61
```

4. 有了cubin后，就可以直接dump-sass了
```
cuobjdump --dump-sass tmpxft_00002f40_00000000-13_ker_fp32_deconv_implicit_gemm_K4_S2_P1_16x64.1.sm_61.cubin
```
除了cuobjdump，还有```nvdisasm```，也可以反汇编cubin文件，得到cubin后也可以反汇编代码，只是不打印控制信息了。
```
nvdisasm tmpxft_00002f40_00000000-13_ker_fp32_deconv_implicit_gemm_K4_S2_P1_16x64.1.sm_61.cubin
```
