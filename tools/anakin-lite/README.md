# Anakin Lite
Anakin Lite支持AOT和通用两种模式。  
AOT模式是使用模型转换器根据具体一个模型生成与模型相关的`*.h`, `*.cpp`和模型文件`*.bin`，然后编译生成模型对应的库。  
通用模式是直接编译生成库，库是通用的，所需的模型文件只需通过模型转换器转换为`*.lite.bin）`(融合模型)或者`*.info, *.bin`(分立模型)即可使用。 
其中`*.info`表示模型的描述文件；`*.bin`表示模型的weights；`*.lite.bin`融合模型包含了模型的weights和模型描述文件。
## 编译模型转换器
1. 为宿主机编译安装protobuf   
protobuf3.4.0 源码从这里[下载](https://github.com/google/protobuf/releases/tag/v3.4.0)    
```bash
$ tar -xzf protobuf-3.4.0.tar.gz  
$ cd protobuf-3.4.0   
$ ./autogen.sh  
$ ./configure    
$ make  
$ make check   
$ make install
```
2. 编译模型转换器
运行tools目录下build_lite.sh，编译完成后，会在output目录下生成generator文件夹

## AOT模式
#### <span id = '0001'> 一、使用模型转换器转换为`*.bin`模型和生成相应`*.h`, `*.cpp` </span> ####
1. 运行generator目录下的gen_code.sh，转换`*.anakin.bin`模型，输出目录选择到`tools/anakin_lite`，  
'-a'参数为1，表示AOT模式。该命令会输出3个文件，`*.h`, `*.cpp`和`*.bin`。  
‘-m’参数为模型(model)所在路径，如“/home/Anakin/mobilenet.anakin.bin”  
'-n'参数为生成三个文件的名字(name)  
'-o'参数为生成文件的路径，一般设置在tools/anakin-lite目录  
‘-d’参数为Debug模式，默认为0，不开启Debug  
```bash
$ sh gen_code.sh -a 1 -m /home/Anakin/mobilenet.anakin.bin -n mobilenet -o ../../tools/anakin-lite -d 0
```
2. 如果有多个模型，重复1的操作即可。

#### <span id = '0002'> 二、使用脚本编译Anakin Lite库</span> ####
1. 编辑tools/anakin_lite目录下的脚本lite_android_build_armv7/8.sh，设置ANDROID_NDK路径。
2. 运行脚本即可生成模型对应的库。

#### <span id = '0001'> 三、测试模型(可选)</span> ####
1. 根据具体的测试模型修改`test/lite/`目录下的`test_lite_aot_model.cpp`，编译完成后，使用adb push将tools/anakin_lite/output/unit_test目录下生成的test_lite_aot_model和模型`*.bin`拷贝到手机目录data/local/tmp
```bash
$ adb push tools/anakin_lite/output/unit_test/test_lite_model data/local/tmp 
$ adb push tools/anakin_lite/*.bin data/local/tmp 
```
2. 使用adb shell命令运行test_lite_aot_model，用法为  
./test_lite_aot_model <模型文件> <batch_size> <预热次数> <执行次数> <大小核> <线程数>   
大小核参数：0代表使用大核，1代表使用小核心。  
如测试model.bin，batch_size=1，预热十次，测试二十次，使用大核，四线程  
```bash
$ adb shell
$ cd data/local/tmp
$ ./test_lite_aot_model model.bin 1 10 20 0 4
```

## 通用模式

#### <span id = '0001'> 一、使用脚本编译Anakin Lite通用库</span> ####
1. 如使用过AOT模式，请删除tools/anakin_lite目录下的`.h`和`.cpp`文件。注释掉`test/lite/test_lite_model.cpp`AOT模式下添加的模型，如果没有编辑过该文件，则不需要修改。
2. 编译Android库：编辑tools/anakin_lite目录下的脚本lite_android_build_armv7/8.sh，设置ANDROID_NDK路径。
3. 编译IOS库：直接运行lite_ios_build_armv7/8.sh。
4. 运行脚本即可生成通用库。

#### <span id = '0002'> 二、使用模型转换器把模型转换为Lite版（已有Lite版模型文件可跳过）</span> ####
1. 运行generator目录下的gen_code.sh，转换`*.anakin.bin`模型，输出目录选择到`tools/anakin_lite`，  
'-a'参数为0，表示通用模式。该命令会输出3个模型文件`*.lite.bin`，`*.bin`, `*.info`，可以选择用融合的模型`*.lite.bin`或者同时使用`*.bin`和`*.info`。  
‘-m’参数为模型(model)所在路径，如“/home/Anakin/mobilenet.anakin.bin”  
'-n'参数为生成模型文件的名字(name)  
'-o'参数为生成文件的路径，一般设置在tools/anakin-lite目录  
‘-d’参数为Debug模式，默认为0，不开启Debug  
```bash
$ sh gen_code.sh -a 0 -m /home/Anakin/mobilenet.anakin.bin -n mobilenet -o ../../tools/anakin-lite -d 0
```

#### <span id = '0003'> 三、测试模型(可选)</span> ####
1. 使用adb push将tools/anakin_lite/output/unit_test目录下生成的test_lite_model或者test_lite_merged_model和模型`*.info, *.bin`或者`*.lite.bin`拷贝到手机目录data/local/tmp。内存加载模式可以参考test_lite_model_from_mem或者test_lite_merged_model_from_mem。
```bash
$ adb push tools/anakin_lite/output/unit_test/test_lite_net data/local/tmp 
$ adb push tools/anakin_lite/*.lite.bin data/local/tmp 
```
2. 使用adb shell命令运行test_lite_net，用法为  
./test_lite_net  <模型文件> <batch_size> <预热次数> <执行次数> <大小核> <线程数>   
大小核参数：0代表使用大核，1代表使用小核心  
如测试model.lite.bin，batch_size=1，预热十次，测试二十次，使用大核，四线程  
```bash
$ adb shell
$ cd data/local/tmp
$ ./test_lite_model model.lite.bin 1 10 20 0 4
```

