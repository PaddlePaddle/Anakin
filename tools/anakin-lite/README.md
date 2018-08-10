# Anakin Lite

## 编译模型转换器

运行tools目录下build_lite.sh，编译完成后，会在output目录下生成generator文件夹

## AOT模式
1. 运行generator目录下的gen_code.sh，转换`*.anakin.bin`模型，输出目录选择到`tools/anakin_lite`，
'-a'参数为1，表示AOT模式。该命令会输出3个文件，`*.h`, `*.cpp`和`*.bin`。
2. 如果有多个模型，重复1的操作即可。
3. 运行tools/anakin_lite目录下的脚本，即可生成模型对应的库。
4. 测试模型请参考test/lite/目录下的`test_lite_model.cpp`

## 通用模式

1. 删除tools/anakin_lite目录下的`.h`和`.cpp`文件，运行tools/anakin_lite目录下的脚本，生成对应平台的通用库
2. 运行generator目录下的gen_code.sh，转换`*.anakin.bin`模型，输出目录可自定义，
   '-a'参数为0，表示通用模式。该命令会输出2个文件，`*.lite.info`和`*.lite.bin`
3. 测试模型请参考test/lite/目录下的`test_lite_net.cpp`


