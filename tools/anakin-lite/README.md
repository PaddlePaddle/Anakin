# Anakin Lite
Anakin Lite是Anakin为移动端打造的轻量化前向计算库，支持AOT和通用两种模式。
```
- AOT模式是使用模型转换器根据具体一个模型生成与其相关的`*.h`, `*.cpp`和模型文件`*.bin`，然后编译生成模型对应的库。
- 通用模式是直接编译生成库，库是通用的，所需的模型文件只需通过模型转换器转换为`*.lite.bin）`(融合模型)或者`*.info, *.bin`(分立模型)即可使用。
- 其中`*.info`表示模型的描述文件；`*.bin`表示模型的weights；`*.lite.bin`融合模型包含了模型的weights和模型描述文件。
```
Anakin Lite 的特性包括：
* 支持ARMv7/v8架构
* 支持Android和ios系统
* 无第三方依赖
* 支持openmp多线程
* 支持大小核调度机制
* 支持从memory加载模型
* 简单易用的API

## 注意事项
<font color=#ff0000 size=12 face="黑体">如果使用Android NDK自带的`android.toolchain.cmake`，需要修改该文件，去掉debug信息。</font>

```bash
# 用编辑器打开 $ANDROID_NDK/build/cmake/android.toolchain.cmake
# 删除 "-g" 这行
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

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
   运行tools目录下build_lite.sh，编译完成后，会在output目录下生成generator文件夹。然后，运行output/generator/gen_code.sh，将`*.anakin.bin`模型转换为 `*.lite.bin` 或 `*.bin`，传入参数的含义如下：
	- `-a` 参数为是否AOT模式，如果是，则生成`*.h`, `*.cpp`和`*.bin`；否则，生成`*.lite.bin`， `*.bin`和 `*.info` 默认是0，即通用模式。
	- `-m` 参数为模型(model)所在路径，如“/home/Anakin/mobilenet.anakin.bin”
	- `-n` 参数为生成三个文件的名字(name)
	- `-o` 参数为生成文件的路径，一般设置在tools/anakin-lite目录
	- `-d` 参数为Debug模式，默认为0，不开启Debug
	- `-b` 参数为batch_size的大小，默认为1
	- `-l` 参数为保持weights的数据类型，目前支持float、int16和int8类型，即0-float，1-int8, 2-int16，默认是0

## AOT模式
#### <span id = '0001'> 一、使用模型转换器转换为`*.bin`模型和生成相应`*.h`, `*.cpp` </span> ####
根据上述描述，生成AOT模式的示例如下：
```bash
$ sh gen_code.sh -a 1 -m /home/Anakin/mobilenet.anakin.bin -n mobilenet -o ../../tools/anakin-lite -d 0
```
如果有多个模型，重复上述操作即可

#### <span id = '0002'> 二、使用脚本编译Anakin Lite库</span> ####
1. 编辑tools/anakin_lite目录下的脚本lite_android_build_armv7/8_*.sh，并设置ANDROID_NDK路径。
2. 运行脚本即可生成模型对应的库。

#### <span id = '0003'> 三、测试模型(可选)</span> ####
1. 根据具体的测试模型修改`test/lite/`目录下的`test_lite_aot_model.cpp`，编译完成后，使用adb push将tools/anakin_lite/output/	  unit_test目录下生成的test_lite_aot_model和模型`*.bin`拷贝到手机目录data/local/tmp
```bash
$ adb push tools/anakin_lite/output/unit_test/test_lite_model data/local/tmp
$ adb push tools/anakin_lite/*.bin data/local/tmp
```
2. 使用adb shell命令运行test_lite_aot_model，用法为：

    ./test_lite_aot_model <模型文件> <batch_size> <预热次数> <执行次数> <核数> <线程数>

	eg: 测试model.bin，batch_size=1，预热十次，测试二十次，使用大核，四线程
```bash
$ adb shell
$ cd data/local/tmp
$ ./test_lite_aot_model model.bin 1 10 20 0 4
```

## 通用模式

#### <span id = '0001'> 一、使用模型转换器转换为`*.lite.bin`模型 </span> ####
根据上述描述，生成通用模式的示例如下：

```bash
# 将weights为float类型的anakin.bin转换为float类型的lite.bin
$ sh gen_code.sh -a 0 -m /home/Anakin/mobilenet.anakin.bin -n mobilenet -o ../../tools/anakin-lite -d 0

# 将weights为float类型的anakin.bin转换并压缩为int8类型的lite.bin
$ sh gen_code.sh -a 0 -m /home/Anakin/mobilenet.anakin.bin -n mobilenet -o ../../tools/anakin-lite -d 0 -l 1

# 将weights为float类型的anakin.bin转换并压缩为int16类型的lite.bin
$ sh gen_code.sh -a 0 -m /home/Anakin/mobilenet.anakin.bin -n mobilenet -o ../../tools/anakin-lite -d 0 -l 2
```
如果有多个模型，重复上述操作即可

#### <span id = '0002'> 二、使用脚本编译Anakin Lite通用库</span> ####
1. 如使用过AOT模式，请删除tools/anakin_lite目录下的`.h`和`.cpp`文件。注释掉`test/lite/test_lite_model.cpp`AOT模式下添加的模型，如果没有编辑过该文件，则不需要修改。
2. 编译Android库：编辑tools/anakin_lite目录下的脚本lite_android_build_armv7/8_*.sh，设置ANDROID_NDK路径。
3. 编译IOS库：直接运行lite_ios_build_armv7/8.sh。
4. 运行脚本即可生成通用库。

#### <span id = '0003'> 三、测试模型(可选)</span> ####
1. 使用adb push将tools/anakin_lite/output/unit_test目录下生成的test_lite_model或者test_lite_merged_model和模型`*.info, *.bin`或者`*.lite.bin`拷贝到手机目录data/local/tmp。内存加载模式可以参考test_lite_model_from_mem或者test_lite_merged_model_from_mem单测。
```bash
$ adb push tools/anakin_lite/output/unit_test/test_lite_net /data/local/tmp
$ adb push tools/anakin_lite/*.lite.bin /data/local/tmp
```
2. 使用adb shell命令运行test_lite_net，用法为

   ./test_lite_net  <模型文件> <batch_size> <预热次数> <执行次数> <核数> <线程数>

   eg: 测试model.lite.bin，batch_size=1，预热十次，测试二十次，使用大核，四线程
```bash
$ adb shell
$ cd data/local/tmp
$ ./test_lite_model model.lite.bin 1 10 20 0 4
```

## API 使用说明

### Net
Net类是Anakin预测库对外的接口。

1. 构造函数`Net(PowerMode mode = SABER_POWER_HIGH, int threads = 1)`：

    说明：构造一个net，net可以加载模型，获取输入输出，并做预测。

	参数：
	* `mode`：可以指定Android端大小核调度。默认参数`SABER_POWER_HIGH`：使用大核；
		- `SABER_POWER_HIGH`：使用大核，数值为0；
		- `SABER_POWER_LOW`:使用小核，数值为1；
		- `SABER_POWER_FULL`：可以同时使用大小核，优先使用大核， 数值为2；
		- `SABER_POWER_NO_BIND`：不绑定大小核， 数值为3。
	* `threads`：指定前向计算的线程数（Android，Openmp），默认1个线程。
		- 当指定大小核时，线程数若超过核的数量，则线程数会设置为相应处理器核的数量；
		- 当模式是`SABER_POWER_FULL`或者`SABER_POWER_NO_BIND`时，输入线程数若超过总的处理器核数量时，线程数量会被设置为总核数。

2. 运行模式设置`set_run_mode(PowerMode mode, int threads)`：

	说明：设置模型运行模式，支持Android系统，可以指定大小核和线程数量。

	参数：参考构造函数。

3. 从文件路径加载融合模型`load_model(const char* lite_model_path)`：

	说明： 从文件路径加载模型，模型为`*.lite.bin`融合模型，包含网络信息和参数；

	参数：
	* `const char* lite_model_path`: 模型路径

	返回：若加载成功，则返回`SaberSuccess`，否则返回错误代码；

4. 从文件路径加载分立模型`load_model(const char* info_path, const char* weights_path)`：

	说明： 从文件路径加载分立模型，分别为网络信息和参数信息；

	参数：
	* `const char* info_path`: 模型网络信息
	* `const char* weights_path`：网络参数信息

	返回： 若加载成功，则返回`SaberSuccess`，否则返回错误代码；

5. 从内存加载融合模型`load_model(const void* merged_memory, size_t mem_size)`：

	说明： 从内存加载融合模型，包含网络信息和参数；

	参数：
	* `const void* merged_memory`: 融合模型
	* `size_t mem_size`：数据长度，单位bytes

	返回： 若加载成功，则返回`SaberSuccess`，否则返回错误代码；

6. 从内存加载分立模型`load_model(const void* info_memory, size_t info_size, const void* weights_memory, size_t weights_size)`：

	说明： 从内存加载分立模型，分别为网络信息和参数信息；

	参数：
	* `const void* info_memory`: 模型网络信息
	* `size_t info_size`：数据长度，单位bytes
	* `const void* weights_memory`：网络参数信息
	* `size_t weights_size`：数据长度，单位bytes

	返回： 若加载成功，则返回`SaberSuccess`，否则返回错误枚举类型；

7. 获取网络输入`std::vector<Tensor<CPU>*> get_input()`：

	说明：获取net所有的输入tensor的指针，可以进行赋值和reshape操作

	返回：返回一个vector存放所有输入tensor的指针，tensor已经分配好空间。

8. 获取网络指定的输入`Tensor<CPU>* get_input(std::string name)`：

	说明：根据输入的名称，获取指定输入tensor指针

	参数：
	* `std::string name`：输入tensor的名称，可以在网络图中获取
	返回：如果存在名字为`name`的tensor，则返回该tensor的指针，否则返回`nullptr`

9. 获取网络输出层的全部输出`std::vector<Tensor<CPU>*> get_output()`：

	说明： 获取网络所有输出tensor的指针

	返回：返回一个vector存放所有输出tensor的指针。

10. 获取网络输出层的指定输出`Tensor<CPU>* get_output(std::string name)`：

	说明：根据输入的名称，获取指定输出tensor指针

	参数：
	* `std::string name`：输出tensor的名称，可以在网络图中获取

	返回：如果存在名字为`name`的tensor，则返回该tensor的指针，否则返回`nullptr`

11. 网络前向计算`prediction()`：

	说明： 网络前向计算

	返回： 如果成功返回`SaberSuccess`，如果有错误返回相应错误枚举类型。

### Tensor
`Tensor`类是Anakin lite的基础数据类型。
- Tensor是一个模板类，支持移动端CPU，GPU，DSP等，支持数据类型有float，int8等。目前lite版仅支持CPU数据，
数据类型为float或int8, 声明Tensor对象时需要指定模板为`Tensor<CPU>`(无需指定数据类型)
- Tensor支持内存的复用，因此Tensor包含当前有效维度信息`valid_shape`和总维度信息`Shape`，
在取数据时，需要注意用`valid_shape`和`valid_size`接口。

1. 构造函数
	Tensor包含4个构造函数：
	* `Tensor(DataType type = AK_FLOAT)`:声明一个空的tensor，没有分配数据空间，默认type是AK_FLOAT；
	* `Tensor(Shape shape, DataType type = AK_FLOAT)`：构造一个维度信息为`shape`的tensor，分配`shape`维度信息的数据空间；
	* `Tensor(Dtype* data_ptr, Shape shape, DataType type = AK_FLOAT)`：从已有的数据构造一个tensor，不分配数据空间；
	* `Tensor(const Tensor<ttype, dtype>& tensor)`：拷贝构造函数，数据为浅拷贝

2. 设置tensor维度信息`set_shape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape())`:
	说明：设置tensor的维度信息，不分配数据空间。
	参数：
	* `valid_shape`：当前tensor有效数据维度信息
	* `shape`：当前tensor真正维度信息。默认为空，表示与valid_shape一致，shape始终要大于等于valid_shape
	* `offset`：表示valid_shape偏移shape的维度信息，默认为空，只有在share_sub_buffer的情况下用到（该参数暂时没有用）。
	返回：如果成功返回`SaberSuccess`，否则返回错误枚举类型。

3. 重新分配空间`re_alloc(Shape shape, DataType dtype)`:
	说明：重新分配tensor内存空间，如果tensor已经分配了内存空间，则先释放该内存，重新申请一块内存。
	如果当前tensor是从别的tensor共享的（调用share_from），在调用此接口时会返回错误。
	参数：
	* `shape`: tensor维度信息，调用该接口后，tensor内部的`valid_shape`和`shape`都变成输入的`shape`.
	* `dtype`: tensor的数据类型，目前支持AK_FLOAT，AK_INT8
	返回：如果成功返回`SaberSuccess`，否则返回错误枚举类型。

4. 调整内存空间`reshape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape())`：
	说明：调整tensor内存空间和有效数据维度信息。该接口可以用于对网络(net)输入维度进行调整。如果tensor是通过`share_from`共享的，
	则输入`shape`的大小不能超过原有tensor的`shape`的大小。
	参数：
	* `valid_shape`：当前tensor有效数据维度信息
	* `shape`：当前tensor真正维度信息。默认为空，表示与valid_shape一致，shape始终要大于等于valid_shape
	* `offset`：表示valid_shape偏移shape的维度信息，默认为空，只有在share_sub_buffer的情况下用到（该参数暂时没有用）。
	返回：如果成功返回`SaberSuccess`，否则返回错误枚举类型。

5. 设置tensor的数据类型`set_dtype(DataType type)`:
	说明：可以自动设置tensor的数据类型，如果该tensor是通过share_from共享的或tensor的已分配空间小于`set_dtype`所需的空间，则不能设置tensor的数据类型
	参数：
	* `type`：数据类型，目前支持float，int8
	返回：如果成功返回`SaberSuccess`；否则返回错误枚举类型。

6. 获取tensor的数据类型`get_dtype()`:
	说明：获取tensor的数据类型
	返回：tensor的数据类型

7. 设置tensor的量化值`set_scale(const std::vector<float> scale)`:
	说明：在量化模型中，设置tensor的量化值
	参数：
	* `scale`: 表示tensor的量化值

8. 获取tensor的量化值`get_scale()`:
	说明：获取tensor的量化值
	返回：获取tensor的量化值

9. 获取有效维度信息`valid_shape()`:
	说明：获取当前tensor有效的数据维度信息。
	返回：维度信息Shape

10. 获取真实维度信息`shape()`:
	说明：获取当前tensor真实的数据维度信息。
	返回：维度信息Shape

11. 获取有效数据长度`valid_size()`：
	说明：获取有效数据的长度
	返回：有效数据长度

12. 获取真实数据长度`size()`：
	说明：获取有效数据的长度
	返回：有效数据长度

13. 获取可修改数据的指针`mutable_data(int index = 0)`：
	说明：获取tensor的数据指针，可读写
	参数：
	* `index`：数据起始地址，默认为0
	返回：数据指针，该指针为void* 类型，如需对数据进行修改，需将改指针强制转换为Tensor的DataType，AK_FLOAT为float*，AK_INT8为signed char*

14. 获取只读数据的指针`data(int index = 0)`：
	说明：获取tensor的数据指针，只读
	参数：
	* `index`：数据起始地址，默认为0
	返回：该指针为void* 类型，如需对数据进行读取，需将改指针强制转换为Tensor的DataType，AK_FLOAT为float*，AK_INT8为signed char*

15. 数据共享`share_from(const Tensor& tensor)`:
	说明：共享tensor的数据空间，要求被共享的tensor的真实数据长度不小于当前tensor真实数据长度。
	参数：
	* `tensor`：被共享的数据空间的tensor
	返回：如果成功返回`SaberSuccess`，否则返回相应的错误枚举类型。

16. 数据拷贝`copy_from(const Tensor& tensor)`:
	说明：tensor至今数据拷贝，要求当前tensor和被拷贝的tensor有效数据长度必须一致。
	参数：
	* `tensor`：被拷贝的数据空间的tensor
	返回：如果成功返回`SaberSuccess`，否则返回相应的错误枚举类型。

17. 获取特定维度信息`num()`, `channel()`, `height()`, `width()`:
	说明：
	* `num()`获取tensor的batch大小；
	* `channel()`获取tensor的通道数；
	* `height()`获取tensor高度大小；
	* `width()`获取tensor宽度大小；
	返回： 返回对应的维度大小

### Shape
`Shape`类用于指定`Tensor`类数据维度信息，Layout类型是NCHW

1. 构造函数
	* `Shape(First first, Args... res)`: 通过可变长参数构造Shape，可以是任意长度。在Tensor中使用时，输入为4维。
	* `Shape(std::vector<int> vsh)`： 从一个vector<int>构造Shape，Shape中的数据从vsh拷贝

### Example示例
本章将简要介绍如何用Anakin lite的产出，跑通public 的caffe 模型mobilenet_v1。

#### Caffe 模型转换 `*.anakin.bin`
```bash
$ python converter.py --framework CAFFE --result_name mobilenet_v1 --caffe_proto_paths    ./model/caffe.proto --caffe_proto_txt_path  ./model/mobilenet_v1.prototxt --caffe_model_path   ./model/mobilenet_v1.caffemodel
```
#### `*.anakin.bin` 转换 `*.lite.bin`
```bash
$ sh output/generator/gen_code.sh -n mobilenet_v1 -m ./output/ mobilenet_v1.anakin.bin -a 0 -d 0 -o ./
```
#### Net demo
1. 在Android或IOS工程中，包含Anakin lite的include目录下的头文件和相应ABI的静态库或动态库文件
2. 建立一个CPP文件，调用Anakin Lite API 接口，demo 示例如下：
```bash
	1) 构造网络：
		Net net = new Net((PowerMode)0, 1)
	2) 设置运行模式：
		net->set_run_mode((PowerMode)3, 4); //选择核数和线程数
	3) 加载模型：
		net->load_model(lite_model_path);
		eg: net->load_model("/data/local/tmp/ mobilenet_v1.lite.bin")
	4) 获取输入tensor：
		std::vector<TensorHf*> vtin = net->get_input()
	5) 可以对输入tensor做reshape处理
		eg: vtin[0]->reshape(Shape(1, 3, 224, 224))
	6) 对输入的tensor赋值：
		* 方法一：赋全一的数据--Anakin 自带
			fill_tensor_const(*vtin[0], 1.f);
		* 方法二：赋随机数据--Anakin 自带
			fill_tensor_rand (*vtin[0], -1.f,1.f);//第二个参数和第三个参数表示赋值范围
		* 方法三：从图片读数据进行赋值--需要自己实现
			fill_tensor_with_cvmat(img, *vtin[0], batch_size, thin.width(), thin.height(), mean_mb, scale_mb);
	7) 前向预测计算：
		net->prediction()
	8) 获取预测结果：
		std::vector<TensorHf*> vtout = net->get_output()
	9) 最后释放net，结束预测：
		delete net
```
