**Anakin 用户手册**
===============

   本手册旨在让Anakin使用者能更好的运用Anakin到自己的项目和实验中去，减少用户的时间和精力成本。
   如果您在使用Anakin过程中遇到了问题，希望您能在本手册中找到原因和解决问题的方法。如果本手册不能帮助到您，您可以发邮件到Anakin邮件组(anakin@baidu.com)，我们会及时回应您的问题。
   为了让您能够高效正确的使用Anakin，建议您仔细阅读本手册；您也可以通过右下方的『内容目录』选择您关心的部分进行查看。

1. Anakin基本介绍
-----------------

   Anakin是一个跨平台的高性能前向推理引擎，支持多种硬件平台和各大主流的深度学习训练框架。
  目前Anakin的特性如下：
   -- 支持x86、NV GPU、AMD GPU、ARM、FPGA多种硬件平台
   -- 支持FP32、INT8精度及动态精度推理策略
   -- 全面支持Caffe、Paddle Fluid、Onnx、 TensorFlow 框架。
 

2. Anakin的编译与部署
-------------------

   
 **2.1 直接拉取产出布署**

 您可以使用以下命令直接拉取Anakin版本的产出：

拉取之后你会得到如下结构的文件目录

    output:
        -anakin_arm_release.tar.gz
        -anakin_release_nv.tar.gz
        -anakin_release_native_x86.tar.gz
   分别代表Anakin在arm、nv、x86上的库，请对应于您所要使用的平台选择解压，解压后您将得到如下文件目录：
   

    output:
         --framework
         --libanakin.so.1.0.0		
         --saber
         --include
         --libanakin_saber_common.so	
         --unit_test
         --libanakin.so			
         --libanakin_saber_common.so.1.0.0	
         --utils

   **2.2 编译Anakin**

   Anakin最新代码在icode维护，每当发布新版本，我们会同步更新到github上，如果您想了解Anakin的最新版本的代码、编译、性能数据等信息，您可前往github上进行查看（https://github.com/PaddlePaddle/Anakin）。

3. 将其他模型转换成Anakin模型
-------------------

在用Anakin进行模型推理时，您首先要将其他框架的模型转换成Anakin自己的模型格式（后缀名为anakin.bin）。Anakin目前全面支持Caffe和Paddle Fluid框架模型，部分支持ONNX、Lego、TensorFlow模型。
Anakin提供了一整套模型转换工具及可视化工具，您只需要进行简单的配置即可进行模型转换，并能够看到转换后的结果。
我们的转换工具位于*Anakin-2.0/tools/external_converter_v2*，转换工具不需要事先编译Anakin，只需要您有相应的python环境即可。

**3.1所需环境**

    --Python 2.7+

    --Protobuf 3.1+（务必注意 Python 与系统环境 Protobuf 版本一致）

    --PaddlePaddle 0.12.0+ (Fluid 模式下)

    --flask, bson, matplotlib, scikit-image（python库，可使用pip安装）

    --tkinter
    
**3.2文件配置**

您需要在external_converter_v2下的config.yaml文件里进行必要的配置，以成功转换模型，示例如下：

**config.yaml文件：**

    OPTIONS:
        Framework: CAFFE       # 依框架类型填写 CAFFE 或 FLUID
        SavePath: ./output     # 转换结束后模型的保存位置
        ResultName: googlenet  # 输出模型的名字，生成的模型会自动带有后缀
        Config:
            LaunchBoard: ON    # 是否生成网络结构预览页面
            Server:
                ip: 0.0.0.0
                port: 8888     # 从一个可用端口访问预览页面
            OptimizedGraph:    # 仅当您执行完预测并使用 Optimized 功能时，才应打开此项
                enable: OFF
                path: /path/to/anakin_optimized_anakin_model/googlenet.anakin.bin.saved
        LOGGER:
            LogToPath: ./log/  # 生成日志的路径
            WithColor: ON
    
    TARGET:
        CAFFE:
            # 当 Framework 为 CAFFE 时需填写
            ProtoPaths:
            - /path/to/caffe/src/caffe/proto/caffe.proto
            PrototxtPath: /path/to/your/googlenet.prototxt
            ModelPath: /path/to/your/googlenet.caffemodel
            Remark:
    
        FLUID:
            # 当 Framework 为 FLUID 时需填写
            Debug: NULL                                # 不需要更改
            ModelPath: /path/to/fluid/inference_model  # 模型所在文件夹路径，里面需要包括 model 和 params 文件
            NetType:                                   # 一般为空。特殊网络类型使用，如 OCR、SSD
        LEGO:
            # path to proto files
            ProtoPath:
            PrototxtPath:
            ModelPath:

    TENSORFLOW:
            ModelPath: /path/to/your/model/
            OutPuts:
    ONNX:
            ModelPath:
        # ...

**3.3 进行转换**

在完成配置文件的修改后，您只需执行 python converter.py 就可以进行模型转换了，而且您可以到配置文件中的网址进行查看转换后的模型。
注：如您在配置文件中的ip是0.0.0.0，则在查看时应换成当前运行机的ip.

4. Anakin常用数据结构及api简介
---------------------

**4.1 Shape**

Shape 是Anakin中表示维度信息的数据结构，在Anakin中，您可以将其看作一个vector，Shape的长度表示了其维度数，Shape中的layout成员记录了Shape所表示的布局信息，您可通过get_layout()方法获取，其返回LayoutType类型。
Anakin支持的所有layout如下表所示：

  |LayoutType | Tensor维度数 | 备注 | 
|---|---|---|
|Layout_W | 1 | 向量 | 
|Layout_HW | 2 | 矩阵 | 
|Layout_WH | 2 |  | 
|Layout_NW | 2 |  | 
|Layout_NHW | 3 |  | 
|(default)Layout_NCHW | 4 | 标准4维张量 | 
|Layout_NHWC | 4 |  | 
|Layout_NCHW_C4 | 5 | 用于int8类型 | 
|Layout_NCHW_C8R | 4 | 目前专用于x86部分op |
|Layout_NCHW_C16R | 4 |  |


**4.1.1. Shape的构造：**

**Shape()**：创建一个空Shape, 其layout为默认的NCHW
**Shape(vector, LayoutType=Layout_NCHW)**：创建对应维度的layout的Shape。

Shape的更多详细方法您可参考*Anakin-2.0/saber/core/shape.h*。

**4.2 Tensor**

Tensor是Anakin最基础的数据结构，用于存储各种计算结果数据，包括各个op的输入输出、op内部的weight等数据。Tensor是与硬件平台相关的，在定义时需要显式指定其所在的平台(TargetType)。

**4.2.1. Tensor的构造：**

**Tensor(DataType=AK_FLOAT)**: 定义给定数据类型的空Tensor
**Tensor(Shape, DataType=AK_FLOAT)**: 定义给定数据类型和维度信息的Tensor，同时分配足够的空间
**Tensor(dtype*, TargetType, int, Shape, DataType=AK_FLOAT)**: 定义给定数据指针，平台，设备id, 维度信息，数据类型的Tensor, Tensor共享数据指针所指向的空间
**Tensor(Tensor)**: 复制给定Tensor的Tensor, 只能复制同平台的Tensor

**4.2.2. Tensor主要属性和方法：**

  -- **data**：Tensor所存储的数据指针
*可以通过data()/mutable_data()方法获取只读/可读可写指针，返回的是void指针类型，需要手动转换成所需要的指针类型*

  --**dtype**：Tensor中数据的类型
  *可以通过get_dtype()方法获取，返回DataType类型。Anakin所支持的DataType类型如下：*

  |DataType | 对应的C++类型 | 说明 | 备注 | 
|---|---|---|---|
|AK_HALF | short | fp16 |  | 
|AK_FLOAT | float | fp32 |  | 
|AK_DOUBLE | double | fp64 |  | 
|AK_INT8 | char | int8 |  | 
|AK_INT16 | short | int16 |  | 
|AK_INT32 | int | int32 |  | 
|AK_INT64 | long | int64 |  | 
|AK_UINT8 | unsigned char | uint8 |  | 
|AK_UINT16 | unsigned short | uint8 |  | 
|AK_UINT32 | unsigned int | uint32 |  | 
|AK_STRING | std::string | -- |  | 
|AK_BOOL | bool | -- |  | 
|AK_SHAPE | -- | Anakin Shape |  | 
|AK_TENSOR | -- | Anakin Tensor |  | 


  -- **shape**：Tensor中数据的维度信息

      *可以通过valid_shape()方法获取，返回的为vector类型，其长度表示了Tensor的维度数，不同的维度数与数学中的向量、矩阵对应关系如下：*

   |维度数 | 数学定义 | 
|---|---|
|1 | 向量 | 
|2 | 矩阵 | 
|3 | 3维张量 | 
|n | n维张量 | 
      
  -- **layout**：Tensor中数据的布局信息

      *可以通过get_layout()方法获取，返回的是LayoutType类型。

其它的Tensor常用api列于下表，更详细的用法可查看*Anakin-2.0/saber/core/tensor.h*。

|api | 说明 | 返回值 | 备注 | 
|---|---|---|---|
|set_dtype(DataType) | 设置Tensor数据类型 | SaberStatus |  | 
|set_layout(LayoutType) | 设置Tensor数据布局 | SaberStatus |  | 
|set_shape(Shape) | 设置Tensor的维度信息 | SaberStatus<br> | 该方法只会设置维度信息，不改变Tensor空间大小 | 
|get_seq_offset() | 获取Tensor的offset信息 | vector<vector<int>> | 专用于word相关的Tensor |
|set_seq_offset() | 设置Tensor的offset信息 | SaberStatus<br> | 专用于word相关的Tensor | 
|re_alloc(Shape, DataType) | 为Tensor分配空间 | SaberStatus<br> | 空间大小由Shape指定 | 
|valid_size() | 返回Tensor的总的数据大小 | long long |  | 
|reshape(Shape) | 重新设置Tensor的维度信息 | SaberStatus<br> | 该方法会重设Tensor的空间大小 | 
|copy_from(Tensor) | 从已有Tensor拷贝数据 | SaberStatus<br> | 源Tensor可以与目标Tensor所在平台不同 | 

**4.2.3. Tensor常用操作**

由于Tensor是与平台有关的，为了让用户更加高效地操作Tensor，Anakin提供了常用的Tensor操作，以下将使用比较多的接口做一介绍，详细接口您可查看*Anakin-2.0/saber/core/tensor_op.h*。
i. 为Tensor填充数据
为了方便测试，Anakin提供了常用了两种Tensor填充函数：
fill_tensor_const(Tensor, v)和fill_tensor_random(Tensor, min, max)，这两个函数可以接受任意平台的Tensor，用法示例如下：


    //定义一个Tensor, 目标平台为NV
    Tensor<NV> tensor_d;
    //定义Tensor的shape, 同时指定其布局信息为NCHW
    //注：如不显式指定布局信息，其默认为NCHW
     Shape tensor_shape({1, 2, 3, 4}, );
     //为tensor_d分配空间，指定其数据类型为float
     //注：如不显式指定数据类型，其默认为AK_FLOAT
     tensor_d.re_alloc(Shape, AK_FLOAT);
     //填充tensor_d为全1
     fill_tensor_const(tensor_d, 1);
     //填充tensor_d为（-1，1）之间的随机数 
     //如不显式指定范围，其默认为(-255, 255)
     fill_tensor_random(tensor_d, -1, 1);

在实际运用中，我们需要在tensor中填充自定义数据结构中的数据，我们可以先获取到Tensor的数据指针，然后对其数据区进行填充。由于Tensor上的数据可能存在于其他平台，为了安全起见，我们需要先维护一个host端的同样大小的Tensor，填充完后再拷贝回目标Tensor， 示例如下：


     //创建用户数据
     //假定用户数据为长度为24的float数组
     float* user_data = new float[24];
     //定义一个host端tensor, 其大小、数据类型与tensor_d一致
     //该定义方式下，tensor_h会自动分配足够空间
     Tensor<X86> tensor_h(tensor_d.valid_shape(), tensor_d.get_dtype());
     //获取tensor_h的数据指针，并转成float*类型
     float* h_data_ptr = static_cast<float*>(tensor_h.mutable_data());
     //逐个填充数据
     //注：为了避免越界，请保证user_data数据个数不小于tensor_h大小
     for (int i=0; i < tensor_h.valid_size(); ++i){
         h_data_ptr[i] = user_data[i];
     } 
     //host端填充完毕，拷贝到device端tensor
     tensor_d.copy_from(tensor_h);
     //释放用户数据
     delete user_data;


   ii.查看Tensor数据
   为方便用户查看Tensor的数据，Anakin提供了与平台无关的函数print_tensor_valid(Tensor)，该函数可以按对应的shape打印出Tensor中的数据，方便用户进行一些调试和查看某些中间数据结果。
   具体用法如下：
   

       //打印出tensor_d的数据
       print_tensor_valid(tensor_d);
       //打印出tensor_h的数据
       print_tensor_valid(tensor_h);

   **4.3 Graph**

   Graph是对神经网络模型的表示图，其中存储了模型的网络结构、op信息、以及权重参数信息，同时Graph类还负责对原始模型的分析优化工作。Graph是与平台和执行精度相关的，在定义时必须显式指定其所在平台（TargetType）和执行精度(Precision)。
   **4.3.1. Graph只有一个空构造函数**
   **Graph()** : 构造一个空的Graph。
   
   **4.3.2. Graph常用方法**

   Graph的常用方法如下表所示，更详细的方法请查看*Anakin-2.0/framework/graph/graph.h*

 |方法 | 说明 | 返回值 | 备注 | 
|---|---|---|---|
|load(std::string)<input type="checkbox" class="rowselector hidden"> | 加载模型 | Status |  | 
|Optimize() | 进行图的分析和优化 | Status | 如果第一次加载模型，则必须调用此函数 | 
|save(std::string) | 模型存储 | Status | 特别的，您可以存储一个优化后的图，之后加载时将不会再进行优化 | 
|get_ins() | 获取图的输入数据名 | vector&lt;std::string&gt; |  | 
|get_outs() | 获取图的输出数据名 | vector&lt;std::string&gt; |  | 
|ResetBatchSize(std::string, int) | 设置输入数据batchsize | void |  | 
|Reshape(std::string, Shape) | 重新设置输入数据的大小 | void |  | 
| |  |  |  | 

**4.3.3. Graph使用方法**

以下是graph类的用法示例：


    //Graph定义，其平台为NV, 执行精度为FP32
    Graph<NV, Precision::FP32> graph;
    //加载Anakin模型
    std::string model_path = "model.anakin.bin";
    graph.load(model_path);
    //进行图的分析优化
    graph.Optimize();
    //存储优化后的模型
    std::string save_path = "model.anakin.bin.saved";
    graph.save(save_path);
    //获取模型的输入名
    std::vector<std::string> in_names = graph.get_ins();
    //获取模型的输出名
    std::vector<std::string> out_names = graph.get_outs();
    
**4.4 Net**

Net主要负责网络的运行时初始化工作及模型推理，是Graph的执行器。和Graph一样，Net是与平台、执行精度相关的，定义时必须显式指定其平台（TargetType）和精度（Precision）。同时，Net还提供了同步和异步两种执行方式，默认情况下是同步执行。
**4.4.1. Net的构造**
**Net( bool  = false)**: 构造空的Net，bool表示是否需要Net提供运行时信息
**Net(Graph, bool = false)**: 用Graph构造Net
**Net(Graph, Context, bool = false)**: 用Graph和上下文构造Net

**4.4.2. Net常用方法**
Net常用的方法如下表所示，更详细的接口您可查看*Anakin-2.0/framework/core/net.h查看。*

|方法 | 说明 | 返回值 | 备注 | 
|---|---|---|---|
|init(Graph) | 网络初始化 | 无 | <br> | 
|prediction() | 网络预测 | 无 |  | 
|get_in(std::string) | 获取对应名称的输入Tensor |  |  | 
|get_out(std::string) | 获取对应名称的输出Tensor |  |  | 
|load_calibrator_config(std::string, std::string) | 加载网络精度配置和量化表 |  | 用在int8精度推理中 | 

Net的示例用法如下：


    //定义Net, 其平台为NV, 执行精度为FP32
    Net<NV, Precision::FP32> net(true);
    //用Graph初始化网络
    net.init(graph);
    //获取网络的输入Tensor指针
    //注：需要事先知道网络的输入Tensor名，您可以通过dashboard查看，更通用的方法将在下节演示
    auto in_tensor_p = net.get_in("input_0");
    //填充Tensor
    fill_tensor_const(*in_tensor_p, 1);
    //进行预测
    net.prediction();
    //如果是rnn模型还需要设置offset,例如输入经过预处理后是一行包含n个float类型的数，那么offset=n
    std::vector<std::vector<int>> seq_offset={{0,offset}};
    in_tensor_p->set_seq_offset(offset);
    //获取输出Tensor指针
    //注：需要事先知道网络的输出Tensor名，您可以通过dashboard查看，更通用的方法将在下节演示
    auto out_tensor_p = net.get_out("output");
    
**4.5 Worker**

Worker是Anakin进行多线程异步推理的执行器，与Net一样，Worker也与平台（TargetType）和执行精度（Precision）相关，在定义时必须显式指定。
我们将在第6节详细介绍Worker的用法，这里仅列出Worker类常用的方法，更详细的接口您可查看*Anakin-2.0/framework/core/worker.h*。

|方法 | 说明 | 返回值 | 备注 | 
|---|---|---|---|
|register_inputs(std::vector&lt;std::string&gt;) | 注册模型输入数据名 | void | <br> | 
|register_outputs(std::vector&lt;std::string&gt;) | 注册模型输出数据名 | void |  | 
|Reshape(std::vector&lt;int&gt;) | 设置输入数据维度信息 | void |  | 
|sync_prediction() | 同步预测 | std::future |  | 
|syn_prediction() | 异步预测 | std::future |  | 

**4.6 子图**

anakin支持子图调用，作为第三方框架的一个子图进行计算，子图API见anakin网站*https://anakin.baidu.com/Anakin/addCustomOp.html*

5. 单线程下使用Anakin前向推理

前面几节我们分别介绍了Anakin的布署、模型的转换、以及常用api的用法，在本节，我们将以一个例子来演示如何使用Anakin进行推理。
我们假设您已经成功布署Anakin、成功转换了anakin.bin模型。
1.首先，我们需要包含必要的头文件

    #inlcude "framework/graph/graph.h"//for Graph
    #include "framework/core/net/net.h"//for Net
    #include "saber/core/tensor.h"//for Tensor
    #include "saber/core/tensor_op.h"//for tensor op funcs
    
    #include <string>

2.接下来，定义一些预定义变量

    //定义目标平台为NV, 对应的host平台为NVHX86
    using Target = NV
    using Target_H = NVHX86
    //定义执行精度为fp32
    Precision Prec = Precision::FP32;
    //定义batchsize =1， epoch =1, 设备id为0
    int batch_size = 1;
    int epoch = 1;
    int device_id = 0;
    std::string model_path = "anakin_model.anakin.bin";
    std::string save_path = "anakin_model.anakin.bin.saved";

3.初始化当前环境，在我们进行跑模型之前，我们要先设置当前的设备id，然后用Env类来进行硬件环境的初始化工作，示例如下

    //环境初始化
    TargetWrapper<Target>::set_device(device_id);
    Env<Target>::env_init();

4.定义graph并进行图优化

    //定义Graph
    Graph<Target,  Prec> graph;
    //加载Anakin模型
    auto status = graph.load(model_path);
    if (!status){
        LOG(FATAL) << "load model failed!!";
    }
    //图的分析优化
    graph.Optimize();
    //存储优化后的模型
    graph.save(save_path);
    //获取模型的输入名
    std::vector<std::string> in_names = graph.get_ins();
    //遍历所有输入，设置其batch_size
    //注：在这里也可以同时设置模型的输入shape
    for (int i=0; i<in_names.size(); ++i){
        graph.ResetBatchSize(in_names[i], batch_size);
    }
    //获取模型的输出名
    std::vector<std::string> out_names = graph.get_outs();

    //如果想获取指定层的输出，需要根据该层和下一层的名字进行注册，并且是在*Optimize*之前做。
    graph->RegistOut(std::string node_name, std::string next_node_name)；
    
4.定义执行器Net

    //定义net
    Net<Target, Prec> net(true);
    //初始化网络
    net.init(graph);
    //获取网络的输入Tensor指针并填充
    //注：这里通过遍历graph的所有输入，来一一给net的输入填充数据
    for (int i=0; i<in_names.size(); ++i){
        auto in_tensor_p = net.get_in(in_names[i]);
        //这里您可以用自己的数据填充，见4.1节
        fill_tensor_random(*in_tensor_p, -1, 1);    
    }
    //进行预测
    net.prediction();
    //获取输出Tensor指针
    //注：这里通过遍历graph的所有输出，来一一获得net的输出数据
    for (int i=0; i<out_names.size(); ++i){
        auto out_tensor_p = net.get_out(out_names[i]);
        //这里您可以对各个输出做后处理
        //我们这里只是取出每一个tensor的只读指针
        const float* out_data = static_cast<const float*>(out_tensor_p->data())；
    }

    //获取指定层的输出，在步骤3注册后，进行下面操作
    auto out_tensor = net.get_out(node_name.c_str());
    print_tensor(*out_tensor);
完整的代码示例请查看Anakin-*2.0/test/framework/net/net_exec_test.cpp*。
    

6.多线程下使用Anakin前向推理

**6.1 Anakin worker类介绍**

为了能进行多线程异步推理，Anakin设计了Worker类，Worker里面维护着多个线程，当我们需要进行某个数据的预测的时候，Worker类便找到其中的某个空闲线程，让它异步地帮我们完成数据的预测。
与Net相似， Worker是与平台和精度相关的，在我们定义Worker时，需要显式指定其所在平台（TargetType）和执行精度（Precision），同时需要传给它维护的模型和内部的线程数。

**6.2多线程示例**

下面我们将以一个例子来演示如何使用Anakin Worker进行推理。
我们假设您已经成功布署Anakin、成功转换了anakin.bin模型。
1.首先，我们需要包含必要的头文件

    #inlcude "framework/core/net/worker.h"//for Worker
    #inlcude "framework/graph/graph.h"//for Graph
    #include "saber/core/tensor.h"//for Tensor
    #include "saber/core/tensor_op.h"//for tensor op funcs
    
    #include <string>

2.接下来，定义一些预定义变量

    //定义目标平台为NV, 对应的host平台为NVHX86
    using Target = NV
    using Target_H = NVHX86
    //定义执行精度为fp32
    Precision Prec = Precision::FP32;
    //定义batchsize =1， epoch =10, 设备id为0, 线程数为5
    int batch_size = 1;
    int epoch = 10;
    int threads = 5;
    int device_id = 0;
    std::string model_path = "anakin_model.anakin.bin";
    
 3.定义graph并获取输入输出

    //定义Graph
    Graph<Target,  Prec> graph;
    //加载Anakin模型
    auto status = graph.load(model_path);
    if (!status){
        LOG(FATAL) << "load model failed!!";
    }
    //获取模型的输入名
    std::vector<std::string> in_names = graph.get_ins();
    //获取模型的输出名
    std::vector<std::string> out_names = graph.get_outs();
 
 4. 定义Worker
 
示例

    //定义worker，执行方式为异步形式
    worker< Target,Prec,OpRunType::ASYNC > worker(model_path,threads);
    //注册输入输出Tensor名
    worker.register_inputs(in_names);
    worker.register_outputs(out_names);
    //设置输入数据大小
    //注：在实际创建过程中，您需要根据网络的输入Shape或您的需要来设置worker的Shape，为了演示方便，我们采用同一个Shape
    Shape in_sh({1, 2, 3, 4});
    for (int i=0; i<in_names.size(); ++i){
        worker.Reshape(in_names[i], in_sh);   
    }
    //设置输入Tensor
    std::vector<Tensor<Target>*> in_tensors;
    for (int i=0; i< in_names.size(); ++i){
        Tensor<Target>* tensor = new Tensor<Target>(in_sh);
        fill_tensor_const(*tensor, 1);
        in_tensors.push_back(tensor);
    }
    //进行异步预测
    //这里我们用同一个输入进行10次预测
    for (int i=0; i<epoch; ++i){
        worker.async_prediction(in_tensors);    
    }
    //接收预测结果
    int iteration = epoch;
    while (iteration){
        if (!worker.empty()) {
            auto result = worker.async_get_result()[0];
            //you can do with result here
            --iteration;
        }
    }

 完整的代码示例请查看*Anakin2.0/test/framework/net/net_exec_multi_thread_test.cpp*

7. Anakin推理性能评估

**7.1 SaberTimer类介绍**

为了统计Anakin在各个平台下的推理时间，我们提供了SaberTimer类，它能够提供各个硬件平台下的基本的性能统计功能。SaberTimer是和硬件平台相关的，在定义时必须显式指定其所在平台（TargetType），下表列出了SaberTimer类常用的方法，更详细的用法您可查看*Anakin-2.0/saber/funcs/timer.h*。

|方法 | 说明 | 返回值 | 备注 | 
|---|---|---|---|
|clear() | 清除时间记录 | 无 | <br> | 
|start(Context) | 开始计时 | 无 |  | 
|end(Context) | 结束计时 | 无 |  | 
|get_average_ms() | 获取开始到结束的时间 | float | 单位为ms | 

**7.2性能评估**

利用SaberTimer类可以很方便的进行模型推理的时间统计，它要用到当前设备的Context，Context代表了当前设备的上下文信息，可以用下面代码定义，示例代码如下：


    //定义Context和SaberTimer，其平台为NV，设备id为0
    int dev_id = 0;
    Context<NV> ctx(dev_id, 0, 0);
    saber::SaberTimer<NV> time;
    //开始计时
    time.start(ctx);
    //做网络预测和其他一些事情
    net.prediction();
    ...
    //结束计时
    time.end(ctx);
    //获取时间间隔
    float elapsed = time.get_average_ms();

8. Anakin demo

**8.1 x86**

通过编译产出或者下载anakin发布产出后，通过gcc指定头文件搜索路径，指定动态库就可以构建自己的工程。例如：Anakin库所在目录为Anakin_Lib,测试cpp为demo_test_x86_net.cpp,gcc使用如下命令：


    gcc demo_test_x86_net.cpp -I Anakin_Lib -I Anakin_Lib/include/ -L Anakin_Lib -L /opt/compiler/gcc-4.8.2/lib -L /usr/lib64/ -L /lib64/  -ldl -lanakin_saber_common -lanakin  -o example_x86_rnn_net -std=c++11

    #include <string>
    #include "framework/graph/graph.h"
    #include "framework/core/net/net.h"
    #include "saber/core/tensor.h"
    
    std::string model_path = "vgg16.anakin.bin";
    int batch_size = 1;
    
    void run_demo(){
        //new graph
        anakin::graph::Graph<anakin::saber::X86, anakin::Precision::FP32>* graph = new 
                                                                  anakin::graph::Graph<anakin::saber::X86, anakin::Precision::FP32>();
        //load model
        auto status = graph->load(model_path);
        std::cout << "model_path:" << model_path << std::endl;
        //get inputs
        std::vector<std::string>& vin_name = graph->get_ins();
        //reset batchsize by man's input shape
        for (auto& in: graph->get_ins()) {
            graph->ResetBatchSize(in, batch_size);
        }
        //graph optimize
        auto status_2 = graph->Optimize();
        //net init
        anakin::Net<anakin::saber::X86, anakin::Precision::FP32> net(*graph, true);
        //set offset
        std::vector<std::vector<int>> seq_offset = {{0, batch_size}};
        for (int i = 0; i < vin_name.size(); i++) {
            anakin::saber::Tensor<anakin::saber::X86>* tensor = net.get_in(vin_name[i]);
            fill_tensor_rand(*tensor);
            tensor->set_seq_offset(seq_offset);
        }
        net.prediction();
        std::vector<std::string>& out_name = graph->get_outs();
        for (int i=0; i < out_name.size(); i++) {
            print_tensor(*net.get_out(out_name[i]));
        }
    }
    
    int main(int argc, const char** argv){
        if (argc == 2){
            model_path = atoi(argv[1]);
        }
        std::cout << "model_path:" << model_path << std::endl;
        if (argc == 3){
        batch_size = atoi(argv[2]);
        }
        anakin::saber::Env<anakin::saber::X86>::env_init();
        run_demo();
        return 0;
    }
