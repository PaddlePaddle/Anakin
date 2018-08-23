# Example
Anakin目前只支持NCHW的格式
示例文件在test/framework/net下

## 在NV的GPU上运行CNN模型
整体流程如下：
- 将模型的的path设置为anakin模型的路径，初始化NV平台的图对象。 anakin模型可以通过转换器转化caffe或fluid的模型得到
```cpp
Graph<NV, AK_FLOAT, Precision::FP32> graph;
auto status = graph.load("Resnet50.anakin.bin");
```

- 根据模型设置网络图的输入尺寸，进行图优化
```cpp
graph.Reshape("input_0", {1, 3, 224, 224});
graph.Optimize();
```

- 根据优化后的网络图初始化网络执行器
```cpp
Net<NV, AK_FLOAT, Precision::FP32> net_executer(graph, true);
```

- 取出网络的输入tensor，将数据拷贝到输入tensor，其中copy_from将数据从内存拷贝到显存
```cpp
auto d_tensor_in_p = net_executer.get_in("input_0");
Tensor4d<X86, AK_FLOAT> h_tensor_in;
h_tensor_in.re_alloc(valid_shape_in);
fill_tensor_host_rand(h_tensor_in, -1.0f, 1.0f);
d_tensor_in_p->copy_from(h_tensor_in);
```

- 运行推导
```cpp
net_executer.prediction();
```

- 取出网络的输出tensor,其中copy_from将数据从显存拷贝到内存
```cpp
auto d_out=net_executer.get_out("prob_out");
Tensor4d<X86, AK_FLOAT> h_tensor_out;
h_tensor_out.re_alloc(d_out->valid_shape());
h_tensor_out.copy_from(*d_out);
```

示例文件为[example_nv_cnn_net.cpp](cuda/example_nv_cnn_net.cpp)  
以NV平台为例演示Anakin框架的使用方法，注意编译时需要打开GPU编译开关和example编译开关，也可以将文件复制到`test/framework/net`下直接编译
- - -
## 在X86上运行RNN模型

整体流程与在NV的GPU上运行CNN模型相似，不同之处如下：
- 使用X86标识初始化图对象和网络执行器对象
```cpp
Graph<X86, AK_FLOAT, Precision::FP32> graph;
Net<X86, AK_FLOAT, Precision::FP32> net_executer(graph, true);
```

- rnn模型的输入尺寸是可变的，初始化图时的输入维度是维度的最大值，输入维度N代表总的词的个数。还需要设置输入tensor的seq_offset来标示这些词是如何划分为句子的，如{0,10,15,30}表示共有12个词，其中第0到第9个词是第一句话，第10到第14个词是第二句话，第15到第29个词是第三句话  
```cpp
h_tensor_in_p->set_seq_offset({0,10,15,30});
```

示例文件为[example_x86_rnn_net.cpp](x86/example_x86_rnn_net.cpp)  
以X86平台为例演示Anakin框架的使用方法，注意编译时需要打开X86编译开关和example编译开关，也可以将文件复制到`test/framework/net`下直接编译
- - -
## 在NV的GPU上使用Anakin的线程池运行CNN模型

整体流程与在NV的GPU上运行CNN模型相似，不同之处如下：
- 用模型地址和线程池大小初始化worker对象，注册输入输出，启动线程池  
```cpp
Worker<NV, AK_FLOAT, Precision::FP32>  workers("Resnet50.anakin.bin", 10);
workers.register_inputs({"input_0"});
workers.register_outputs({"prob_out"});
workers.Reshape("input_0", {1, 3, 224, 224});
workers.launch();
```
- 将输入tensor注入任务队列,获得输出tensor  
```cpp
auto d_tensor_p_out_list = workers.sync_prediction(host_tensor_p_in_list);
auto d_tensor_p = d_tensor_p_out_list[0];
```

示例文件为[example_nv_cnn_net_multi_thread.cpp](cuda/example_nv_cnn_net_multi_thread.cpp) 示例使用worker的同步预测接口  

以NV平台为例演示Anakin框架的使用方法，注意编译时需要打开GPU编译开关和example编译开关，也可以将文件复制到`test/framework/net`下直接编译
