# 模型转换指南

Anakin 支持不同框架的模型预测。但由于格式的差别，Anakin 需要您预先转换模型。本文档介绍如何转换模型。

## 简介

Anakin 模型转换器输入支持 Caffe、Fluid、Tensorflow、ONNX 和 Lego 多种格式的预测模型，模型包含网络结构（model 或 prototxt）和权重参数（param 或 caffemodel）。

模型转换的输出是一个 anakin.bin 文件，它作为 Anakin 框架的 graph 参数导入。

您还可以使用模型转换器的 launch board 功能生成网络结构的 HTML 预览。


## 系统要求

- Python 2.7+
- Protobuf 3.1+（务必注意 Python 与系统环境 Protobuf 版本一致）
- PaddlePaddle 0.12.0+ (Fluid 模式下)
- Tensorflow 1.0.0+（Tensorflow模式下）
- ONNX 1.2.0+ (ONNX 模式下)
- flask, bson, matplotlib, scikit-image
- tkinter


## 用法

### 1、环境
转换器所需的依赖标注于 *系统要求* 一节。

### 2、配置
您需要对 *config.yaml* 文件进行修改以告知您的需求。工程中给出了 *config.yaml* 示例，下面作进一步说明。

#### config.yaml
```bash
OPTIONS:
    Framework: CAFFE       # 依框架类型填写 CAFFE、FLUID、LEGO、TENSORFLOW、ONNX
    SavePath: ./output     # 转换结束后模型的保存位置
    ResultName: googlenet  # 输出模型的名字
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
        Remark:  #一般不用填写，如果是Training模式，需要填写

    FLUID:
        # 当 Framework 为 FLUID 时需填写
        Debug: NULL                                # 不需要更改
        ModelPath: /path/to/fluid/inference_model  # 此路径通常包括 model 和 params 两个文件
        NetType:                                   # 填写网络类型，如 OCR、SSD
    # ...
    LEGO:
        # 当 Framework 为 LEGO 时需填写
        ProtoPath:
        PrototxtPath:
        ModelPath:

    TENSORFLOW:
        # 当 Framework 为 TENSORFLOW 时需填写
        ModelPath: /path/to/your/model/
        OutPuts:

    ONNX:
        # 当 Framework 为 ONNX 时需填写
        ModelPath: /path/to/your/model/
        TxtPath:
```

### 3、转换
在完成配置文件的修改后，您只需执行 ```python converter.py``` 就可以进行模型转换了。
还有一种方法，不用修改配置文件，通过传参数方式，直接运行```converter.py``` 就可以，具体方法请见 *命令行方式* 示例

#### 命令行方式
介绍每个参数的含义
+ 通用参数的含义
    - `--framework` 待转换模型的原框架, 例如: Caffe、Fluid、Tensorflow、ONNX 和 Lego
    - `--save_path` 保存转换后模型的路径，默认是 ./output
    - `--result_name` 转换后模型的name，默认是 googlenet
    - `--open_launch_board` 是否将生成的模型在网页上打印出来，默认是ON打印网络结构
    - `--board_server_ip` 显示网络结构的IP地址，默认是0.0.0.0 即本地IP的地址
    - `--board_server_port` 显示网络结构的端口地址，默认是8888
    - `--optimized_graph_enable` 是否显示优化后的网络图，默认是OFF关闭
    - `--optimized_graph_path` 保持优化后网络的路径
+ CAFFE框架
    - `--caffe_proto_paths` caffe.proto的路径
    - `--caffe_proto_txt_path` prototxt的路径
    - `--caffe_model_path` model的路径
    - `caffe_remark` 是否是Training模式，如果是，需要备注；否则，可以不填

    example：
    ```bash
    1) $ python converter.py --framework CAFFE --result_name mobilenet_ssd  --caffe_proto_paths    ./model/caffe.proto  --caffe_proto_txt_path  ./model/mobilenet_ssd.prototxt --caffe_model_path   ./model/mobilenet_ssd.caffemodel -- caffe_remark Training
    2) $ python converter.py --framework CAFFE --result_name mobilenet_ssd  --caffe_proto_paths    ./model/caffe.proto  --caffe_proto_txt_path  ./model/mobilenet_ssd.prototxt --caffe_model_path   ./model/mobilenet_ssd.caffemodel
    ```
+ FLUID框架
    - `--fluid_debug` 不需要填写，默认NULL
    - `--fluid_model_path` 包括 model 和 params 两个文件的路径
    - `fluid_net_type` 填写网络类型，如 OCR、SSD

    example：
    ```bash
    1) $ python converter.py --framework FLUID --result_name mobilenet_ssd --fluid_model_path  ./model/mobilenet_ssd/ --fluid_net_type  SSD
    2) $ python converter.py --framework FLUID --result_name mobilenet_v1 --fluid_model_path  ./model/mobilenet_v1/
    ```
+ TENSORFLOW框架
    - `--tensorflow_model_path` 模型路径

    example：
    ```bash
    1) $ python converter.py --framework TENSORFLOW --result_name mobilenet_ssd --tensorflow_model_path  ./model/mobilenet_ssd.pb
    ```
+ ONNX框架
    - `--onnx_model_path` 模型路径

    example：
    ```bash
    1) $ python converter.py --framework TENSORFLOW --result_name mobilenet_ssd --onnx_model_path  ./model/mobilenet_ssd.onnx
    ```
+ LEGO框架
    - `--lego_proto_paths` lego.proto的路径
    - `--lego_proto_txt_path` prototxt的路径
    - `--lego_model_path` model的路径

    example：
    ```bash
    1) $ python converter.py --framework LEGO --result_name mobilenet_ssd  --lego_proto_paths    ./model/lego.proto  --lego_proto_txt_path  ./model/mobilenet_ssd.prototxt --lego_model_path   ./model/mobilenet_ssd.model
    ```

### 4、预览
最后一步，就是在浏览器中查看令人振奋的转换结果！网址是在 *config.yaml* 中配置的，例如 http://0.0.0.0:8888 。

> 注意：若您使用了默认的 IP 地址 0.0.0.0，请在预览时使用真实的服务器地址 real_ip:port 替代它。
