# 模型转换指南

paddle模型转换inference模型指南。

## 简介

该转换器支持将 Fluid 预测模型转为专有预测模型，以提升预测性能。   

模型转换的输出是一个 bin 文件，它作为 paddle预测 框架的 graph 参数导入。   

您还可以使用模型转换器的 launch board 功能生成网络结构的 HTML 预览。   


## 系统要求

- Python 2.7+
- Protobuf 3.1+（务必注意 Python 与系统环境 Protobuf 版本一致）
- PaddlePaddle 0.12.0+ (Fluid 模式下)
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
    Framework: FLUID
    SavePath: ./output
    ResultName: googlenet
    Config:
        LaunchBoard: ON
        Server:
            ip: 0.0.0.0
            port: 8888
        OptimizedGraph:
            enable: OFF
            path: /path/to/paddle_inference_model_optimized/googlenet.paddle_inference_model.bin.saved
    LOGGER:
        LogToPath: ./log/
        WithColor: ON 

TARGET:
    FLUID:
        # path of fluid inference model
        Debug: NULL                            # Generally no need to modify.
        ModelPath: /path/to/your/model/        # The upper path of a fluid inference model.
        NetType:                               # Generally no need to modify.
```

### 3、转换
在完成配置文件的修改后，您只需执行 ```python converter.py``` 就可以进行模型转换了。


### 4、预览
最后一步，就是在浏览器中查看令人振奋的转换结果！网址是在 *config.yaml* 中配置的，例如 http://0.0.0.0:8888 。

> 注意：若您使用了默认的 IP 地址 0.0.0.0，请在预览时使用真实的服务器地址 real_ip:port 替代它。
