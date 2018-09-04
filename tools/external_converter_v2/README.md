# External Converter
---
## Introduction

## Requirements
---
```bash
	0. python version 2.7+
	1. pyyaml
	2. flask
```

## User Manual
---

### 1. Configuration

#### Caffe Case
```bash
OPTIONS:
    Framework: CAFFE # select a target dl-framework you want parsing
    SavePath: ./output
    ResultName: googlenet # the name you want when saving the parsed model
    Config:
        LaunchBoard: ON  # should be on if you want to launch graph board
        Server:
            ip: 0.0.0.0
            port: 8888
        OptimizedGraph:  # only enable(set enable(ON) and path) when you have optimized graph model.
            enable: ON
            path: /path/to/anakin_optimized_anakin_model/googlenet.anakin.bin.saved
    LOGGER:
        LogToPath: ./log/ # the path where log
        WithColor: ON  # colorful log message

TARGET:
    CAFFE:
        # path to proto files
        ProtoPaths:
            - /path/to/caffe/src/caffe/proto/caffe.proto
        PrototxtPath: /path/to/your/googlenet.prototxt
        ModelPath: /path/to/your/googlenet.caffemodel
   
	# not support yet 
    PADDLE:
        # path to proto files   
        ProtoPath:
            - /path/to/proto_0
            - /path/to/proto_1
            - /path/to/proto_n
        PrototxtPath: /path/to/prototxt
        ModelPath: /path/to/model
	# ... 
```

>**Reference to config.yaml**

### 2. How To Run 
After configuring the converter , you just need to call python script ```python converter.py```  to complete transfromation.

#### launch the dash board
Anakin external converter will be launched on site http://0.0.0.0:8888 (configurable).
Then open you browser and search http://0.0.0.0:8888, amazing things will happen!

> if you set ip to 0.0.0.0 in remote server, you need to open local browser and search the server real ip:port, not the 0.0.0.0.

#### launch nothing

### 4. Note

> 1.We support caffe so far


