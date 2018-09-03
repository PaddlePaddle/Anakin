# External Converter

This guide will show you how to convert your models to Anakin models.

## Introduction

Before using Anakin, you must convert your models to Anakin ones. If you don't, Anakin won't work properly.

## Requirements

- python 2.7+
- pyyaml
- flask

## Downloading Converter Source

```bash
git clone https://xxxxxxxxx
```

## Usage

### 1. Configuration
Configure your *config.yaml* file. Find example *config.yaml* file in the `converter source` directory. The example below explains how to configure your config.yaml file.
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

    FLUID:
        # path to proto files   
        ProtoPath:
            - /path/to/proto_0
            - /path/to/proto_1
            - /path/to/proto_n
        PrototxtPath: /path/to/prototxt
        ModelPath: /path/to/model
	# ...
```

### 2. Converting
After finishing configuration , you just need to call python script ```python converter.py```  to complete transfromation.

### 3. Launching dash board
Anakin external converter will be launched on site http://0.0.0.0:8888 (configurable).
Then open you browser and search http://0.0.0.0:8888, amazing things will happen!

> if you set ip to 0.0.0.0 in remote server, you need to open local browser and search the server real ip:port, not the 0.0.0.0.


### 4. Note

> 1.We support caffe so far
