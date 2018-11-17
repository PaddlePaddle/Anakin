#ifdef USE_TENSORRT
#include "framework/core/net/rt_net.h"
#include <string> 
using namespace nvinfer1;

namespace anakin {

class RTLogger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        if (severity != Severity::kINFO)
           LOG(INFO) << msg;
    }
} rt_gLogger;

class ICaffePoolOutputDimensionsFormula: public IOutputDimensionsFormula
{
public:
    virtual DimsHW compute(DimsHW inputDims, DimsHW kernelSize, DimsHW stride, DimsHW padding, DimsHW dilation, const char* layerName) const {
        const int kernel_extent_h = dilation.d[0] * (kernelSize.d[0] - 1) + 1;
        const int kernel_extent_w = dilation.d[1] * (kernelSize.d[1] - 1) + 1;
        int h = ceil((inputDims.d[0] + 2* padding.d[0] - kernel_extent_h)*1.0 /stride.d[0]) + 1;
        int w = ceil((inputDims.d[1] + 2* padding.d[1] - kernel_extent_w)*1.0 /stride.d[1]) + 1;
      return DimsHW(h, w);
    }

    ICaffePoolOutputDimensionsFormula() {}
    ~ICaffePoolOutputDimensionsFormula() {}
};

//template<typename X86, Precision Precision::FP32, OpRunType RunType>
RTNet::~RTNet() {
    if (_graph) {
        delete _graph;
        _network->destroy();
        _builder->destroy();
        _graph = nullptr;
    }
}

RTNet::RTNet(graph::Graph<X86, Precision::FP32>& graph, nvinfer1::IInt8Calibrator* calibrator) { 
    _builder = nvinfer1::createInferBuilder(rt_gLogger);
    _network = _builder->createNetwork();
    ICaffePoolOutputDimensionsFormula poolFormula;
    _network->setPoolingOutputDimensionsFormula(&poolFormula);
    std::map<std::string, ITensor*> tensor_map;
    std::map<std::string, DimsCHW>  tensor_dims_map;
    std::map<std::string, DimsCHW> _input_dims_map;
    auto node_name_in_exec_order = graph.get_nodes_in_order();

    /*prepare inputs*/
    for(auto input :  graph.get_ins()){
        auto input_dim = graph[input]->template get_attr<PTuple<int>>("input_shape");
        _batch_size = input_dim[0];
        DimsCHW dims = nvinfer1::DimsCHW{input_dim[1], input_dim[2], input_dim[3]};
        _input_dims_map.insert(std::pair<std::string, DimsCHW>(input, dims));
        auto data = _network->addInput(input.c_str(), nvinfer1::DataType::kFLOAT, dims);
        CHECK(data != nullptr) << "rt input is not valid";
        auto node_ptr = graph[input];
        auto edge_out_its = graph.get_out_arc_its(input);
        data->setName(edge_out_its[0]->name().c_str());
        tensor_dims_map.insert(std::pair<std::string, DimsCHW>(input, _input_dims_map[input]));
        tensor_map.insert(std::pair<std::string, ITensor*>(edge_out_its[0]->name().c_str(), data));
        _input_names.push_back(edge_out_its[0]->name().c_str());
    }

    for (auto output :  graph.get_outs()) {
        auto edge_in_its = graph.get_in_arc_its(output);
        _output_names.push_back(edge_in_its[0]->name().c_str());
    }
    /*construct net**/
    for(int i = 0; i < node_name_in_exec_order.size(); i++ ){
        auto node_name = node_name_in_exec_order[i];
        auto node_ptr = graph[node_name];
        auto edge_in_its = graph.get_in_arc_its(node_name);
        auto edge_out_its = graph.get_out_arc_its(node_name);
        auto bottom_size = edge_in_its.size();
        //node_ptr->template get_attr<PTuple<int>>(bottom_size);
        ITensor* inputs[bottom_size];
        for (int j = 0; j < bottom_size; j++) {
            CHECK(tensor_map[edge_in_its[j]->name()] != nullptr) << " " << node_name << "input tensor does not exist";
            inputs[j] = tensor_map[edge_in_its[j]->name()];
        }
        if (node_ptr->get_op_name() == "Input") {
            continue;
        }
        addLayer(node_ptr, edge_in_its, edge_out_its, inputs, bottom_size, _network, tensor_map, tensor_dims_map);
    }

    /*trt output*/

    for (auto& s : _output_names) {
        _network->markOutput(*tensor_map[s]);
    }
    cudaStreamCreate(&_stream);
    _workspace_size = 1<<20;

    _builder->setMaxBatchSize(_batch_size);
    _builder->setMaxWorkspaceSize(_workspace_size);
    _builder->setInt8Mode(calibrator != nullptr);
    _builder->setInt8Calibrator(calibrator);
    _builder->setDebugSync(true);
    bool mode = calibrator != nullptr;
    LOG(INFO)<<"int8 mode"<< mode;

    ICudaEngine * engine = _builder->buildCudaEngine(*_network);
    _context = engine->createExecutionContext();
    _engine = &(_context->getEngine());

    _buffers.resize(_input_names.size() + _output_names.size());
    int num =  _engine->getNbBindings();
    LOG(INFO) << "binging num" << num;
    for (auto input: _input_names) {
        size_t bindingIndex = _engine->getBindingIndex(input.c_str());
        CHECK_LT(bindingIndex, _buffers.size());
        DimsCHW dims = static_cast<DimsCHW&&>(_engine->getBindingDimensions((int)bindingIndex)); 
        int count = dims.c() * dims.h() * dims.w() * _batch_size;
        Shape shape({_batch_size, dims.c(), dims.h(), dims.w()}, Layout_NCHW);
        Tensor<NV>* tensor = new Tensor<NV>(shape);
        _input_tensors.push_back(tensor);
        _buffers[bindingIndex] = tensor->data();
    }

    for (auto output: _output_names) {
        size_t bindingIndex = _engine->getBindingIndex(output.c_str());
        CHECK_LT(bindingIndex, _buffers.size());
        DimsCHW dims = static_cast<DimsCHW&&>(_engine->getBindingDimensions((int)bindingIndex));
        int count = dims.c() * dims.h() * dims.w() * _batch_size;
        Shape shape({_batch_size, dims.c(), dims.h(), dims.w()}, Layout_NCHW);
        Tensor<NV>* tensor = new Tensor<NV>(shape);
        _output_tensors.push_back(tensor);
        _buffers[bindingIndex] = tensor->data();
    }
}

void RTNet::prediction() {
    _context->enqueue(_batch_size, &_buffers[0], _stream, nullptr);
}


Tensor4dPtr<NV> RTNet::get_out(std::string out_name) {
    return _output_tensors[_output_names_id_map[out_name]];
}

std::vector<Tensor4dPtr<NV> > RTNet::get_out_list() {
    return _output_tensors;

}

Tensor4dPtr<NV> RTNet::get_in(std::string in_name) {
    return _input_tensors[_input_names_id_map[in_name]];
}

std::vector<Tensor4dPtr<NV> > RTNet::get_in_list() {
    return _input_tensors;
}

 
void RTNet::addConvLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map) {
    //ConvParam param;
    //parser_conv_param(conv,  param);
    auto num_output = edge_out_its.size();
    auto paddings = node_ptr->template get_attr<PTuple<int>>("padding");
    auto strides = node_ptr->template get_attr<PTuple<int>>("strides");
    auto dilation = node_ptr->template get_attr<PTuple<int>>("dilation_rate");
    auto filter_num = node_ptr->template get_attr<int>("filter_num");
    auto kernel_size = node_ptr->template get_attr<PTuple<int>>("kernel_size");
    auto group = node_ptr->template get_attr<int>("group");
    auto bias_term = node_ptr->template get_attr<bool>("bias_term");

    using pblock_type = PBlock<X86>;
    auto weights = node_ptr->template get_attr<PBlock<X86>>("weight_1");
    Weights filter_weight{nvinfer1::DataType::kFLOAT, weights.d_tensor().data(), weights.d_tensor().valid_size()};
    IConvolutionLayer* convLayer = NULL;
    if (bias_term) {
        auto bias = node_ptr->template get_attr<pblock_type>("weight_2");
        nvinfer1::Weights bias_weight{nvinfer1::DataType::kFLOAT, bias.d_tensor().data(), bias.count()};
        convLayer = net->addConvolution(*inputs[0], filter_num, DimsHW{kernel_size[0], kernel_size[1]}, filter_weight, bias_weight);
    } else {
        nvinfer1::Weights bias_weight{nvinfer1::DataType::kFLOAT, nullptr, 0};
        convLayer = net->addConvolution(*inputs[0], filter_num, DimsHW{kernel_size[0], kernel_size[1]}, filter_weight, bias_weight);
    }
    convLayer->setStride(DimsHW{strides[0], strides[1]});
    convLayer->setPadding(DimsHW{paddings[1], paddings[1]});
    convLayer->setNbGroups(group);
    convLayer->setName(node_ptr->name().c_str());
    convLayer->setDilation(DimsHW{dilation[0], dilation[1]});
    auto top_name = (*edge_out_its[0]).name();
    convLayer->getOutput(0)->setName(top_name.c_str());
    tensor_map.insert(std::pair<std::string, ITensor*>(top_name, convLayer->getOutput(0)));
}

void RTNet::addPoolLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map) {
    //ConvParam param;
    //parser_conv_param(conv,  param);
    auto num_output = edge_out_its.size();
    auto paddings = node_ptr->template get_attr<PTuple<int>>("padding");
    auto strides = node_ptr->template get_attr<PTuple<int>>("strides");
    auto kernel_size = node_ptr->template get_attr<PTuple<int>>("pool_size");
    auto pool_type = node_ptr->template get_attr<std::string>("method");
    auto global_pooling = node_ptr->template get_attr<bool>("global_pooling");

    IPoolingLayer* poolLayer = NULL;
    nvinfer1::PoolingType pooling_type;
    if (pool_type == "AVG") {
        pooling_type = nvinfer1::PoolingType::kAVERAGE;
    } else if (pool_type == "MAX")
        pooling_type = nvinfer1::PoolingType::kMAX;
    else {
        LOG(FATAL) << "pooling type is not valid";
    }
    poolLayer = net->addPooling(*inputs[0], pooling_type, DimsHW{kernel_size[0], kernel_size[1]});
    poolLayer->setStride(DimsHW{strides[0], strides[1]});
    poolLayer->setPadding(DimsHW{paddings[1], paddings[1]});
    poolLayer->setName(node_ptr->name().c_str());
    auto top_name = (*edge_out_its[0]).name();
    poolLayer->getOutput(0)->setName(top_name.c_str());
    tensor_map.insert(std::pair<std::string, ITensor*>(top_name, poolLayer->getOutput(0)));
}

void RTNet::addActiveLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map) {
    //nvinfer1::ActivationType type = nvinfer1::ActivationType::kSIGMOID;
    //auto ak_type = node_ptr->template get_attr<std::string>("type");
    //if (ak_type == "Sigmoid") {
    //    type = nvinfer1::ActivationType::kSIGMOID;
    //} else if (ak_type == "TanH") {
    //    type = nvinfer1::ActivationType::kTANH;
    //} else if (ak_type == "ReLU") {
    //    type = nvinfer1::ActivationType::kRELU;
    //} else {
    //    LOG(FATAL) << "unknown type";
    //}
    IActivationLayer* layer = net->addActivation(*inputs[0], ActivationType::kRELU);
    layer->setName(node_ptr->name().c_str());
    auto top_name = (*edge_out_its[0]).name();
    layer->getOutput(0)->setName(top_name.c_str());
    tensor_map.insert(std::pair<std::string, ITensor*>(top_name, layer->getOutput(0)));
}

void RTNet::addSoftmaxLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map) {
    ISoftMaxLayer* layer = net->addSoftMax(*inputs[0]);
    layer->setName(node_ptr->name().c_str());
    auto top_name = (*edge_out_its[0]).name();
    layer->getOutput(0)->setName(top_name.c_str());
    tensor_map.insert(std::pair<std::string, ITensor*>(top_name, layer->getOutput(0)));
}


void RTNet::addInnerProductLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map) {
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto axis = node_ptr->template get_attr<int>( "axis");
    auto out_dim = node_ptr->template get_attr<int>( "out_dim");
    auto bias_term = node_ptr->template get_attr<bool>( "bias_term");
    using pblock_type = PBlock<X86>;
    auto ak_weights = node_ptr->template get_attr<pblock_type>("weight_1");
    nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, ak_weights.d_tensor().data(), ak_weights.count()};
    
    IFullyConnectedLayer* layer = net->addFullyConnected(*inputs[0], out_dim, weights, bias);
    layer->setName(node_ptr->name().c_str());
    if (bias_term) {
        auto ak_bias = node_ptr->template get_attr<pblock_type>("weight_2");
        nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, ak_bias.d_tensor().data(), ak_bias.count()};
        layer->setBiasWeights(bias);
    }
    auto top_name = (*edge_out_its[0]).name();
    layer->getOutput(0)->setName(top_name.c_str());
    tensor_map.insert(std::pair<std::string, ITensor*>(top_name, layer->getOutput(0)));

}

void RTNet::addLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor* const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map) {
    if (node_ptr->get_op_name() == "Convolution") {
        addConvLayer(node_ptr, edge_in_its, edge_out_its, inputs, nbInputs, net, tensor_map, tensor_dims_map);
    } else if (node_ptr->get_op_name() == "Pooling") {
        addPoolLayer(node_ptr, edge_in_its, edge_out_its, inputs, nbInputs, net, tensor_map, tensor_dims_map);
    } else if (node_ptr->get_op_name() == "Activation" || node_ptr->get_op_name() == "ReLU") {
        addActiveLayer(node_ptr, edge_in_its, edge_out_its, inputs, nbInputs, net, tensor_map, tensor_dims_map);
    } else if (node_ptr->get_op_name() == "Softmax") {
        addSoftmaxLayer(node_ptr, edge_in_its, edge_out_its, inputs, nbInputs, net, tensor_map, tensor_dims_map);
    } else if (node_ptr->get_op_name() == "Dense") {
        addInnerProductLayer(node_ptr, edge_in_its, edge_out_its, inputs, nbInputs, net, tensor_map, tensor_dims_map);
    } else if (node_ptr->get_op_name() == "Input" || node_ptr->get_op_name() == "Output"){
    } else {
        std::cout << "unknown layer type:" << node_ptr->get_op_name() << std::endl;
    }
}

}
#endif
 /* namespace anakin_rt */
