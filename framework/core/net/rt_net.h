/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifdef USE_TENSORRT
#ifndef ANAKIN_RTNET_H
#define ANAKIN_RTNET_H

#include "framework/graph/graph.h"
#include "framework/core/net/operator_func.h"
#include "framework/core/net/calibrator_factory.h"
#include "saber/core/tensor_op.h"
#include "third-party/tensorrt5/include/NvInfer.h"

using namespace nvinfer1;

namespace anakin {

using namespace anakin::graph;

typedef std::map<std::string, std::vector<nvinfer1::Weights>> WeightMap;
typedef std::map<std::string, ITensor*> TensorMap;
typedef std::map<std::string, nvinfer1::DimsCHW> TensorDimsMap;

template<typename NV>
class Calibrator;

/** 
 *  \brief Net class used for execution of graph and it is thread safety.
 */
class RTNet {
public:
	typedef std::vector<Arc_iterator<std::string, 
									 Tensor4dPtr<X86>, 
									 Edge<X86> > > ArcsIteratorList;

    RTNet(graph::Graph<X86, Precision::FP32>&, nvinfer1::IInt8Calibrator* calibrator);

    ~RTNet();

public:
    
    /** 
     * \brief do inference.   
     */
    void prediction();

public:

    /**
     *  \brief Get out by name.
     */
    Tensor4dPtr<NV> get_out(std::string out_name);
    std::vector<Tensor4dPtr<NV> > get_out_list();
    
    /**
     *  \brief Get in by name.
     */
    Tensor4dPtr<NV> get_in(std::string in_name);

    std::vector<Tensor4dPtr<NV> > get_in_list();

private:
void addConvLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map);

void addPoolLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map);

void addActiveLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map);

void addSoftmaxLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map);

void addInnerProductLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map);

void addLayer(NodePtr node_ptr,
           ArcsIteratorList& edge_in_its,
           ArcsIteratorList& edge_out_its,
           ITensor*const* inputs,
           int nbInputs,
           INetworkDefinition* net,
           TensorMap& tensor_map,
           TensorDimsMap& tensor_dims_map);


private:
    ///< executor for operators in node.
    //std::vector<OperatorFunc<X86, Precision::FP32> > _exec_funcs;
    ///< The pointer to Context.
    OpContextPtr<NV> _ctx_p;

    graph::Graph<X86, Precision::FP32>* _graph{nullptr};
    ///< Input
    std::vector<std::string> _input_names;
    ///< Output
    std::vector<std::string> _output_names;

    std::map<std::string, int> _input_names_id_map;
    std::map<std::string, int> _output_names_id_map;
   
    ///< A list of in tensor.
    std::vector<Tensor4dPtr<NV> > _input_tensors;
    ///< A list of out tensor.
    std::vector<Tensor4dPtr<NV> > _output_tensors;

    ///< all tensor names 
    std::vector<std::string > _tensor_name_list;
    ///< network definition
    INetworkDefinition* _network;
    ///< create an optimized engine
    IBuilder* _builder;
    ///< engine
    ICudaEngine* _engine;
    IExecutionContext* _context;
    //< inference
    //void doInference(ICudaEngine& engine);
    int _batch_size; 
    int _workspace_size;
    std::vector<void*> _buffers;
    cudaStream_t _stream;
    IInt8Calibrator* _calibrator;
};

}
#endif
#endif

