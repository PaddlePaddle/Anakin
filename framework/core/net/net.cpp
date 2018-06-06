#include "framework/core/net/net.h"
#include "saber/funcs/timer.h"
namespace anakin {

template<typename Ttype, DataType Dtype>
double tensor_average(Tensor4dPtr<Ttype, Dtype>& out_tensor_p) {
    double sum = 0.0f;
    Tensor4d<X86, AK_FLOAT> h_tensor_result;
    float* h_data = new float[out_tensor_p->valid_size()];
    const float* d_data = out_tensor_p->data();
    CUDA_CHECK(cudaMemcpy(h_data, d_data, out_tensor_p->valid_size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i=0; i<out_tensor_p->valid_size(); i++) {
	sum+=h_data[i];
    }
    return sum/out_tensor_p->valid_size();
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Net<Ttype, Dtype, Ptype, RunType>::Net(bool need_summary) {
    _graph_p = new graph::Graph<Ttype, Dtype, Ptype>();
    _need_summary = need_summary;
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Net<Ttype, Dtype, Ptype, RunType>::Net(graph::Graph<Ttype, Dtype, Ptype>& graph, bool need_summary) {
    _graph_p = new graph::Graph<Ttype, Dtype, Ptype>();
    _need_summary = need_summary;
    init_env(graph);
    init(graph);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Dtype, Ptype, RunType>::init(graph::Graph<Ttype, Dtype, Ptype>& graph) {
    init_env(graph);
    // shallow copy
    _graph_p->CopyFrom(graph);
     
    auto node_names_in_exec_order = graph.get_nodes_in_order();
    // infer basic shape and parsing parameter from graph
    for (auto& node_name : node_names_in_exec_order) {
        auto& node_ptr = (*_graph_p)[node_name];
        if (node_ptr->get_op_name() == "Output") {
            continue;
        }
#ifdef ENABLE_OP_TIMER
        if (std::string::npos != (node_ptr->get_op_name()).find("Conv") 
                || std::string::npos != (node_ptr->get_op_name()).find("Deconv")) {
            std::string key = "kernel_size";
            std::string strides = "strides";
            std::string group = "group";
            std::string dilation_rate = "dilation_rate";
            std::string padding = "padding";
            auto kernel_size = node_ptr->template get_attr<PTuple<int>>(key);
            auto stride_size = node_ptr->template get_attr<PTuple<int>>(strides);
            auto group_val =  node_ptr->template get_attr<int>(group);
            auto dilation = node_ptr->template get_attr<PTuple<int>>(dilation_rate);
            auto padding_size = node_ptr->template get_attr<PTuple<int>>(padding);
            char buf[100];
            if (std::string::npos != (node_ptr->get_op_name()).find("Conv")) {
                sprintf(buf, "%s_%d*%d_%d*%d_%d*%d_%d*%d_%d", "Conv", kernel_size[0], kernel_size[1], padding_size[0], padding_size[1], stride_size[0], stride_size[1], dilation[0], dilation[1], group_val);
            } else {
                sprintf(buf, "%s_%d*%d_%d*%d_%d*%d_%d*%d_%d", "Deconv", kernel_size[0], kernel_size[1], padding_size[0], padding_size[1], stride_size[0], stride_size[1], dilation[0], dilation[1], group_val);
            }
            _op_param.push_back(buf);
        } else if (std::string::npos != (node_ptr->get_op_name()).find("Pooling")){
            std::string key = "pool_size";
            std::string strides = "strides";
            std::string padding = "padding";
            auto kernel_size = node_ptr->template get_attr<PTuple<int>>(key);
            auto stride_size = node_ptr->template get_attr<PTuple<int>>(strides);
            auto padding_size = node_ptr->template get_attr<PTuple<int>>(padding);
            char buf[100];
            sprintf(buf, "%s_%d*%d_%d*%d_%d*%d", "Pooling", kernel_size[0], kernel_size[1], padding_size[0], padding_size[1], stride_size[0], stride_size[1]);
            _op_param.push_back(buf);
        } else {
            _op_param.push_back(node_ptr->get_op_name());   
        }
#endif
        // create operations
#if 1
        if (node_ptr->get_op_name() == "ConvReluPool"|| node_ptr->get_op_name() == "ConvBatchnormScaleRelu" || node_ptr->get_op_name() == "ConvBatchnormScaleReluPool" || node_ptr->get_op_name() == "ConvRelu" || node_ptr->get_op_name() == "Convolution") {
        std::string key = "kernel_size";
        std::string strides = "strides";
        std::string group = "group";
        std::string dilation_rate = "dilation_rate";
        auto kernel_size = node_ptr->template get_attr<PTuple<int>>(key);
        auto stride_size = node_ptr->template get_attr<PTuple<int>>(strides);
        auto group_val =  node_ptr->template get_attr<int>(group);
        auto dilation = node_ptr->template get_attr<PTuple<int>>(dilation_rate);
        //if (dilation[0] == 1 && dilation[1] == 1 && kernel_size[0] == 3 && kernel_size[1] == 3 && stride_size[0] == 1 && stride_size[1] == 1 /*&& count > 3*/ && group_val == 1) {
        if (group_val == 1) {
            node_ptr->set_op(OpFactory<Ttype, Dtype, Ptype>::Global()["Sass"+node_ptr->get_op_name()]);
            node_ptr->get_op_name() = "Sass" + node_ptr->get_op_name();
        } else {
            LOG(ERROR) << "node_ptr->get_op_name()  sass not support yet.";
            auto* op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()];
            node_ptr->set_op(op_pointer);
        }
        } else {
            auto* op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()];
            node_ptr->set_op(op_pointer);
        }
#else
        auto* op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()];
        node_ptr->set_op(op_pointer);
#endif
        // bind parameter structure
        static_cast<Operator<Ttype, Dtype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
        // parsing parameter
        static_cast<Operator<Ttype, Dtype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
    }

    // remove null op node
    for (auto it = node_names_in_exec_order.begin(); it != node_names_in_exec_order.end(); ){
        if (!(*_graph_p)[*it]->Op()) {
            it = node_names_in_exec_order.erase(it);
        } else {
            ++it;
        }
    }
    _exec_funcs.resize(node_names_in_exec_order.size());
    for(int i = 0; i < node_names_in_exec_order.size(); i++) {
        auto& node_name = node_names_in_exec_order[i];
        auto& op_func = _exec_funcs[i];
        op_func.name = node_name;
        auto& edge_in_its = _graph_p->get_in_arc_its(node_name);
        DLOG(ERROR) << " node : " << op_func.name << " (" << (*_graph_p)[node_name]->get_op_name() << ") ";
        for(auto& edge_it : edge_in_its) { 
            DLOG(INFO) << "  => find in arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
            op_func.ins.push_back(edge_it->weight().get()); 
            op_func.in_lanes.push_back(edge_it->lane());
        }
        auto& edge_out_its = _graph_p->get_out_arc_its(node_name);
        for(auto& edge_it : edge_out_its) { 
            DLOG(INFO) << "  <= find out arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
            op_func.outs.push_back(edge_it->weight().get()); 
            op_func.out_lanes.push_back(edge_it->lane());
        }
        op_func.current_lane = (*_graph_p)[node_name]->lane();
        op_func.need_sync = (*_graph_p)[node_name]->need_wait();
        op_func.op = static_cast<Operator<Ttype, Dtype, Ptype>* >((*_graph_p)[node_name]->Op());
        op_func.op_name = (*_graph_p)[node_name]->get_op_name();
        op_func.ctx_p = std::make_shared<Context<Ttype>>(TargetWrapper<Ttype>::get_device_id(), 
                                                         op_func.current_lane, 
                                                         op_func.current_lane);
        // call init of operator
        CHECK_NOTNULL_S(op_func.op) << "Node(node_name) doesn't have op pointer! ";

        op_func.op->_helper->InferShape(op_func.ins, op_func.outs);

#ifdef ENABLE_DEBUG
        for(auto& in : op_func.ins) {
                LOG(INFO) << "  => [shape]: " << in->valid_shape()[0] 
                          << " " << in->valid_shape()[1] 
                          << " " << in->valid_shape()[2] 
                          << " " << in->valid_shape()[3];
        }
        for(auto& out : op_func.outs) {
                LOG(INFO) << "  <= [shape]: " << out->valid_shape()[0] 
                          << " " << out->valid_shape()[1] 
                          << " " << out->valid_shape()[2] 
                          << " " << out->valid_shape()[3];
        }
#endif
        op_func.op->_helper->Init(*(op_func.ctx_p), op_func.ins, op_func.outs);
    }
    
    // init memory of _graph_p
    init_memory();
#ifdef ENABLE_OP_TIMER
    _op_time = std::vector<float>(_exec_funcs.size(), 0.0f);
#endif

#ifdef ENABLE_DEBUG
    LOG(ERROR) << "Checking memroy...";
    for(auto& executer : _exec_funcs) {
        if (executer.need_sync) {
            for(int i = 0; i < executer.ins.size(); i++) {
                // record
                executer.ins[i]->record_event(executer.ctx_p->get_compute_stream());
                //Env<TargetType>::cur_env()[]
                executer.ins[i]->sync();
            }
        }
        LOG(WARNING) << " Inspect memory of " << executer.name << " (" << executer.op_name << ") ";
        executer.infer_shape();
    
	    for (auto out : executer.outs) {
	        LOG(INFO) << "    |-- out tensor avg " << tensor_average(out);
	    }
	    CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());
    }
#endif
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Dtype, Ptype, RunType>::prediction() {
#ifdef ENABLE_OP_TIMER
    int op_id = 0;
#endif
    for(auto& executer : _exec_funcs) {
        if (RunType == OpRunType::SYNC || executer.need_sync) {
            for(int i = 0; i < executer.ins.size(); i++) {
                // record
                executer.ins[i]->record_event(executer.ctx_p->get_compute_stream());
                //Env<TargetType>::cur_env()[]
                executer.ins[i]->sync();
            }
        }
#ifdef ENABLE_DEBUG
        LOG(ERROR) << " executer : " << executer.name << " (" << executer.op_name << ") ";
        for(auto in : executer.ins) {
                LOG(ERROR) << "    \\in shape " << in->valid_shape()[0] 
                           << " " << in->valid_shape()[1] 
                           << " " << in->valid_shape()[2] 
                           << " " << in->valid_shape()[3] 
			   << " valid_size: " << in->valid_size() 
			   << " realsize: " << in->size()
                << " offset_size "<<in->get_seq_offset().size();
        }
#endif
#ifdef ENABLE_OP_TIMER   
	Context<NV> ctx(0, 0, 0);
	saber::SaberTimer<NV> my_time;
	my_time.start(ctx);
#endif
      executer.infer_shape();
      executer.launch();

      for(int i = 0; i < executer.outs.size(); i++) {
          executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
      }
#ifdef ENABLE_OP_TIMER
    for (int i = 0; i < executer.outs.size(); i++) {
        // record
        executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
        //Env<TargetType>::cur_env()[]
        executer.outs[i]->sync();
    }
	my_time.end(ctx);
    _op_time[op_id++] += my_time.get_average_ms();
#endif
	//LOG(INFO)<< "op: " << executer.name<<"(" << executer.op_name <<")  ===  infer+launch time "<<my_time.get_average_ms() << " ms";
#ifdef ENABLE_DEBUG	
	cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
	for (auto out : executer.outs) {
	    LOG(ERROR) << "    |---out avg " << tensor_average(out);
	}
	cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
#endif
    }
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype, Dtype> Net<Ttype, Dtype, Ptype, RunType>::get_out(std::string out_name) {
    auto& edge_it_list = _graph_p->get_in_arc_its(out_name);
    CHECK_EQ(edge_it_list.size(), 1) << " Node(" << out_name << ") should have 1 in edge.";
    return edge_it_list[0]->weight().get();
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype, Dtype> > Net<Ttype, Dtype, Ptype, RunType>::get_out_list() {
    auto& out_list_vec = _graph_p->get_outs();
    for (auto& out : out_list_vec) {
        _out_tensor_list.push_back(get_out(out.c_str()));
    }
    return _out_tensor_list;
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype, Dtype> Net<Ttype, Dtype, Ptype, RunType>::get_in(std::string in_name) {
    auto& edge_it_list = _graph_p->get_out_arc_its(in_name);
    CHECK_EQ(edge_it_list.size(), 1) << " Node(" << in_name << ") should have 1 out edge.";
    return edge_it_list[0]->weight().get();
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype, Dtype> > Net<Ttype, Dtype, Ptype, RunType>::get_in_list() {
    auto& in_list_vec = _graph_p->get_ins();
    for (auto& in : in_list_vec) {
        _in_tensor_list.push_back(get_in(in.c_str()));
    }
    return _in_tensor_list;
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype, Dtype> Net<Ttype, Dtype, Ptype, RunType>::get_tensor_from_edge(const char* from, const char* to) {
    return _graph_p->get_arc(std::string(from), std::string(to)).weight().get();
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Status Net<Ttype, Dtype, Ptype, RunType>::init_memory() {
    auto alloc_memory = [this](graph::Edge<Ttype, Dtype>& edge) {
        auto& tensor_p = edge.weight();
        if(!edge.shared()) {
            tensor_p->re_alloc(tensor_p->shape());
        }
        return 0;
    };
    _graph_p->Scanner->BFS_Edge(alloc_memory);

    auto share_memory = [this](graph::Edge<Ttype, Dtype>& edge) {
        if(edge.shared()) {
            auto& edge_name = edge.share_from();
	    bool continue_search = true;
	    while(continue_search) {
            	auto match_edge = [&](graph::Edge<Ttype, Dtype>& inner_edge) {
            	    if(inner_edge.name() == edge_name) {
	    	        if(inner_edge.shared()) {
	    	    	    edge_name = inner_edge.share_from();
	    	    	    return Status::EXIT(" Continue to find next . ");
	    	        }
            	        if (inner_edge.weight()->size() < edge.weight()->valid_size()) {
            	            auto inner_original_shape = inner_edge.weight()->valid_shape();
            	            inner_edge.weight()->re_alloc(edge.weight()->valid_shape());
            	            inner_edge.weight()->set_shape(inner_original_shape, inner_edge.weight()->shape());
            	        }
            	        edge.weight()->share_from(*(inner_edge.weight()));
	    	            continue_search = false;
            	        return Status::EXIT(" Find the matched target edge. ");
            	    }
            	    return Status::OK();
            	};
            	this->_graph_p->Scanner->BFS_Edge(match_edge);
	    }
        }
    };
    _graph_p->Scanner->BFS_Edge(share_memory);

    if (_need_summary) {
        size_t temp_mem_in_mbytes = 0;
        size_t ori_temp_mem_in_mbytes = 0;
        auto analysis_used_of_temp_mem = [&](graph::Edge<Ttype, Dtype>& edge) {
            auto& tensor_p = edge.weight();
            if (!edge.shared()) {
                temp_mem_in_mbytes += (tensor_p->size() * 4);
            }
            ori_temp_mem_in_mbytes += (tensor_p->valid_shape().count() * 4);
        };
        this->_graph_p->Scanner->BFS_Edge(analysis_used_of_temp_mem);
        LOG(ERROR) << " temp !!!!!! " << temp_mem_in_mbytes / 1e6 << "  mb ";
        LOG(ERROR) << " origin temp !!!!!! " << ori_temp_mem_in_mbytes / 1e6 << "  mb ";
    }
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Status Net<Ttype, Dtype, Ptype, RunType>::init_env(graph::Graph<Ttype, Dtype, Ptype>& graph) {
    LOG(WARNING) << "Detect and initial " << graph.get_ins().size() << " lanes.";
    Env<NV>::env_init(graph.get_ins().size()); 
    LOG(WARNING) << "Current used device id : " << TargetWrapper<Ttype>::get_device_id();
    return Status::OK();
}


template class Net<NV, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Net<NV, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Net<NV, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;

template class Net<ARM, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Net<ARM, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Net<ARM, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;

template class Net<NV, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
template class Net<NV, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
template class Net<NV, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;

template class Net<ARM, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
template class Net<ARM, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
template class Net<ARM, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;

} /* namespace anakin */

