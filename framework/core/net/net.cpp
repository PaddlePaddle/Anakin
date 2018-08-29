#include "framework/core/net/net.h"
#include "saber/funcs/timer.h"
#include "framework/core/mem_info.h"

namespace anakin {

template<typename Ttype, Precision Ptype, OpRunType RunType>
Net<Ttype, Ptype, RunType>::~Net() {
	if(_graph_p) {
		delete _graph_p;
		_graph_p = nullptr;
	}
}

template<typename Ttype>
double tensor_average(Tensor4dPtr<Ttype>& out_tensor_p) {
    double sum = 0.0f;
    Shape shin = out_tensor_p->valid_shape();
    PBlock<Ttype> tensorptr();
    tensorptr.h_tensor().copy_from(*out_tensor_p);
    auto* h_data_p = tensorptr.h_tensor().data();
    for (int i=0; i<out_tensor_p->valid_size(); i++) {
		sum += h_data_p[i];
    }
    return sum/out_tensor_p->valid_size();
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Net<Ttype, Ptype, RunType>::Net(bool need_summary) {
    _graph_p = new graph::Graph<Ttype, Ptype>();
    _need_summary = need_summary;
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Net<Ttype, Ptype, RunType>::Net(graph::Graph<Ttype, Ptype>& graph, bool need_summary) {
    _graph_p = new graph::Graph<Ttype, Ptype>();
    _need_summary = need_summary;
    //init_env(graph);
    init(graph);
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Net<Ttype, Ptype, RunType>::Net(\
    graph::Graph<Ttype, Ptype>& graph, OpContextPtr<Ttype> ctx, bool need_summary) {
    _graph_p = new graph::Graph<Ttype, Ptype>();
    _need_summary = need_summary;
    //init_env(graph);
    init(graph, ctx);
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::init(graph::Graph<Ttype, Ptype>& graph, \
    OpContextPtr<Ttype> ctx) {

    init_env(graph);
    // shallow copy
    _graph_p->CopyFrom(graph);
    auto node_names_in_exec_order = graph.get_nodes_in_order();
    // infer basic shape and parsing parameter from graph
    for (auto& node_name : node_names_in_exec_order) {
        auto node_ptr = (*_graph_p)[node_name];
        //LOG(ERROR) << "get node " << node_name << ", op type " << node_ptr->get_op_name();
        if (node_ptr->get_op_name() == "Output") {
            continue;
        }

        // create operations
        auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
        if (op_pointer == nullptr) {
            LOG(FATAL) << node_name << ", type " << node_ptr->get_op_name() << " is null";
        }
        node_ptr->set_op(op_pointer);
        //LOG(ERROR) << "set op";
        op_pointer = nullptr;

        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
        //LOG(ERROR) << "bind param";
        // parsing parameter
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
        //LOG(ERROR) << "init param";
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
        op_func.op = static_cast<Operator<Ttype, Ptype>* >((*_graph_p)[node_name]->Op());
        op_func.op_name = (*_graph_p)[node_name]->get_op_name();
        op_func.ctx_p = ctx;
        // call init of operator
        CHECK_NOTNULL(op_func.op) << "Node(node_name) doesn't have op pointer! ";

        op_func.op->_helper->InferShape(op_func.ins, op_func.outs);
        op_func.op->_helper->Init(*(op_func.ctx_p), op_func.ins, op_func.outs);
    }

    // init memory of _graph_p
    init_memory();
}


template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::init(graph::Graph<Ttype, Ptype>& graph) {
    init_env(graph);
    // shallow copy
    _graph_p->CopyFrom(graph);
	
	double curr_mem_in_mb_start = MemoryInfo<Ttype>::Global().get_used_mem_in_mb(); 

    auto node_names_in_exec_order = graph.get_nodes_in_order();
    // infer basic shape and parsing parameter from graph
    for (auto& node_name : node_names_in_exec_order) {
        auto node_ptr = (*_graph_p)[node_name];
        //LOG(ERROR) << "get node " << node_name << ", op type " << node_ptr->get_op_name();
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
#if defined(USE_CUDA)
       	if (node_ptr->get_op_name() == "ConvBatchnormScale" || node_ptr->get_op_name() == "ConvBatchnormScaleRelu" || node_ptr->get_op_name() == "ConvRelu" || node_ptr->get_op_name() == "Convolution") {
        	std::string group = "group";
        	auto group_val =  node_ptr->template get_attr<int>(group);
			using pblock_type = PBlock<Ttype>;
			std::string weight_name = "weight_1";
			auto weights = node_ptr->template get_attr<pblock_type>(weight_name);
			//int c = weights.d_tensor().channel();
			
        	if ((group_val == 1)) {
            	node_ptr->set_op(OpFactory<Ttype, Ptype>::Global()["Sass"+node_ptr->get_op_name()]);
            	node_ptr->get_op_name() = "Sass" + node_ptr->get_op_name();
        	} else {
            	LOG(ERROR) << "node_ptr->get_op_name()  sass not support yet.";
            	auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
            	node_ptr->set_op(op_pointer);
        	}
        } else {
            auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
            node_ptr->set_op(op_pointer);
        }
#else
        auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
        if (op_pointer == nullptr) {
            LOG(FATAL) << node_name << ", type " << node_ptr->get_op_name() << " is null";
        }
        node_ptr->set_op(op_pointer);
        //LOG(ERROR) << "set op";
		op_pointer = nullptr;
#endif
        // bind parameter structure
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
        // parsing parameter
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
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
        op_func.op = static_cast<Operator<Ttype, Ptype>* >((*_graph_p)[node_name]->Op());
        op_func.op_name = (*_graph_p)[node_name]->get_op_name();
        op_func.ctx_p = std::make_shared<Context<Ttype>>(TargetWrapper<Ttype>::get_device_id(), 
                                                         op_func.current_lane, 
                                                         op_func.current_lane);
        // call init of operator
        CHECK_NOTNULL(op_func.op) << "Node(node_name) doesn't have op pointer! ";

        op_func.op->_helper->InferShape(op_func.ins, op_func.outs);

#ifdef ENABLE_DEBUG
        for(auto& in : op_func.ins) {
                LOG(INFO) << "  => [shape]: " << in->valid_shape()[0] 
                          << " " << in->valid_shape()[1] 
                          << " " << in->valid_shape()[2] 
                          << " " << in->valid_shape()[3];
                LOG(INFO) <<"in offset size = "<<in->get_seq_offset().size();
        }
        for(auto& out : op_func.outs) {
                LOG(INFO) << "  <= [shape]: " << out->valid_shape()[0] 
                          << " " << out->valid_shape()[1] 
                          << " " << out->valid_shape()[2] 
                          << " " << out->valid_shape()[3];
                LOG(INFO) <<"out offset size = "<<out->get_seq_offset().size();
        }

#endif
        op_func.op->_helper->Init(*(op_func.ctx_p), op_func.ins, op_func.outs);
    }
    
	double curr_mem_in_mb_end = MemoryInfo<Ttype>::Global().get_used_mem_in_mb(); 
	this->_graph_p->statistics.template set_info<graph::SYSTEM_MEM>(curr_mem_in_mb_end - curr_mem_in_mb_start);
    // init memory of _graph_p
    init_memory();
	
	graph.statistics = _graph_p->statistics; // copy statistic back
	LOG(INFO) << "Temp mem used:        " << this->_graph_p->statistics.template get_info<graph::TEMP_MEM>() << " MB"; 
	LOG(INFO) << "Original mem used:    " << this->_graph_p->statistics.template get_info<graph::ORI_TEMP_MEM>() << " MB";
	LOG(INFO) << "Model mem used:       " << this->_graph_p->statistics.template get_info<graph::MODEL_MEM>() << " MB";
	LOG(INFO) << "System mem used:      " << this->_graph_p->statistics.template get_info<graph::SYSTEM_MEM>() << " MB";

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
                executer.ins[i]->sync();
            }
        }
        LOG(WARNING) << " Inspect memory of " << executer.name << " (" << executer.op_name << ") ";
        executer.infer_shape();
    
	    for (auto out : executer.outs) {
	        LOG(INFO) << "    |-- out tensor avg " << tensor_average(out);
	    }
#ifdef USE_CUDA
	    CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());
#endif
    }
#endif
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::prediction() {
    int i = 0;
    for(auto& executer : _exec_funcs) {
        if (RunType == OpRunType::SYNC || executer.need_sync) {
            for(int i = 0; i < executer.ins.size(); i++) {
                executer.ins[i]->record_event(executer.ctx_p->get_compute_stream());
                executer.ins[i]->sync();
            }
        }
#ifdef ENABLE_DEBUG
        LOG(ERROR) << " executer : " << executer.name << " (" << executer.op_name << ") ";
        for(auto in : executer.ins) {
                LOG(ERROR) << "    \\__ in shape: [" << in->valid_shape()[0] 
                           << "," << in->valid_shape()[1] 
                           << "," << in->valid_shape()[2] 
                           << "," << in->valid_shape()[3] 
                           << "] valid_size: " << in->valid_size() 
                           << " realsize: " << in->size() 
                           << " offset_size: "<<in->get_seq_offset().size();
        }
        for (auto out : executer.outs) {
	        LOG(ERROR) << "    \\__ out tensor avg " << tensor_average(out);
	    }
#endif

#ifdef ENABLE_OP_TIMER
        Context<Ttype> ctx(0, 0, 0);
	    saber::SaberTimer<Ttype> my_time;
	    my_time.start(ctx);
#endif
        if (executer.op_name != "Input") {
            executer.infer_shape();
            executer.launch();
        }
        for(int i = 0; i < executer.outs.size(); i++) {
            executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
        }

#ifdef ENABLE_OP_TIMER
        for (int i = 0; i < executer.outs.size(); i++) {
            // record event
            executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
            executer.outs[i]->sync();
        }
	    my_time.end(ctx);
        _op_time[op_id++] += my_time.get_average_ms();
#endif
    }
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::execute_stop_at_node(std::string node_name) {
	if(_suspended_point==-1) { 
		for(int i=0; i<_exec_funcs.size(); i++) {
			if(_exec_funcs[i].name == node_name) {
				_suspended_point = i;
			}
		}
	}
	for(int i=0; i<_suspended_point; i++) {
		auto& executer = _exec_funcs[i];
        if (RunType == OpRunType::SYNC || executer.need_sync) {
            for(int i = 0; i < executer.ins.size(); i++) {
                // record
                executer.ins[i]->record_event(executer.ctx_p->get_compute_stream());
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
		for (auto out : executer.outs) {
	        LOG(INFO) << "    |-- out tensor avg " << tensor_average(out);
	    }

#endif 
		if (executer.op_name != "Input") { 
			executer.infer_shape(); 
			executer.launch(); 
		} 
      
		for(int i = 0; i < executer.outs.size(); i++) { 
			executer.outs[i]->record_event(executer.ctx_p->get_compute_stream()); 
		} 
	}
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::execute_start_from_node(std::string node_name) {
	if(_start_point == -1) {
		for(int i=0; i<_exec_funcs.size(); i++) {
			if(_exec_funcs[i].name == node_name) {
				_start_point = i;
			}
		}
	}
	for(int i=_start_point; i<_exec_funcs.size(); i++) {
		auto& executer = _exec_funcs[i];
        if (RunType == OpRunType::SYNC || executer.need_sync) {
            for(int i = 0; i < executer.ins.size(); i++) {
                // record
                executer.ins[i]->record_event(executer.ctx_p->get_compute_stream());
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
		for (auto out : executer.outs) {
	        LOG(INFO) << "    |-- out tensor avg " << tensor_average(out);
	    }

#endif 
		if (executer.op_name != "Input") { 
			executer.infer_shape(); 
			executer.launch(); 
		} 
      
		for(int i = 0; i < executer.outs.size(); i++) { 
			executer.outs[i]->record_event(executer.ctx_p->get_compute_stream()); 
		} 
	}
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype> Net<Ttype, Ptype, RunType>::get_out(std::string out_name) {
    auto& edge_it_list = _graph_p->get_in_arc_its(out_name);
    CHECK_EQ(edge_it_list.size(), 1) << " Node(" << out_name << ") should have 1 in edge.";
    return edge_it_list[0]->weight().get();
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype> > Net<Ttype, Ptype, RunType>::get_out_list() {
    auto& out_list_vec = _graph_p->get_outs();
    for (auto& out : out_list_vec) {
        _out_tensor_list.push_back(get_out(out.c_str()));
    }
    return _out_tensor_list;
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype> Net<Ttype, Ptype, RunType>::get_in(std::string in_name) {
    auto& edge_it_list = _graph_p->get_out_arc_its(in_name);
    CHECK_EQ(edge_it_list.size(), 1) << " Node(" << in_name << ") should have 1 out edge.";
    return edge_it_list[0]->weight().get();
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype> > Net<Ttype, Ptype, RunType>::get_in_list() {
    auto& in_list_vec = _graph_p->get_ins();
    for (auto& in : in_list_vec) {
        _in_tensor_list.push_back(get_in(in.c_str()));
    }
    return _in_tensor_list;
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype> Net<Ttype, Ptype, RunType>::get_tensor_from_edge(const char* from, const char* to) {
    return _graph_p->get_arc(std::string(from), std::string(to)).weight().get();
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Status Net<Ttype, Ptype, RunType>::init_memory() {
    auto alloc_memory = [this](graph::Edge<Ttype>& edge) {
        auto& tensor_p = edge.weight();
        if(!edge.shared()) {
            tensor_p->re_alloc(tensor_p->shape());
        }
        return 0;
    };
    _graph_p->Scanner->BFS_Edge(alloc_memory);

    auto share_memory = [this](graph::Edge<Ttype>& edge) {
        if(edge.shared()) {
            auto& edge_name = edge.share_from();
	    bool continue_search = true;
	    while(continue_search) {
            	auto match_edge = [&](graph::Edge<Ttype>& inner_edge) {
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
        auto analysis_used_of_temp_mem = [&](graph::Edge<Ttype>& edge) {
            auto& tensor_p = edge.weight();
            if (!edge.shared()) {
                temp_mem_in_mbytes += (tensor_p->size() * 4);
            }
            ori_temp_mem_in_mbytes += (tensor_p->valid_shape().count() * 4);
        };
        this->_graph_p->Scanner->BFS_Edge(analysis_used_of_temp_mem);

		this->_graph_p->statistics.template set_info<graph::TEMP_MEM>(temp_mem_in_mbytes / 1e6);
		this->_graph_p->statistics.template set_info<graph::ORI_TEMP_MEM>(ori_temp_mem_in_mbytes / 1e6);
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Status Net<Ttype, Ptype, RunType>::init_env(graph::Graph<Ttype, Ptype>& graph) {
    LOG(WARNING) << "Detect and initial " << graph.get_ins().size() << " lanes.";
    Env<Ttype>::env_init(graph.get_ins().size());
    LOG(WARNING) << "Current used device id : " << TargetWrapper<Ttype>::get_device_id();
    return Status::OK();
}


#ifdef USE_CUDA
template class Net<NV, Precision::FP32, OpRunType::ASYNC>;
template class Net<NV, Precision::FP16, OpRunType::ASYNC>;
template class Net<NV, Precision::INT8, OpRunType::ASYNC>;

template class Net<NV, Precision::FP32, OpRunType::SYNC>;
template class Net<NV, Precision::FP16, OpRunType::SYNC>;
template class Net<NV, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_X86_PLACE
template class Net<X86, Precision::FP32, OpRunType::ASYNC>;
template class Net<X86, Precision::FP16, OpRunType::ASYNC>;
template class Net<X86, Precision::INT8, OpRunType::ASYNC>;

template class Net<X86, Precision::FP32, OpRunType::SYNC>;
template class Net<X86, Precision::FP16, OpRunType::SYNC>;
template class Net<X86, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef AMD_GPU
template class Net<AMD, Precision::FP32, OpRunType::ASYNC>;
template class Net<AMD, Precision::FP16, OpRunType::ASYNC>;
template class Net<AMD, Precision::INT8, OpRunType::ASYNC>;

template class Net<AMD, Precision::FP32, OpRunType::SYNC>;
template class Net<AMD, Precision::FP16, OpRunType::SYNC>;
template class Net<AMD, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class Net<ARM, Precision::FP32, OpRunType::ASYNC>;
template class Net<ARM, Precision::FP32, OpRunType::SYNC>;
#endif

#ifdef ANAKIN_TYPE_FP16
template class Net<ARM, Precision::FP16, OpRunType::ASYNC>;
template class Net<ARM, Precision::FP16, OpRunType::SYNC>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class Net<ARM, Precision::INT8, OpRunType::ASYNC>;
template class Net<ARM, Precision::INT8, OpRunType::SYNC>;
#endif //int8

#endif //arm

} /* namespace anakin */

