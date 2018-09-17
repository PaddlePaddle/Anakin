#include "framework/core/net/net.h"
#include "saber/funcs/timer.h"
#include "saber/funcs/debug.h"
#include "framework/core/mem_info.h"

namespace anakin {

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Net<Ttype, Dtype, Ptype, RunType>::~Net() {
    if(_graph_p) {
        delete _graph_p;
        _graph_p = nullptr;
    }
}

template<typename Ttype, DataType Dtype>
double tensor_average(Tensor4dPtr<Ttype, Dtype>& out_tensor_p) {
    double sum = 0.0f;
    typedef typename DataTrait<Ttype, Dtype>::dtype dtype;
    const dtype* hptr = nullptr;

    Shape shin = out_tensor_p->valid_shape();
    PBlock<dtype, Ttype> tensorptr(shin);
    tensorptr.h_tensor().copy_from(*out_tensor_p);
    hptr = tensorptr.h_tensor().data();
    for (int i=0; i<out_tensor_p->valid_size(); i++) {
        sum += hptr[i];
    }
    return sum/out_tensor_p->valid_size();
}
template <>
double tensor_average<X86,AK_FLOAT>(Tensor4dPtr<X86,AK_FLOAT>& out_tensor_p) {
    double sum = 0.0f;
    CHECK_NOTNULL(out_tensor_p)<<"out_tensor_p can not be null";
    const float* hptr = out_tensor_p->data();
    for (int i=0; i<out_tensor_p->valid_size(); i++) {
        sum += hptr[i];
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
    //init_env(graph);
    init(graph);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Net<Ttype, Dtype, Ptype, RunType>::Net(\
    graph::Graph<Ttype, Dtype, Ptype>& graph, OpContextPtr<Ttype> ctx, bool need_summary) {
    _graph_p = new graph::Graph<Ttype, Dtype, Ptype>();
    _need_summary = need_summary;
    //init_env(graph);
    init(graph, ctx);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Dtype, Ptype, RunType>::init(graph::Graph<Ttype, Dtype, Ptype>& graph, \
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
        auto* op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()];
        if (op_pointer == nullptr) {
            LOG(FATAL) << node_name << ", type " << node_ptr->get_op_name() << " is null";
        }
        node_ptr->set_op(op_pointer);
        op_pointer = nullptr;

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
        op_func.ctx_p = ctx;
        // call init of operator
        CHECK_NOTNULL(op_func.op) << "Node(node_name) doesn't have op pointer! ";

        op_func.op->_helper->InferShape(op_func.ins, op_func.outs);
        op_func.op->_helper->Init(*(op_func.ctx_p), op_func.ins, op_func.outs);
    }

    // init memory of _graph_p
    init_memory();
}


template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Dtype, Ptype, RunType>::init(graph::Graph<Ttype, Dtype, Ptype>& graph) {
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

        if(std::is_same<Ttype,NV>::value) {
            if (node_ptr->get_op_name() == "ConvBatchnormScale" ||
                node_ptr->get_op_name() == "ConvBatchnormScaleRelu" || node_ptr->get_op_name() == "ConvRelu" ||
                node_ptr->get_op_name() == "Convolution") {
                std::string group = "group";    
                auto group_val = node_ptr->template get_attr<int>(group);
                std::string dilation = "dilation_rate";
                auto dilation_rate_val =  node_ptr->template get_attr<PTuple<int> >(dilation);
                using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;
                std::string weight_name = "weight_1";
                auto weights = node_ptr->template get_attr<pblock_type>(weight_name);
                
                int k_w = weights.d_tensor().width();
                int k_h = weights.d_tensor().height();
                int dil_h = dilation_rate_val.vector()[0];
                int dil_w = dilation_rate_val.vector()[1];

            if ((group_val == 1) && (k_w == 3 && k_h == 3 && dil_h == 1 && dil_w == 1)) {
                node_ptr->set_op(OpFactory<Ttype, Dtype, Ptype>::Global()["Sass"+node_ptr->get_op_name()]);
                node_ptr->get_op_name() = "Sass" + node_ptr->get_op_name();
            } else {
                    LOG(ERROR) << "node_ptr->get_op_name()  sass not support yet.";
                    auto *op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()];
                    node_ptr->set_op(op_pointer);
                }
            } else {
                auto *op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()];
                node_ptr->set_op(op_pointer);
            }
        }
        else {
            auto *op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()];
            if (op_pointer == nullptr) {
                CHECK(false)<< node_name << ", type " << node_ptr->get_op_name() << " is null";
                        LOG(FATAL) << node_name << ", type " << node_ptr->get_op_name() << " is null";
            }
            node_ptr->set_op(op_pointer);
        }
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
#ifdef ENABLE_DEBUG
        DLOG(INFO)<<"op init success "<<op_func.name;
#endif
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

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Dtype, Ptype, RunType>::prediction() {
#ifdef ENABLE_OP_TIMER
    int op_id = 0;
#endif

    int i = 0;
    for(auto& executer : _exec_funcs) {
        if (RunType == OpRunType::SYNC || executer.need_sync) {
            for(int i = 0; i < executer.ins.size(); i++) {
                // sync event record in multi_stream
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
        // record
        executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
        executer.outs[i]->sync();
    }
    my_time.end(ctx);
    _op_time[op_id++] += my_time.get_average_ms();
#endif
    //LOG(INFO)<< "op: " << executer.name<<"(" << executer.op_name <<")  ===  infer+launch time "<<my_time.get_average_ms() << " ms";
#ifdef ENABLE_DEBUG
#ifdef USE_CUDA
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());
#endif
    for (auto out : executer.outs) {
        std::vector<int> offset=out->get_seq_offset();
        LOG(INFO)<<"print offset of "<<executer.name <<",size = "<<offset.size();
        for(int i=0;i<offset.size();++i){
            LOG(INFO)<<offset[i]<<",";
        }
        LOG(INFO)<<"  end print offset of "<<executer.name;
#define RECORD_INNER
#if defined(RECORD_INNER) && defined(USE_X86_PLACE)
        record_tensor_to_file(*out,("record_"+executer.name).c_str());
#endif
        LOG(INFO) <<executer.name <<" d_tensor_out_p :" <<out->data();
#ifdef USE_CUDA
    record_tensor_to_file(*out,("record_"+executer.name).c_str());
#endif
#ifdef USE_X86_PLACE
//        for (int i = 0; i < 10; ++i) {
//            std::cout << out->data()[i]<<" ";
//        }
#endif
        LOG(ERROR) << "    |---out avg " << tensor_average(out);
    }

#ifdef USE_ARM_PLACE
        int idx = 0;
        for (auto out : executer.outs) {
            int size = out->valid_size();
            const float* ptr_data = out->data();
            int num = out->num();
            int c = out->channel();
            int w = out->width();
            int h = out->height();
            double sum = 0;
            for (int j = 0; j < num * c; ++j) {
                double sum_c = 0;
                for (int k = 0; k < w * h; ++k) {
                    sum_c += *ptr_data;
                    sum += *ptr_data;
                    ptr_data++;
                }
                //LOG(INFO) << "channel: " << j << ", mean value :" << sum_c / (w * h);
            }

            LOG(INFO) << executer.name << ", tensor idx: " << idx << ", mean value :" << sum / size << ", num: " << out->num() << \
                 ", channel: " << out->channel() << ", height: " << out->height() << ", width: " << out->width();
            idx++;
        }


        i++;
        if (0/*executer.name == "prob"*/) {
            for (auto out : executer.outs) {
                printf("output size: dims=%d, ", out->dims());
                for (int i = 0; i < out->dims(); i++){
                    printf("%d ", out->valid_shape()[i]);
                }
                printf("\n");

                printf("extract data: size: %d, num: %d, channel: %d, height=%d, width=%d\n", \
                     out->valid_size(), out->num(), out->channel(), out->height(), out->width());

                int ch_get = 0;
                int size_channel = out->width() * out->height();
                int start = ch_get * size_channel;
                int end = start + 1000;//size_channel;
                end = (end > out->valid_size())? (out->valid_size() - start) : end;
                const float* ptr_in = out->data(start);
#if 1
                //FILE* fp = fopen("conv0_relu_anakin.txt", "w+");

                for (int i = 0; i < end - start; i++)
                {
                    //fprintf(fp, "%f ", ptr_in[i]);
                    printf("%0.4f  ", ptr_in[i]);
                    if ((i + 1) % 10 == 0) {
                        //fprintf(fp, "\n");
                        printf("\n");
                    }
                }
                printf("\n");
                //fflush(fp);
                //fclose(fp);
#endif
            }
            LOG(FATAL) << "exit";
        }
#endif
#endif //debug
    }
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Dtype, Ptype, RunType>::execute_stop_at_node(std::string node_name) {
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

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Dtype, Ptype, RunType>::execute_start_from_node(std::string node_name) {
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

        this->_graph_p->statistics.template set_info<graph::TEMP_MEM>(temp_mem_in_mbytes / 1e6);
        this->_graph_p->statistics.template set_info<graph::ORI_TEMP_MEM>(ori_temp_mem_in_mbytes / 1e6);
    }
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Status Net<Ttype, Dtype, Ptype, RunType>::init_env(graph::Graph<Ttype, Dtype, Ptype>& graph) {
    LOG(WARNING) << "Detect and initial " << graph.get_ins().size() << " lanes.";
    // fixme, multi_stream error
    //Env<Ttype>::env_init(graph.get_ins().size());
    Env<Ttype>::env_init(1);
    LOG(WARNING) << "Current used device id : " << TargetWrapper<Ttype>::get_device_id();
    return Status::OK();
}


#ifdef USE_CUDA
template class Net<NV, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Net<NV, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Net<NV, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;

template class Net<NV, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
template class Net<NV, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
template class Net<NV, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_X86_PLACE
template class Net<X86, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Net<X86, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Net<X86, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;

template class Net<X86, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
template class Net<X86, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
template class Net<X86, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class Net<ARM, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Net<ARM, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
#endif

#ifdef ANAKIN_TYPE_FP16
template class Net<ARM, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Net<ARM, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class Net<ARM, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;
template class Net<ARM, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;
#endif //int8

#endif //arm

} /* namespace anakin */
