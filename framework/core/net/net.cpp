#include "framework/core/net/net.h"
#include "saber/funcs/debug.h"
#include "framework/core/mem_info.h"
#include "framework/core/net/auto_layout_config.h"
#ifdef ENABLE_OP_TIMER
#include "saber/funcs/timer.h"
#endif

namespace anakin {

template<typename Ttype, Precision Ptype, OpRunType RunType>
Net<Ttype, Ptype, RunType>::~Net() {
    if (_graph_p) {
        delete _graph_p;
        _graph_p = nullptr;
    }
    if (_fusion) {
        delete _fusion;
        _fusion = nullptr;
    }
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


#ifndef USE_SGX
template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::
load_calibrator_config(graph::Graph<Ttype, Ptype>& graph, bool load_layout_from_graph,
                       bool auto_layout_config) {
    //clear calibrator info
    //load node precision
    auto load_node_precision = [&, this](graph::NodePtr & node_p) {
        auto type = node_p -> bit_type();
        _calibrator_parser.set_precision(node_p -> name(), type);
    };
    graph.Scanner -> BFS(load_node_precision);
    //load edge scale
    auto load_edge_scale = [&, this](graph::Edge<Ttype>& edge) {
        if (edge.scale().size() > 0) {
            float scale = edge.scale()[0];
            _calibrator_parser.set_scale(edge.name(), scale);
        }
    };
    graph.Scanner -> BFS_Edge(load_edge_scale);

    if (load_layout_from_graph) {
        //load edge layout
        auto load_edge_layout = [&, this](graph::Edge<Ttype>& edge) {
            auto layout = edge.layout();
            _calibrator_parser.set_layout(edge.name(), layout);
        };
        graph.Scanner->BFS_Edge(load_edge_layout);
    }

    if (auto_layout_config && std::is_same<Ttype, X86>::value) {
        bool is_all_nchw = true;
        auto search_layout = [&, this](graph::Edge<Ttype>& edge) {
            is_all_nchw = is_all_nchw && (_calibrator_parser.get_layout(edge.name()) == Layout_NCHW);
        };
        graph.Scanner->BFS_Edge(search_layout);
        bool is_edge_scale = false;
        auto search_scale = [&, this](graph::Edge<Ttype>& edge) {
            if (edge.scale().size() > 0 && edge.scale()[0] != 1.f) {
                is_edge_scale = true;
            }
        };
        graph.Scanner->BFS_Edge(search_scale);
                LOG(INFO) << "is_edge_scale " << is_edge_scale;

        if (is_edge_scale) {
            AutoLayoutConfigHelper<Ttype, Ptype> helper;
            auto layout_map = helper.auto_config_node_dtype(graph);

            for (auto k : layout_map) {
                LOG(INFO)<<"deduce "<<k.first<<","<<k.second;
                _calibrator_parser.set_precision(k.first, k.second);
            }

        } else if (is_all_nchw) {
                    LOG(INFO) << "ready to config layout";
            AutoLayoutConfigHelper<Ttype, Ptype> helper;
            helper.scane_dfs_from_input(graph);
            helper.print_layout();

            if (helper.check_merge(graph)) {
                auto configed_layout = helper.get_config_layout();
                auto set_edge_layout = [&, this](graph::Edge<Ttype>& edge) {
                    auto layout = configed_layout[edge.name()];
                            DLOG(ERROR) << edge.name() << " loaded layout: " << layout;
                    CHECK(layout != "");
                    _calibrator_parser.set_layout(edge.name(), layout);
                };

                graph.Scanner->BFS_Edge(set_edge_layout);
            } else {
                        LOG(ERROR) << "auto layout config cancel";
            }

        }
    }
}
#endif

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::init(graph::Graph<Ttype, Ptype>& graph, \
                                      OpContextPtr<Ttype> ctx, bool auto_config_layout) {

    init_env(graph);
    // shallow copy
    _graph_p->CopyFrom(graph);
    auto node_names_in_exec_order = graph.get_nodes_in_order();

#ifndef USE_SGX
    load_calibrator_config(graph,!_has_loaded_layout_from_file,auto_config_layout);
#endif

    // infer basic shape and parsing parameter from graph
    for (auto& node_name : node_names_in_exec_order) {
        auto node_ptr = (*_graph_p)[node_name];
        // create operations
        //auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
        auto* op_pointer = calibrator_op<Ttype>(node_ptr->get_op_name(), node_ptr->name(), _calibrator_parser);

        if (op_pointer == nullptr) {
            LOG(FATAL) << node_name << ", type " << node_ptr->get_op_name() << " is null";
        }

        node_ptr->set_op(op_pointer);
        op_pointer = nullptr;

        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
        // parsing parameter
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
    }

    // remove null op node
    for (auto it = node_names_in_exec_order.begin(); it != node_names_in_exec_order.end();) {
        if (!(*_graph_p)[*it]->Op()) {
            it = node_names_in_exec_order.erase(it);
        } else {
            ++it;
        }
    }

    _exec_funcs.resize(node_names_in_exec_order.size());


    std::vector<std::string> tensor_names;
    std::vector<saber::LayoutType> layouts;

    for (int i = 0; i < node_names_in_exec_order.size(); i++) {
        auto& node_name = node_names_in_exec_order[i];
        auto& op_func = _exec_funcs[i];
        op_func.name = node_name;
        auto& edge_in_its = _graph_p->get_in_arc_its(node_name);
        DLOG(WARNING) << " node : " << op_func.name << " (" << (*_graph_p)[node_name]->get_op_name() << ") ";
        for (auto& edge_it : edge_in_its) {
            DLOG(INFO) << "  => find in arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
            DLOG(INFO)<<"set "<<edge_it->name()<<" scale :"<< _calibrator_parser.get_calibrator(edge_it->name());
            op_func.ins.push_back(edge_it->weight().get());
            op_func.in_lanes.push_back(edge_it->lane());
            _tensor_name_list.push_back(edge_it->name());
        }

        auto& edge_out_its = _graph_p->get_out_arc_its(node_name);
        for (auto& edge_it : edge_out_its) {
            DLOG(INFO) << "  <= find out arc : " << edge_it->bottom() << "  -->  " << edge_it->top();

            tensor_names.push_back(edge_it->name());
            layouts.push_back(edge_it->weight()->get_layout());
#ifndef USE_SGX
            set_calibrator_info(edge_it);
#endif
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
void Net<Ttype, Ptype, RunType>::init(graph::Graph<Ttype, Ptype>& graph,bool auto_config_layout) {
    init_env(graph);
    // shallow copy
    _graph_p->CopyFrom(graph);

    double curr_mem_in_mb_start = MemoryInfo<Ttype>::Global().get_used_mem_in_mb();

    auto node_names_in_exec_order = graph.get_nodes_in_order();

#ifndef USE_SGX
    load_calibrator_config(graph,!_has_loaded_layout_from_file,auto_config_layout);
#endif

    // infer basic shape and parsing parameter from graph
    for (auto& node_name : node_names_in_exec_order) {
        auto node_ptr = (*_graph_p)[node_name];
#ifdef ENABLE_OP_TIMER

        if ((std::string::npos != (node_ptr->get_op_name()).find("Conv")
                || std::string::npos != (node_ptr->get_op_name()).find("Deconv")) && std::string::npos == (node_ptr->get_op_name()).find("ConvUnpadding")) {
            LOG(INFO) <<"op name:"<<node_ptr->get_op_name();
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
                sprintf(buf, "%s_%d*%d_%d*%d_%d*%d_%d*%d_%d", "Conv", kernel_size[0], kernel_size[1],
                        padding_size[0], padding_size[1], stride_size[0], stride_size[1], dilation[0], dilation[1],
                        group_val);
            } else {
                sprintf(buf, "%s_%d*%d_%d*%d_%d*%d_%d*%d_%d", "Deconv", kernel_size[0], kernel_size[1],
                        padding_size[0], padding_size[1], stride_size[0], stride_size[1], dilation[0], dilation[1],
                        group_val);
            }

            _op_param.push_back(buf);
        } else if (std::string::npos != (node_ptr->get_op_name()).find("Pooling") && std::string::npos == (node_ptr->get_op_name()).find("TopK")) {
            std::string key = "pool_size";
            std::string strides = "strides";
            std::string padding = "padding";
            auto kernel_size = node_ptr->template get_attr<PTuple<int>>(key);
            auto stride_size = node_ptr->template get_attr<PTuple<int>>(strides);
            auto padding_size = node_ptr->template get_attr<PTuple<int>>(padding);
            char buf[100];
            sprintf(buf, "%s_%d*%d_%d*%d_%d*%d", "Pooling", kernel_size[0], kernel_size[1], padding_size[0],
                    padding_size[1], stride_size[0], stride_size[1]);
            _op_param.push_back(buf);
        } else {
            _op_param.push_back(node_ptr->get_op_name());
        }

#endif
        //* create operations with target the same as this net
        //auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
        auto* op_pointer = calibrator_op<Ttype>(node_ptr->get_op_name(), node_ptr->name(), _calibrator_parser);
        if (op_pointer == nullptr) {
            CHECK(false) << node_name << ", type " << node_ptr->get_op_name() << " is null";
            LOG(FATAL) << node_name << ", type " << node_ptr->get_op_name() << " is null";
        }
        node_ptr->set_op(op_pointer);
        // bind parameter structure
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
        // parsing parameter
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
    }

    // remove null op node
    for (auto it = node_names_in_exec_order.begin(); it != node_names_in_exec_order.end();) {
        if (!(*_graph_p)[*it]->Op()) {
            it = node_names_in_exec_order.erase(it);
        } else {
            ++it;
        }
    }

    _exec_funcs.resize(node_names_in_exec_order.size());


    std::vector<std::string> tensor_names;
    std::vector<saber::LayoutType> layouts;

    //_calibrator_parser.layout_parse(_layout_config_path);
    for (int i = 0; i < node_names_in_exec_order.size(); i++) {
        auto& node_name = node_names_in_exec_order[i];
        auto& op_func = _exec_funcs[i];
        op_func.name = node_name;
        auto& edge_in_its = _graph_p->get_in_arc_its(node_name);
        DLOG(WARNING) << " node : " << op_func.name << " (" << (*_graph_p)[node_name]->get_op_name() << ") ";

        for (auto& edge_it : edge_in_its) {
            DLOG(INFO) << "  => find in arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
            DLOG(INFO)<<"set "<<edge_it->name()<<" scale :"<< _calibrator_parser.get_calibrator(edge_it->name());

            op_func.ins.push_back(edge_it->weight().get());
            op_func.in_lanes.push_back(edge_it->lane());
        }

        auto& edge_out_its = _graph_p->get_out_arc_its(node_name);

        for (auto& edge_it : edge_out_its) {
            DLOG(INFO) << "  <= find out arc : " << edge_it->bottom() << "  -->  " << edge_it->top();

            tensor_names.push_back(edge_it->name());
            layouts.push_back(edge_it->weight()->get_layout());
#ifndef USE_SGX
            set_calibrator_info(edge_it);
#endif
            op_func.outs.push_back(edge_it->weight().get());
            op_func.out_lanes.push_back(edge_it->lane());
            _tensor_name_list.push_back(edge_it->name());
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

        for (auto& in : op_func.ins) {
            LOG(INFO) << "  => [dtype]: " << in->get_dtype();
            LOG(INFO) << "  => [shape]: " << in->valid_shape();
            LOG(INFO) << "in offset size = " << in->get_seq_offset().size();
        }

        for (auto& out : op_func.outs) {
            LOG(INFO) << "  <= [dtype]: " << out->get_dtype();
            LOG(INFO) << "  <= [shape]: " << out->valid_shape();
            LOG(INFO) << "out offset size = " << out->get_seq_offset().size();
        }

#endif
        op_func.op->_helper->Init(*(op_func.ctx_p), op_func.ins, op_func.outs);
#ifdef ENABLE_DEBUG
        DLOG(INFO) << "op init success " << op_func.name;
#endif
    }

    double curr_mem_in_mb_end = MemoryInfo<Ttype>::Global().get_used_mem_in_mb();
    this->_graph_p->statistics.template set_info<graph::SYSTEM_MEM>(curr_mem_in_mb_end - curr_mem_in_mb_start);
    // init memory of _graph_p
    init_memory();

    graph.statistics = _graph_p->statistics; // copy statistic back
    LOG(INFO) << "Temp mem used:        " << this->_graph_p->statistics.template
            get_info<graph::TEMP_MEM>() << " MB";
    LOG(INFO) << "Original mem used:    " << this->_graph_p->statistics.template
            get_info<graph::ORI_TEMP_MEM>() << " MB";
    LOG(INFO) << "Model mem used:       " << this->_graph_p->statistics.template
            get_info<graph::MODEL_MEM>() << " MB";
    LOG(INFO) << "System mem used:      " << this->_graph_p->statistics.template
            get_info<graph::SYSTEM_MEM>() << " MB";



#ifdef ENABLE_OP_TIMER
    _op_time = std::vector<float>(_exec_funcs.size(), 0.0f);
#endif

#ifdef ENABLE_DEBUG
    LOG(WARNING) << "Checking memory...";

    for (auto& executer : _exec_funcs) {
        if (executer.need_sync) {
            for (int i = 0; i < executer.ins.size(); i++) {
                // record
                executer.ins[i]->record_event(executer.ctx_p->get_compute_stream());
                executer.ins[i]->sync();
            }
        }

        LOG(WARNING) << " Inspect memory of " << executer.name << " (" << executer.op_name << ") ";
        executer.infer_shape();

        for (auto out : executer.outs) {
            LOG(INFO) <<  executer.name << "  |-- out tensor avg " << tensor_mean_value_valid(*out);
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
#ifdef ENABLE_OP_TIMER
    int op_id = 0;
#endif
#ifdef ENABLE_DEBUG
    int op_cnt = 0;
#endif
    for (auto& executer : _exec_funcs) {
        if (RunType == OpRunType::SYNC || executer.need_sync || executer.op_name == "Output") {
            for (int i = 0; i < executer.ins.size(); i++) {
                // sync event record in multi_stream or syn when encountering output op
                executer.ins[i]->sync();
            }
        }

#ifdef ENABLE_DEBUG
        LOG(WARNING) << "[Num: "<< op_cnt++ << "] executer: " << executer.name << " (" << executer.op_name << ") ";

        for (auto in : executer.ins) {
            LOG(INFO) << "    \\ in shape (" << in->valid_shape() << ")"<<",data type "<<in->get_dtype()<<" , "
                       << " valid_size: " << in->valid_size()
                       << " realsize: " << in->size()
                       << " offset_size " << in->get_seq_offset().size();
        }

#endif

#ifdef ENABLE_OP_TIMER
        Context<Ttype> ctx(0, 0, 0);
        saber::SaberTimer<Ttype> my_time;
        my_time.start(ctx);
#endif

        if (executer.op_name != "Input" && executer.op_name != "Output") {
            executer.infer_shape();
            executer.launch();
        }

        for (int i = 0; i < executer.outs.size(); i++) {
            executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
        }

#ifdef ENABLE_DEBUG
#ifdef USE_CUDA
        if (std::is_same<Ttype, NV>::value) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
#endif
        for (auto out : executer.outs) {
            if (executer.name=="detection_out"){
                print_tensor(*out);
                LOG(INFO)<<"===============================";
            }
            LOG(INFO) << "    \\ out shape (" << out->valid_shape() << ") "<<",data type "<<out->get_dtype()<<" , "
                         << "executer name:"<< executer.name << " avg: " << tensor_mean_value_valid(*out);
        }
#ifdef USE_CUDA
        if (std::is_same<Ttype, NV>::value) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaPeekAtLastError());
        }
#endif

#ifndef USE_SGX
#if defined(RECORD_TENSOR_IN_NET)
        for (int i = 0; i < executer.ins.size(); i++) {
            record_tensor_in_format(*executer.ins[i], executer.op_name,executer.name,false,i);
        }
        for (int i = 0; i < executer.outs.size(); i++) {
            record_tensor_in_format(*executer.outs[i], executer.op_name,executer.name,true,i);
        }
#endif
#endif

#endif

#ifdef ENABLE_OP_TIMER

        for (int i = 0; i < executer.outs.size(); i++) {
            // record
            executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
            executer.outs[i]->sync();
        }

        my_time.end(ctx);
        float timmer_time = my_time.get_average_ms();
        //LOG(INFO)<<"[OP TIMER]  name = "<< _op_param[op_id]<<"  ,  time = "<<timmer_time<<" ms";
        _op_time[op_id++] += timmer_time;
#endif

    } // for
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::fusion_prediction() {
//	ASIC_CHECK(Ttype);
    if (!std::is_same<Ttype, MLU>::value && !std::is_same<Ttype, BM>::value) {
        LOG(FATAL) << "only support bm and mlu right now!";
    }
    if (_fusion->ctx_p->fusion()) {
        _fusion->launch();
    }
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
std::unique_ptr<Net<Ttype, Ptype, RunType> > Net<Ttype, Ptype, RunType>::Clone() {
    auto ret_net = std::unique_ptr<Net<Ttype, Ptype, RunType> >(new Net<Ttype, Ptype, RunType>);
    ret_net->_graph_p->CopyFrom(*(this->_graph_p));
    return ret_net;
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::init() {
    init_env(*_graph_p);

    double curr_mem_in_mb_start = MemoryInfo<Ttype>::Global().get_used_mem_in_mb();

    auto node_names_in_exec_order = _graph_p->get_nodes_in_order();

#ifndef USE_SGX
    load_calibrator_config(*_graph_p,!_has_loaded_layout_from_file);
#endif

    // infer basic shape and parsing parameter from graph
    for (auto& node_name : node_names_in_exec_order) {
        auto node_ptr = (*_graph_p)[node_name];

       //* create operations with target the same as this net
        //auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
        auto* op_pointer = calibrator_op<Ttype>(node_ptr->get_op_name(), node_ptr->name(), _calibrator_parser);
        if (op_pointer == nullptr) {
            CHECK(false) << node_name << ", type " << node_ptr->get_op_name() << " is null";
            LOG(FATAL) << node_name << ", type " << node_ptr->get_op_name() << " is null";
        }
        node_ptr->set_op(op_pointer);
        // bind parameter structure
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
        // parsing parameter
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
    }

    // remove null op node
    for (auto it = node_names_in_exec_order.begin(); it != node_names_in_exec_order.end();) {
        if (!(*_graph_p)[*it]->Op()) {
            it = node_names_in_exec_order.erase(it);
        } else {
            ++it;
        }
    }

    _exec_funcs.resize(node_names_in_exec_order.size());


    std::vector<std::string> tensor_names;
    std::vector<saber::LayoutType> layouts;

    //_calibrator_parser.layout_parse(_layout_config_path);
    for (int i = 0; i < node_names_in_exec_order.size(); i++) {
        auto& node_name = node_names_in_exec_order[i];
        auto& op_func = _exec_funcs[i];
        op_func.name = node_name;
        auto& edge_in_its = _graph_p->get_in_arc_its(node_name);
        DLOG(WARNING) << " node : " << op_func.name << " (" << (*_graph_p)[node_name]->get_op_name() << ") ";

        for (auto& edge_it : edge_in_its) {
            DLOG(INFO) << "  => find in arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
            DLOG(INFO)<<"set "<<edge_it->name()<<" scale :"<< _calibrator_parser.get_calibrator(edge_it->name());

            op_func.ins.push_back(edge_it->weight().get());
            op_func.in_lanes.push_back(edge_it->lane());
        }

        auto& edge_out_its = _graph_p->get_out_arc_its(node_name);

        for (auto& edge_it : edge_out_its) {
            DLOG(INFO) << "  <= find out arc : " << edge_it->bottom() << "  -->  " << edge_it->top();

            tensor_names.push_back(edge_it->name());
            layouts.push_back(edge_it->weight()->get_layout());

            set_calibrator_info(edge_it);

            op_func.outs.push_back(edge_it->weight().get());
            op_func.out_lanes.push_back(edge_it->lane());
            _tensor_name_list.push_back(edge_it->name());
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

        for (auto& in : op_func.ins) {
            LOG(INFO) << "  => [layout]: " << in->get_layout();
            LOG(INFO) << "  => [shape]: " << in->valid_shape();
            LOG(INFO) << "in offset size = " << in->get_seq_offset().size();
        }

        for (auto& out : op_func.outs) {
            LOG(INFO) << "  <= [layout]: " << out->get_layout();
            LOG(INFO) << "  <= [shape]: " << out->valid_shape();
            LOG(INFO) << "out offset size = " << out->get_seq_offset().size();
        }

#endif
        op_func.op->_helper->Init(*(op_func.ctx_p), op_func.ins, op_func.outs);
    }

    double curr_mem_in_mb_end = MemoryInfo<Ttype>::Global().get_used_mem_in_mb();
    this->_graph_p->statistics.template set_info<graph::SYSTEM_MEM>(curr_mem_in_mb_end - curr_mem_in_mb_start);
    // init memory of _graph_p
    init_memory();

    LOG(INFO) << "Temp mem used:        " << this->_graph_p->statistics.template
            get_info<graph::TEMP_MEM>() << " MB";
    LOG(INFO) << "Original mem used:    " << this->_graph_p->statistics.template
            get_info<graph::ORI_TEMP_MEM>() << " MB";
    LOG(INFO) << "Model mem used:       " << this->_graph_p->statistics.template
            get_info<graph::MODEL_MEM>() << " MB";
    LOG(INFO) << "System mem used:      " << this->_graph_p->statistics.template
            get_info<graph::SYSTEM_MEM>() << " MB";
}



template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::execute_stop_at_node(std::string node_name) {
    if (_suspended_point == -1) {
        for (int i = 0; i < _exec_funcs.size(); i++) {
            if (_exec_funcs[i].name == node_name) {
                _suspended_point = i;
            }
        }
    }

    for (int i = 0; i < _suspended_point; i++) {
        auto& executer = _exec_funcs[i];

        if (RunType == OpRunType::SYNC || executer.need_sync) {
            for (int i = 0; i < executer.ins.size(); i++) {
                // record
                executer.ins[i]->record_event(executer.ctx_p->get_compute_stream());
                executer.ins[i]->sync();
            }
        }

#ifdef ENABLE_DEBUG
        LOG(WARNING) << " executer: " << executer.name << " (" << executer.op_name << ") ";

        for (auto in : executer.ins) {
            LOG(INFO) << "    \\ in shape (" << in->valid_shape() << ")"
                       << " valid_size: " << in->valid_size()
                       << " realsize: " << in->size()
                       << " offset_size " << in->get_seq_offset().size();
        }

        for (auto out : executer.outs) {
            LOG(INFO) << "    |-- out tensor avg " << tensor_mean_value_valid(*out);
        }

#endif

        if (executer.op_name != "Input") {
            executer.infer_shape();
            executer.launch();
        }

        for (int i = 0; i < executer.outs.size(); i++) {
            executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
        }
    }
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::execute_start_from_node(std::string node_name) {
    if (_start_point == -1) {
        for (int i = 0; i < _exec_funcs.size(); i++) {
            if (_exec_funcs[i].name == node_name) {
                _start_point = i;
            }
        }
    }

    for (int i = _start_point; i < _exec_funcs.size(); i++) {
        auto& executer = _exec_funcs[i];

        if (RunType == OpRunType::SYNC || executer.need_sync) {
            for (int i = 0; i < executer.ins.size(); i++) {
                // record
                executer.ins[i]->record_event(executer.ctx_p->get_compute_stream());
                executer.ins[i]->sync();
            }
        }

#ifdef ENABLE_DEBUG
        LOG(WARNING) << " executer: " << executer.name << " (" << executer.op_name << ") ";

        for (auto in : executer.ins) {
            LOG(INFO) << "    \\ in shape (" << in->valid_shape() << ")"
                       << " valid_size: " << in->valid_size()
                       << " realsize: " << in->size()
                       << " offset_size " << in->get_seq_offset().size();
        }

        for (auto out : executer.outs) {
            LOG(INFO) << "    |-- out tensor avg " << tensor_mean_value_valid(*out);
        }

#endif

        if (executer.op_name != "Input") {
            executer.infer_shape();
            executer.launch();
        }

        for (int i = 0; i < executer.outs.size(); i++) {
            executer.outs[i]->record_event(executer.ctx_p->get_compute_stream());
        }
    }
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype> Net<Ttype, Ptype, RunType>::get_out(std::string out_name) {
    auto& edge_it_list = _graph_p->get_in_arc_its(out_name);
    CHECK_EQ(edge_it_list.size(), 1) << " Node (" << out_name << ") should have 1 in edge.";
    return edge_it_list[0]->weight().get();
}


template<typename Ttype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype> > Net<Ttype, Ptype, RunType>::get_out_list() {
    _out_tensor_list.clear();
    auto& out_list_vec = _graph_p->get_outs();

    for (auto& out : out_list_vec) {
        _out_tensor_list.push_back(get_out(out.c_str()));
    }

    return _out_tensor_list;
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype> Net<Ttype, Ptype, RunType>::get_in(std::string in_name) {
    auto& edge_it_list = _graph_p->get_out_arc_its(in_name);
    CHECK_EQ(edge_it_list.size(), 1) << " Node (" << in_name << ") should have 1 out edge.";
    return edge_it_list[0]->weight().get();
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype> > Net<Ttype, Ptype, RunType>::get_in_list() {
    _in_tensor_list.clear();
    auto& in_list_vec = _graph_p->get_ins();

    for (auto& in : in_list_vec) {
        _in_tensor_list.push_back(get_in(in.c_str()));
    }

    return _in_tensor_list;
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Tensor4dPtr<Ttype> Net<Ttype, Ptype, RunType>::get_tensor_from_edge(const char* from,
                                                                    const char* to) {
    return _graph_p->get_arc(std::string(from), std::string(to)).weight().get();
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Status Net<Ttype, Ptype, RunType>::alloc_memory_first(graph::Graph<Ttype, Ptype>& graph) {
    _graph_p->CopyFrom(graph);
    auto alloc_memory = [this](graph::Edge<Ttype>& edge) {
        auto& tensor_p = edge.weight();

        if (!edge.shared()) {
            if(tensor_p->mutable_data() == nullptr) {
				anakin::saber::Shape tmp_shape({1, 1 , 1, 1});
                tensor_p->re_alloc(tmp_shape, saber::AK_FLOAT);
                return Status::EXIT();
            }
        }

        return Status::OK();
    };
    _graph_p->Scanner->BFS_Edge(alloc_memory);
    return Status::OK();
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Status Net<Ttype, Ptype, RunType>::init_memory() {
    auto alloc_memory = [this](graph::Edge<Ttype>& edge) {
        auto& tensor_p = edge.weight();

        if (!edge.shared()) {
            if(tensor_p->mutable_data() == nullptr) {
                tensor_p->re_alloc(tensor_p->shape(), tensor_p->get_dtype());
            }
        }

        return 0;
    };
    
    _graph_p->Scanner->BFS_Edge(alloc_memory);

    auto share_memory = [this](graph::Edge<Ttype>& edge) {
        if (edge.shared()) {
            auto& edge_name = edge.share_from();
            bool continue_search = true;

            while (continue_search) {
                auto match_edge = [&](graph::Edge<Ttype>& inner_edge) {
                    if (inner_edge.name() == edge_name) {
                        if (inner_edge.shared()) {
                            edge_name = inner_edge.share_from();
                            return Status::EXIT(" Continue to find next.");
                        }
                        if ((inner_edge.weight()->size() * inner_edge.weight()->get_buf_dtype_size()
                                < edge.weight()->valid_size() * edge.weight()->get_dtype_size()) ||
                                (inner_edge.weight()->capacity() < edge.weight()->valid_size() * edge.weight()->get_dtype_size())) {
                            if(inner_edge.weight()->size() * inner_edge.weight()->get_buf_dtype_size() >
                                    edge.weight()->valid_size() * edge.weight()->get_dtype_size()) {
                                // this will be invoked when use API(alloc_memory_first)
                                inner_edge.weight()->re_alloc(inner_edge.weight()->valid_shape(),
                                                              inner_edge.weight()->get_dtype());
                            } else {
                                // normal mode
                                auto inner_original_shape = inner_edge.weight()->valid_shape();
                                auto inner_edge_dtype = inner_edge.weight()->get_dtype();
                                inner_edge.weight()->re_alloc(edge.weight()->valid_shape(),
                                                              edge.weight()->get_dtype());
                                inner_edge.weight()->set_dtype(inner_edge_dtype);
                                inner_edge.weight()->set_shape(inner_original_shape, inner_edge.weight()->shape());
                            }
                        }

                        edge.weight()->share_from(*(inner_edge.weight()));
                        continue_search = false;
                        return Status::EXIT(" Find the matched target edge.");
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
                temp_mem_in_mbytes += (tensor_p->size() * tensor_p->get_dtype_size());
                DLOG(WARNING) << "Edge("<< edge.bottom() << " ==> "
                          << edge.top() << ") shape("
                          << tensor_p->shape()[0] <<", "
                          << tensor_p->shape()[1] <<", "
                          << tensor_p->shape()[2] <<", "
                          << tensor_p->shape()[3] <<") . size: "
                          << tensor_p->size() * tensor_p->get_dtype_size() / 1024.0 / 1024.0 << " MB";
            }

            ori_temp_mem_in_mbytes += (tensor_p->valid_shape().count() * tensor_p->get_dtype_size());
        };
        this->_graph_p->Scanner->BFS_Edge(analysis_used_of_temp_mem);

        this->_graph_p->statistics.template set_info<graph::TEMP_MEM>(temp_mem_in_mbytes / 1024.0 / 1024.0);
        this->_graph_p->statistics.template set_info<graph::ORI_TEMP_MEM>(ori_temp_mem_in_mbytes / 1024.0 / 1024.0);
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

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Net<Ttype, Ptype, RunType>::fusion_init(graph::Graph<Ttype, Ptype>& graph, OpContextPtr<Ttype> ctx, bool auto_config_layout) {
    
//	ASIC_CHECK(Ttype);
    if (!std::is_same<Ttype, MLU>::value && !std::is_same<Ttype, BM>::value) {
        LOG(FATAL) << "only support mlu and bm right now!";
    }

    init_env(graph);
    // shallow copy
    _graph_p->CopyFrom(graph);
    auto node_names_in_exec_order = graph.get_nodes_in_order();
#ifndef USE_SGX
    load_calibrator_config(graph,!_has_loaded_layout_from_file,auto_config_layout);
#endif
#ifndef USE_BM_PLACE // anbl add 
    // infer basic shape and parsing parameter from graph
    for (auto& node_name : node_names_in_exec_order) {
        auto node_ptr = (*_graph_p)[node_name];
        // create operations
        //auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
        auto* op_pointer = calibrator_op<Ttype>(node_ptr->get_op_name(), node_ptr->name(), _calibrator_parser);

        if (op_pointer == nullptr) {
            LOG(FATAL) << node_name << ", type " << node_ptr->get_op_name() << " is null";
        }

        node_ptr->set_op(op_pointer);
        op_pointer = nullptr;

        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
        // parsing parameter
        static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
    }

    // remove null op node
    for (auto it = node_names_in_exec_order.begin(); it != node_names_in_exec_order.end();) {
        if (!(*_graph_p)[*it]->Op()) {
            it = node_names_in_exec_order.erase(it);
        } else {
            ++it;
        }
    }

    _exec_funcs.resize(node_names_in_exec_order.size());

    std::vector<std::string> tensor_names;
    std::vector<saber::LayoutType> layouts;

    for (int i = 0; i < node_names_in_exec_order.size(); i++) {
        auto& node_name = node_names_in_exec_order[i];
        auto& op_func = _exec_funcs[i];
        op_func.name = node_name;
        auto& edge_in_its = _graph_p->get_in_arc_its(node_name);
        DLOG(WARNING) << " node : " << op_func.name << " (" << (*_graph_p)[node_name]->get_op_name() << ") ";
        for (auto& edge_it : edge_in_its) {
            DLOG(INFO) << "  => find in arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
            DLOG(INFO)<<"set "<<edge_it->name()<<" scale :"<< _calibrator_parser.get_calibrator(edge_it->name());
            edge_it->weight()->set_dtype(_calibrator_parser.get_dtype(edge_it->bottom(), edge_it->top()));//set tensor precision
            op_func.ins.push_back(edge_it->weight().get());
            op_func.in_lanes.push_back(edge_it->lane());
            _tensor_name_list.push_back(edge_it->name());
        }

        auto& edge_out_its = _graph_p->get_out_arc_its(node_name);
        for (auto& edge_it : edge_out_its) {
            DLOG(INFO) << "  <= find out arc : " << edge_it->bottom() << "  -->  " << edge_it->top();

            tensor_names.push_back(edge_it->name());
            layouts.push_back(edge_it->weight()->get_layout());
            set_calibrator_info(edge_it);
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
#else
    
    auto ins_list = _graph_p->get_ins();
    auto outs_list = _graph_p->get_outs();
    
    auto in_tensor_list = get_in_list();
    auto out_tensor_list = get_out_list();

    for (int i = 0; i < ins_list.size(); ++i) {
        auto& node_name = ins_list[i];
        auto node_ptr = (*_graph_p)[node_name];
         
        auto vec_shape = node_ptr->template get_attr<PTuple<int>>("input_shape");
        Shape input_s({vec_shape[0], vec_shape[1], vec_shape[2], vec_shape[3]}, Layout_NCHW);
        in_tensor_list[i]->re_alloc(input_s, AK_FLOAT);
    }
#endif

    // init memory of _graph_p
    // there's bug when a network has split op and need to be fixed.
    // init_memory();
    if (ctx->fusion()) {
        if (!_fusion) {
            _fusion = new OperatorFunc<Ttype, Ptype>();
            _fusion->op = OpFactory<Ttype, Ptype>::Global()["Fusion"];
        }
        std::vector<Tensor4dPtr<Ttype>> in_list;
        std::vector<Tensor4dPtr<Ttype>> out_list;
        auto in_tensor_list = get_in_list();
        auto out_tensor_list = get_out_list();

        for (int i = 0; i < in_tensor_list.size(); ++i) {
            in_list.push_back(in_tensor_list[i]);
        }
        for (int i = 0; i < out_tensor_list.size(); ++i) {
            out_list.push_back(out_tensor_list[i]);
        }
        _fusion->ins = in_list;
        _fusion->outs = out_list;
        _fusion->ctx_p = ctx;
        _fusion->op->_helper->Init(*ctx, in_list, out_list);

#ifdef USE_BM_PLACE // anbl add
        for (int i = 0; i < out_list.size(); ++i) {
            out_tensor_list[i]->re_alloc(out_list[i]->valid_shape(), AK_FLOAT);
        }
#endif
    }
}

#ifdef USE_CUDA
template class Net<NV, Precision::FP32, OpRunType::ASYNC>;
template class Net<NV, Precision::FP16, OpRunType::ASYNC>;
template class Net<NV, Precision::INT8, OpRunType::ASYNC>;

template class Net<NV, Precision::FP32, OpRunType::SYNC>;
template class Net<NV, Precision::FP16, OpRunType::SYNC>;
template class Net<NV, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_MLU
template class Net<MLU, Precision::FP32, OpRunType::ASYNC>;
template class Net<MLU, Precision::FP16, OpRunType::ASYNC>;
template class Net<MLU, Precision::INT8, OpRunType::ASYNC>;

template class Net<MLU, Precision::FP32, OpRunType::SYNC>;
template class Net<MLU, Precision::FP16, OpRunType::SYNC>;
template class Net<MLU, Precision::INT8, OpRunType::SYNC>;
#endif  // USE_MLU
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
template class Net<ARM, Precision::FP32, OpRunType::ASYNC>;
template class Net<ARM, Precision::FP16, OpRunType::ASYNC>;
template class Net<ARM, Precision::INT8, OpRunType::ASYNC>;

template class Net<ARM, Precision::FP32, OpRunType::SYNC>;
template class Net<ARM, Precision::FP16, OpRunType::SYNC>;
template class Net<ARM, Precision::INT8, OpRunType::SYNC>;
#endif //arm

#ifdef USE_BM_PLACE
template class Net<BM, Precision::FP32, OpRunType::ASYNC>;
template class Net<BM, Precision::FP16, OpRunType::ASYNC>;
template class Net<BM, Precision::INT8, OpRunType::ASYNC>;

template class Net<BM, Precision::FP32, OpRunType::SYNC>;
template class Net<BM, Precision::FP16, OpRunType::SYNC>;
template class Net<BM, Precision::INT8, OpRunType::SYNC>;
#endif //bm

} /* namespace anakin */
