#include "framework/graph/graph.h"
#include "framework/model_parser/parser/parser.h"
#include "framework/graph/llvm/scheduler.h"
#include "framework/graph/llvm/optimizer/conv_elewise_fusion_scheduler.h"
#include "framework/graph/llvm/optimizer/parall_scheduler.h"
#include "framework/graph/llvm/optimizer/memory_scheduler.h"
#include "framework/graph/llvm/fusion/graph_pattern.h"
#include "framework/core/operator/operator.h"
#include "framework/graph/llvm/optimizer/optimize_strategy.h"

namespace anakin {

namespace graph {

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::load(std::string model_path) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
    std::unique_lock<std::mutex> lock(this->_mut);
    Status ret = Status::OK();
    if (model_path != _model_path) {
        this->Clean();
        ret = parser::load<Ttype>(this, model_path);
        _model_path = model_path;
    }

    return ret;
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::load(const char* model_path) {
    return parser::load<Ttype>(this, model_path);
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::load(const char* buffer, size_t len) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
    std::unique_lock<std::mutex> lock(this->_mut);
    Status ret = Status::OK();
    this->Clean();
    ret = parser::load<Ttype, Ptype>(this, buffer, len);
    return ret;
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::save(std::string model_path) {
    return parser::save<Ttype>(this, model_path);
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::save(const char* model_path) {
    return parser::save<Ttype>(this, model_path);
}

template<typename Ttype, Precision Ptype>
std::vector<std::string>& Graph<Ttype, Ptype>::get_nodes_in_order() {
    return _nodes_exec_order;
}

template<typename Ttype, Precision Ptype>
void Graph<Ttype, Ptype>::Reshape(std::string in_name,
        std::vector<int> shape) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
    std::unique_lock<std::mutex> lock(this->_mut);
    auto input_node_p = (*this)[in_name];
    std::string in_shape = "input_shape";
    auto input_dim = input_node_p->template get_attr<PTuple<int>>(in_shape);
    CHECK_EQ(input_dim.size(), shape.size()) << "Target shape parameter's dim should equal to " <<
             input_dim.size();

    for (int i = 0; i < input_dim.size(); i++) {
        input_dim[i] = shape[i];
    }

    input_node_p->remove_attr(in_shape);
    input_node_p->set_attr(in_shape, input_dim);
}

template<typename Ttype, Precision Ptype>
void Graph<Ttype, Ptype>::ResetBatchSize(std::string in_name,
        const int batch_size) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
    std::unique_lock<std::mutex> lock(this->_mut);
    auto input_node_p = (*this)[in_name];
    std::string in_shape = "input_shape";
    auto input_dim = input_node_p->template get_attr<PTuple<int>>(in_shape);
    input_dim[0] = batch_size;
    input_node_p->remove_attr(in_shape);
    input_node_p->set_attr(in_shape, input_dim);
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::AddOp(const std::string& name, const std::string& type,
                                const std::vector<std::string>& inputs,
                                const std::vector<std::string>& outputs) {
    NodePtr node_p = std::make_shared<graph::Node>();
    node_p->set_name(name);
    node_p->get_op_name() = type;
    this->add_vertex(name, node_p);
    node_ins[name] = inputs;
    node_outs[name] = outputs;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::RegistBlock(PBlock<Ttype> * block_p) {
    graph::GraphGlobalMem<Ttype>::Global().register_block(block_p);
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::SetOpPrec(const std::string& name, DataType dtype) {
    if(this->has_vertex(name)) {
        NodePtr node_p = (*this)[name];
        node_p->set_bit_type(dtype);
        return Status::OK();
    }
    return Status::ANAKINFAIL("[EEROR]: SetOpPrec is called on an unknown op name");
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::SetWeightsScale(const std::string& name, const std::vector<float>& scales, bool is_bias) {
    if(this->has_vertex(name)) {
        NodePtr node_p = (*this)[name];
        if(is_bias) {
            bool bias_term = node_p->get_attr<bool>("bias_term");
            if(bias_term) {
                auto bias = node_p->get_attr<PBlock<Ttype>>("weight_2");
                bias.d_tensor().set_scale(scales);
                bias.h_tensor().set_scale(scales);
                return Status::OK();
            }
            return Status::OK("[WARNING]: SetWeightsScale is called to set bias scales in node which doesn't have it.");
        } else { // is weight
            if(node_p->inspect_attr("weight_1")) {
                auto weight = node_p->get_attr<PBlock<Ttype>>("weight_1");
                weight.d_tensor().set_scale(scales);
                weight.h_tensor().set_scale(scales);
                return Status::OK();
            }
            return Status::OK("[WARNING]: SetWeightsScale is called to set weight scales in node which doesn't have it.");
        }
    }
    return Status::ANAKINFAIL("[EEROR]: SetOpPrec is called on an unknown op name");
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::SetVarScale(const std::string& var, float scale) {
    std::unordered_map<std::string, std::vector<std::string> > in_to_op_map;
    std::unordered_map<std::string, std::vector<std::string> > out_to_op_map;
    for(const auto& pair: node_ins) {
        for(auto& in : pair.second) {
            in_to_op_map[in].push_back(pair.first);
        }
    }
    for(const auto& pair: node_outs) {
        for(auto& out : pair.second) {
            out_to_op_map[out].push_back(pair.first);
        }
    }
    for(const auto& pair : in_to_op_map) {
        if(in_to_op_map.count(var) > 0) {
            for(auto top : in_to_op_map[var]) {
                auto bottom = out_to_op_map[var][0];
                if(this->has_arc(bottom, top)) {
                    auto& edge = this->get_arc(bottom, top);
                    edge.set_scale({scale});
                    NodePtr node_p = (*this)[top];
                    if(node_p->get_op_name() == "Split") {
                        for(auto& edge_it : this->get_out_arc_its(top)) {
                            edge_it->set_scale({scale});
                        }
                    }
                }
            }
        }
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::RegistVar(const std::string& var) {
    auto regist_new_output = [&, this] () {
        this->add_out(var);
        this->AddOp(var, "Output", {var}, {});
    };

    std::unordered_map<std::string, std::vector<std::string> > in_to_op_map;
    std::unordered_map<std::string, std::vector<std::string> > out_to_op_map;
    for(const auto& pair: node_ins) {
        for(auto& in : pair.second) {
            in_to_op_map[in].push_back(pair.first);
        }
    }
    for(const auto& pair: node_outs) {
        for(auto& out : pair.second) {
            out_to_op_map[out].push_back(pair.first);
        }
    }
    for(const auto& pair : in_to_op_map) {
        if(in_to_op_map.count(var) > 0) {
            for(auto top : in_to_op_map[var]) {
                auto bottom = out_to_op_map[var][0];
                std::pair<std::string, std::string> tmp_pair(bottom, top);
                _registed_outs.push_back(tmp_pair);
                regist_new_output();
            }
        }
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::Freeze() {
    std::unordered_map<std::string, std::vector<std::string> > in_to_op_map;
    std::unordered_map<std::string, std::vector<std::string> > out_to_op_map;
    for(const auto& pair: node_ins) {
        for(auto& in : pair.second) {
            in_to_op_map[in].push_back(pair.first);
        }
    }
    for(const auto& pair: node_outs) {
        for(auto& out : pair.second) {
            out_to_op_map[out].push_back(pair.first);
        }
    }
    std::unordered_map<std::string, std::vector<std::string> > op_map_ins;
    std::unordered_map<std::string, std::vector<std::string> > op_map_outs;
    std::unordered_map<std::string, std::vector<std::string> > split_map_ins;
    std::unordered_map<std::string, std::vector<std::string> > split_map_outs;

    for(const auto& pair: node_ins) {
        for(auto& in : pair.second) {
            if(out_to_op_map.count(in) <= 0) {
                op_map_ins[in] = std::vector<std::string>{};
                op_map_outs[in] = std::vector<std::string>{in};
            }
        }
    }
    for(const auto& pair: op_map_ins) {
        auto op_name = pair.first;
        if(!this->has_vertex(op_name)) {
            this->add_in(op_name);
            this->AddOp(op_name, "Input", op_map_ins[op_name], op_map_outs[op_name]);
        }
    }
    op_map_ins.clear();
    op_map_outs.clear();
    auto auto_replace_split_ins = [&, this](const std::string split_variable,
                                            const std::vector<std::string>& outputs,
                                            const std::vector<std::string>& split_nexts) {
        for(int i=0; i < split_nexts.size(); i++) {
            for(auto& in : node_ins[split_nexts[i]]) {
                if(in == split_variable) {
                    in = outputs[i];
                }
            }
        }
    };

    // automatically add Split and Output
    for(const auto& pair : node_outs) {
        for(auto& out : pair.second) {
            if(in_to_op_map.count(out) <=0) {
                op_map_ins[out] = std::vector<std::string>{out};
                op_map_outs[out] = std::vector<std::string>{};
                continue;
            }
            if (in_to_op_map[out].size() > 1) {
               // find one to multi edge
                std::vector<std::string> inputs;
                std::vector<std::string> outputs;
                inputs.push_back(out);
                int split_num = in_to_op_map[out].size();
                std::string output_name_base = out + std::string("_split_");
                for(int i=0; i < split_num; i++) {
                    outputs.push_back(output_name_base + std::to_string(i));
                }
                std::string split_name = out + std::string("split");
                split_map_ins[split_name] = inputs;
                split_map_outs[split_name] = outputs;
                auto_replace_split_ins(out, outputs, in_to_op_map[out]);
            }
        }
    }
    for(const auto& pair: op_map_ins) {
        auto op_name = pair.first;
        if(!this->has_vertex(op_name)) {
            this->add_out(op_name);
            this->AddOp(op_name, "Output", op_map_ins[op_name], op_map_outs[op_name]);
        }
    }
    for(const auto& pair : split_map_ins) {
        auto split_name = pair.first;
        if(!this->has_vertex(split_name)) {
            this->AddOp(split_name, "Split", split_map_ins[split_name], split_map_outs[split_name]);
            this->AddOpAttr(split_name, "split_num", (int)(split_map_outs[split_name].size()));
        }
    }

    in_to_op_map.clear();
    out_to_op_map.clear();
    for(const auto& pair: node_ins) {
        for(auto& in : pair.second) {
            in_to_op_map[in].push_back(pair.first);
        }
    }
    for(const auto& pair: node_outs) {
        for(auto& out : pair.second) {
            out_to_op_map[out].push_back(pair.first);
        }
    }
    // those code logic with loop belown can't merge with that above
    for (const auto& pair: node_ins) {
        for (auto& in : pair.second) {
            if(out_to_op_map.count(in) > 0) {
                graph::Edge<Ttype> edge(out_to_op_map[in][0], pair.first);
                this->add_in_arc(edge);
            }
        }
    }
    for (const auto& pair: node_outs) {
        for (auto& out : pair.second) {
            if(in_to_op_map.count(out) > 0) {
                graph::Edge<Ttype> edge(pair.first, in_to_op_map[out][0]);
                this->add_out_arc(edge);
            }
        }
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::RegistOut(std::string node_bottom_name,
        std::string node_top_name) {
    std::pair<std::string, std::string> tmp_pair(node_bottom_name, node_top_name);
    _registed_outs.push_back(tmp_pair);
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::RegistAllOut() {
    auto register_edge = [&, this](Edge<Ttype>& edge) {
        this->RegistOut(edge.bottom(), edge.top());
        return Status::OK();
    };

    // regist all edge tensor
    this->Scanner->BFS_Edge(register_edge);

    return Status::OK();;
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::Optimize(bool with_fusion) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
    std::unique_lock<std::mutex> lock(this->_mut);

    if (!_has_graph_optimized) {
        DLOG(WARNING) << "Get virtual graph of graph ... ";
        get_vgraph();
        DLOG(INFO) << _vgraph->to_string();

        //! decide wheter the vgraph is optimized
        auto is_optimized = statistics.get_info<IS_OPTIMIZED>();
        is_optimized = false;

        if (is_optimized && (_registed_outs.size() == 0)) {
            // schedule for exec order
            Scheduler scheduler;
            scheduler.RegIOResource(_vgraph);
            scheduler.Run();
            // get node exec in order
            _nodes_exec_order = scheduler.get_exec_node_in_order();
        } else {
            if (with_fusion) {
                // xiaogang rang wo jia de
                DLOG(WARNING) << "Exe the graph fusion and combination [ SUPPORT IN-ORDER PATTERM ]";
                // TODO ...
                auto in_ordered_fusion_op_name_vec = FusionOpRegister::Global().get_list_op_name_in_fusion_order_of(
                        IN_ORDER);
                for (auto &fusion_name : in_ordered_fusion_op_name_vec) {
                    //in x86, we ignore two fusion patterns
                    if (std::is_same<Ttype, X86>::value &&
                        (fusion_name == "ConvReluPool" || fusion_name == "ConvBatchnormScaleReluPool")) {
                        continue;
                    }

                    if (std::is_same<Ttype, NV>::value && Precision::INT8 == Ptype &&
                        (fusion_name == "ConvReluPool" || fusion_name == "ConvBatchnormScaleReluPool")) {
                        continue;
                    }
                    if (std::is_same<Ttype, ARM>::value && Precision::INT8 == Ptype &&
                        (fusion_name == "ConvReluPool" || fusion_name == "ConvBatchnormScaleReluPool")) {
                        continue;
                    }
                    DLOG(INFO) << " processing in-ordered fusion : " << fusion_name;
                    _vgraph->Match(FusionOpRegister::Global()[fusion_name]);

                }
            }

            ///*
            restore_from_vgraph(_vgraph);
            graph_strategy<Ttype, Ptype> _strategy;
            if (std::is_same<Ttype,X86>::value) {
                LOG(INFO)<<"x86 close horizontal_combine";
            }else if (std::is_same<Ttype, ARM>::value){
                LOG(INFO)<<"arm close horizontal_combine";
            } else {
                _strategy.apply_horizontal_combine(this);
            }
            _strategy.apply_stride_up(this);
            *_vgraph = this->get_vgraph();
            //*/

            DLOG(WARNING) <<
                          "Schedule the vgraph for memory optimization and exec lanes ,as well as sync flags.";

            // schedule for exec order
            Scheduler scheduler;
            scheduler.RegIOResource(_vgraph);
            scheduler.Run();

            //LOG(ERROR) << "gen exe order";

            _nodes_exec_order = scheduler.get_exec_node_in_order();
//#if 0
#ifndef BUILD_LITE // enable conv+eltwise fusion
#ifndef USE_ARM_PLACE
            // optimization
            // xiaogang rang wo jia de
            if (with_fusion) {
                if ((std::is_same<Ttype, NV>::value||std::is_same<Ttype, X86>::value) && Precision::INT8 == Ptype) {
                } else {
                    ConvElsFusionScheduler conv_eltwise_fusion_scheduler;
                    conv_eltwise_fusion_scheduler.RegIOResource(_vgraph);
                    conv_eltwise_fusion_scheduler.Run();
                    // get node exec in order
                    _nodes_exec_order = conv_eltwise_fusion_scheduler.get_exec_node_in_order();
                }
            }
#endif
#endif
            // optimization again
            ParallScheduler para_scheduler;
            para_scheduler.RegIOResource(_vgraph);
            para_scheduler.Run();
            MemoryScheduler mem_scheduler;
            mem_scheduler.RegIOResource(_vgraph);
            mem_scheduler.Run();

            // set info for graph
            statistics.set_info<IS_OPTIMIZED>(true);
            DLOG(INFO) << " model size : " << graph::GraphGlobalMem<Ttype>::Global().get_sum_mbyte() << " mb ";
            statistics.set_info<MODEL_MEM>(graph::GraphGlobalMem<Ttype>::Global().get_sum_mbyte());

            DLOG(WARNING) << "Restore graph from virtual graph of ... ";
            restore_from_vgraph(_vgraph);
        }

        _has_graph_optimized = true;
    }

#ifdef ENABLE_DEBUG
    auto print_edge_debug_string = [](Edge<Ttype>& edge) {
        DLOG(INFO) << "Real Graph Edge : " << edge.ToString();
        return Status::OK();
    };
    this->Scanner->BFS_Edge(print_edge_debug_string);
    auto print_Node_debug_string = [](NodePtr& target_node) {
        DLOG(INFO) << "Real Graph Node : " << target_node->ToString();
        return Status::OK();
    };
    this->Scanner->BFS(print_Node_debug_string);
#endif
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
VGraph& Graph<Ttype, Ptype>::get_vgraph() {
    _vgraph = new VGraph();
    auto set_nodes = [&](NodePtr& node_p) {
        node v_node;
        v_node.name = node_p->name();
        v_node.opName = node_p->get_op_name();
        _vgraph->add_vertex(node_p->name(), v_node);
        return Status::OK();
    };
    // add node
    this->Scanner->BFS(set_nodes);

    auto set_edge_io_in = [&](NodePtr& node_p) {
        auto& edge_its = this->get_in_arc_its(node_p->name());
        for (auto& edge_it : edge_its) {
            io v_io;
            v_io.name = edge_it->name();
            Arc<std::string, io> arc(edge_it->bottom(), edge_it->top(), v_io);
            _vgraph->add_in_arc(arc);
        }

        return Status::OK();
    };

    auto set_edge_io_out = [&](NodePtr& node_p) {
        auto& edge_its = this->get_out_arc_its(node_p->name());

        for (auto& edge_it : edge_its) {
            io v_io;
            v_io.name = edge_it->name();
            Arc<std::string, io> arc(edge_it->bottom(), edge_it->top(), v_io);
            _vgraph->add_out_arc(arc);
        }

        return Status::OK();
    };


    // register graph out edge
    for (auto& out : _registed_outs) {
        _vgraph->register_outs(out.first, out.second);
    }

    // add edge io
    this->Scanner->BFS(set_edge_io_in);
    this->Scanner->BFS(set_edge_io_out);
    return *_vgraph;
}

//get graph scale maps
template<typename Ttype, Precision Ptype>
std::unordered_map<std::string, std::vector<float>>
Graph<Ttype, Ptype>::get_scale_map(){
    std::unordered_map<std::string, std::vector<float>> scale_map;
    auto get_scale = [&, this](NodePtr& node_p){
        auto& arc_its = this->get_in_arc_its(node_p->name());
        for (auto arc : arc_its){
            std::string edge_s = arc -> name();
            std::vector<float> scales = arc -> scale();
            scale_map[edge_s] = scales;
        }
    };

    this->Scanner->BFS(get_scale);
    return scale_map;
}
//get graph scale maps
template<typename Ttype, Precision Ptype>
std::unordered_map<std::string, saber::LayoutType>
Graph<Ttype, Ptype>::get_layout_map(){
    std::unordered_map<std::string, saber::LayoutType> layout_map;
    auto get_layout = [&, this](Edge<Ttype>& edge){
            layout_map[edge.name()] = edge.layout();
    };

    this->Scanner->BFS_Edge(get_layout);
    return layout_map;
}

template <typename Ttype, Precision Ptype>
void Graph<Ttype, Ptype>::load_calibrator_config(
    std::string config_file, std::string cal_file){
    CalibratorParser cal_parser;
#ifndef USE_SGX
    cal_parser.parse_from_file(config_file, cal_file);
#endif

    auto set_node_info = [&](NodePtr& node_p){
        node_p->set_bit_type(cal_parser.get_dtype_of_precision(node_p->name()));
    };
    this->Scanner->BFS(set_node_info);

    auto set_edge_scale = [&](Edge<Ttype>& edge){
        edge.set_scale({cal_parser.get_calibrator(edge.name())});
    };
    this->Scanner->BFS_Edge(set_edge_scale);
}

#ifndef USE_SGX
template <typename Ttype, Precision Ptype>
void Graph<Ttype, Ptype>::load_layout_config(std::string config_file){
    CalibratorParser cal_parser;
    cal_parser.layout_parse(config_file);

    auto set_edge_info = [&](Edge<Ttype>& edge){
        LOG(ERROR)<<"load layout :: " << edge.name() <<","<< cal_parser.get_layout(edge.name());
        edge.set_layout(cal_parser.get_layout(edge.name()));
    };
    this->Scanner->BFS_Edge(set_edge_info);
}
#endif

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::restore_from_vgraph(VGraph* vgraph) {
    //! need to clear graph edge first
    auto graph_scale_map = this->get_scale_map();
    auto graph_layout_map = this->get_layout_map();

    this->arcs_clear();

    auto interpreter_io_in = [&, this](node& target_node) {
        auto & arc_its = vgraph->get_in_arc_its(target_node.name);
        for (auto& arc_it : arc_its) {
            auto& tmp_io = arc_it->weight();
            auto& bottom = arc_it->bottom();
            auto& top = arc_it->top();
            Edge<Ttype> edge(bottom, top);
            auto& shared = edge.shared();
            shared = tmp_io.shared;
            auto& share_from = edge.share_from();
            share_from = tmp_io.share_from;
            auto& lane = edge.lane();
            lane = tmp_io.lane;
            this->add_in_arc(edge);
        }
        return Status::OK();
    };

    vgraph->Scanner->BFS(interpreter_io_in);  // this will change this real graph

    auto interpreter_io_out = [&, this](node & target_node) {
        auto& arc_its = vgraph->get_out_arc_its(target_node.name);

        for (auto& arc_it : arc_its) {
            auto& tmp_io = arc_it->weight();
            auto& bottom = arc_it->bottom();
            auto& top = arc_it->top();
            Edge<Ttype> edge(bottom, top);
            auto& shared = edge.shared();
            shared = tmp_io.shared;
            auto& share_from = edge.share_from();
            share_from = tmp_io.share_from;
            auto& lane = edge.lane();
            lane = tmp_io.lane;
            //edge.weight() = new Tensor4d<Ttype>();
            //edge.weight() = std::make_shared<Tensor4d<Ttype> >();
            this->add_out_arc(edge);
        }

        return Status::OK();
    };

    vgraph->Scanner->BFS(interpreter_io_out);  // this will change this real graph

    // interpreter for node, more complicated
    //first, we need clear all merge node info
    for (auto& v : this->_node_merges){
        v.second.clear();
    }
    for (auto& v : this->_pattern_name_merges){
        v.second.clear();
    }
    for (auto& v : this->_node_merges_keep){
        v.second.clear();
    }

    auto map_node_to_node_ptr = [this](NodePtr& node_p,
    node & target_node) -> Status {
        if (node_p->name() == target_node.name) {
            CHECK_EQ(target_node.mergeNodes.size(), target_node.mergeNodeNames.size())
                << "Merge node must have same size with merged pattern name";

            if (target_node.mergeNodes.size()) { // target node is merged nodes.
                for (int i = 0; i < target_node.mergeNodes.size(); i++) {
                    this->_node_merges[target_node.name].push_back(target_node.mergeNodes[i].name);
                    this->_pattern_name_merges[target_node.name].push_back(target_node.mergeNodeNames[i]);
                }
            }
            if (target_node.idx_keep_in_merge_nodes.size()) {
                for (auto& idx : target_node.idx_keep_in_merge_nodes) {
                    this->_node_merges_keep[target_node.name].push_back(idx);
                }
            }

            auto& need_wait = node_p->need_wait();
            need_wait = target_node.need_wait;
            auto& lane = node_p->lane();
            lane = target_node.lane;
            auto& op_name = node_p->get_op_name();
            op_name = target_node.opName;
            return Status::EXIT(" Find the matched target arc io. ");
        }

        return Status::OK();
    };
    auto interpreter_node = [&, this](node & target_node) -> Status {
        this->Scanner->BFS(map_node_to_node_ptr, target_node);
        return Status::OK();
    };
    vgraph->Scanner->BFS(interpreter_node);

    //! merge the attr of nodes which need to merge
    auto merge_node_attrs = [&, this](NodePtr& node_p) -> Status {
        auto& target_node_name = node_p->name();

        if (this->_node_merges.count(target_node_name) > 0 && this->_node_merges[target_node_name].size()) {
            for (int i = 0; i < this->_node_merges[target_node_name].size(); i++) {
                auto& tmp_node_p = this->operator[](this->_node_merges[target_node_name][i]);
                (*node_p).Merge(*tmp_node_p,
                                this->_pattern_name_merges[target_node_name][i]); // add the merge node's attr

                // detect if the i-th node in _node_merges should be saved in Graph
                auto ret = std::find(this->_node_merges_keep[target_node_name].begin(),
                                     this->_node_merges_keep[target_node_name].end(),
                                     i);
                if (ret == this->_node_merges_keep[target_node_name].end()) {
                    this->remove(this->_node_merges[target_node_name][i]); // remove merge node which is useless
                }
            }
        }
        //here: we insert merged node slice info
        PTuple<int> slice_info;
        auto arc_out_its = vgraph -> get_out_arc_its(target_node_name);
        if (arc_out_its.size() > 1){
            for (int i = 0;i < arc_out_its.size(); ++i){
                auto name = arc_out_its[i]->top();
                auto v_vertex = (*vgraph)[name];
                slice_info.push_back(v_vertex.mergeNodes.size() + 1);
            }
            int sum = 0;
            for (int i = 0; i < slice_info.size(); ++i){
                sum += slice_info[i];
            }
            //LOG(INFO) << "nodename:" << target_node_name;
            //LOG(WARNING) << "sum:" << sum;
            //LOG(WARNING) << this->_node_merges[target_node_name].size(); 
            if (sum == this->_node_merges[target_node_name].size() + 1){
                node_p->set_attr<PTuple<int>>("slice_info", slice_info);
            }
        }

        return Status::OK();
    };
    this->Scanner->BFS(merge_node_attrs);

    //recover scales to edge
    auto recover_scale = [&, this](Edge<Ttype>& edge){
        std::string edge_name = edge.name();
        std::string old_name = vgraph -> get_fusion_old_edge(edge_name);
        if (old_name != ""){
            edge_name = old_name;
        }
        if (graph_scale_map.count(edge_name) > 0){
            auto scales = graph_scale_map[edge_name];
            edge.set_scale(scales);
        } else {
            LOG(ERROR) << "when recover scale: the edge has no scale to map:" << edge_name;
        }

    };
    this->Scanner->BFS_Edge(recover_scale);

    //recover layout to edge
    auto recover_layout = [&, this](Edge<Ttype>& edge){
        std::string edge_name = edge.name();
        std::string old_name = vgraph -> get_fusion_old_edge(edge_name);
        if (old_name != ""){
            edge_name = old_name;
        }
        if (graph_layout_map.count(edge_name) > 0){
            auto layout = graph_layout_map[edge_name];
            edge.set_layout(layout);
        } else {
            LOG(ERROR) << "when recover layout: the edge has no layout to map:" << edge_name;
        }

    };
    this->Scanner->BFS_Edge(recover_layout);

    //for conv_eltwise, we deal scale to one node
    auto conv_eltwise_deal_scale = [this](NodePtr& node_p) -> Status {
        if (node_p->get_op_name() == "Gather"){
            auto in_edge_its = this->get_in_arc_its(node_p->name());
            float scale_0 = 1.f;
            float scale_3 = 1.f;
            DataType be_eltwise_dtype = AK_INVALID;
            CHECK_EQ(in_edge_its.size(), 2);
            auto eltwise_node_name = in_edge_its[0]->bottom();

            if ((*this)[in_edge_its[0]->bottom()]->get_op_name() == "ConvEltwise"){
                if (in_edge_its[1]->scale().size() > 0){
                    scale_0 = in_edge_its[1]->scale()[0];
                }
                be_eltwise_dtype = (*this)[in_edge_its[1]->bottom()]->bit_type();
            } else {
                if (in_edge_its[0]->scale().size() > 0){
                    scale_0 = in_edge_its[0]->scale()[0];
                }
                be_eltwise_dtype = (*this)[in_edge_its[0]->bottom()]->bit_type();
                eltwise_node_name = in_edge_its[1]->bottom();
            }
            auto out_edge_its = this->get_out_arc_its(node_p->name());
            CHECK_EQ(out_edge_its.size(), 1);
            if (in_edge_its[1]->scale().size() > 0){
                scale_3 = out_edge_its[0]->scale()[0];

            }
            auto eltwise_node = (*this)[eltwise_node_name];
            eltwise_node->template set_attr<float>("scale_0", scale_0);
            eltwise_node->template set_attr<float>("scale_3", scale_3);
            eltwise_node->template set_attr<DataType>("be_eltwise_dtype", be_eltwise_dtype);
        }

        return Status::OK();
    };
    this->Scanner->BFS(conv_eltwise_deal_scale);


    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::CopyFrom(Graph<Ttype, Ptype>& graph) {
    if(this->size() == graph.size()) {
        return Status::OK();
    }
    // this clear all the edges and nodes
    this->all_clear();
    auto shallow_copy_node = [&, this](NodePtr& node_p) {
        // create and copy node
        NodePtr node_new_p = std::make_shared<graph::Node>();
        *node_new_p = *node_p;
        this->add_vertex(node_new_p->name(), node_new_p);
    };
    graph.Scanner->BFS(shallow_copy_node);

    auto shallow_copy_edge = [&, this](NodePtr& node_p) {
        // create and copy edges
        auto edge_in_its = graph.get_in_arc_its(node_p->name());

        for (auto in_it : edge_in_its) {
            in_it->weight() = std::make_shared<Tensor4d<Ttype> >();
            this->add_in_arc(*in_it);
        }

        auto edge_out_its = graph.get_out_arc_its(node_p->name());

        for (auto out_it : edge_out_its) {
            out_it->weight() = std::make_shared<Tensor4d<Ttype> >();
            this->add_out_arc(*out_it);
        }
    };
    graph.Scanner->BFS(shallow_copy_edge);
    // get node execution order
    _nodes_exec_order = graph.get_nodes_in_order();
    // get graph inputs and outputs
    _ins = graph._ins;
    _outs = graph._outs;
    // get statistic
    statistics = graph.statistics;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::Clean() {
    // this clear all the edges and nodes
    this->all_clear();
    // delete _vgraph pointer
    delete _vgraph;
    _vgraph = nullptr;
    // clenn all weights
    graph::GraphGlobalMem<Ttype>::Global().clean_all();

    return Status::OK();
}
template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::fusion_optimize(bool with_fusion) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
    
    if (!std::is_same<Ttype, MLU>::value && !std::is_same<Ttype, BM>::value) {
        LOG(FATAL) << "only support bm and mlu right now!";
    }
    std::unique_lock<std::mutex> lock(this->_mut);
    if (!_has_graph_optimized) {
        DLOG(WARNING) << "Get virtual graph of graph ... ";
        get_vgraph();
        DLOG(INFO) << _vgraph->to_string();

        auto is_optimized = statistics.get_info<IS_OPTIMIZED>();

        Scheduler scheduler;
        scheduler.RegIOResource(_vgraph);
        scheduler.Run();
        _nodes_exec_order = scheduler.get_exec_node_in_order();

        _has_graph_optimized = true;
    }

#ifdef ENABLE_DEBUG
    auto print_edge_debug_string = [](Edge<Ttype>& edge) {
        DLOG(INFO) << "Real Graph Edge : " << edge.ToString();
        return Status::OK();
    };
    this->Scanner->BFS_Edge(print_edge_debug_string);
    auto print_Node_debug_string = [](NodePtr& target_node) {
        DLOG(INFO) << "Real Graph Node : " << target_node->ToString();
        return Status::OK();
    };
    this->Scanner->BFS(print_Node_debug_string);
#endif
    return Status::OK();
}

#ifdef USE_CUDA
template class Graph<NV, Precision::FP32>;
template class Graph<NV, Precision::FP16>;
template class Graph<NV, Precision::INT8>;
#endif

#ifdef USE_BM_PLACE
template class Graph<BM, Precision::FP32>;
template class Graph<BM, Precision::FP16>;
template class Graph<BM, Precision::INT8>;
#endif

#ifdef USE_MLU
template class Graph<MLU, Precision::FP32>;
template class Graph<MLU, Precision::FP16>;
template class Graph<MLU, Precision::INT8>;
#endif  // USE_MLU
#if defined USE_X86_PLACE || defined BUILD_LITE
template class Graph<X86, Precision::FP32>;
template class Graph<X86, Precision::FP16>;
template class Graph<X86, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class Graph<ARM, Precision::FP32>;
template class Graph<ARM, Precision::FP16>;
template class Graph<ARM, Precision::INT8>;
#endif

#ifdef AMD_GPU
template class Graph<AMD, Precision::FP32>;
template class Graph<AMD, Precision::FP16>;
template class Graph<AMD, Precision::INT8>;
#endif
} /* namespace graph */

} /* namespace anakin */
