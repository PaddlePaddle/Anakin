#include "framework/graph/graph.h"
#include "framework/model_parser/parser/parser.h"
#include "framework/graph/llvm/scheduler.h"
#include "framework/graph/llvm/optimizer/conv_elewise_fusion_scheduler.h"
#include "framework/graph/llvm/optimizer/parall_scheduler.h"
#include "framework/graph/llvm/optimizer/memory_scheduler.h"
#include "framework/graph/llvm/fusion/graph_pattern.h"
#include "framework/core/operator/operator.h"

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

#ifndef USE_NANOPB
template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::save(std::string model_path) {
    return parser::save<Ttype>(this, model_path);
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::save(const char* model_path) {
    return parser::save<Ttype>(this, model_path);
}
#endif

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
Status Graph<Ttype, Ptype>::RegistOut(std::string node_bottom_name,
        std::string node_top_name) {
    std::pair<std::string, std::string> tmp_pair(node_bottom_name, node_top_name);
    _registed_outs.push_back(tmp_pair);
    return Status::OK();;
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
Status Graph<Ttype, Ptype>::Optimize() EXCLUSIVE_LOCKS_REQUIRED(_mut) {
    std::unique_lock<std::mutex> lock(this->_mut);

    if (!_has_graph_optimized) {
        DLOG(WARNING) << "Get virtual graph of graph ... ";
        get_vgraph();
        DLOG(INFO) << _vgraph->to_string();

        //! decide wheter the vgraph is optimized
        auto is_optimized = statistics.get_info<IS_OPTIMIZED>();

        if (is_optimized && (_registed_outs.size() == 0)) {
            // schedule for exec order
            Scheduler scheduler;
            scheduler.RegIOResource(_vgraph);
            scheduler.Run();
            // get node exec in order
            _nodes_exec_order = scheduler.get_exec_node_in_order();
        } else {
            DLOG(WARNING) << "Exe the graph fusion and combination [ SUPPORT IN-ORDER PATTERM ]";
            // TODO ...
            auto in_ordered_fusion_op_name_vec = FusionOpRegister::Global().get_list_op_name_in_fusion_order_of(IN_ORDER);
            for (auto& fusion_name : in_ordered_fusion_op_name_vec) {
                //in x86, we ignore two fusion patterns
                if (std::is_same<Ttype, X86>::value &&
                    (fusion_name == "ConvReluPool" || fusion_name == "ConvBatchnormScaleReluPool")){
                    continue;
                }

                LOG(INFO) << " processing in-ordered fusion : " << fusion_name;
                _vgraph->Match(FusionOpRegister::Global()[fusion_name]);
            }

            DLOG(WARNING) <<
                          "Schedule the vgraph for memory optimization and exec lanes ,as well as sync flags.";

            // schedule for exec order
            Scheduler scheduler;
            scheduler.RegIOResource(_vgraph);
            scheduler.Run();

            //LOG(ERROR) << "gen exe order";

            _nodes_exec_order = scheduler.get_exec_node_in_order();


#ifndef BUILD_LITE // enable conv+eltwise fusion
            // optimization
            ConvElsFusionScheduler conv_eltwise_fusion_scheduler;
            conv_eltwise_fusion_scheduler.RegIOResource(_vgraph);
            conv_eltwise_fusion_scheduler.Run();
            // get node exec in order
            _nodes_exec_order = conv_eltwise_fusion_scheduler.get_exec_node_in_order();
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
template <typename Ttype, Precision Ptype>
void Graph<Ttype, Ptype>::load_calibrator_config(
    std::string config_file, std::string cal_file){
    CalibratorParser cal_parser;
    cal_parser.parse_from_file(config_file, cal_file);

    auto set_node_info = [&](NodePtr& node_p){
        node_p->set_bit_type(cal_parser.get_dtype_of_precision(node_p->name()));
    };
    this->Scanner->BFS(set_node_info);

    auto set_edge_scale = [&](Edge<Ttype>& edge){
        edge.set_scale({cal_parser.get_calibrator(edge.name())});
    };
    this->Scanner->BFS_Edge(set_edge_scale);
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::restore_from_vgraph(VGraph* vgraph) {
    //! need to clear graph edge first
    auto graph_scale_map = this->get_scale_map();

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
    auto merge_node_attrs = [this](NodePtr& node_p) -> Status {
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

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Graph<Ttype, Ptype>::CopyFrom(Graph<Ttype, Ptype>& graph) {
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

#ifdef USE_CUDA
template class Graph<NV, Precision::FP32>;
template class Graph<NV, Precision::FP16>;
template class Graph<NV, Precision::INT8>;
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
template class Graph<X86, Precision::FP32>;
template class Graph<X86, Precision::FP16>;
template class Graph<X86, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class Graph<ARM, Precision::FP32>;
#endif
#ifdef ANAKIN_TYPE_FP16
template class Graph<ARM, Precision::FP16>;
#endif
#ifdef ANAKIN_TYPE_INT8
template class Graph<ARM, Precision::INT8>;
#endif
#endif

#ifdef AMD_GPU
template class Graph<AMD, Precision::FP32>;
template class Graph<AMD, Precision::FP16>;
template class Graph<AMD, Precision::INT8>;
#endif
} /* namespace graph */

} /* namespace anakin */
