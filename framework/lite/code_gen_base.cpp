#include "framework/lite/code_gen_base.h"
#include "framework/graph/graph_global_mem.h"
#include "framework/core/net/net.h"

namespace anakin {

namespace lite {

/**
 * this full specialization use for help generating lite device running api
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
bool CodeGenBase<Ttype, Dtype, Ptype>::init_graph(Graph<Ttype, Dtype, Ptype>& graph) {
	_graph.CopyFrom(graph);
	// getting execution order
	auto& node_names_in_exec_order = _graph.get_nodes_in_order();
	for (auto& node_name : node_names_in_exec_order) { 
		auto node_ptr = _graph[node_name];
		if(node_ptr->get_op_name() == "Output") {
			continue;
		} 
		auto* op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()]; 
		node_ptr->set_op(op_pointer); 
		op_pointer = nullptr;
		static_cast<Operator<Ttype, Dtype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr); 
		// parsing parameter
		static_cast<Operator<Ttype, Dtype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
		// op name execution order
		_exec_op_order.push_back(node_ptr->get_op_name());
	} 
	// remove null op node
	for (auto it = node_names_in_exec_order.begin(); it != node_names_in_exec_order.end(); ){ 
		if (!_graph[*it]->Op()) { 
			it = node_names_in_exec_order.erase(it); 
		} else { 
			++it; 
		} 
	}

	// compute in/out shape and initialize the _graph
	std::vector<OperatorFunc<Ttype, Dtype, Ptype> > exec_funcs;
    exec_funcs.resize(node_names_in_exec_order.size());
    for(int i = 0; i < node_names_in_exec_order.size(); i++) {
        auto& node_name = node_names_in_exec_order[i];
        auto& op_func = exec_funcs[i];
        op_func.name = node_name;
        auto& edge_in_its = _graph.get_in_arc_its(node_name);
        DLOG(ERROR) << " node : " << op_func.name << " (" << _graph[node_name]->get_op_name() << ") ";
        for(auto& edge_it : edge_in_its) {
            DLOG(INFO) << "  => find in arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
            op_func.ins.push_back(edge_it->weight().get());
            op_func.in_lanes.push_back(edge_it->lane());
        }
        auto& edge_out_its = _graph.get_out_arc_its(node_name);
        for(auto& edge_it : edge_out_its) {
            DLOG(INFO) << "  <= find out arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
            op_func.outs.push_back(edge_it->weight().get());
            op_func.out_lanes.push_back(edge_it->lane());
        }
        op_func.current_lane = _graph[node_name]->lane();
        op_func.need_sync = _graph[node_name]->need_wait();
        op_func.op = static_cast<Operator<Ttype, Dtype, Ptype>* >(_graph[node_name]->Op());
        op_func.op_name = _graph[node_name]->get_op_name();
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
	return true;
}

template<typename Ttype, DataType Dtype, Precision Ptype>
bool CodeGenBase<Ttype, Dtype, Ptype>::init_memory_info() { 
	auto alloc_memory = [this](graph::Edge<Ttype, Dtype>& edge) { 
		auto& tensor_p = edge.weight(); 
		if(!edge.shared()) {
            tensor_p->re_alloc(tensor_p->shape());
        }
        return 0;
    };
    _graph.Scanner->BFS_Edge(alloc_memory);

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
				_graph.Scanner->BFS_Edge(match_edge); 
			} 
		} 
	};
    _graph.Scanner->BFS_Edge(share_memory);	
	return true;
}

template<typename Ttype, DataType Dtype, Precision Ptype>
bool CodeGenBase<Ttype, Dtype, Ptype>::serialize_weights() {
	return true;
}



} /* namespace lite */

} /* namespace anakin */

