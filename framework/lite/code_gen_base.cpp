#include "framework/lite/code_gen_base.h"
#include "framework/graph/graph_global_mem.h"
#include "framework/core/net/net.h"
#include "framework/graph/llvm/scheduler.h"
#include "framework/graph/llvm/optimizer/parall_scheduler.h"
#include "framework/graph/llvm/optimizer/memory_scheduler.h"

namespace anakin {

namespace lite {

/**
 * this full specialization use for help generating lite device running api
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
bool CodeGenBase<Ttype, Dtype, Ptype>::extract_graph(std::string model_path) {
	graph::Graph<Ttype, Dtype, Ptype> graph;
	auto status = graph.load(model_path);
	if(!status ) { 
		LOG(ERROR) << " [ERROR] " << status.info(); 
		return false;
	}
	// Optimize
#if USE_ARM_PLACE
	auto vgraph = graph.get_vgraph();
	graph::Scheduler scheduler; 
	// schedule for exec order
	scheduler.RegIOResource(&vgraph); 
	scheduler.Run();
	scheduler.get_exec_node_in_order();
	// optimize mem
	graph::MemoryScheduler mem_scheduler; 
	mem_scheduler.RegIOResource(&vgraph); 
	mem_scheduler.Run(); 
	// analyse parallel
	graph::ParallScheduler para_scheduler; 
	para_scheduler.RegIOResource(&vgraph); 
	para_scheduler.Run();
	// restore from vgraph
	graph.restore_from_vgraph(&vgraph);
#else
	// Optimize
	graph.Optimize();
#endif

	// copy graph
	_graph.CopyFrom(graph);
	// getting execution order
	auto& node_names_in_exec_order = _graph.get_nodes_in_order();
	for (auto& node_name : node_names_in_exec_order) { 
		auto node_ptr = _graph[node_name];
		if(node_ptr->get_op_name() == "Output") {
			continue;
		} 
		// op execution order
		_exec_node_order.push_back(node_name);
		_graph_node_map[node_name].name = node_name;
		_graph_node_map[node_name].op_name = node_ptr->get_op_name();
		// set node op pointer
		auto* op_pointer = OpFactory<Ttype, Dtype, Ptype>::Global()[node_ptr->get_op_name()]; 
		node_ptr->set_op(op_pointer); 
		op_pointer = nullptr; 
		// bind parameter structure
		static_cast<Operator<Ttype, Dtype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
		// parsing parameter
		static_cast<Operator<Ttype, Dtype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
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
        auto& edge_in_its = _graph.get_in_arc_its(node_name);
        DLOG(ERROR) << " node : " << node_name << " (" << _graph[node_name]->get_op_name() << ") ";
        for(auto& edge_it : edge_in_its) {
            DLOG(INFO) << "  => find in arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
			_graph_node_map[node_name].ins.push_back(edge_it->name());
			op_func.ins.push_back(edge_it->weight().get()); 
			op_func.in_lanes.push_back(edge_it->lane());
        }
        auto& edge_out_its = _graph.get_out_arc_its(node_name);
        for(auto& edge_it : edge_out_its) {
            DLOG(INFO) << "  <= find out arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
			_graph_node_map[node_name].outs.push_back(edge_it->name());
			op_func.outs.push_back(edge_it->weight().get()); 
			op_func.out_lanes.push_back(edge_it->lane());
        }
		op_func.current_lane = _graph[node_name]->lane(); 
		op_func.need_sync = _graph[node_name]->need_wait(); 
		op_func.op = static_cast<Operator<Ttype, Dtype, Ptype>* >(_graph[node_name]->Op()); 
		op_func.op_name = _graph[node_name]->get_op_name(); 

		CHECK_NOTNULL(op_func.op) << "Node(node_name) doesn't have op pointer! ";
		op_func.op->_helper->InferShape(op_func.ins, op_func.outs);
    }

	// initialize memory info
	if(!init_memory_info()) {
		return false;
	}
	return true;
}

template<typename Ttype, DataType Dtype, Precision Ptype>
bool CodeGenBase<Ttype, Dtype, Ptype>::init_memory_info() { 
	auto alloc_memory = [this](graph::Edge<Ttype, Dtype>& edge) { 
		EdgeInfo edge_info;
		edge_info.name = edge.name();

		auto& tensor_p = edge.weight(); 
		if(!edge.shared()) {
            tensor_p->re_alloc(tensor_p->shape());

			edge_info.valid_shape = tensor_p->shape();
			edge_info.real_shape = tensor_p->shape();
			edge_info.is_shared = false;
        } else {
			edge_info.is_shared = true;
		}
		_tensor_map[edge_info.name] = edge_info;
        return 0;
    };
    _graph.Scanner->BFS_Edge(alloc_memory);

    auto share_memory = [this](graph::Edge<Ttype, Dtype>& edge) {
        if(edge.shared()) { 
			auto& edge_name = edge.share_from(); 

			_tensor_map[edge.name()].valid_shape = edge.weight()->valid_shape(); 
			_tensor_map[edge.name()].real_shape = edge.weight()->shape();

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

							_tensor_map[edge_name].valid_shape = inner_edge.weight()->valid_shape();
							_tensor_map[edge_name].real_shape = edge.weight()->valid_shape();
						} 
						edge.weight()->share_from(*(inner_edge.weight())); 
						_tensor_map[edge.name()].share_from= edge_name;
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

#ifdef USE_CUDA
template class CodeGenBase<NV, AK_FLOAT, Precision::FP32>;
template class CodeGenBase<NV, AK_FLOAT, Precision::FP16>;
template class CodeGenBase<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class CodeGenBase<X86, AK_FLOAT, Precision::FP32>;
template class CodeGenBase<X86, AK_FLOAT, Precision::FP16>;
template class CodeGenBase<X86, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class CodeGenBase<ARM, AK_FLOAT, Precision::FP32>;
template class CodeGenBase<ARM, AK_FLOAT, Precision::FP16>;
template class CodeGenBase<ARM, AK_FLOAT, Precision::INT8>;
#endif

} /* namespace lite */

} /* namespace anakin */

