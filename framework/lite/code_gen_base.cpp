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
template<typename Ttype, Precision Ptype>
bool CodeGenBase<Ttype, Ptype>::extract_graph(const std::string& model_path, const int batch_size) {
    graph::Graph<Ttype, Ptype> graph;
	auto status = graph.load(model_path);
	if (!status ) {
		LOG(ERROR) << " [ERROR] " << status.info();
		return false;
	}

	// change graph node and edge name to standard of c(or others)variable name
	change_name(graph);
	//add batchsize
	std::vector<std::string>& ins = graph.get_ins();
	for (int i = 0; i < ins.size(); i++){
		graph.ResetBatchSize(ins[i], batch_size);
	}
	// Optimize
#ifdef USE_ARM_PLACE
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
    LOG(ERROR) << "finish fusion";

	// get graph io
	_ins =  graph.get_ins();
	_outs = graph.get_outs();

	// copy graph
	_graph.CopyFrom(graph);

	// getting execution order
	auto& node_names_in_exec_order = _graph.get_nodes_in_order();
	for (auto& node_name : node_names_in_exec_order) {
		auto node_ptr = _graph[node_name];
		//if(node_ptr->get_op_name() == "Output") {
		//	continue;
		//}
		// op execution order
		_exec_node_order.push_back(node_name);
		_graph_node_map[node_name].name = node_name;
		_graph_node_map[node_name].op_name = node_ptr->get_op_name();
		_graph_node_map[node_name].dtype = node_ptr->bit_type();
		// set node op pointer
		auto* op_pointer = OpFactory<Ttype, Ptype>::Global()[node_ptr->get_op_name()];
		node_ptr->set_op(op_pointer);
		op_pointer = nullptr;
		// bind parameter structure
		static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->BindParam(node_ptr);
		// parsing parameter
		static_cast<Operator<Ttype, Ptype>*>(node_ptr->Op())->_helper->InitParam();
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
	std::vector<OperatorFunc<Ttype, Ptype> > exec_funcs;
	exec_funcs.resize(node_names_in_exec_order.size());
    for (int i = 0; i < node_names_in_exec_order.size(); i++) {
        auto& node_name = node_names_in_exec_order[i];
		auto& op_func = exec_funcs[i];
        auto& edge_in_its = _graph.get_in_arc_its(node_name);
        DLOG(ERROR) << " node : " << node_name << " (" << _graph[node_name]->get_op_name() << ") ";
        for (auto& edge_it : edge_in_its) {
            DLOG(INFO) << "  => find in arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
			_graph_node_map[node_name].ins.push_back(edge_it->name());
			op_func.ins.push_back(edge_it->weight().get());
			op_func.in_lanes.push_back(edge_it->lane());
        }
        auto& edge_out_its = _graph.get_out_arc_its(node_name);
        for (auto& edge_it : edge_out_its) {
            DLOG(INFO) << "  <= find out arc : " << edge_it->bottom() << "  -->  " << edge_it->top();
			_graph_node_map[node_name].outs.push_back(edge_it->name());
			op_func.outs.push_back(edge_it->weight().get());
			op_func.out_lanes.push_back(edge_it->lane());
        }
		op_func.current_lane = _graph[node_name]->lane();
		op_func.need_sync = _graph[node_name]->need_wait();
		op_func.op = static_cast<Operator<Ttype, Ptype>* >(_graph[node_name]->Op());
		op_func.op_name = _graph[node_name]->get_op_name();

		CHECK_NOTNULL(op_func.op) << "Node(node_name) doesn't have op pointer! ";
		LOG(INFO)<<"OPNAME:"<<op_func.op_name << ", node_name:" << node_name;
		op_func.op->_helper->InferShape(op_func.ins, op_func.outs);
    }
	// initialize memory info
	if (!init_memory_info()) {
		return false;
	}
	return true;
}


template<typename Ttype, Precision Ptype>
void CodeGenBase<Ttype, Ptype>::change_name(graph::Graph<Ttype, Ptype>& graph) {
	auto convert2underline = [&](std::string& name, char converter_char) -> std::string {
		char* target_p = strdup(name.c_str());
		for (char* p = strchr(target_p + 1, converter_char); p!=NULL; p = strchr(p + 1, converter_char)) {
			*p = '_';
		}
		return std::string(target_p);
	};
	auto change_node_name = [&, this](graph::NodePtr& node_p) {
		auto & name = node_p->name();
		// add_alias is an important api for changing node's name and edge
		// and add_alias is useful only at this place so far.
		graph.add_alias(name, convert2underline(name, '/'));
		name = convert2underline(name, '/');
		graph.add_alias(name, convert2underline(name, '-'));
		name = convert2underline(name, '-');
	};
	graph.Scanner->BFS(change_node_name);
	auto change_edge_name = [&, this](graph::Edge<Ttype>& edge) {
		auto & first = edge.first();
		auto & second = edge.second();
		first = convert2underline(first, '/');
		second = convert2underline(second, '/');
		first = convert2underline(first, '-');
		second = convert2underline(second, '-');
	};
	graph.Scanner->BFS_Edge(change_edge_name);
}

template<typename Ttype, Precision Ptype>
bool CodeGenBase<Ttype, Ptype>::init_memory_info() {
	auto alloc_memory = [this](graph::Edge<Ttype>& edge) {
		EdgeInfo edge_info;
		edge_info.name = edge.name();

		auto& tensor_p = edge.weight();
		if (!edge.shared()) {
            tensor_p->re_alloc(tensor_p->shape());

			edge_info.valid_shape = tensor_p->shape();
			edge_info.real_shape = tensor_p->shape();
			edge_info.is_shared = false;
        } else {
			edge_info.is_shared = true;
		}
		edge_info.in_node = edge.first();
		edge_info.out_node = edge.second();
		edge_info.scale = edge.scale();
		auto in_node_ptr = _graph[edge_info.in_node];
		auto out_node_ptr = _graph[edge_info.out_node];
		auto in_node_dtype = in_node_ptr->bit_type();
		auto out_node_dtype = out_node_ptr->bit_type();
		if (in_node_dtype == AK_INT8 && out_node_dtype == AK_INT8) {
            edge_info.dtype = AK_INT8;
        } else {
		    edge_info.dtype = AK_FLOAT;
		}
		//edge_info.dtype = edge.data()->get_dtype();
		_tensor_map[edge_info.name] = edge_info;
        return 0;
    };
    _graph.Scanner->BFS_Edge(alloc_memory);
    auto share_memory = [this](graph::Edge<Ttype>& edge) {
        if (edge.shared()) {
			auto& edge_name = edge.share_from();

			_tensor_map[edge.name()].valid_shape = edge.weight()->valid_shape();
			_tensor_map[edge.name()].real_shape = edge.weight()->shape();
			bool continue_search = true;
			while (continue_search) {
                auto match_edge = [&](graph::Edge<Ttype>& inner_edge) {
                    if (inner_edge.name() == edge_name) {
                    	if (inner_edge.shared()) {
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
template class CodeGenBase<NV, Precision::FP32>;
template class CodeGenBase<NV, Precision::FP16>;
template class CodeGenBase<NV, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class CodeGenBase<X86, Precision::FP32>;
template class CodeGenBase<X86, Precision::FP16>;
template class CodeGenBase<X86, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class CodeGenBase<ARM, Precision::FP32>;
template class CodeGenBase<ARM, Precision::FP16>;
template class CodeGenBase<ARM, Precision::INT8>;
#endif

template class CodeGenBase<X86, Precision::FP32>;

} /* namespace lite */

} /* namespace anakin */

