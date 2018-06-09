#include "framework/lite/code_gen_base.h"

namespace anakin {

namespace lite {

/**
 * this full specialization use for help generating lite device running api
 */
template
bool CodeGenBase::init_graph<ARM, AK_FLOAT, Precision::FP32>(Graph<ARM, AK_FLOAT, Precision::FP32>& graph) {
	auto& node_names_in_exec_order = graph.get_nodes_in_order();
	for (auto& node_name : node_names_in_exec_order) { 
		auto node_ptr = graph[node_name];
		if(node_ptr->get_op_name() == "Output") {
			continue;
		}
		_exec_op_order.push_back(node_ptr->get_op_name());
	}	
	return true;
}

} /* namespace lite */

} /* namespace anakin */

