#include "framework/lite/code_gen_cpp.h"

namespace anakin {

namespace lite {

void GenCPP::gen_header_start() {
	auto code_name = this->get_code_name();
	_code.Clean();
	_code.feed("#ifndef ANAKIN_%s_H \n", code_name.c_str());
	_code.feed("#define ANAKIN_%s_H \n\n", code_name.c_str());
	_code<<"namepsace anakin \{ \n\n";
}	

void GenCPP::gen_header_end() {
	_code<<"\} /* namespace anakin */\n";
	_code<<"#endif\n";
}

void GenCPP::gen_source_start() {
	auto code_name = this->get_code_name();
	_code.Clean();
	_code.feed("#include \"%s.h\" \n\n", code_name.c_str());
	_code<<"namepsace anakin \{ \n\n";
	// add running impl for model api
}	

void GenCPP::gen_source_end() {
	_code<<"\} /* namespace anakin */\n";
}

void GenCPP::gen_tensors() {
	_code<<"// generating and setting tensors ";
	for(auto it = _tensor_map.begin(); it != _tensor_map.end(); ++it) {
		auto& edge_name = it->first;
		auto& edge_info = it->second;
		if(! edge_info.is_shared) {
			_code.feed("tensor %s;\n", edge_name);
			_code.feed("%s.set_valid_shape(%s);", edge_name, edge_info.valid_shape);
			_code.feed("%s.set_real_shape(%s);", edge_name, edge_info.real_shape);
		}
	}
	for(auto it = _tensor_map.begin(); it != _tensor_map.end(); ++it) {
		auto& edge_name = it->first;
		auto& edge_info = it->second;
		if(edge_info.is_shared) {
			_code.feed("tensor %s;\n", edge_name);
			_code.feed("%s.set_valid_shape(%s);", edge_name, edge_info.valid_shape);
			_code.feed("%s.set_real_shape(%s);", edge_name, edge_info.real_shape);
			_code.feed("%s.share_from(%s);", edge_name, edge_info.share_from);
		}
	}
}

void GenCPP::gen_model_ios() {
	_code<<"// generating model's io \n";
	for(auto & node_name : _exec_node_order) {
		auto& node_info = _graph_node_map[node_name];
		_code.feed("std::vector<tensor*> %s_ins;\n", node_name);
		for(auto &edge_in : node_info.ins) {
			_code.feed("%s_ins.push_back(&%s);", node_name, edge_in);
		}
		_code.feed("std::vector<tensor*> %s_outs;\n", node_name);
		for(auto &edge_out : node_info.outs) {
			_code.feed("%s_outs.push_back(&%s);", node_name, edge_out);
		}
	}
}

void GenCPP::gen_ops() {
	_code<<"// generating model's io \n";
	for(auto & node_name : _exec_node_order) {
		auto& node_info = _graph_node_map[node_name];
		auto& attr_info = _graph[node_name]->attr();
		_code<<OPERATION_MAP[node_info.op_name](attr_info, node_name);
	}
}

void GenCPP::gen_init_impl() {
}

void GenCPP::gen_api_impl() {
}


} /* namespace lite */

} /* namespace anakin */

