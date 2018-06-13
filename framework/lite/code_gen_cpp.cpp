#include "framework/lite/code_gen_cpp.h"

namespace anakin {

namespace lite {

void GenCPP::gen_header_start() {
	_code.Clean();
	_code.feed("#ifndef ANAKIN_%s_H \n", _code_name.c_str());
	_code.feed("#define ANAKIN_%s_H \n\n", _code_name.c_str());
	_code<<"namespace anakin { \n\n";
}	

void GenCPP::gen_header_end() {
	_code<<"} /* namespace anakin */\n";
	_code<<"#endif\n";
}

void GenCPP::gen_source_start() {
	_code.Clean();
	_code.feed("#include \"%s.h\" \n\n", _code_name.c_str());
	_code<<"namespace anakin { \n\n";
	// add running impl for model api
}	

void GenCPP::gen_source_end() {
	_code<<"} /* namespace anakin */\n";
}

void GenCPP::gen_tensors() {
	_code<<"// generating and setting tensors \n";
	for(auto it = _tensor_map.begin(); it != _tensor_map.end(); ++it) {
		auto& edge_name = it->first;
		auto& edge_info = it->second;
		if(! edge_info.is_shared) {
			_code.feed("tensor %s;\n", edge_name.c_str());
			_code.feed("%s.set_valid_shape(%d);\n", edge_name.c_str(), edge_info.valid_shape[0]);
			_code.feed("%s.set_real_shape(%d);\n", edge_name.c_str(), edge_info.real_shape[0]);
		}
	}
	for(auto it = _tensor_map.begin(); it != _tensor_map.end(); ++it) {
		auto& edge_name = it->first;
		auto& edge_info = it->second;
		if(edge_info.is_shared) {
			_code.feed("tensor %s;\n", edge_name.c_str());
			_code.feed("%s.set_valid_shape(%d);\n", edge_name.c_str(), edge_info.valid_shape[0]);
			_code.feed("%s.set_real_shape(%d);\n", edge_name.c_str(), edge_info.real_shape[0]);
			_code.feed("%s.share_from(%s);\n", edge_name.c_str(), edge_info.share_from.c_str());
		}
	}
}

void GenCPP::gen_model_ios() {
	_code<<"// generating model's I/O \n";
	for(auto & node_name : _exec_node_order) {
		auto& node_info = _graph_node_map[node_name];
		_code.feed("std::vector<tensor*> %s_ins;\n", node_name.c_str());
		for(auto &edge_in : node_info.ins) {
			_code.feed("%s_ins.push_back(&%s);\n", node_name.c_str(), edge_in.c_str());
		}
		_code.feed("std::vector<tensor*> %s_outs;\n", node_name.c_str());
		for(auto &edge_out : node_info.outs) {
			_code.feed("%s_outs.push_back(&%s);\n", node_name.c_str(), edge_out.c_str());
		}
	}
}

void GenCPP::gen_and_parse_ops() {
	_code<<"// generating model's operations\n";
	for(auto & node_name : _exec_node_order) {
		if(_graph_node_map[node_name].op_name == "Input" || _graph_node_map[node_name].op_name == "Output") {
			continue;
		}
		auto& node_info = _graph_node_map[node_name];
		auto& attr_info = _graph[node_name]->attr();
		if(OPERATION_MAP.count(node_info.op_name) > 0) {
			_code<<OPERATION_MAP[node_info.op_name].parse(attr_info, node_info.op_name, node_name, _weights);
		} else {
			LOG(ERROR) << "Target op type : " << _graph_node_map[node_name].op_name << " not support";
		}
	}
	return true;
}

void GenCPP::gen_init_impl() {
}

void GenCPP::gen_api_impl() {
}

void GenCPP::gen_header() {
	_code.Clean();
	_code.open(_h_file_name);
	gen_header_start();
	//
    gen_header_end();
	_code.save();	
}

void GenCPP::gen_source() {
	_code.Clean();
	_code.open(_cpp_file_name);
	gen_source_start(); 
	// generate tensors 
	gen_tensors();
	// generate i/o
	gen_model_ios();
	// generate and parse ops
	gen_and_parse_ops();
	gen_source_end();
	_code.save();	
}

} /* namespace lite */

} /* namespace anakin */

