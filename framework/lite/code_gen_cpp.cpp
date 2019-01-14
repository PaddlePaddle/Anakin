#include <fstream>
#include "framework/lite/code_gen_cpp.h"
#include "framework/core/net/calibrator_parse.h"

namespace anakin {

namespace lite {

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_license() {
	_code<< "/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.\n\n   Licensed under the Apache License, Version 2.0 (the \"License\");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an \"AS IS\" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n*/\n\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_header_start() {
	_code.Clean();
	gen_license();
	_code.feed("#ifndef ANAKIN_%s_H \n", _code_name.c_str());
	_code.feed("#define ANAKIN_%s_H \n\n", _code_name.c_str());
    _code<<"#include <saber/lite/core/tensor_op_lite.h>\n";
	_code<<"#include <saber/lite/core/common_lite.h>\n";
    _code<<"#include <saber/lite/core/context_lite.h>\n";
	_code<<"using namespace anakin;\n";
	_code<<"using namespace anakin::saber;\n";
	_code<<"using namespace anakin::saber::lite;\n\n";
    _code<<"namespace anakin { \n\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_header_end() {
	_code<<"} /* namespace anakin */\n";
	_code<<"\n#endif\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_source_start() {
	_code.Clean();
	_code.feed("#include \"%s.h\" \n\n", _code_name.c_str());
    _code<<"#include <saber/lite/funcs/op_param.h>\n";
    _code<<"#include <saber/lite/funcs/op_base.h>\n";
    _code<<"#include <saber/lite/funcs/detection_lite.h>\n";
    _code<<"#include <saber/lite/funcs/saber_activation.h>\n";
    _code<<"#include <saber/lite/funcs/saber_concat.h>\n";
    _code<<"#include <saber/lite/funcs/saber_detection_output.h>\n";
    _code<<"#include <saber/lite/funcs/saber_eltwise.h>\n";
    _code<<"#include <saber/lite/funcs/saber_eltwise_act.h>\n";
    _code<<"#include <saber/lite/funcs/saber_permute.h>\n";
    _code<<"#include <saber/lite/funcs/saber_power.h>\n";
    _code<<"#include <saber/lite/funcs/saber_priorbox.h>\n";
    _code<<"#include <saber/lite/funcs/saber_scale.h>\n";
    _code<<"#include <saber/lite/funcs/saber_slice.h>\n";
    _code<<"#include <saber/lite/funcs/timer_lite.h>\n";
    _code<<"#include <saber/lite/funcs/saber_conv.h>\n";
    _code<<"#include <saber/lite/funcs/saber_deconv.h>\n";
    _code<<"#include <saber/lite/funcs/saber_conv_pooling.h>\n";
    _code<<"#include <saber/lite/funcs/saber_fc.h>\n";
    _code<<"#include <saber/lite/funcs/saber_pooling.h>\n";
    _code<<"#include <saber/lite/funcs/saber_split.h>\n";
    _code<<"#include <saber/lite/funcs/saber_flatten.h>\n";
    _code<<"#include <saber/lite/funcs/saber_reshape.h>\n";
    _code<<"#include <saber/lite/funcs/saber_shuffle_channel.h>\n";
    _code<<"#include <saber/lite/funcs/saber_softmax.h>\n\n";
	_code<<"namespace anakin { \n\n";
	// add running impl for model api
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_source_end() {
	_code<<"} /* namespace anakin */\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_tensors() {
	_code<<"\n// generating tensors \n";
	for(auto it = this->_tensor_map.begin(); it != this->_tensor_map.end(); ++it) {
		auto& edge_name = it->first;
		auto& edge_info = it->second;
		if(! edge_info.is_shared) {
			_code.feed("Tensor<CPU, AK_FLOAT> %s_%s;\n", _code_name.c_str(), edge_name.c_str());
			_code.feed("Shape %s_%s_real_shape(%d,%d,%d,%d);\n", _code_name.c_str(),
                       edge_name.c_str(),
                       edge_info.real_shape[0],
                       edge_info.real_shape[1],
                       edge_info.real_shape[2],
                       edge_info.real_shape[3]);
			_code.feed("Shape %s_%s_valid_shape(%d,%d,%d,%d);\n", _code_name.c_str(),
                       edge_name.c_str(),
                       edge_info.valid_shape[0],
                       edge_info.valid_shape[1],
                       edge_info.valid_shape[2],
                       edge_info.valid_shape[3]);
		}
	}
	for(auto it = this->_tensor_map.begin(); it != this->_tensor_map.end(); ++it) {
		auto& edge_name = it->first;
		auto& edge_info = it->second;
		if(edge_info.is_shared) {
			_code.feed("Tensor<CPU, AK_FLOAT> %s_%s;\n", _code_name.c_str(), edge_name.c_str());
			_code.feed("Shape %s_%s_valid_shape(%d,%d,%d,%d);\n", _code_name.c_str(),
                       edge_name.c_str(),
                       edge_info.valid_shape[0],
                       edge_info.valid_shape[1],
                       edge_info.valid_shape[2],
                       edge_info.valid_shape[3]);
		}
	}
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::tensors_init() {
	_code<<"\n// initialize tensors \n";
	_code.feed("void %s_tensors_init() {\n", _code_name.c_str());
	for(auto it = this->_tensor_map.begin(); it != this->_tensor_map.end(); ++it) {
		auto& edge_name = it->first;
		auto& edge_info = it->second;
		if(! edge_info.is_shared) {
			_code.feed("    %s_%s.re_alloc(%s_%s_real_shape);\n", _code_name.c_str(), edge_name.c_str(), _code_name.c_str(), edge_name.c_str());
			_code.feed("    %s_%s.set_shape(%s_%s_valid_shape);\n", _code_name.c_str(), edge_name.c_str(), _code_name.c_str(), edge_name.c_str());
		}
	}
	for(auto it = this->_tensor_map.begin(); it != this->_tensor_map.end(); ++it) {
		auto& edge_name = it->first;
		auto& edge_info = it->second;
		if(edge_info.is_shared) {
			_code.feed("    %s_%s.set_shape(%s_%s_valid_shape);\n", _code_name.c_str(), edge_name.c_str(), _code_name.c_str(), edge_name.c_str());
			_code.feed("    %s_%s.share_from(%s_%s);\n", _code_name.c_str(), edge_name.c_str(), _code_name.c_str(), edge_info.share_from.c_str());
		}
	}
	_code<<"}\n";

}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_model_ios() {
	_code<<"\n// generating model's I/O \n";
    _code.feed("std::vector<std::vector<Tensor<CPU, AK_FLOAT>*>> %s_tensor_ins;\n", _code_name.c_str());
    _code.feed("std::vector<std::vector<Tensor<CPU, AK_FLOAT>*>> %s_tensor_outs;\n", _code_name.c_str());
//	for(auto & node_name : this->_exec_node_order) {
//		auto& node_info = this->_graph_node_map[node_name];
//		_code.feed("std::vector<Tensor<CPU, AK_FLOAT>*> %s_ins;\n", node_name.c_str());
//		_code.feed("std::vector<Tensor<CPU, AK_FLOAT>*> %s_outs;\n", node_name.c_str());
//	}
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::model_ios_init() {
	_code<<"\n// initialize model's I/O \n";
    _code.feed("void %s_model_ios_init() {\n", _code_name.c_str());
    _code.feed("    %s_tensor_ins.resize(%d);\n", _code_name.c_str(), this->_exec_node_order.size());
    _code.feed("    %s_tensor_outs.resize(%d);\n", _code_name.c_str(), this->_exec_node_order.size());
    _code.feed("    for(int i = 0; i < %d; i++) {\n", this->_exec_node_order.size());
    _code.feed("        %s_tensor_ins[i].clear();\n", _code_name.c_str());
    _code.feed("        %s_tensor_outs[i].clear();\n", _code_name.c_str());
    _code.feed("    }\n");
    _code.feed("    int i = 0;\n");
    for(auto & node_name : this->_exec_node_order) {
        if(this->_graph_node_map[node_name].op_name == "Input" || this->_graph_node_map[node_name].op_name == "Output") {
            continue;
        }
        auto& node_info = this->_graph_node_map[node_name];
        for(auto &edge_in : node_info.ins) {
            _code.feed("    %s_tensor_ins[i].push_back(&%s_%s);\n", _code_name.c_str(), _code_name.c_str(), edge_in.c_str());
        }
        for(auto &edge_out : node_info.outs) {
            _code.feed("    %s_tensor_outs[i].push_back(&%s_%s);\n", _code_name.c_str(), _code_name.c_str(), edge_out.c_str());
        }
        _code.feed("    i++;\n");
    }
	_code<<"}\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_ops() {
	_code<<"\n// generating model's operations\n";
    _code<<"\n// create vector of ops\n";
    _code.feed("std::vector<OpBase*> %s_g_ops;\n", _code_name.c_str());
    _code.feed("void %s_gen_ops() {\n", _code_name.c_str());
    _code.feed("    if (%s_g_ops.size() > 0) {\n", _code_name.c_str());
    _code.feed("        return;\n");
    _code.feed("    }\n");
	for(auto & node_name : this->_exec_node_order) {
		if(this->_graph_node_map[node_name].op_name == "Input" || this->_graph_node_map[node_name].op_name == "Output") {
			continue;
		}
		auto& node_info = this->_graph_node_map[node_name];
		if(OPERATION_MAP.count(node_info.op_name) > 0) {
			_code.feed("    OpBase* %s = new %s; \n", node_name.c_str(), OPERATION_MAP[node_info.op_name].OpClassName.c_str());
            _code.feed("#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG) \n");
            _code.feed("    %s->set_op_name(\"%s\"); \n", node_name.c_str(), node_name.c_str());
            _code.feed("#endif \n");
            _code.feed("    %s_g_ops.push_back(%s);\n", _code_name.c_str(), node_name.c_str());
		}
	}
    _code << "}\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_init_impl() {
	_code<<"// initial function for model.\n";
	_code.feed("bool %s_init(Context& ctx) {\n", _code_name.c_str());
    _code.feed("    bool flag = false;\n");
    _code.feed("    for (int i = 0; i < %s_g_ops.size(); i++) {\n", _code_name.c_str());
    _code.feed("        %s_g_ops[i]->compute_output_shape(%s_tensor_ins[i], %s_tensor_outs[i]);\n", _code_name.c_str(), _code_name.c_str(), _code_name.c_str());
    _code.feed("        flag = %s_g_ops[i]->init(%s_tensor_ins[i], %s_tensor_outs[i], ctx);\n", _code_name.c_str(), _code_name.c_str(), _code_name.c_str());
    _code.feed("        if (!flag) {\n");
    _code.feed("#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG) \n");
    _code.feed("            printf(\"%s op init failed;\\n\", %s_g_ops[i]->get_op_name());\n", "%s", _code_name.c_str());
    _code.feed("#endif \n");
    _code.feed("            return false;\n");
    _code.feed("        }\n");
    _code << "    }\n";
//	for(auto & node_name : this->_exec_node_order) {
//		if(this->_graph_node_map[node_name].op_name == "Input" || this->_graph_node_map[node_name].op_name == "Output") {
//			continue;
//		}
//		auto& node_info = this->_graph_node_map[node_name];
//		if(OPERATION_MAP.count(node_info.op_name) > 0) {
//			_code.feed("    %s.compute_output_shape(%s_ins,%s_outs); \n", node_name.c_str(),
//																	  	  node_name.c_str(),
//																		  node_name.c_str());
//			_code.feed("    %s.init(%s_ins,%s_outs,ctx); \n", node_name.c_str(),
//														  	  node_name.c_str(),
//															  node_name.c_str());
//		}
//	}
    _code << "    return true;\n";
	_code << "}\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_run_impl(const bool debug_mode) {
	_code << "// Running prediction for model. \n";
	_code.feed("bool %s_prediction() {\n", _code_name.c_str());
    _code.feed("    bool flag = false;\n");
    _code.feed("    for (int i = 0; i < %s_g_ops.size(); i++) {\n", _code_name.c_str());
    _code.feed("        flag = %s_g_ops[i]->dispatch(%s_tensor_ins[i], %s_tensor_outs[i]);\n", _code_name.c_str(), _code_name.c_str(), _code_name.c_str());
    _code.feed("        if (!flag) {\n");
    _code.feed("#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG) \n");
    _code.feed("            printf(\"%s op dispatch failed;\\n\", %s_g_ops[i]->get_op_name());\n", "%s", _code_name.c_str());
    _code.feed("#endif \n");
    _code.feed("            return false;\n");
    _code.feed("        }\n");
    if (debug_mode) {
        _code.feed("        for(int j = 0; j < %s_tensor_outs[i].size(); j++) {\n", _code_name.c_str());
        _code.feed("            double mean_val = tensor_mean(*%s_tensor_outs[i][0]); \n", _code_name.c_str());
        _code.feed("#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG) \n");
        _code.feed("            printf(\"mean_val in %s ops: %s \\n\", %s_g_ops[i]->get_op_name(), mean_val);\n", "%s", "%.6f", _code_name.c_str());
        _code.feed("#else \n");
        _code.feed("            printf(\"mean_val in ops: %s \\n\", mean_val);\n", "%.6f");
        _code.feed("#endif \n");
        _code.feed("        }\n");
    }
    _code << "    }\n";

//	for(auto & node_name : this->_exec_node_order) {
//		if(this->_graph_node_map[node_name].op_name == "Input" || this->_graph_node_map[node_name].op_name == "Output") {
//			continue;
//		}
//		auto& node_info = this->_graph_node_map[node_name];
//		if(OPERATION_MAP.count(node_info.op_name) > 0) {
//            /*
//			_code.feed("    %s.compute_output_shape(%s_ins,%s_outs); \n", node_name.c_str(),
//																	  	  node_name.c_str(),
//																		  node_name.c_str());
//																		  */
//			_code.feed("    %s.dispatch(%s_ins,%s_outs); \n", node_name.c_str(),
//														  	  node_name.c_str(),
//															  node_name.c_str());
//            if (debug_mode) {
//                _code.feed("    double mean_%s = tensor_mean(*%s_outs[0]); \n", node_name.c_str(), node_name.c_str());
//                _code.feed("    printf(\"%s run mean_val: %s %s\", mean_%s);\n", node_name.c_str(), "%.6f", "\\n", node_name.c_str());
//            }
//		}
//	}
    _code << "    return true;\n";
	_code << "}\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_head_api() {
	// gen gloss for graph ins
	_code << "/// Model "<< _code_name << " have  " << this->_ins.size() << " inputs.\n";
	for(auto in : this->_ins) {
		auto& node_info = this->_graph_node_map[in];
		auto& edge_info = this->_tensor_map[node_info.outs[0]];
		_code << "///  |-- input name : " << in << "  -- Shape(";
		std::string shape_str;
		for(int i=0; i<edge_info.valid_shape.size() - 1; i++) {
			_code << edge_info.valid_shape[i] << ",";
		}
		if(edge_info.valid_shape.size() > 0) {
			_code << edge_info.valid_shape[edge_info.valid_shape.size() - 1] << ")\n";
		} else {
			_code << ")\n";
		}
	}

	// gen api for getting graph input tensor
	_code.feed("LITE_EXPORT std::vector<Tensor<CPU, AK_FLOAT>*> %s_get_in();\n\n", _code_name.c_str());

	// gen gloss for graph outs
	_code << "/// Model " << _code_name << " have  " << this->_outs.size() << " outputs.\n";
	for(auto out : this->_outs) {
		auto& node_info = this->_graph_node_map[out];
		auto& edge_info = this->_tensor_map[node_info.ins[0]];
		_code << "///  |-- output name : " << out << "  -- Shape(";
		for(int i=0; i<edge_info.valid_shape.size() - 1; i++) {
			_code << edge_info.valid_shape[i] << ",";
		}
		if(edge_info.valid_shape.size() > 0) {
			_code << edge_info.valid_shape[edge_info.valid_shape.size() - 1] << ")\n";
		} else {
			_code << ")\n";
		}
	}
	// gen api for getting graph output tensor

	_code.feed("LITE_EXPORT std::vector<Tensor<CPU, AK_FLOAT>*> %s_get_out();\n\n", _code_name.c_str());

	// gen weights loading function
	_code.feed("LITE_EXPORT bool %s_load_param(const char* param_path);\n\n", _code_name.c_str());

    // gen weights loading function from memory
    _code.feed("LITE_EXPORT bool %s_load_weights(const void* weights);\n\n", _code_name.c_str());

	// gen api for model init
	_code.feed("/// %s_init should only be invoked once when input shape changes.\n", _code_name.c_str());
	_code.feed("LITE_EXPORT bool %s_init(Context& ctx);\n\n", _code_name.c_str());

	// gen api for model prediction
	_code.feed("/// Running prediction for model %s.\n", _code_name.c_str());
	_code.feed("LITE_EXPORT bool %s_prediction();\n\n", _code_name.c_str());

	// gen free function
	_code.feed("/// Release all resource used by model %s.\n", _code_name.c_str());
	_code.feed("LITE_EXPORT void %s_release_resource();\n\n", _code_name.c_str());

}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_head_api_impl() {
	// gen api for getting graph input tensor
	_code << "\n// gen api for getting graph input tensor \n";
	_code.feed("std::vector<Tensor<CPU, AK_FLOAT>*> %s_get_in() {\n", _code_name.c_str());
    _code.feed("    std::vector<Tensor<CPU, AK_FLOAT>*> vin;\n", this->_ins[0].c_str());
    for(int i = 0; i < this->_ins.size(); i++) {
        auto node_info = this->_graph_node_map[this->_ins[i]];
        auto edge_info = this->_tensor_map[node_info.outs[0]];
        _code.feed("    vin.push_back(&%s_%s);\n", _code_name.c_str(), edge_info.name.c_str());
    }
    _code.feed("    return vin;\n");

//	_code.feed("    if(strcmp(in_name, \"%s\") == 0) {\n", this->_ins[0].c_str());
//	auto node_info = this->_graph_node_map[this->_ins[0]];
//	auto edge_info = this->_tensor_map[node_info.outs[0]];
//	_code.feed("        return &%s;\n    }", edge_info.name.c_str());
//	for(int i = 1; i < this->_ins.size(); i++) {
//		node_info = this->_graph_node_map[this->_ins[i]];
//		edge_info = this->_tensor_map[node_info.outs[0]];
//		_code.feed(" else if(strcmp(in_name, \"%s\") == 0) {\n", this->_ins[i].c_str());
//		_code.feed("        return &%s;\n    }\n", edge_info.name.c_str());
//	}
//	_code <<" else {\n        return nullptr;\n    }\n";
	_code <<"}\n";

	// gen api for getting graph output tensor
	_code << "\n// gen api for getting graph output tensor \n";
	_code.feed("std::vector<Tensor<CPU, AK_FLOAT>*> %s_get_out() {\n", _code_name.c_str());
    _code.feed("    std::vector<Tensor<CPU, AK_FLOAT>*> vout;\n");
    for(int i = 0; i < this->_outs.size(); i++) {
        auto node_info = this->_graph_node_map[this->_outs[i]];
        auto edge_info = this->_tensor_map[node_info.ins[0]];
        _code.feed("    vout.push_back(&%s_%s);\n", _code_name.c_str(), edge_info.name.c_str());
    }
    _code.feed("    return vout;\n");

//	_code.feed("    if(strcmp(out_name, \"%s\") == 0) {\n", this->_outs[0].c_str());
//	node_info = this->_graph_node_map[this->_outs[0]];
//	edge_info = this->_tensor_map[node_info.ins[0]];
//	_code.feed("        return &%s;\n    }", edge_info.name.c_str());
//	for(int i = 1; i < this->_outs.size(); i++) {
//		node_info = this->_graph_node_map[this->_outs[i]];
//		edge_info = this->_tensor_map[node_info.ins[0]];
//		_code.feed(" else if(strcmp(out_name ,\"%s\") == 0) {\n", this->_outs[i].c_str());
//		_code.feed("        return &%s;\n    }\n", edge_info.name.c_str());
//	}
//	_code <<" else {\n        return nullptr;\n    }\n";
	_code <<"}\n\n";

	// gen weights loading function
	_code.feed("float *%s = nullptr; // global weights start pointer \n", _g_weights_ptr_name.c_str());
    _code.feed("std::vector<ParamBase*> %s_g_param; // global vector of param \n", _code_name.c_str());

    _code.feed("bool %s_load_param(const char* param_path) {\n", _code_name.c_str());
    _code <<   "    FILE *f = fopen(param_path, \"rb\"); \n";
    _code <<   "    if(!f) {\n";
    _code <<   "        return false;\n    }\n";
    _code <<   "    fseek(f, 0, SEEK_END);\n";
    _code <<   "    long fsize = ftell(f);\n";
    _code <<   "    fseek(f, 0, SEEK_SET);\n";
    _code.feed("    if(%s) {\n", _g_weights_ptr_name.c_str());
    _code.feed("        delete [] %s;\n", _g_weights_ptr_name.c_str());
    _code.feed("        %s = nullptr;\n", _g_weights_ptr_name.c_str());
    _code.feed("    }\n");
    _code.feed("    %s = new float[fsize + 1];\n", _g_weights_ptr_name.c_str());
    _code.feed("    fread(%s, fsize, sizeof(float), f);\n", _g_weights_ptr_name.c_str());
    _code <<   "    fclose(f);\n";
    _code.feed("    %s_load_weights((const void*)%s);\n", _code_name.c_str(), _g_weights_ptr_name.c_str());
    _code << "}";

	_code.feed("bool %s_load_weights(const void* weights) {\n", _code_name.c_str());
    _code.feed("    if (weights == nullptr) {\n"); // invoke (model_name)_tensors_init()
    _code.feed("        return false;\n"); // invoke (model_name)_tensors_init()
    _code.feed("    }\n"); // invoke (model_name)_tensors_init()
	_code.feed("    %s_tensors_init();\n", _code_name.c_str()); // invoke (model_name)_tensors_init()
	_code.feed("    %s_model_ios_init();\n", _code_name.c_str()); // invoke (model_name)_model_ios_init()
    _code.feed("    for (int i = 0; i < %s_g_param.size(); i++) {\n", _code_name.c_str());
    _code.feed("        if (%s_g_param[i]) {\n", _code_name.c_str());
    _code.feed("            delete %s_g_param[i];\n", _code_name.c_str());
    _code.feed("        }\n");
    _code.feed("        %s_g_param[i] = nullptr;\n", _code_name.c_str());
    _code.feed("    }\n");
    _code.feed("    %s_g_param.clear();\n", _code_name.c_str());
    _code.feed("    const char* weights_ptr = (const char*)weights;\n");
    std::string local_weight_string = "weights_ptr";

	for(auto & node_name : this->_exec_node_order) {
		if(this->_graph_node_map[node_name].op_name == "Input" || this->_graph_node_map[node_name].op_name == "Output") {
			continue;
		}

		auto& node_info = this->_graph_node_map[node_name];
		auto& attr_info = this->_graph[node_name]->attr();
		if(OPERATION_MAP.count(node_info.op_name) > 0) {
			LOG(INFO) << "node name: " << node_name;
			LOG(INFO) << "Target op type : " << this->_graph_node_map[node_name].op_name << " parsing ...";
			if (this->_graph[node_name]->bit_type() == AK_INVALID){
			    this->_graph[node_name]->set_bit_type(AK_FLOAT);
			}
			auto str = OPERATION_MAP[node_info.op_name].parse(attr_info, _code_name,
                                                              OPERATION_MAP[node_info.op_name].OpClassName,
															  node_name,
															  local_weight_string,
															  _weights,
                                                              false,
                                                              _lite_mode,
                                                              this->_graph[node_name]->bit_type());
			if(!str.empty()) {
				_code.feed("    %s", str.c_str());
			}
		} else {
			LOG(FATAL) << "Target op type : " << this->_graph_node_map[node_name].op_name << " not support";
		}
	}
    _code.feed("    %s_gen_ops();\n", _code_name.c_str());
    _code.feed("    for (int i = 0; i < %s_g_ops.size(); i++) {\n", _code_name.c_str());
    _code.feed("        SaberStatus state = %s_g_ops[i]->load_param(%s_g_param[i]);\n", _code_name.c_str(), _code_name.c_str());
    _code.feed("        if (state != SaberSuccess) { \n");
    _code.feed("            printf(\"load param failed\\n\");\n");
    _code.feed("        }\n");
    _code.feed("    }\n");

	_code << "    return true;\n";
	_code <<"}\n\n";

	// release all resource function impl
	_code.feed("void %s_release_resource() {\n", _code_name.c_str());
    _code.feed("    for (int i = 0; i < %s_g_ops.size(); i++) {\n", _code_name.c_str());
    _code.feed("        if (%s_g_ops[i]) {\n", _code_name.c_str());
    _code.feed("            delete %s_g_ops[i];\n", _code_name.c_str());
    _code.feed("            %s_g_ops[i] = nullptr;\n", _code_name.c_str());
    _code.feed("        }\n");
    _code.feed("    }\n");
    _code.feed("    %s_g_ops.clear();\n", _code_name.c_str());
    _code.feed("    for (int i = 0; i < %s_g_param.size(); i++) {\n", _code_name.c_str());
    _code.feed("        if (%s_g_param[i]) {\n", _code_name.c_str());
    _code.feed("            delete %s_g_param[i];\n", _code_name.c_str());
    _code.feed("            %s_g_param[i] = nullptr;\n", _code_name.c_str());
    _code.feed("        }\n");
    _code.feed("    }\n");
    _code.feed("    %s_g_param.clear();\n", _code_name.c_str());
    _code.feed("    if (%s) {\n", _g_weights_ptr_name.c_str());
	_code.feed("        delete [] %s;\n", _g_weights_ptr_name.c_str());
	_code.feed("        %s = nullptr;\n", _g_weights_ptr_name.c_str());
    _code.feed("    }\n", _g_weights_ptr_name.c_str());
	_code <<"}\n\n";
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_header() {
	_code.Clean();
	_code.open(_h_file_name);
	gen_header_start();
	// gen api
	gen_head_api();
	gen_header_end();
	_code.save();
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_source(const bool debug_mode) {
	_code.Clean();
	_code.open(_cpp_file_name);
	gen_source_start();
	// generate tensors
	gen_tensors();
	// tensors init
	tensors_init();
	// generate i/o
	gen_model_ios();
	// initial model i/o
	model_ios_init();
	// generate ops
	gen_ops();
	// gen head api implement
	gen_head_api_impl();
	// gen initial api impl
	gen_init_impl();
	// gen running api impl
	gen_run_impl(debug_mode);
	gen_source_end();
	_code.save();
    gen_opt_model();
    if (!_flag_aot) {
        gen_merge_model();
    }
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_opt_model() {

	//parse config file
	bool flag_precision = false;
	bool flag_calibrator = false;
	int flag_lite_mode = _lite_mode;
	CalibratorParser parser;
	if (_precision_path == ""){
		flag_precision = false;
	}else {
		parser.parse_from_file(_precision_path, "");
		flag_precision = true;
	}

	if (_calibrator_path == ""){
		flag_calibrator = false;
	}else {
		parser.parse_from_file("", _calibrator_path);
		flag_calibrator = true;
	}
	auto get_op_precision = [&](NodeInfo& node_info)->std::string{
		if (flag_precision){
		    std::string op_precision = parser.get_precision(node_info.name);
		    if (op_precision == "fp32"){
		        node_info.dtype = AK_FLOAT;
		    } else if (op_precision == "int8"){
		        node_info.dtype = AK_INT8;
		    } else{
		        node_info.dtype = AK_FLOAT;
		    }
			return op_precision;
		} else {
			auto dtype = node_info.dtype;
			if (dtype == AK_FLOAT){
				return "fp32";
			} else if (dtype == AK_INT8){
				return "int8";
			} else {
				//LOG(FATAL) << "unsupport precision type";
                node_info.dtype = AK_FLOAT;
				return "fp32";
			}
		}
	};
	auto get_tensor_precision = [&](EdgeInfo& edge_info)->std::string{
		if (flag_precision){
			auto dtype = parser.get_dtype(edge_info.in_node, edge_info.out_node);
			if (dtype == AK_FLOAT){
				return "fp32";
			} else if (dtype == AK_INT8) {
				return "int8";
			} else {
				//LOG(FATAL) << "unsupport precision type";
				return "fp32";
			}
		} else {
			auto dtype = edge_info.dtype;
			if (dtype == AK_FLOAT){
				return "fp32";
			} else if (dtype == AK_INT8) {
				return "int8";
			} else {
				//LOG(FATAL) << "unsupport precision type";
                edge_info.dtype = AK_FLOAT;
				return "fp32";
			}
		}
	};

	auto get_tensor_calibrator = [&](EdgeInfo& edge_info)->float{
		if (flag_calibrator){
			auto calibrator_scale = parser.get_calibrator(edge_info.name);
			return calibrator_scale;
		} else {
			std::vector<float> calibrator_scale = edge_info.scale;
			if (calibrator_scale.size() == 0){
				return 1.f;
			} else {
				return calibrator_scale[0];
			}
		}
	};

	//!generate Version Number
	int version_num = MAJOR * 100 + MINOR * 10 + REVISION;
	_opt_param_write << "Version: " << version_num << "\n";
    //! generate Tensors
    LOG(INFO) << "gen opt model tensors";
    _opt_param_write << "Tensor_number " << this->_tensor_map.size() << "\n";
    //! firstly, gen tensor withnot shared
    for(auto it = this->_tensor_map.begin(); it != this->_tensor_map.end(); ++it) {
        auto& edge_name = it->first;
        auto& edge_info = it->second;
        if(! edge_info.is_shared) {
            //tensor info format: tensor_name tensor_precision valid_shape real_shape is_shared shared_tensor_name
            _opt_param_write << edge_name << " ";
            //tensor precision info
            auto t_precision = get_tensor_precision(edge_info);
            _opt_param_write << t_precision << " ";
 			//tensor calibrator info
 			auto t_calibrator = get_tensor_calibrator(edge_info);
 			_opt_param_write << t_calibrator << " ";
            //tensor valid shape
            _opt_param_write << edge_info.valid_shape.size() << " ";
            for (int i = 0; i < edge_info.valid_shape.size(); ++i) {
                _opt_param_write << edge_info.valid_shape[i] << " ";
            }
            //tensor shape
            _opt_param_write << edge_info.real_shape.size() << " ";
            for (int i = 0; i < edge_info.real_shape.size(); ++i) {
                _opt_param_write << edge_info.real_shape[i] << " ";
            }
            _opt_param_write << 0 << " " << "null" << "\n";
        }
    }
    //! then gen tensor shared memory
    for(auto it = this->_tensor_map.begin(); it != this->_tensor_map.end(); ++it) {
        auto& edge_name = it->first;
        auto& edge_info = it->second;
        if(edge_info.is_shared) {
            //tensor info format: tensor_name valid_shape real_shape is_shared shared_tensor_name

            _opt_param_write << edge_name << " ";

            //tensor precision info
            auto t_precision = get_tensor_precision(edge_info);
            _opt_param_write << t_precision << " ";
            //tensor calibrator info
 			auto t_calibrator = get_tensor_calibrator(edge_info);
 			_opt_param_write << t_calibrator << " ";
            //tensor valid shape
            _opt_param_write << edge_info.valid_shape.size() << " ";
            for (int i = 0; i < edge_info.valid_shape.size(); ++i) {
                _opt_param_write << edge_info.valid_shape[i] << " ";
            }
            //tensor shape
            _opt_param_write << edge_info.valid_shape.size() << " ";
            for (int i = 0; i < edge_info.valid_shape.size(); ++i) {
                _opt_param_write << edge_info.valid_shape[i] << " ";
            }
            _opt_param_write << 1 << " " << edge_info.share_from << "\n";
        }
    }
    //! gen inputs and outputs tensor name and precision
    _opt_param_write << "inputs " << this->_ins.size();
    for(auto in : this->_ins) {
        auto node_info = this->_graph_node_map[in];
        auto edge_info = this->_tensor_map[node_info.outs[0]];
        _opt_param_write << " " << edge_info.name;
        _opt_param_write << " " << "fp32";
    }
    _opt_param_write << "\n";

    //! gen outputs and outputs tensor name and precision
    _opt_param_write << "outputs " << this->_outs.size();
    for(auto out : this->_outs) {
        auto node_info = this->_graph_node_map[out];
        auto edge_info = this->_tensor_map[node_info.ins[0]];
        _opt_param_write << " " << edge_info.name;
        _opt_param_write << " " << "fp32";
    }
    _opt_param_write << "\n";

    //! gen ops and params
    int op_num = this->_exec_node_order.size();
    for(auto & node_name : this->_exec_node_order) {
        if (this->_graph_node_map[node_name].op_name == "Input" ||
            this->_graph_node_map[node_name].op_name == "Output") {
            op_num--;
        }
    }
    _opt_param_write << "OPS " << op_num << "\n";
    for(auto & node_name : this->_exec_node_order) {
        if(this->_graph_node_map[node_name].op_name == "Input" || this->_graph_node_map[node_name].op_name == "Output") {
            continue;
        }
        auto& node_info = this->_graph_node_map[node_name];
        auto& attr_info = this->_graph[node_name]->attr();
        if (OPERATION_MAP.count(node_info.op_name) > 0) {
            LOG(INFO) << "node name: " << node_name;
            LOG(INFO) << "Target op type : " << this->_graph_node_map[node_name].op_name << " parsing ...";
            _opt_param_write << OPERATION_MAP[node_info.op_name].OpClassName << " " << node_name << " ";
        	_opt_param_write << get_op_precision(node_info) << " ";
            _opt_param_write << node_info.ins.size() << " ";
            _opt_param_write << node_info.outs.size() << " ";
            for(auto &edge_in : node_info.ins) {
                _opt_param_write << edge_in << " ";
                // auto edge_in_name = this->_tensor_map[edge_in].in_node;
                // auto edge_out_name = this->_tensor_map[edge_in].out_node;
                // auto t_precision = get_tensor_precision(edge_in_name, edge_out_name);
                // _opt_param_write << t_precision << " ";
            }
            for(auto &edge_out : node_info.outs) {
                _opt_param_write << edge_out.c_str() << " ";
                // auto edge_in_name = this->_tensor_map[edge_out].in_node;
                // auto edge_out_name = this->_tensor_map[edge_out].out_node;
                // auto t_precision = get_tensor_precision(edge_in_name, edge_out_name);
                // _opt_param_write << t_precision << " ";
            }
            std::string local_weighs_string = "null";
            auto str = OPERATION_MAP[node_info.op_name].parse(attr_info, _code_name,
                                                              OPERATION_MAP[node_info.op_name].OpClassName,
                                                              node_name,
                                                              local_weighs_string,
                                                              _opt_weights,
                                                              true,
                                                              flag_lite_mode,
                                                              node_info.dtype);
            _opt_param_write << str;
        } else {
            LOG(FATAL) << "Target op type : " << this->_graph_node_map[node_name].op_name << " not support";
        }
    }

    _opt_param_write.save();
}

template<typename Ttype, Precision Ptype>
void GenCPP<Ttype, Ptype>::gen_merge_model() {
    FILE* fp_merge = fopen(_merge_opt_file.c_str(), "wb");
    FILE* fp_weight = fopen(_model_file_name.c_str(), "rb");
    FILE* fp_info = fopen(_model_opt_file_name.c_str(), "rb");
    fseek(fp_weight, 0, SEEK_END);
    long wsize = ftell(fp_weight);
    fseek(fp_weight, 0, SEEK_SET);
    char* wbuffer = new char[wsize + 1];
    fread(wbuffer, wsize, 1, fp_weight);

    fseek(fp_info, 0, SEEK_END);
    long isize = ftell(fp_info);
    fseek(fp_info, 0, SEEK_SET);
    char* ibuffer = new char[isize + 1];
    fread(ibuffer, isize, 1, fp_info);

    fprintf(fp_merge, "Wsize %lu\n", wsize);
    fwrite(wbuffer, wsize, 1, fp_merge);

    fwrite(ibuffer, isize, 1, fp_merge);

    fflush(fp_merge);
    fclose(fp_merge);

    fclose(fp_weight);
    fclose(fp_info);

    delete [] wbuffer;
    delete [] ibuffer;
}

#ifdef USE_CUDA
template class GenCPP<NV, Precision::FP32>;
template class GenCPP<NV, Precision::FP16>;
template class GenCPP<NV, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class GenCPP<X86, Precision::FP32>;
template class GenCPP<X86, Precision::FP16>;
template class GenCPP<X86, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class GenCPP<ARM, Precision::FP32>;
template class GenCPP<ARM, Precision::FP16>;
template class GenCPP<ARM, Precision::INT8>;
#endif

template class GenCPP<X86, Precision::FP32>;

} /* namespace lite */

} /* namespace anakin */

