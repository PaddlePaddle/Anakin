#include "framework/lite/op_map.h"

namespace anakin {

namespace lite {

std::string not_impl_yet(graph::AttrInfo&, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	LOG(ERROR) << "Parsing not impl yet. continue ...";
	return "";
}

// SaberConv2D
std::string ParserConvolution(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<float, ARM>>("weight_1", attr);
	writter.register_weights(node_name, weights);
	if(bias_term) {
		auto bias = get_attr<PBlock<float, ARM>>("weight_2", attr);
		writter.register_weights(node_name, bias);
	}

	// gen cpp code
	CodeWritter code_w;
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%d,%d);\n", node_name.c_str(), 
										   group,
										   bias_term,
										   padding[0],
										   padding[1],
										   kernel_size[0],
										   kernel_size[1],
										   strides[0],
										   strides[1],
										   dilation_rate[0],
										   dilation_rate[1]);
	return code_w.get_code_string();
}

// ParserConvolutionRelu
std::string ParserConvolutionRelu(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<float, ARM>>("weight_1", attr);
	writter.register_weights(node_name, weights);
	if(bias_term) {
		auto bias = get_attr<PBlock<float, ARM>>("weight_2", attr);
		writter.register_weights(node_name, bias);
	}

	// gen cpp code
	CodeWritter code_w;
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%d,%d, Active_relu);\n", node_name.c_str(), 
										   group,
										   bias_term,
										   padding[0],
										   padding[1],
										   kernel_size[0],
										   kernel_size[1],
										   strides[0],
										   strides[1],
										   dilation_rate[0],
										   dilation_rate[1]);
	return code_w.get_code_string();
}

// SaberConcat
std::string ParserConcat(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
	auto axis = get_attr<int>("axis", attr);
	// gen cpp code
	CodeWritter code_w;
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%d);\n", node_name.c_str(), axis);
	return code_w.get_code_string();
}

// SaberDectionOutput
std::string ParserDectionOutput(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
    auto flag_share_location = get_attr<bool>("share_location", attr);
    auto flag_var_in_target  = get_attr<bool>("variance_encode_in_target", attr);
    auto classes_num         = get_attr<int>("class_num", attr);
    auto background_id       = get_attr<int>("background_id", attr);
    auto keep_top_k          = get_attr<int>("keep_top_k", attr);
    auto code_type           = get_attr<std::string>("code_type", attr);
    auto conf_thresh         = get_attr<float>("conf_thresh", attr);
    auto nms_top_k           = get_attr<int>("nms_top_k", attr);
    auto nms_thresh          = get_attr<float>("nms_thresh", attr);
    auto nms_eta             = get_attr<float>("nms_eta", attr);

    // gen cpp code
	CodeWritter code_w; 
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str()); 
	code_w.feed("%s.load_param(%s,%s,%d,%d,%d,%s,%f,%d,%f,%f);\n", node_name.c_str(), 
										 flag_share_location ? "true":"false",
										 flag_var_in_target ? "true":"false",
										 classes_num,
										 background_id,
										 keep_top_k,
										 code_type,
										 conf_thresh,
										 nms_top_k,
										 nms_thresh,
										 nms_eta); 
	return code_w.get_code_string();
}

// SaberEltwise
std::string ParserEltwise(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
    auto type = get_attr<std::string>("type", attr); 
    auto coeff = get_attr<PTuple<float>>("coeff", attr);

	CodeWritter coeff_vec_code;
	coeff_vec_code<<"{";
	for(int i=0; i<coeff.vector().size()-1; i++) {
		coeff_vec_code<<coeff.vector()[i]<<",";
	}
	coeff_vec_code<<coeff.vector()[coeff.vector().size()-1] << "}";

	// gen cpp code
	CodeWritter code_w; 
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%s, %s);\n", node_name.c_str(), type, coeff_vec_code.get_code_string());
	return code_w.get_code_string();
}

// SaberActivation
std::string ParserActivation(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
	auto type = get_attr<std::string>("type", attr);

	std::string act_type("Active_unknow");

	if (type == "TanH") { 
		act_type = "Active_tanh";
	} else if (type == "Sigmoid") { 
		act_type = "Active_sigmoid";
	} else { 
		LOG(FATAL) << "Other Activation type" << type << " unknown."; 
	}	

	// gen cpp code
	CodeWritter code_w;
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%s);\n", node_name.c_str(), act_type);
	return code_w.get_code_string();
}

// SaberFc
std::string ParserFc(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
    auto axis = get_attr<int>("axis", attr); 
    auto out_dim = get_attr<int>("out_dim", attr); 
    auto bias_term = get_attr<bool>("bias_term", attr);
	
	auto weights = get_attr<PBlock<float, ARM>>("weight_1", attr);
	writter.register_weights(node_name, weights);
	if(bias_term) {
		auto bias = get_attr<PBlock<float, ARM>>("weight_2", attr);
		writter.register_weights(node_name, bias);
	}

	// gen cpp code
	CodeWritter code_w;
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%d, %d, %s);\n", node_name.c_str(), axis, out_dim,
												bias_term ? "true":"false");
	return code_w.get_code_string();
}

// SaberPermute
std::string ParserPermute(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
	auto dims = get_attr<PTuple<int>>("dims", attr);

	CodeWritter dims_vec_code;
	dims_vec_code<<"{";
	for(int i=0; i<dims.vector().size()-1; i++) {
		dims_vec_code<<dims.vector()[i]<<",";
	}
	dims_vec_code<<dims.vector()[dims.vector().size()-1] << "}";

	// gen cpp code
	CodeWritter code_w; 
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str()); 
	code_w.feed("%s.load_param(%s);\n", node_name.c_str(), dims_vec_code.get_code_string()); 
	return code_w.get_code_string();
}

// SaberPooling
std::string ParserPooling(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
    auto global_pooling = get_attr<bool>("global_pooling", attr);
    auto pool_padding = get_attr<PTuple<int>>("padding", attr);
    auto pool_strides = get_attr<PTuple<int>>("strides", attr);
    auto pool_size = get_attr<PTuple<int>>("pool_size", attr);
    auto pool_method = get_attr<std::string>("method", attr);	

	// gen cpp code
    CodeWritter code_w; 
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str()); 
	code_w.feed("%s.load_param(%s,%s,%d,%d,%d,%d,%d,%d);\n", node_name.c_str(), pool_method,
										    global_pooling ? "true" : "false",
											pool_size[1],
											pool_size[0],
											pool_strides[1],
											pool_strides[0],
											pool_padding[1],
											pool_padding[0]); 
	return code_w.get_code_string();
}

// SaberPrelu
std::string ParserPrelu(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
	auto channel_shared = get_attr<bool>("channel_shared", attr);

	auto weights = get_attr<PBlock<float, ARM>>("weight_1", attr);
	writter.register_weights(node_name, weights);

	// gen cpp code
    CodeWritter code_w; 
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%s);\n", node_name.c_str(), channel_shared ? "true":"false");
	return code_w.get_code_string();
}

// SaberPriorBox
std::string ParserPriorBox(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
    auto min_size  = get_attr<PTuple<float>>("min_size", attr); 
	auto max_size  = get_attr<PTuple<float>>("max_size", attr); 
	auto as_ratio  = get_attr<PTuple<float>>("aspect_ratio", attr); 
	auto flip_flag = get_attr<bool>("is_flip", attr); 
	auto clip_flag = get_attr<bool>("is_clip", attr); 
	auto var       = get_attr<PTuple<float>>("variance", attr); 
	auto image_h   = get_attr<int>("img_h", attr); 
	auto image_w   = get_attr<int>("img_w", attr); 
	auto step_h    = get_attr<float>("step_h", attr); 
	auto step_w    = get_attr<float>("step_w", attr); 
	auto offset    = get_attr<float>("offset", attr);

	auto gen_vec_code = [](PTuple<float>& ptuple) -> std::string {
		CodeWritter dims_vec_code;
		dims_vec_code<<"{";
		for(int i=0; i<ptuple.vector().size()-1; i++) {
			dims_vec_code<<ptuple.vector()[i]<<",";
		}
		dims_vec_code<<ptuple.vector()[ptuple.vector().size()-1] << "}";
		return dims_vec_code.get_code_string();
	};

	// gen cpp code
	CodeWritter code_w; 
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str()); 
	code_w.feed("%s.load_param(%s,%s,%s,%s,%s,%s,%d,%d,%f,%f,%f);\n", node_name.c_str(),
										 flip_flag ? "ture":"false",
										 clip_flag ? "true":"false",
										 gen_vec_code(min_size),
										 gen_vec_code(max_size),
										 gen_vec_code(as_ratio),
										 gen_vec_code(var),
										 image_w, 
										 image_h,
										 step_w,
										 step_h,
										 offset); 
	return code_w.get_code_string();
}

// SaberSlice
std::string ParserSlice(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
    // parsing parameter 
	auto slice_dim = get_attr<int>("slice_dim", attr); 
	auto slice_point = get_attr<PTuple<int>>("slice_point", attr); 
	auto axis = get_attr<int>("axis", attr);

	CodeWritter slice_point_vec_code;
	slice_point_vec_code<<"{";
	for(int i=0; i<slice_point.vector().size()-1; i++) {
		slice_point_vec_code<<slice_point.vector()[i]<<",";
	}
	slice_point_vec_code<<slice_point.vector()[slice_point.vector().size()-1] << "}";

	// gen cpp code
    CodeWritter code_w; 
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%d,%s);\n", node_name.c_str(), axis, slice_point_vec_code.get_code_string());
	return code_w.get_code_string();
}

// SaberSoftmax
std::string ParserSoftmax(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
    auto axis = get_attr<int>("axis", attr);

	// gen cpp code
    CodeWritter code_w; 
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str()); 
	code_w.feed("%s.load_param(%d);\n", node_name.c_str(), axis); 
	return code_w.get_code_string();
}

std::unordered_map<std::string, OpParser> OPERATION_MAP({
	{"Input", {"Input", not_impl_yet} },
	{"Convolution", {"SaberConv2D", ParserConvolution} },
	{"Activation", {"SaberActivation", ParserActivation} },
	{"ConvRelu", {"SaberConvAct2D", ParserConvolutionRelu} }, 
	{"Concat", {"SaberConcat", ParserConcat} }, 
	{"DetectionOutput", {"SaberDectionOutput", ParserDectionOutput} }, // not certain
	{"Eltwise", {"SaberEltwise", ParserEltwise} },
	{"Dense", {"SaberFc", ParserFc} },
	{"Permute", {"SaberPermute", ParserPermute} },
	{"Pooling", {"SaberPooling", ParserPooling} },
	{"ReLU", {"SaberPrelu", ParserPrelu} },
	{"PriorBox", {"SaberPriorBox", ParserPriorBox} },
	{"Slice", {"SaberSlice", ParserSlice} },
	{"Softmax", {"SaberSoftmax", ParserSoftmax} }
});

} /* namespace lite */

} /* namespace anakin */

