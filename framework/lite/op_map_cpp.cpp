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

// SaberActivation
std::string ParserActivation(graph::AttrInfo& attr, std::string& op_class_name, std::string& node_name, WeightsWritter& writter) {
	// parsing parameter
	auto type = get_attr<std::string>("type", attr);
	// gen cpp code
	CodeWritter code_w;
	if (type == "TanH") {
		code_w.feed("    ActiveType %s_act_t = %d;\n", node_name.c_str(), Active_tanh);
	} else if (type == "Sigmoid") {
		code_w.feed("    ActiveType %s_act_t = %d;\n", node_name.c_str(), Active_sigmoid);
	} else {
		LOG(FATAL) << "Other Activation type" << type << " should be replace by other ops.";
	}
	code_w.feed("%s %s; \n", op_class_name.c_str(), node_name.c_str());
	code_w.feed("%s.load_param(%s_act_t);\n", node_name.c_str(), node_name.c_str());
	return code_w.get_code_string();
}

std::unordered_map<std::string, OpParser> OPERATION_MAP({
	{"Input", {"Input", not_impl_yet} },
	{"Convolution", {"SaberConv2D", ParserConvolution} },
	{"Activation", {"SaberActivation", ParserActivation} },
	{"ConvRelu", {"SaberConvAct2D", not_impl_yet} }, 
	{"Concat", {"SaberConcat", not_impl_yet} }, 
	{"DetectionOutput", {"SaberDectionOutput", not_impl_yet} },
	{"Eltwise", {"SaberEltwise", not_impl_yet} },
	{"Dense", {"SaberFc", not_impl_yet} },
	{"Permute", {"SaberPermute", not_impl_yet} },
	{"Pooling", {"SaberPooling", not_impl_yet} },
	{"ReLU", {"SaberPrelu", not_impl_yet} },
	{"PriorBox", {"SaberPriorBox", not_impl_yet} },
	{"Slice", {"SaberSlice", not_impl_yet} },
	{"Softmax", {"SaberSoftmax", not_impl_yet} }
});

} /* namespace lite */

} /* namespace anakin */

