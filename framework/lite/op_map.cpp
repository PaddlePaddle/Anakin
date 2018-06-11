#include "framework/lite/op_map.h"

namespace anakin {

namespace lite {

std::string not_impl_yet(graph::AttrInfo&) {
	LOG(ERROR) << "Parsing not impl yet. continue ...";
	return "";
}

std::string ParserConvolution(graph::AttrInfo& attr) {
	// parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = GET_PARAMETER(int, axis);
	// gen cpp code
	CodeWritter code_w;
	code_w.feed("    int group = %d;\n", group);
	code_w.feed("    bool flag_bias = %s;\n", bias_term ? "true":"false");
	code_w.feed("    int pad_h = %d;\n", padding[0]);
	code_w.feed("    int pad_w = %d;\n", padding[1]);
	code_w.feed("    int kh = %d;\n", kernel_size[0]);
	code_w.feed("    int kw = %d;\n", kernel_size[1]);
	code_w.feed("    int stride_h = %d;\n", strides[0]);
	code_w.feed("    int stride_w = %d;\n", strides[0]);
	code_w.feed("    int dila_h = %d;\n", dilation_rate[0]);
	code_w.feed("    int dila_w = %d;\n", dilation_rate[0]);
	return code_w.get_code_string();
}

std::string ParserActivation() {
	// parsing parameter
	auto type = get_attr<std::string>("type", attr);
	// gen cpp code
	CodeWritter code_w;
	if (type == "TanH") {
		code_w.feed("    ActiveType act_t = %d;\n", Active_tanh);
	} else if (type == "Sigmoid") {
		code_w.feed("    ActiveType act_t = %d;\n", Active_sigmoid);
	} else {
		LOG(FATAL) << "Other Activation type" << type << " should be replace by other ops.";
	}
	return code_w.get_code_string();
}

std::unordered_map<std::string, ParseParamFunctor> OPERATION_MAP {
	{"Convolution", ParserConvolution},
	{"Activation", ParserActivation},
	{"",}
	{"",}
	{"",}
	{"",}
	{"",}
	{"",}
	{"",}
	{"",}
	{"",}
	{"",}
	{"",}
};

} /* namespace lite */

} /* namespace anakin */

