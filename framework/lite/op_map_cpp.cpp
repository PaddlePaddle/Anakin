#include "framework/lite/op_map.h"
#include "framework/lite/utils.h"

namespace anakin {

namespace lite {

std::string not_impl_yet(graph::AttrInfo&,
                         std::string& code_name,
						 std::string& op_class_name, 
						 std::string& node_name, 
						 std::string& weights_ptr_name, 
						 WeightsWritter& writter,
                         bool gen_param) {
	LOG(INFO) << "Target "<< op_class_name << "Parsing not impl yet. continue ...";
	return "";
}

// SaberConv2D
std::string ParserConvolution(graph::AttrInfo& attr,
                              std::string& code_name,
							  std::string& op_class_name, 
							  std::string& node_name, 
							  std::string& weights_ptr_name, 
							  WeightsWritter& writter,
                              bool gen_param) {
	// parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();//weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0];//*weights_shape[1];

	writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count();
	if(bias_term) {
		auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
		writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
	}

	auto offset_info = writter.get_weights_by_name(node_name);

    CodeWritter code_w;
    if (gen_param) {
        // gen cpp code
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    bias_term ? 1 : 0,
                    offset_info.weights[0].offset,
                    bias_term ? offset_info.weights[1].offset : 0);
    } else {
        // gen cpp code
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s+%d,%s+%d);\n", node_name.c_str(),
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    bias_term ? "true":"false",
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[1].offset : 0);

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

	return code_w.get_code_string();
}
    // SaberPower
std::string ParserPower(graph::AttrInfo& attr,
                            std::string& code_name,
                            std::string& op_class_name,
                            std::string& node_name,
                            std::string& weights_ptr_name,
                            WeightsWritter& writter,
                            bool gen_param) {
        // parsing parameter
        auto power = get_attr<float>("power", attr);
        auto scale = get_attr<float>("scale", attr);
        auto shift = get_attr<float>("shift", attr);
        
        // gen cpp code
        CodeWritter code_w;

        if (gen_param) {
            code_w.feed("%f,%f,%f\n", scale, shift, power);
        } else {
            code_w.feed("ParamBase* %s_param = new PowerParam(%f,%f,%f);\n", node_name.c_str(), scale, shift, power);
            code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
        }
        return code_w.get_code_string();
    }
// SaberDeconv2D
std::string ParserDeconvolution(graph::AttrInfo& attr,
                              std::string& code_name,
                              std::string& op_class_name, 
                              std::string& node_name, 
                              std::string& weights_ptr_name, 
                              WeightsWritter& writter,
                              bool gen_param) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();//weights_shape[2]*weights_shape[3];
    int num_output = filter_num;//*weights_shape[1];

    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count();
    if(bias_term) {
        auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    auto offset_info = writter.get_weights_by_name(node_name);

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    bias_term ? 1 : 0,
                    offset_info.weights[0].offset,
                    bias_term ? offset_info.weights[1].offset : 0);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s+%d,%s+%d);\n", node_name.c_str(),
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    bias_term ? "true":"false",
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[1].offset : 0);

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

    return code_w.get_code_string();
}

// ParserConvolutionRelu
std::string ParserConvolutionRelu(graph::AttrInfo& attr,
                                  std::string& code_name,
								  std::string& op_class_name, 
								  std::string& node_name, 
								  std::string& weights_ptr_name, 
								  WeightsWritter& writter,
                                  bool gen_param) {
	// parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();//weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0];//*weights_shape[1];

	writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count();
	if(bias_term) {
		auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
		writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
	}

    auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
    if(gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    bias_term ? 1 : 0,
                    (int)Active_relu,
                    1, //set flag_relu true
                    offset_info.weights[0].offset,
                    bias_term ? offset_info.weights[1].offset : 0);
    } else {
        code_w.feed("ParamBase* %s_param = new ConvAct2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,Active_relu,%s,%s+%d,%s+%d);\n",
                    node_name.c_str(),
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    bias_term ? "true":"false",
                    "true", //set flag_relu true
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[1].offset : 0);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

	return code_w.get_code_string();
}

// ParserConvolutionRelu
std::string ParserConvolutionReluPool(graph::AttrInfo& attr,
                                  std::string& code_name,
                                  std::string& op_class_name,
                                  std::string& node_name,
                                  std::string& weights_ptr_name,
                                  WeightsWritter& writter,
                                  bool gen_param) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();//weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0];//*weights_shape[1];

    writter.register_weights(node_name, weights);
    if(bias_term) {
        auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
    }

    // parsing pooling parameter
    auto global_pooling = get_attr<bool>("pooling_0_global_pooling", attr);
    auto pool_padding = get_attr<PTuple<int>>("pooling_0_padding", attr);
    auto pool_strides = get_attr<PTuple<int>>("pooling_0_strides", attr);
    auto pool_size = get_attr<PTuple<int>>("pooling_0_pool_size", attr);
    auto pool_method = get_attr<std::string>("pooling_0_method", attr);

    std::string str_pool_method;

    PoolingType pool_type;
    if (pool_method == "MAX") {
        pool_type = Pooling_max;
        str_pool_method = "Pooling_max";
    }
    if (pool_method == "AVG") {
        pool_type = Pooling_average_include_padding;
        str_pool_method = "Pooling_average_include_padding";
    }

    auto offset_info = writter.get_weights_by_name(node_name);

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    bias_term ? 1 : 0,
                    (int)Active_relu,
                    1, //set flag_relu true
                    (int)pool_type,
                    global_pooling? 1 : 0,
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    offset_info.weights[0].offset,
                    bias_term ? offset_info.weights[1].offset : 0);
    } else {
        code_w.feed("ParamBase* %s_param = new ConvActPool2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,Active_relu,%s,%s,%s,%d,%d,%d,%d,%d,%d,%s+%d,%s+%d);\n",
                    node_name.c_str(),
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    bias_term ? "true":"false",
                    "true", //set flag_relu true
                    str_pool_method.c_str(),
                    global_pooling? "true" : "false",
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[1].offset : 0);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

    return code_w.get_code_string();
}

//conv batchnorm
std::string ParserConvBatchnorm(graph::AttrInfo& attr,
                                std::string& code_name,
                                std::string& op_class_name,
                                std::string& node_name,
                                std::string& weights_ptr_name,
                                WeightsWritter& writter,
                                bool gen_param) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();//weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0];//*weights_shape[1];

    // get batchnorm param
    auto epsilon = get_attr<float>("batchnorm_0_epsilon", attr);
    auto momentum = get_attr<float>("batchnorm_0_momentum", attr);
    auto batch_norm_weight_1 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_1", attr);
    auto batch_norm_weight_1_vector = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_2", attr);
    auto batch_norm_weight_2_vector = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_3", attr);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector();

    if(bias_term) {
        auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
        update_weights(weights, bias,
                       weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                       bias_term,
                       batch_norm_weight_3_vector[0], epsilon,
                       batch_norm_weight_1_vector,
                       batch_norm_weight_2_vector);


        writter.register_weights(node_name, weights);
                LOG(INFO) << node_name << " write weights: " << weights.count();
        writter.register_weights(node_name, bias);
                LOG(INFO) << node_name << " write bias: " << bias.count();
    } else {
        auto bias = PBlock<float, X86>();
        update_weights(weights, bias,
                       weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                       false,
                       batch_norm_weight_3_vector[0], epsilon,
                       batch_norm_weight_1_vector,
                       batch_norm_weight_2_vector);

        writter.register_weights(node_name, weights);
                LOG(INFO) << node_name << " write weights: " << weights.count();
        writter.register_weights(node_name, bias);
                LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    auto offset_info = writter.get_weights_by_name(node_name);

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    1,//bias term always true
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset); //always has bias
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s+%d,%s+%d);\n", node_name.c_str(), \
                weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    "true",//bias term always true
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset); //always has bias
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

std::string ParserConvBatchnormScale(graph::AttrInfo& attr,
                                     std::string& code_name,
					                 std::string& op_class_name, 
					                 std::string& node_name, 
					                 std::string& weights_ptr_name, 
    				                 WeightsWritter& writter,
                                     bool gen_param) {
    // parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();//weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0];//*weights_shape[1];

    // get batchnorm param
    auto epsilon = get_attr<float>("batchnorm_0_epsilon", attr);
    auto momentum = get_attr<float>("batchnorm_0_momentum", attr);
    auto batch_norm_weight_1 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_1", attr);
    auto batch_norm_weight_1_vector = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_2", attr);
    auto batch_norm_weight_2_vector = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_3", attr);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector();

    // get scale param
    auto scale_num_axes = get_attr<int>("scale_0_num_axes", attr);
    auto scale_bias_term = get_attr<bool>("scale_0_bias_term", attr);
    auto scale_axis = get_attr<int>("scale_0_axis", attr);
    auto scale_weight_1 = get_attr<PBlock<float, X86>>("scale_0_weight_1", attr);
    auto scale_weight_1_vector = scale_weight_1.vector();
    auto scale_weight_2 = get_attr<PBlock<float, X86>>("scale_0_weight_2", attr);
    auto scale_weight_2_vector = scale_weight_2.vector();


	if(bias_term) {
		auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
		update_weights(weights, bias,
					   weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
					   bias_term, 
					   batch_norm_weight_3_vector[0], epsilon, 
					   batch_norm_weight_1_vector, 
					   batch_norm_weight_2_vector,
					   scale_weight_1_vector,
					   scale_weight_2_vector,
					   scale_bias_term);
	
		
		writter.register_weights(node_name, weights);
        LOG(INFO) << node_name << " write weights: " << weights.count();
		writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
	} else {
		auto bias = PBlock<float, X86>();
		update_weights(weights, bias,
					   weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
					   false, 
					   batch_norm_weight_3_vector[0], epsilon, 
					   batch_norm_weight_1_vector, 
					   batch_norm_weight_2_vector,
					   scale_weight_1_vector,
					   scale_weight_2_vector,
					   scale_bias_term);

		writter.register_weights(node_name, weights);
        LOG(INFO) << node_name << " write weights: " << weights.count();
		writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
	}

    auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    1,//bias term always true
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset); //always has bias
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s+%d,%s+%d);\n", node_name.c_str(), \
                weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    "true",//bias term always true
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset); //always has bias
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

	return code_w.get_code_string();
}

// SaberConvBatchnormScaleRelu
std::string ParserConvBatchnormScaleRelu(graph::AttrInfo& attr,
                                         std::string& code_name,
					                     std::string& op_class_name, 
					                     std::string& node_name, 
					                     std::string& weights_ptr_name, 
    				                     WeightsWritter& writter,
                                         bool gen_param) {
    // parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();//weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0];//*weights_shape[1];

    // get batchnorm param
    auto epsilon = get_attr<float>("batchnorm_0_epsilon", attr);
    auto momentum = get_attr<float>("batchnorm_0_momentum", attr);
    auto batch_norm_weight_1 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_1", attr);
    auto batch_norm_weight_1_vector = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_2", attr);
    auto batch_norm_weight_2_vector = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_3", attr);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector(); 

    // get scale param
    auto scale_num_axes = get_attr<int>("scale_0_num_axes", attr);
    auto scale_bias_term = get_attr<bool>("scale_0_bias_term", attr);
    auto scale_axis = get_attr<int>("scale_0_axis", attr);
    auto scale_weight_1 = get_attr<PBlock<float, X86>>("scale_0_weight_1", attr);
    auto scale_weight_1_vector = scale_weight_1.vector();
    auto scale_weight_2 = get_attr<PBlock<float, X86>>("scale_0_weight_2", attr);
    auto scale_weight_2_vector = scale_weight_2.vector();

	if(bias_term) {
		auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
		update_weights(weights, bias,
					   weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
					   bias_term, 
					   batch_norm_weight_3_vector[0], epsilon, 
					   batch_norm_weight_1_vector, 
					   batch_norm_weight_2_vector,
					   scale_weight_1_vector,
					   scale_weight_2_vector,
					   scale_bias_term);
	
		
		writter.register_weights(node_name, weights);
        LOG(INFO) << node_name << " write weights: " << weights.count();
		writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
	} else {
		auto bias = PBlock<float, X86>();
		update_weights(weights, bias,
					   weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
					   false, 
					   batch_norm_weight_3_vector[0], epsilon, 
					   batch_norm_weight_1_vector, 
					   batch_norm_weight_2_vector,
					   scale_weight_1_vector,
					   scale_weight_2_vector,
					   scale_bias_term);

		writter.register_weights(node_name, weights);
        LOG(INFO) << node_name << " write weights: " << weights.count();
		writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
	}

    auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    1, // set bias to true
                    (int)Active_relu,
                    1, //set flag_relu true
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset);
    } else {
        code_w.feed("ParamBase* %s_param = new ConvAct2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,Active_relu,%s,%s+%d,%s+%d);\n",
                    node_name.c_str(),
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    "true", // set bias to true
                    "true", //set flag_relu true
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset);

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberConvBatchnormScaleRelu
std::string ParserConvBatchnormScaleReluPool(graph::AttrInfo& attr,
                                         std::string& code_name,
                                         std::string& op_class_name,
                                         std::string& node_name,
                                         std::string& weights_ptr_name,
                                         WeightsWritter& writter,
                                         bool gen_param) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();//weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0];//*weights_shape[1];

    // get batchnorm param
    auto epsilon = get_attr<float>("batchnorm_0_epsilon", attr);
    auto momentum = get_attr<float>("batchnorm_0_momentum", attr);
    auto batch_norm_weight_1 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_1", attr);
    auto batch_norm_weight_1_vector = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_2", attr);
    auto batch_norm_weight_2_vector = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = get_attr<PBlock<float, X86>>("batchnorm_0_weight_3", attr);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector();

    // get scale param
    auto scale_num_axes = get_attr<int>("scale_0_num_axes", attr);
    auto scale_bias_term = get_attr<bool>("scale_0_bias_term", attr);
    auto scale_axis = get_attr<int>("scale_0_axis", attr);
    auto scale_weight_1 = get_attr<PBlock<float, X86>>("scale_0_weight_1", attr);
    auto scale_weight_1_vector = scale_weight_1.vector();
    auto scale_weight_2 = get_attr<PBlock<float, X86>>("scale_0_weight_2", attr);
    auto scale_weight_2_vector = scale_weight_2.vector();

    if(bias_term) {
        auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
        update_weights(weights, bias,
                       weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                       bias_term,
                       batch_norm_weight_3_vector[0], epsilon,
                       batch_norm_weight_1_vector,
                       batch_norm_weight_2_vector,
                       scale_weight_1_vector,
                       scale_weight_2_vector,
                       scale_bias_term);


        writter.register_weights(node_name, weights);
        LOG(INFO) << node_name << " write weights: " << weights.count();
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    } else {
        auto bias = PBlock<float, X86>();
        update_weights(weights, bias,
                       weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                       false,
                       batch_norm_weight_3_vector[0], epsilon,
                       batch_norm_weight_1_vector,
                       batch_norm_weight_2_vector,
                       scale_weight_1_vector,
                       scale_weight_2_vector,
                       scale_bias_term);

        writter.register_weights(node_name, weights);
        LOG(INFO) << node_name << " write weights: " << weights.count();
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    // parsing pooling parameter
    auto global_pooling = get_attr<bool>("pooling_0_global_pooling", attr);
    auto pool_padding = get_attr<PTuple<int>>("pooling_0_padding", attr);
    auto pool_strides = get_attr<PTuple<int>>("pooling_0_strides", attr);
    auto pool_size = get_attr<PTuple<int>>("pooling_0_pool_size", attr);
    auto pool_method = get_attr<std::string>("pooling_0_method", attr);

	std::string str_pool_method;
	PoolingType pool_type;
	if (pool_method == "MAX") {
		pool_type = Pooling_max;
		str_pool_method = "Pooling_max";
	}
	if (pool_method == "AVG") {
		pool_type = Pooling_average_include_padding;
		str_pool_method = "Pooling_average_include_padding";
	}

    auto offset_info = writter.get_weights_by_name(node_name);

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    1, // set bias to true
                    (int)Active_relu,
                    1, //set flag_relu true
                    (int)pool_type,
                    global_pooling? 1 : 0,
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset);
    } else {
        code_w.feed("ParamBase* %s_param = new ConvActPool2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,Active_relu,%s,%s,%s,%d,%d,%d,%d,%d,%d,%s+%d,%s+%d);\n",
                    node_name.c_str(),
                    weights_size,
                    num_output,
                    group,
                    kernel_size[1],
                    kernel_size[0],
                    strides[1],
                    strides[0],
                    padding[1],
                    padding[0],
                    dilation_rate[1],
                    dilation_rate[0],
                    "true", // set bias to true
                    "true", //set flag_relu true
                    str_pool_method.c_str(),
                    global_pooling? "true" : "false",
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset);

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

    return code_w.get_code_string();
}

// SaberConcat
std::string ParserConcat(graph::AttrInfo& attr,
                         std::string& code_name,
						 std::string& op_class_name, 
						 std::string& node_name, 
						 std::string& weights_ptr_name, 
						 WeightsWritter& writter,
                         bool gen_param) {
	// parsing parameter
	auto axis = get_attr<int>("axis", attr);
	// gen cpp code
	CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d\n", axis);
    } else  {
        code_w.feed("ParamBase* %s_param = new ConcatParam(%d);\n",
                    node_name.c_str(), axis);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberDectionOutput
std::string ParserDectionOutput(graph::AttrInfo& attr,
                                std::string& code_name,
								std::string& op_class_name, 
								std::string& node_name, 
								std::string& weights_ptr_name, 
								WeightsWritter& writter,
                                bool gen_param) {
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

    CodeType cd_type;
    if (code_type == "CORNER") {
        cd_type = CORNER;
    } else if (code_type == "CORNER_SIZE") {
        cd_type = CORNER_SIZE;
    } else if (code_type == "CENTER_SIZE") {
        cd_type = CENTER_SIZE;
    } else {
        LOG(FATAL) << "unsupport code type in detection output param: " << code_type;
    }
    // gen cpp code
	CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%f,%d,%d,%d,%d,%f,%f,%d,%d\n",
                    classes_num,
                    conf_thresh,
                    nms_top_k,
                    background_id,
                    keep_top_k,
                    (int)cd_type,
                    nms_thresh,
                    nms_eta,
                    flag_share_location? 1 : 0,
                    flag_var_in_target? 1 : 0);
    } else {
        code_w.feed("ParamBase* %s_param = new DetectionOutputParam(%d,%f,%d,%d,%d,%s,%f,%f,%s,%s);\n",
                    node_name.c_str(),
                    classes_num,
                    conf_thresh,
                    nms_top_k,
                    background_id,
                    keep_top_k,
                    code_type.c_str(),
                    nms_thresh,
                    nms_eta,
                    flag_share_location? "true" : "false",
                    flag_var_in_target? "true" : "false");
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberEltwise
std::string ParserEltwise(graph::AttrInfo& attr,
                          std::string& code_name,
						  std::string& op_class_name, 
						  std::string& node_name, 
						  std::string& weights_ptr_name, 
						  WeightsWritter& writter,
                          bool gen_param) {
	// parsing parameter
    auto type = get_attr<std::string>("type", attr); 
    auto coeff = get_attr<PTuple<float>>("coeff", attr);

	std::string eltwise_type_str("Eltwise_unknow");
    EltwiseType et_type;
	if (type == "Add") {
        eltwise_type_str = "Eltwise_sum";
        et_type = Eltwise_sum;
    } else if (type == "Max") {
        eltwise_type_str = "Eltwise_max";
        et_type = Eltwise_max;
    } else {
        eltwise_type_str = "Eltwise_prod";
        et_type = Eltwise_prod;
    }

	CodeWritter coeff_vec_code;
	coeff_vec_code<<"{";
	for(int i=0; i<coeff.size()-1; i++) {
		coeff_vec_code<<coeff.vector()[i]<<",";
	}
	if(coeff.size() > 0) {
		coeff_vec_code<<coeff.vector()[coeff.size()-1] << "}";
	} else {
		coeff_vec_code<<"}";
	}

	// gen cpp code
	CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d, %d ", (int)et_type,
                    coeff.size());
        for (int i = 0; i < coeff.size(); ++i) {
            code_w << coeff[i] << " ";
        }
        code_w << "\n";
    } else  {
        code_w.feed("ParamBase* %s_param = new EltwiseParam(%s, %s);\n",
                    node_name.c_str(),
                    eltwise_type_str.c_str(),
                    coeff_vec_code.get_code_string().c_str());

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberActivation
std::string ParserActivation(graph::AttrInfo& attr,
                             std::string& code_name,
							 std::string& op_class_name, 
							 std::string& node_name, 
							 std::string& weights_ptr_name, 
							 WeightsWritter& writter,
                             bool gen_param) {
	// parsing parameter
	auto type = get_attr<std::string>("type", attr);

	std::string act_type("Active_unknow");

    //! ActiveType act_type, float neg_slope = 0.f, float coef = 1.f, bool channel_shared = false, const float* weights = nullptr
    // gen cpp code
    CodeWritter code_w;
	if (type == "TanH") {
        if (gen_param) {
            code_w << (int)Active_tanh << "," << 0.f << "," << 0.f << "," << 0 << "," << 0 << "\n";
        } else {
            act_type = "Active_tanh";
            code_w.feed("ParamBase* %s_param = new ActivationParam(%s);\n",
                        node_name.c_str(),
                        act_type.c_str());
            code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
        }

	} else if (type == "Sigmoid") {
        if (gen_param) {
            code_w << (int)Active_sigmoid << "," << 0.f << "," << 0.f << "," << 0 << "," << 0 << "\n";
        } else {
            act_type = "Active_sigmoid";
            code_w.feed("ParamBase* %s_param = new ActivationParam(%s);\n",
                        node_name.c_str(),
                        act_type.c_str());

            code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
        }

    } else if (type == "ReLU") {
        if (gen_param) {
            code_w << (int)Active_relu << "," << 0.f << "," << 0.f << "," << 0 << "," << 0 << "\n";
        } else {
            act_type = "Active_relu";
            code_w.feed("ParamBase* %s_param = new ActivationParam(%s);\n",
                        node_name.c_str(),
                        act_type.c_str());

            code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
        }

    }  else if (type == "PReLU") {
        act_type = "Active_prelu";
        auto prelu_channel_shared = get_attr<bool>("channel_shared", attr);
        // auto prelu_weights = get_attr<bool>("weights", attr);
        auto prelu_weights = get_attr<PBlock<float, X86>>("weight_1", attr);

        writter.register_weights(node_name, prelu_weights);
                LOG(INFO) << node_name << " write weights: " << prelu_weights.count();

        auto offset_info = writter.get_weights_by_name(node_name);
        if (gen_param) {
            code_w << (int)Active_prelu << "," << 0.f << "," << 0.f << "," << \
                (prelu_channel_shared ? 1 : 0) << "," << offset_info.weights[0].offset << "\n";
        } else {
            code_w.feed("ParamBase* %s_param = new ActivationParam(%s, %f, %f, %s, %s+%d);\n",
                        node_name.c_str(),
                        act_type.c_str(),
                        0.f,
                        0.f,
                        prelu_channel_shared ? "true" : "false",
                        weights_ptr_name.c_str(),
                        offset_info.weights[0].offset);

            code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
        }

	} else { 
		LOG(FATAL) << "Other Activation type" << type << " unknown."; 
	}	
	return code_w.get_code_string();
}

std::string ParserRelu(graph::AttrInfo& attr,
                       std::string& code_name,
					   std::string& op_class_name, 
					   std::string& node_name, 
					   std::string& weights_ptr_name, 
					   WeightsWritter& writter, bool gen_param) {
    // parsing parameter
    auto alpha = get_attr<float>("alpha", attr);

	std::string act_type("Active_relu");

	// gen cpp code
	CodeWritter code_w;
    if (gen_param) {
        code_w << (int)Active_relu << "," << 0.f << "," << 0.f << "," << 0 << "," << 0 << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new ActivationParam(%s);\n",
                    node_name.c_str(),
                    act_type.c_str());

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

	return code_w.get_code_string();
}

// SaberFc
std::string ParserFc(graph::AttrInfo& attr,
                     std::string& code_name,
					 std::string& op_class_name, 
					 std::string& node_name, 
					 std::string& weights_ptr_name, 
					 WeightsWritter& writter,
                     bool gen_param) {
	// parsing parameter
    auto axis = get_attr<int>("axis", attr); 
    auto out_dim = get_attr<int>("out_dim", attr); 
    auto bias_term = get_attr<bool>("bias_term", attr);
	
	auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
	writter.register_weights(node_name, weights);
	if(bias_term) {
		auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
		writter.register_weights(node_name, bias);
	}

	auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d\n",
                    axis,
                    out_dim,
                    bias_term ? 1 : 0,
                    offset_info.weights[0].offset,
                    bias_term ? offset_info.weights[1].offset : 0,
                    0);
    } else {
        code_w.feed("ParamBase* %s_param = new FcParam(%d,%d,%s,%s+%d,%s+%d,%s);\n",
                    node_name.c_str(),
                    axis,
                    out_dim,
                    bias_term ? "true":"false",
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[1].offset : 0,
                    "false");

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberPermute
std::string ParserPermute(graph::AttrInfo& attr,
                          std::string& code_name,
						  std::string& op_class_name, 
						  std::string& node_name, 
						  std::string& weights_ptr_name, 
						  WeightsWritter& writter,
                          bool gen_param) {
	// parsing parameter
	auto dims = get_attr<PTuple<int>>("dims", attr);

	CodeWritter dims_vec_code;
	dims_vec_code<<"{";
	for(int i=0; i<dims.size()-1; i++) {
		dims_vec_code<<dims.vector()[i]<<",";
	}
	if(dims.size() > 0) {
		dims_vec_code<<dims.vector()[dims.size()-1] << "}";
	} else {
		dims_vec_code<< "}";
	}

	// gen cpp code
	CodeWritter code_w;
    if (gen_param) {
        code_w << dims.size() << " ";
        for (int i = 0; i < dims.size(); ++i) {
            code_w << dims[i] << " ";
        }
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new PermuteParam(%s);\n",
                    node_name.c_str(),
                    dims_vec_code.get_code_string().c_str());
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberPooling
std::string ParserPooling(graph::AttrInfo& attr,
                          std::string& code_name,
						  std::string& op_class_name, 
						  std::string& node_name, 
						  std::string& weights_ptr_name, 
						  WeightsWritter& writter,
                          bool gen_param) {
	// parsing parameter
    auto global_pooling = get_attr<bool>("global_pooling", attr);
    auto pool_padding = get_attr<PTuple<int>>("padding", attr);
    auto pool_strides = get_attr<PTuple<int>>("strides", attr);
    auto pool_size = get_attr<PTuple<int>>("pool_size", attr);
    auto pool_method = get_attr<std::string>("method", attr);	

    PoolingType pool_type;
    std::string str_pool_method;
    if (pool_method == "MAX") {
        pool_type = Pooling_max;
        str_pool_method = "Pooling_max";
    }
    if (pool_method == "AVG") {
        pool_type = Pooling_average_include_padding;
        str_pool_method = "Pooling_average_include_padding";
    }

	// gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d,%d,%d,%d\n",
                    (int)pool_type,
                    global_pooling ? 1 : 0,
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0]);
    } else {
        code_w.feed("ParamBase* %s_param = new PoolParam(%s,%s,%d,%d,%d,%d,%d,%d);\n",
                    node_name.c_str(),
                    str_pool_method.c_str(),
                    global_pooling ? "true" : "false",
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0]);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberPrelu
std::string ParserPrelu(graph::AttrInfo& attr,
                        std::string& code_name,
						std::string& op_class_name, 
						std::string& node_name, 
						std::string& weights_ptr_name, 
						WeightsWritter& writter,
                        bool gen_param) {
	// parsing parameter
	auto channel_shared = get_attr<bool>("channel_shared", attr);

	auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);
	writter.register_weights(node_name, weights);

    auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << (int)Active_prelu << "," << 0.f << "," << 0.f << "," << \
                (channel_shared ? 1 : 0) << "," << offset_info.weights[0].offset << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new ActivationParam(%s, %f, %f, %s, %s+%d);\n",
                        node_name.c_str(),
                        "Active_prelu",
                        0.f,
                        0.f,
                        channel_shared ? "true" : "false",
                        weights_ptr_name.c_str(),
                        offset_info.weights[0].offset);

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberPriorBox
std::string ParserPriorBox(graph::AttrInfo& attr,
                           std::string& code_name,
						   std::string& op_class_name, 
						   std::string& node_name, 
						   std::string& weights_ptr_name, 
						   WeightsWritter& writter,
                           bool gen_param) {
	// parsing parameter
    auto min_size  = get_attr<PTuple<float>>("min_size", attr); 
    auto max_size  = get_attr<PTuple<float>>("max_size", attr); 
    auto as_ratio  = get_attr<PTuple<float>>("aspect_ratio", attr);
    //add
    auto fixed_size  = get_attr<PTuple<float>>("fixed_size", attr); 
    auto fixed_ratio  = get_attr<PTuple<float>>("fixed_ratio", attr); 
    auto density  = get_attr<PTuple<float>>("density", attr);

    auto flip_flag = get_attr<bool>("is_flip", attr); 
    auto clip_flag = get_attr<bool>("is_clip", attr); 
    auto var       = get_attr<PTuple<float>>("variance", attr); 
    auto image_h   = get_attr<int>("img_h", attr); 
    auto image_w   = get_attr<int>("img_w", attr); 
    auto step_h    = get_attr<float>("step_h", attr);  
    auto step_w    = get_attr<float>("step_w", attr); 
    auto offset    = get_attr<float>("offset", attr);
    auto order     = get_attr<PTuple<std::string>>("order", attr);

    std::vector<PriorType> order_;
    CodeWritter order_string;
    order_string << "{";

    int order_size = order.size();
    for (int i = 0; i < order_size - 1; i++) {
        if (order[i] == "MIN") {
            order_.push_back(PRIOR_MIN);
            order_string << "PRIOR_MIN, ";
        } else if (order[i] == "MAX") {
            order_.push_back(PRIOR_MAX);
            order_string << "PRIOR_MAX, ";
        } else if (order[i] == "COM") {
            order_.push_back(PRIOR_COM);
            order_string << "PRIOR_COM, ";
        }
    }
   if (order[order_size - 1] == "MIN") {
        order_.push_back(PRIOR_MIN);
        order_string << "PRIOR_MIN";
    } else if (order[order_size - 1] == "MAX") {
        order_.push_back(PRIOR_MAX);
        order_string << "PRIOR_MAX";
    } else if (order[order_size - 1] == "COM") {
        order_.push_back(PRIOR_COM);
        order_string << "PRIOR_COM";
    }

    order_string << "}";

    auto gen_vec_code_0 = [](PTuple<PriorType> ptuple) -> std::string {
        CodeWritter dims_vec_code;
        dims_vec_code<<"{";
        for(int i=0; i<ptuple.size()-1; i++) {
            dims_vec_code<<ptuple.vector()[i]<<",";
        }
        if(ptuple.size() > 0) {
            dims_vec_code<<ptuple.vector()[ptuple.size()-1] << "}";
        } else {
            dims_vec_code<< "}";
        }
        return dims_vec_code.get_code_string();
    };

	auto gen_vec_code = [](PTuple<float> ptuple) -> std::string {
		CodeWritter dims_vec_code;
		dims_vec_code<<"{";
		for(int i=0; i<ptuple.size()-1; i++) {
			dims_vec_code<<ptuple.vector()[i]<<",";
		}
		if(ptuple.size() > 0) {
			dims_vec_code<<ptuple.vector()[ptuple.size()-1] << "}";
		} else {
			dims_vec_code<< "}";
		}
		return dims_vec_code.get_code_string();
	};


	// gen cpp code
	CodeWritter code_w;
    if (gen_param) {
       // printf("**************\n");
        code_w << var.size() << " ";
        for (int i = 0; i < var.size(); ++i) {
            code_w << var[i] << " ";
        }
        code_w.feed(",%d,%d,%d,%d,%f,%f,%f,%d,%d,%d",
                    gen_vec_code(var).c_str(),
                    flip_flag ? 1 : 0,
                    clip_flag ? 1 : 0,
                    image_w,
                    image_h,
                    step_w,
                    step_h,
                    offset,
                    (int)order_[0], (int)order_[1], (int)order_[2]);

        code_w << ", " << min_size.size() << " ";
        for (int i = 0; i < min_size.size(); ++i) {
            code_w << min_size[i] << " ";
        }
        code_w << ", " << max_size.size() << " ";
        for (int i = 0; i < max_size.size(); ++i) {
            code_w << max_size[i] << " ";
        }
        code_w << ", " << as_ratio.size() << " ";
        for (int i = 0; i < as_ratio.size(); ++i) {
            code_w << as_ratio[i] << " ";
        }
        code_w << ", " << fixed_size.size() << " ";
        for (int i = 0; i < fixed_size.size(); ++i) {
            code_w << fixed_size[i] << " ";
        }
        code_w << ", " << fixed_ratio.size() << " ";
        for (int i = 0; i < fixed_ratio.size(); ++i) {
            code_w << fixed_ratio[i] << " ";
        }
        code_w << ", " << density.size() << " ";
        for (int i = 0; i < density.size(); ++i) {
            code_w << density[i] << " ";
        }
        
        code_w << "\n";
    } else {
      //  printf("===============\n");
        code_w.feed("ParamBase* %s_param = new PriorBoxParam(%s,%s,%s,%d,%d,%f,%f,%f,%s,%s,%s,%s,%s,%s,%s);\n",
                    node_name.c_str(),
                    gen_vec_code(var).c_str(),
                    flip_flag ? "true":"false",
                    clip_flag ? "true":"false",
                    image_w,
                    image_h,
                    step_w,
                    step_h,
                    offset,
                    order_string.get_code_string().c_str(),
                    gen_vec_code(min_size).c_str(),
                    gen_vec_code(max_size).c_str(),
                    gen_vec_code(as_ratio).c_str(),
                    gen_vec_code(fixed_size).c_str(),
                    gen_vec_code(fixed_ratio).c_str(),
                    gen_vec_code(density).c_str());

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

	return code_w.get_code_string();
}

// SaberSlice
std::string ParserSlice(graph::AttrInfo& attr,
                        std::string& code_name,
						std::string& op_class_name, 
						std::string& node_name, 
						std::string& weights_ptr_name, 
						WeightsWritter& writter,
                        bool gen_param) {
    // parsing parameter 
	auto slice_dim = get_attr<int>("slice_dim", attr); 
	auto slice_point = get_attr<PTuple<int>>("slice_point", attr); 
	auto axis = get_attr<int>("axis", attr);

	CodeWritter slice_point_vec_code;
	slice_point_vec_code<<"{";
	for(int i=0; i<slice_point.size()-1; i++) {
		slice_point_vec_code<<slice_point.vector()[i]<<",";
	}
	if(slice_point.size() > 0) {
		slice_point_vec_code<<slice_point.vector()[slice_point.size()-1] << "}";
	} else {
		slice_point_vec_code<< "}";
	}

	// gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << axis << ", " << slice_point.size() << " ";
        for (int i = 0; i < slice_point.size(); ++i) {
            code_w << slice_point[i] << " ";
        }
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new SliceParam(%d,%s);\n",
                    node_name.c_str(),
                    axis,
                    slice_point_vec_code.get_code_string().c_str());
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberSlice
std::string ParserScale(graph::AttrInfo& attr,
                        std::string& code_name,
                        std::string& op_class_name,
                        std::string& node_name,
                        std::string& weights_ptr_name,
                        WeightsWritter& writter,
                        bool gen_param) {
    // parsing parameter 
    auto num_axes = get_attr<int>("num_axes", attr);
    auto axis = get_attr<int>("axis", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto weights = get_attr<PBlock<float, X86>>("weight_1", attr);

    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count();
    
    if (bias_term) {
        auto bias = get_attr<PBlock<float, X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
      }

      auto offset_info = writter.get_weights_by_name(node_name);

  // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d,%d,%d,%d,%d\n",
                    offset_info.weights[0].offset,
                    bias_term ? offset_info.weights[1].offset : 0,
                    bias_term ? 1 : 0,
                    axis,
                    num_axes);
    } else {
        code_w.feed("ParamBase* %s_param = new ScaleParam(%s+%d, %s+%d, %s, %d, %d);\n",
                    node_name.c_str(),
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[1].offset : 0,
                    bias_term ? "true":"false",
                    axis,
                    num_axes);

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}


// SaberSoftmax
std::string ParserSoftmax(graph::AttrInfo& attr,
                          std::string& code_name,
						  std::string& op_class_name, 
						  std::string& node_name, 
						  std::string& weights_ptr_name, 
						  WeightsWritter& writter,
                          bool gen_param) {
	// parsing parameter
    auto axis = get_attr<int>("axis", attr);

	// gen cpp code
    CodeWritter code_w;

    if (gen_param) {
        code_w << axis;
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new SoftmaxParam(%d);\n",
                    node_name.c_str(),
                    axis);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
	return code_w.get_code_string();
}

// SaberSplit
std::string ParserSplit(graph::AttrInfo& attr,
                          std::string& code_name,
                          std::string& op_class_name,
                          std::string& node_name,
                          std::string& weights_ptr_name,
                          WeightsWritter& writter,
                          bool gen_param) {
    // parsing parameter
    // no param
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new SplitParam;\n",
                    node_name.c_str());
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

// SaberFlatten
std::string ParserFlatten(graph::AttrInfo& attr,
                          std::string& code_name,
                          std::string& op_class_name,
                          std::string& node_name,
                          std::string& weights_ptr_name,
                          WeightsWritter& writter,
                          bool gen_param) {
    // parsing parameter
    // no param
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new FlattenParam;\n",
                    node_name.c_str());
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

// Parser reshape
std::string ParserReshape(graph::AttrInfo& attr,
                               std::string& code_name,
                               std::string& op_class_name,
                               std::string& node_name,
                               std::string& weights_ptr_name,
                               WeightsWritter& writter,
                               bool gen_param) {
    // parsing parameter
    auto dims = get_attr<PTuple<int>>("dims", attr);
    std::vector<int> vdims = dims.vector();

    CodeWritter reshape_dims_vec_code;
    reshape_dims_vec_code << "{";
    for(int i = 0; i < vdims.size() - 1; i++) {
        reshape_dims_vec_code << vdims[i] << ",";
    }
    if (vdims.size() > 0) {
        reshape_dims_vec_code << vdims[vdims.size() - 1] << "}";
    } else {
        reshape_dims_vec_code<< "}";
    }

    CodeWritter code_w;
    if (gen_param) {
        code_w << dims.size() << " ";
        for (int i = 0; i < dims.size(); ++i) {
            code_w << dims[i] << " ";
        }
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new ReshapeParam(%s);\n", node_name.c_str(), reshape_dims_vec_code.get_code_string().c_str());
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

std::unordered_map<std::string, OpParser> OPERATION_MAP({
	{"Input", {"Input", not_impl_yet} },
	{"Convolution", {"SaberConv2D", ParserConvolution} }, // done
	{"Deconvolution", {"SaberDeconv2D", ParserDeconvolution}}, //done
	{"Activation", {"SaberActivation", ParserActivation} }, // done
	{"ReLU", {"SaberActivation",ParserRelu}}, // done
	{"ConvRelu", {"SaberConvAct2D", ParserConvolutionRelu} },  // done
	{"ConvReluPool", {"SaberConvActPooling2D", ParserConvolutionReluPool} },  // done
	{"ConvBatchnormScaleRelu", {"SaberConvAct2D", ParserConvBatchnormScaleRelu}}, // done have question ??
	{"ConvBatchnormScaleReluPool", {"SaberConvActPooling2D", ParserConvBatchnormScaleReluPool}}, // done have question ??
	{"ConvBatchnormScale", {"SaberConv2D", ParserConvBatchnormScale}}, //done
	{"ConvBatchnorm", {"SaberConv2D", ParserConvBatchnorm}}, //done
	{"Concat", {"SaberConcat", ParserConcat} },  // done
	{"DetectionOutput", {"SaberDetectionOutput", ParserDectionOutput} }, // done 
	{"Eltwise", {"SaberEltwise", ParserEltwise} }, //done
	{"Eltwise", {"SaberEltwiseRelu", not_impl_yet}}, // not impl ??
	{"Dense", {"SaberFc", ParserFc} }, // done
	{"Permute", {"SaberPermute", ParserPermute} }, // done
	{"Pooling", {"SaberPooling", ParserPooling} }, // done
	{"PReLU", {"SaberPrelu", ParserPrelu} }, // done
	{"PriorBox", {"SaberPriorBox", ParserPriorBox} }, // done
	{"Power", {"SaberPower", ParserPower} }, // done
	{"Scale", {"SaberScale", ParserScale} }, // done
	{"Slice", {"SaberSlice", ParserSlice} }, // done
    {"Flatten", {"SaberFlatten", ParserFlatten}}, //done
    {"Reshape", {"SaberReshape", ParserReshape}}, //done
	{"Softmax", {"SaberSoftmax", ParserSoftmax}}, //done
	{"Split", {"SaberSplit", ParserSplit}} // done
});

} /* namespace lite */

} /* namespace anakin */

