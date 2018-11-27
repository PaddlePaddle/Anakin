#include "framework/lite/op_map.h"
#include "framework/utils/parameter_fusion.h"

namespace anakin {

namespace lite {

std::string not_impl_yet(graph::AttrInfo&, 
						 std::string& op_class_name, 
						 std::string& node_name, 
						 std::string& weights_ptr_name, 
						 WeightsWritter& writter) {
	LOG(INFO) << "Target "<< op_class_name << "Parsing not impl yet. continue ...";
	return "";
}

// SaberConv2D
std::string ParserConvolution(graph::AttrInfo& attr, 
							  std::string& op_class_name, 
							  std::string& node_name, 
							  std::string& weights_ptr_name, 
							  WeightsWritter& writter) {
	// parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<NV>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0]*weights_shape[1];

	writter.register_weights(node_name, weights);
	if(bias_term) {
		auto bias = get_attr<PBlock<NV>>("weight_2", attr);
		writter.register_weights(node_name, bias);
	}

	auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
    code_w.feed("%s.load_param(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s+%d,%s+%d);\n", node_name.c_str(),
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
	return code_w.get_code_string();
}

// ParserConvolutionRelu
std::string ParserConvolutionRelu(graph::AttrInfo& attr, 
								  std::string& op_class_name, 
								  std::string& node_name, 
								  std::string& weights_ptr_name, 
								  WeightsWritter& writter) {
	// parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<NV>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0]*weights_shape[1];

	writter.register_weights(node_name, weights);
	if(bias_term) {
		auto bias = get_attr<PBlock<NV>>("weight_2", attr);
		writter.register_weights(node_name, bias);
	}

    auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
    code_w.feed("%s.load_param(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,Active_relu,%s+%d,%s+%d);\n", node_name.c_str(),
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
	return code_w.get_code_string();
}

std::string ParserConvBatchnormScale(graph::AttrInfo& attr, 
					                 std::string& op_class_name, 
					                 std::string& node_name, 
					                 std::string& weights_ptr_name, 
    				                 WeightsWritter& writter) {
    // parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<NV>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0]*weights_shape[1];

    // get batchnorm param
    auto epsilon = get_attr<float>("batchnorm_0_epsilon", attr);
    auto momentum = get_attr<float>("batchnorm_0_momentum", attr);
    auto batch_norm_weight_1 = get_attr<PBlock<NV>>("batchnorm_0_weight_1", attr);
    auto batch_norm_weight_1_vector = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = get_attr<PBlock<NV>>("batchnorm_0_weight_2", attr);
    auto batch_norm_weight_2_vector = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = get_attr<PBlock<NV>>("batchnorm_0_weight_3", attr);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector();

    // get scale param
    auto scale_num_axes = get_attr<int>("scale_0_num_axes", attr);
    auto scale_bias_term = get_attr<bool>("scale_0_bias_term", attr);
    auto scale_axis = get_attr<int>("scale_0_axis", attr);
    auto scale_weight_1 = get_attr<PBlock<NV>>("scale_0_weight_1", attr);
    auto scale_weight_1_vector = scale_weight_1.vector();
    auto scale_weight_2 = get_attr<PBlock<NV>>("scale_0_weight_2", attr);
    auto scale_weight_2_vector = scale_weight_2.vector();


	if(bias_term) {
		auto bias = get_attr<PBlock<NV>>("weight_2", attr);
		update_weights<float, NV>(weights, bias,
					              weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
					              bias_term, 
					              batch_norm_weight_3_vector[0], epsilon, 
					              batch_norm_weight_1_vector, 
					              batch_norm_weight_2_vector,
					              scale_weight_1_vector,
					              scale_weight_2_vector,
					              scale_bias_term);
	
		
		writter.register_weights(node_name, weights);
		writter.register_weights(node_name, bias);
	} else {
		auto bias = PBlock<NV>();
		update_weights<float, NV>(weights, bias,
					              weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
					              false, 
					              batch_norm_weight_3_vector[0], epsilon, 
					              batch_norm_weight_1_vector, 
					              batch_norm_weight_2_vector,
					              scale_weight_1_vector,
					              scale_weight_2_vector,
					              scale_bias_term);

		writter.register_weights(node_name, weights);
		writter.register_weights(node_name, bias);
	}

    auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
    code_w.feed("%s.load_param(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s+%d,%s+%d);\n", node_name.c_str(),
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
                                           "true",
                                           weights_ptr_name.c_str(),
                                           offset_info.weights[0].offset,
                                           weights_ptr_name.c_str(),
                                           bias_term ? offset_info.weights[1].offset : 0);
	return code_w.get_code_string();
}

// SaberConvBatchnormScaleRelu
std::string ParserConvBatchnormScaleRelu(graph::AttrInfo& attr, 
					                     std::string& op_class_name, 
					                     std::string& node_name, 
					                     std::string& weights_ptr_name, 
    				                     WeightsWritter& writter) {
    // parsing parameter
	auto group = get_attr<int>("group", attr);
	auto bias_term = get_attr<bool>("bias_term", attr);
	auto padding = get_attr<PTuple<int>>("padding", attr);
	auto strides = get_attr<PTuple<int>>("strides", attr);
	auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
	auto filter_num = get_attr<int>("filter_num", attr);
	auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
	auto axis = get_attr<int>("axis", attr);

	auto weights = get_attr<PBlock<NV>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape[2]*weights_shape[3];
    int num_output = weights_shape[0]*weights_shape[1];

    // get batchnorm param
    auto epsilon = get_attr<float>("batchnorm_0_epsilon", attr);
    auto momentum = get_attr<float>("batchnorm_0_momentum", attr);
    auto batch_norm_weight_1 = get_attr<PBlock<NV>>("batchnorm_0_weight_1", attr);
    auto batch_norm_weight_1_vector = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = get_attr<PBlock<NV>>("batchnorm_0_weight_2", attr);
    auto batch_norm_weight_2_vector = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = get_attr<PBlock<NV>>("batchnorm_0_weight_3", attr);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector(); 

    // get scale param
    auto scale_num_axes = get_attr<int>("scale_0_num_axes", attr);
    auto scale_bias_term = get_attr<bool>("scale_0_bias_term", attr);
    auto scale_axis = get_attr<int>("scale_0_axis", attr);
    auto scale_weight_1 = get_attr<PBlock<NV>>("scale_0_weight_1", attr);
    auto scale_weight_1_vector = scale_weight_1.vector();
    auto scale_weight_2 = get_attr<PBlock<NV>>("scale_0_weight_2", attr);
    auto scale_weight_2_vector = scale_weight_2.vector();

	if(bias_term) {
		auto bias = get_attr<PBlock<NV>>("weight_2", attr);
		update_weights<float, NV>(weights, bias,
					              weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
					              bias_term, 
					              batch_norm_weight_3_vector[0], epsilon, 
					              batch_norm_weight_1_vector, 
					              batch_norm_weight_2_vector,
					              scale_weight_1_vector,
					              scale_weight_2_vector,
					              scale_bias_term);
	
		
		writter.register_weights(node_name, weights);
		writter.register_weights(node_name, bias);
	} else {
		auto bias = PBlock<NV>();
		update_weights<float, NV>(weights, bias,
					              weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
					              false, 
					              batch_norm_weight_3_vector[0], epsilon, 
					              batch_norm_weight_1_vector, 
					              batch_norm_weight_2_vector,
					              scale_weight_1_vector,
					              scale_weight_2_vector,
					              scale_bias_term);

		writter.register_weights(node_name, weights);
		writter.register_weights(node_name, bias);
	}

    auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
    code_w.feed("%s.load_param(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,Active_relu,%s+%d,%s+%d);\n", node_name.c_str(),
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
                                           "true",
                                           weights_ptr_name.c_str(),
                                           offset_info.weights[0].offset,
                                           weights_ptr_name.c_str(),
                                           bias_term ? offset_info.weights[1].offset : 0);
	return code_w.get_code_string();
}

// SaberConcat
std::string ParserConcat(graph::AttrInfo& attr, 
						 std::string& op_class_name, 
						 std::string& node_name, 
						 std::string& weights_ptr_name, 
						 WeightsWritter& writter) {
	// parsing parameter
	auto axis = get_attr<int>("axis", attr);
	// gen cpp code
	CodeWritter code_w;
	code_w.feed("%s.load_param(%d);\n", node_name.c_str(), axis);
	return code_w.get_code_string();
}

// SaberDectionOutput
std::string ParserDectionOutput(graph::AttrInfo& attr, 
								std::string& op_class_name, 
								std::string& node_name, 
								std::string& weights_ptr_name, 
								WeightsWritter& writter) {
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
	code_w.feed("%s.load_param(%s,%s,%d,%d,%d,%s,%f,%d,%f,%f);\n", node_name.c_str(), 
										 flag_share_location ? "true":"false",
										 flag_var_in_target ? "true":"false",
										 classes_num,
										 background_id,
										 keep_top_k,
										 code_type.c_str(),
										 conf_thresh,
										 nms_top_k,
										 nms_thresh,
										 nms_eta); 
	return code_w.get_code_string();
}

// SaberEltwise
std::string ParserEltwise(graph::AttrInfo& attr, 
						  std::string& op_class_name, 
						  std::string& node_name, 
						  std::string& weights_ptr_name, 
						  WeightsWritter& writter) {
	// parsing parameter
    auto type = get_attr<std::string>("type", attr); 
    auto coeff = get_attr<PTuple<float>>("coeff", attr);

	std::string eltwise_type_str("Eltwise_unknow");

	if (type == "Add") {
        eltwise_type_str = "Eltwise_sum";
    } else if (type == "Max") {
        eltwise_type_str = "Eltwise_max";
    } else {
        eltwise_type_str = "Eltwise_prod";
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
	code_w.feed("%s.load_param(%s, %s);\n", node_name.c_str(), eltwise_type_str.c_str(), 
											coeff_vec_code.get_code_string().c_str());
	return code_w.get_code_string();
}

// SaberActivation
std::string ParserActivation(graph::AttrInfo& attr, 
							 std::string& op_class_name, 
							 std::string& node_name, 
							 std::string& weights_ptr_name, 
							 WeightsWritter& writter) {
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
	code_w.feed("%s.load_param(%s);\n", node_name.c_str(), act_type.c_str());
	return code_w.get_code_string();
}

std::string ParserRelu(graph::AttrInfo& attr, 
					   std::string& op_class_name, 
					   std::string& node_name, 
					   std::string& weights_ptr_name, 
					   WeightsWritter& writter) {
    // parsing parameter
    auto alpha = get_attr<float>("alpha", attr);

	std::string act_type("Active_relu");

	// gen cpp code
	CodeWritter code_w;
	code_w.feed("%s.load_param(%s,%f);\n", node_name.c_str(), act_type.c_str(),alpha);
	return code_w.get_code_string();
}

// SaberFc
std::string ParserFc(graph::AttrInfo& attr, 
					 std::string& op_class_name, 
					 std::string& node_name, 
					 std::string& weights_ptr_name, 
					 WeightsWritter& writter) {
	// parsing parameter
    auto axis = get_attr<int>("axis", attr); 
    auto out_dim = get_attr<int>("out_dim", attr); 
    auto bias_term = get_attr<bool>("bias_term", attr);
	
	auto weights = get_attr<PBlock<NV>>("weight_1", attr);
	writter.register_weights(node_name, weights);
	if(bias_term) {
		auto bias = get_attr<PBlock<NV>>("weight_2", attr);
		writter.register_weights(node_name, bias);
	}

	auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
	CodeWritter code_w;
	code_w.feed("%s.load_param(%d,%d,false,%s,%s+%d,%s+%d);\n", node_name.c_str(), axis, out_dim,
												    bias_term ? "true":"false",
                                                    weights_ptr_name.c_str(),
                                                    offset_info.weights[0].offset,
                                                    weights_ptr_name.c_str(),
                                                    bias_term ? offset_info.weights[1].offset : 0);
	return code_w.get_code_string();
}

// SaberPermute
std::string ParserPermute(graph::AttrInfo& attr, 
						  std::string& op_class_name, 
						  std::string& node_name, 
						  std::string& weights_ptr_name, 
						  WeightsWritter& writter) {
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
	code_w.feed("%s.load_param(%s);\n", node_name.c_str(), dims_vec_code.get_code_string().c_str()); 
	return code_w.get_code_string();
}

// SaberPooling
std::string ParserPooling(graph::AttrInfo& attr, 
						  std::string& op_class_name, 
						  std::string& node_name, 
						  std::string& weights_ptr_name, 
						  WeightsWritter& writter) {
	// parsing parameter
    auto global_pooling = get_attr<bool>("global_pooling", attr);
    auto pool_padding = get_attr<PTuple<int>>("padding", attr);
    auto pool_strides = get_attr<PTuple<int>>("strides", attr);
    auto pool_size = get_attr<PTuple<int>>("pool_size", attr);
    auto pool_method = get_attr<std::string>("method", attr);	

	// gen cpp code
    CodeWritter code_w; 
	code_w.feed("%s.load_param(%s,%s,%d,%d,%d,%d,%d,%d);\n", node_name.c_str(), 
                                                             pool_method.c_str(),
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
std::string ParserPrelu(graph::AttrInfo& attr, 
						std::string& op_class_name, 
						std::string& node_name, 
						std::string& weights_ptr_name, 
						WeightsWritter& writter) {
	// parsing parameter
	auto channel_shared = get_attr<bool>("channel_shared", attr);

	auto weights = get_attr<PBlock<NV>>("weight_1", attr);
	writter.register_weights(node_name, weights);

    auto offset_info = writter.get_weights_by_name(node_name);

	// gen cpp code
    CodeWritter code_w; 
	code_w.feed("%s.load_param(%s,%s+%d);\n", node_name.c_str(), channel_shared ? "true":"false",
                                        weights_ptr_name.c_str(),
                                        offset_info.weights[0].offset);
	return code_w.get_code_string();
}

// SaberPriorBox
std::string ParserPriorBox(graph::AttrInfo& attr, 
						   std::string& op_class_name, 
						   std::string& node_name, 
						   std::string& weights_ptr_name, 
						   WeightsWritter& writter) {
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
	code_w.feed("%s.load_param(%s,%s,%s,%s,%s,%s,%d,%d,%f,%f,%f);\n", node_name.c_str(),
										 flip_flag ? "ture":"false",
										 clip_flag ? "true":"false",
										 gen_vec_code(min_size).c_str(),
										 gen_vec_code(max_size).c_str(),
										 gen_vec_code(as_ratio).c_str(),
										 gen_vec_code(var).c_str(),
										 image_w, 
										 image_h,
										 step_w,
										 step_h,
										 offset); 
	return code_w.get_code_string();
}

// SaberSlice
std::string ParserSlice(graph::AttrInfo& attr, 
						std::string& op_class_name, 
						std::string& node_name, 
						std::string& weights_ptr_name, 
						WeightsWritter& writter) {
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
	code_w.feed("%s.load_param(%d,%s);\n", node_name.c_str(), axis, slice_point_vec_code.get_code_string().c_str());
	return code_w.get_code_string();
}

// SaberSoftmax
std::string ParserSoftmax(graph::AttrInfo& attr, 
						  std::string& op_class_name, 
						  std::string& node_name, 
						  std::string& weights_ptr_name, 
						  WeightsWritter& writter) {
	// parsing parameter
    auto axis = get_attr<int>("axis", attr);

	// gen cpp code
    CodeWritter code_w; 
	code_w.feed("%s.load_param(%d);\n", node_name.c_str(), axis); 
	return code_w.get_code_string();
}

std::unordered_map<std::string, OpParser> OPERATION_MAP({
	{"Input", {"Input", not_impl_yet} },
	{"Convolution", {"SaberConv2D", ParserConvolution} }, // done
	{"Activation", {"SaberActivation", ParserActivation} }, // done
    {"ReLU", {"SaberActivation",ParserRelu}}, // done
	{"ConvRelu", {"SaberConvAct2D", ParserConvolutionRelu} },  // done
	{"ConvBatchnormScaleRelu", {"SaberConvAct2D", ParserConvBatchnormScaleRelu}}, // done have question ??
	{"ConvBatchnormScale", {"SaberConv2D", ParserConvBatchnormScale}}, //done
	{"Concat", {"SaberConcat", ParserConcat} },  // done
	{"DetectionOutput", {"SaberDectionOutput", ParserDectionOutput} }, // done 
	{"Eltwise", {"SaberEltwise", ParserEltwise} }, //done
	{"Eltwise", {"SaberEltwiseRelu", not_impl_yet}}, // not impl ??
	{"Dense", {"SaberFc", ParserFc} }, // done
	{"Permute", {"SaberPermute", ParserPermute} }, // done
	{"Pooling", {"SaberPooling", ParserPooling} }, // done
	{"ReLU", {"SaberPrelu", ParserPrelu} }, // done
	{"PriorBox", {"SaberPriorBox", ParserPriorBox} }, // done
	{"Slice", {"SaberSlice", ParserSlice} }, // done
	{"Softmax", {"SaberSoftmax", ParserSoftmax} } // done
});

} /* namespace lite */

} /* namespace anakin */

