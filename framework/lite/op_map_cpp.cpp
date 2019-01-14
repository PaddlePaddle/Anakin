#include "framework/lite/op_map.h"
#include "framework/lite/utils.h"

namespace anakin {

namespace lite {

//using namespace anakin;
//using namespace anakin::lite;

std::string not_impl_yet(graph::AttrInfo&,
                         std::string& code_name,
                         std::string& op_class_name,
                         std::string& node_name,
                         std::string& weights_ptr_name,
                         WeightsWritter& writter,
                         bool gen_param,
                         int lite_mode,
                         DataType op_precision) {
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
                              bool gen_param,
                              int lite_mode,
                              DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto &weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    auto offset_info = writter.get_weights_by_name(node_name);
    float alpha = 0.f;
    if (find_attr("relu_0_alpha", attr) == SaberSuccess) {
        alpha = get_attr<float>("relu_0_alpha", attr);
    }

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d\n",
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
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    bias_term ? offset_info.weights[2].offset : 0,
                    0, //flag_eltwise
                    0, //set flag_act true
                    (int)Active_relu,
                    alpha, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0/*prelu weights*/);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d);\n",
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
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[2].offset : 0,
                    "false", //flag_eltwise
                    "false", //set flag_act true
                    "Active_relu", alpha, 0.f, "false", weights_ptr_name.c_str(), 0);
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
                        bool gen_param,
                        int lite_mode,
                        DataType op_precision) {
    // parsing parameter
    auto power = get_attr<float>("power", attr);
    auto scale = get_attr<float>("scale", attr);
    auto shift = get_attr<float>("shift", attr);

    // gen cpp code
    CodeWritter code_w;

    if (gen_param) {
        code_w.feed("%f %f %f\n", scale, shift, power);
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
                              bool gen_param,
                              int lite_mode,
                              DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto& weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_deconv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    auto offset_info = writter.get_weights_by_name(node_name);
    float alpha = 0.f;
    if (find_attr("relu_0_alpha", attr) == SaberSuccess) {
        alpha = get_attr<float>("relu_0_alpha", attr);
    }
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d\n",
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
                    (int) weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    bias_term ? offset_info.weights[2].offset : 0,
                    0, //flag_eltwise
                    0, //set flag_act
                    (int)Active_relu,
                    alpha, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0/*prelu weights*/);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d);\n",
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
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[2].offset : 0,
                    "false", //flag_eltwise
                    "false", //set flag_act true
                    "Active_relu", alpha, 0.f, "false", weights_ptr_name.c_str(), 0);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

// ParserDeConvolutionRelu
std::string ParserDeConvolutionRelu(graph::AttrInfo& attr,
                                  std::string& code_name,
                                  std::string& op_class_name,
                                  std::string& node_name,
                                  std::string& weights_ptr_name,
                                  WeightsWritter& writter,
                                  bool gen_param,
                                  int lite_mode,
                                  DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto& weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_deconv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    //write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    auto offset_info = writter.get_weights_by_name(node_name);
    float alpha = 0.f;
    if (find_attr("relu_0_alpha", attr) == SaberSuccess) {
        alpha = get_attr<float>("relu_0_alpha", attr);
    }
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d\n",
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
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    bias_term ? offset_info.weights[2].offset : 0,
                    0, //flag_eltwise
                    1, //set flag_act true
                    (int)Active_relu,
                    alpha, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0/*prelu weights*/);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d);\n",
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
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[2].offset : 0,
                    "false", //flag_eltwise
                    "true", //set flag_act true
                    "Active_relu", alpha, 0.f, "false", weights_ptr_name.c_str(), 0);
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
                                  bool gen_param,
                                  int lite_mode,
                                  DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto& weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    auto offset_info = writter.get_weights_by_name(node_name);
    float alpha = 0.f;
    if (find_attr("relu_0_alpha", attr) == SaberSuccess) {
        alpha = get_attr<float>("relu_0_alpha", attr);
    }
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d\n",
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
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    bias_term ? offset_info.weights[2].offset : 0,
                    0, //flag_eltwise
                    1, //set flag_act true
                    (int)Active_relu,
                    alpha, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0/*prelu weights*/);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d);\n",
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
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[2].offset : 0,
                    "false", //flag_eltwise
                    "true", //set flag_act true
                    "Active_relu", alpha, 0.f, "false", weights_ptr_name.c_str(), 0);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

// ParserConvAct //also with eltwise
std::string ParserConvAct(graph::AttrInfo& attr,
                                  std::string& code_name,
                                  std::string& op_class_name,
                                  std::string& node_name,
                                  std::string& weights_ptr_name,
                                  WeightsWritter& writter,
                                  bool gen_param,
                                  int lite_mode,
                                  DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto& weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    float alpha = 0.f;
    float coef = 0.f;
    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    // get act param
    ActiveType act_type = Active_unknow;
    std::string act_type_str;
    int prelu_size = 0;
    bool act_shared = false;
    int act_weights_offset = 0;
    auto type = get_attr<std::string>("act_0_type", attr);
    if (type == "TanH") {
        act_type = Active_tanh;
        act_type_str = "Active_tanh";
        //LOG(FATAL) << "Activation TanH not supported now.";
    } else if (type == "Sigmoid") {
        act_type = Active_sigmoid;
        act_type_str = "Active_sigmoid";
        //LOG(FATAL) << "Activation Sigmoid not supported now.";
    } else if (type == "PReLU") {
        act_type = Active_prelu;
        act_shared = get_attr<bool>("act_0_channel_shared", attr);
        auto prelu_weights = get_attr<PBlock<X86>>("act_0_weight_1", attr);
        writter.register_weights(node_name, prelu_weights);
        prelu_size = prelu_weights.count();
        LOG(INFO) << node_name << " write prelu weights: " << prelu_weights.count();
        act_type_str = "Active_prelu";
    } else if (type == "Stanh") {
        LOG(FATAL) << "Activation Stanh not supported now.";
    } else if (type == "Relu") {
        act_type = Active_relu;
        act_type_str = "Active_relu";
        if (find_attr("relu_0_alpha", attr) == SaberSuccess) {
            alpha = get_attr<float>("relu_0_alpha", attr);
        }
    } else if (type == "ClippedRelu") {
        act_type = Active_clipped_relu;
        act_type_str = "Active_clipped_relu";
        coef = get_attr<float>("clip_relu_num", attr);
    } else if (type == "Elu") {
        LOG(FATAL) << "Activation Elu not supported now.";
    } else {
        LOG(FATAL) << "Other Activation type" << type << " should be replace by other ops.";
    }

    auto offset_info = writter.get_weights_by_name(node_name);

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d\n",
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
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    bias_term ? offset_info.weights[2].offset : 0,
                    0, //flag_eltwise
                    1, //set flag_act true
                    (int)act_type,
                    alpha, //neg slope
                    coef, //act_coef
                    act_shared, //prelu, channel_shared
                    act_weights_offset,/*prelu weights*/
                    offset_info.weights[3].offset/*prelu weights size*/);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d);\n",
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
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[2].offset : 0,
                    "false", //flag_eltwise
                    "true", //set flag_act true
                    act_type_str.c_str(), alpha, coef, act_shared? "true" : "false", weights_ptr_name.c_str(), offset_info.weights[3].offset);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }

    return code_w.get_code_string();
}

// ParserConvolutionReluPool
std::string ParserConvolutionReluPool(graph::AttrInfo& attr,
                                  std::string& code_name,
                                  std::string& op_class_name,
                                  std::string& node_name,
                                  std::string& weights_ptr_name,
                                  WeightsWritter& writter,
                                  bool gen_param,
                                  int lite_mode,
                                  DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto& weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
    }

    // parsing pooling parameter
    auto ceil_mode = !get_attr<bool>("cmp_out_shape_floor_as_conv", attr);
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

    //! activation param
    float alpha = 0.f;
    if (find_attr("relu_0_alpha", attr) == SaberSuccess) {
        alpha = get_attr<float>("relu_0_alpha", attr);
    }
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d %d %d %d %d %d %d %d %d\n",
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
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    bias_term ? offset_info.weights[2].offset : 0,
                    0, //flag_eltwise
                    1, //set flag_act true
                    (int)Active_relu,
                    alpha, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0,/*prelu weights*/
                    (int)pool_type,
                    global_pooling? 1 : 0,
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    (int)ceil_mode);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d,%s,%s,%d,%d,%d,%d,%d,%d);\n",
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
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[2].offset : 0,
                    "false", //flag_eltwise
                    "true", //set flag_act true
                    "Active_relu", alpha, 0.f, "false", weights_ptr_name.c_str(), 0,
                    str_pool_method.c_str(), global_pooling? "true" : "false",
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    (int)ceil_mode);
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
                                bool gen_param,
                                int lite_mode,
                                DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    auto offset_info = writter.get_weights_by_name(node_name);
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d\n",
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
                    1, //BIAS term
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    offset_info.weights[2].offset,
                    0, //flag_eltwise
                    0, //set flag_act true
                    (int)Active_relu,
                    0.f, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0/*prelu weights*/);

    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d);\n",
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
                    "true",
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[2].offset,
                    "false", //flag_eltwise
                    "false", //set flag_act true
                    "Active_relu", 0.f, 0.f, "false", weights_ptr_name.c_str(), 0);
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
                                     bool gen_param,
                                     int lite_mode,
                                     DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto& weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    auto offset_info = writter.get_weights_by_name(node_name);
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d\n",
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
                    1, //BIAS term
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    offset_info.weights[2].offset,
                    0, //flag_eltwise
                    0, //set flag_act false
                    (int)Active_relu,
                    0.f, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0/*prelu weights*/);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d);\n",
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
                    "true",
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[2].offset,
                    "false", //flag_eltwise
                    "false", //set flag_act true
                    "Active_relu", 0.f, 0.f, "false", weights_ptr_name.c_str(), 0);
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
                                         bool gen_param,
                                         int lite_mode,
                                         DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto &weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }
    auto offset_info = writter.get_weights_by_name(node_name);

    //! activation param
    float alpha = 0.f;
    if (find_attr("relu_0_alpha", attr) == SaberSuccess) {
        alpha = get_attr<float>("relu_0_alpha", attr);
    }
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d\n",
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
                    1, //BIAS term
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    offset_info.weights[2].offset,
                    0, //flag_eltwise
                    1, //set flag_act false
                    (int)Active_relu,
                    alpha, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0/*prelu weights*/);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d);\n",
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
                    "true",
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[2].offset,
                    "false", //flag_eltwise
                    "true", //set flag_act true
                    "Active_relu", alpha, 0.f, "false", weights_ptr_name.c_str(), 0);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

// SaberConvBatchnormScaleReluPool
std::string ParserConvBatchnormScaleReluPool(graph::AttrInfo& attr,
                                         std::string& code_name,
                                         std::string& op_class_name,
                                         std::string& node_name,
                                         std::string& weights_ptr_name,
                                         WeightsWritter& writter,
                                         bool gen_param,
                                         int lite_mode,
                                         DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto padding = get_attr<PTuple<int>>("padding", attr);
    auto strides = get_attr<PTuple<int>>("strides", attr);
    auto dilation_rate = get_attr<PTuple<int>>("dilation_rate", attr);
    auto filter_num = get_attr<int>("filter_num", attr);
    auto kernel_size = get_attr<PTuple<int>>("kernel_size", attr);
    auto axis = get_attr<int>("axis", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto& weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();
    int num_output = weights_shape[0];

    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
    }

    // parsing pooling parameter
    auto ceil_mode = !get_attr<bool>("cmp_out_shape_floor_as_conv", attr);
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

    //! activation param
    float alpha = 0.f;
    if (find_attr("relu_0_alpha", attr) == SaberSuccess) {
        alpha = get_attr<float>("relu_0_alpha", attr);
    }
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f %f %d %d %d %d %d %d %d %d %d %d\n",
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
                    1, //bias term
                    (int)weights_dtype,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    offset_info.weights[2].offset,
                    0, //flag_eltwise
                    1, //set flag_act true
                    (int)Active_relu,
                    alpha, //neg slope
                    0.f, //act_coef
                    0, //prelu, channel_shared
                    0,/*prelu weights*/
                    (int)pool_type,
                    global_pooling? 1 : 0,
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    (int)ceil_mode);
    } else {
        code_w.feed("ParamBase* %s_param = new Conv2DParam(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s+%d,%s+%d,%s+%d,%s,%s,%s,%f,%f,%s,%s+%d,%s,%s,%d,%d,%d,%d,%d,%d);\n",
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
                    "true",
                    (int)weights_dtype,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[2].offset,
                    "false", //flag_eltwise
                    "true", //set flag_act true
                    "Active_relu", alpha, 0.f, "false", weights_ptr_name.c_str(), 0,
                    str_pool_method.c_str(), global_pooling? "true" : "false",
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    (int)ceil_mode);
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
                         bool gen_param,
                         int lite_mode,
                         DataType op_precision) {
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
                                bool gen_param,
                                int lite_mode,
                                DataType op_precision) {
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
        code_w.feed("%d %f %d %d %d %d %f %f %d %d\n",
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
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
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
    for (int i=0; i<coeff.size()-1; i++) {
        coeff_vec_code<<coeff.vector()[i]<<",";
    }
    if (coeff.size() > 0) {
        coeff_vec_code<<coeff.vector()[coeff.size()-1] << "}";
    } else {
        coeff_vec_code<<"}";
    }

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d ", (int)et_type,
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

// SaberEltwiseAct
std::string ParserEltwiseRelu(graph::AttrInfo& attr,
                          std::string& code_name,
                          std::string& op_class_name,
                          std::string& node_name,
                          std::string& weights_ptr_name,
                          WeightsWritter& writter,
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
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
    for (int i=0; i<coeff.size()-1; i++) {
        coeff_vec_code<<coeff.vector()[i]<<",";
    }
    if (coeff.size() > 0) {
        coeff_vec_code<<coeff.vector()[coeff.size()-1] << "}";
    } else {
        coeff_vec_code<<"}";
    }

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d ", (int)et_type,
                    coeff.size());
        for (int i = 0; i < coeff.size(); ++i) {
            code_w << coeff[i] << " ";
        }
        code_w << (int)Active_relu << " " << 0.f << " " << 0.f << " " << \
                0 << " " << 0 << " " << 0 << "\n";
    } else  {
        code_w.feed("ParamBase* %s_param = new EltwiseActParam(%s, %s, %s, %f, %f, %s, %s, %d);\n",
                    node_name.c_str(),
                    eltwise_type_str.c_str(),
                    coeff_vec_code.get_code_string().c_str(),
                    "Active_relu",
                    0.f,
                    0.f,
                    "false",
                    "nullptr",
                    0);

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}
// SaberEltwiseAct
std::string ParserEltwisePRelu(graph::AttrInfo& attr,
                          std::string& code_name,
                          std::string& op_class_name,
                          std::string& node_name,
                          std::string& weights_ptr_name,
                          WeightsWritter& writter,
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
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
    for (int i=0; i<coeff.size()-1; i++) {
        coeff_vec_code<<coeff.vector()[i]<<",";
    }
    if (coeff.size() > 0) {
        coeff_vec_code<<coeff.vector()[coeff.size()-1] << "}";
    } else {
        coeff_vec_code<<"}";
    }
    //prelu
    auto prelu_channel_shared = get_attr<bool>("prelu_0_channel_shared", attr);
    auto prelu_weights = get_attr<PBlock<X86>>("prelu_0_weight_1", attr);

    writter.register_weights(node_name, prelu_weights);
    LOG(INFO) << node_name << " write weights: " << prelu_weights.count();

    auto offset_info = writter.get_weights_by_name(node_name);
    // gen cpp code
    CodeWritter code_w;

    if (gen_param) {
        code_w.feed("%d %d ", (int)et_type,
                    coeff.size());
        for (int i = 0; i < coeff.size(); ++i) {
            code_w << coeff[i] << " ";
        }
        code_w << (int)Active_prelu << " " << 0.f << " " << 0.f << " " << \
                (prelu_channel_shared ? 1 : 0) << " " << offset_info.weights[0].offset << " " << prelu_weights.count() << "\n";
        //code_w << "\n";
    } else  {
        code_w.feed("ParamBase* %s_param = new EltwiseActParam(%s, %s, %s, %f, %f, %s, %s+%d, %d);\n",
                    node_name.c_str(),
                    eltwise_type_str.c_str(),
                    coeff_vec_code.get_code_string().c_str(),
                    "Active_prelu",
                    0.f,
                    0.f,
                    (prelu_channel_shared ? "true" : "false"),
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    prelu_weights.count());

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
                             bool gen_param,
                             int lite_mode,
                             DataType op_precision) {
    // parsing parameter
    auto type = get_attr<std::string>("type", attr);

    std::string act_type("Active_unknow");

    //! ActiveType act_type, float neg_slope = 0.f, float coef = 1.f, bool channel_shared = false, const float* weights = nullptr, const size = prelu_size
    // gen cpp code
    CodeWritter code_w;
    if (type == "TanH") {
        if (gen_param) {
            code_w << (int)Active_tanh << " " << 0.f << " " << 0.f << " " << 0 << " " << 0 << " " << 0 << "\n";
        } else {
            act_type = "Active_tanh";
            code_w.feed("ParamBase* %s_param = new ActivationParam(%s);\n",
                        node_name.c_str(),
                        act_type.c_str());
            code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
        }

    } else if (type == "Sigmoid") {
        if (gen_param) {
            code_w << (int)Active_sigmoid << " " << 0.f << " " << 0.f << " " << 0 << " " << 0 << " " << 0 << "\n";
        } else {
            act_type = "Active_sigmoid";
            code_w.feed("ParamBase* %s_param = new ActivationParam(%s);\n",
                        node_name.c_str(),
                        act_type.c_str());

            code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
        }

    } else if (type == "ReLU") {
        auto alpha = get_attr<float>("alpha", attr);
        if (gen_param) {
            code_w << (int)Active_relu << " " << alpha << " " << 0.f << " " << 0 << " " << 0 << " " << 0 << "\n";
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
        auto prelu_weights = get_attr<PBlock<X86>>("weight_1", attr);

        writter.register_weights(node_name, prelu_weights);
        LOG(INFO) << node_name << " write weights: " << prelu_weights.count();
        auto offset_info = writter.get_weights_by_name(node_name);
        if (gen_param) {
            code_w << (int)Active_prelu << " " << 0.f << " " << 0.f << " " << \
                (prelu_channel_shared ? 1 : 0) << " " << offset_info.weights[0].offset << " " << prelu_weights.count() << "\n";
        } else {
            code_w.feed("ParamBase* %s_param = new ActivationParam(%s, %f, %f, %s, %s+%d, %d);\n",
                        node_name.c_str(),
                        act_type.c_str(),
                        0.f,
                        0.f,
                        prelu_channel_shared ? "true" : "false",
                        weights_ptr_name.c_str(),
                        offset_info.weights[0].offset,
                        prelu_weights.count());

            code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
        }

    } else if (type == "ClippedRelu") {
        auto coef = get_attr<float>("clip_relu_num", attr);
        if (gen_param) {
            code_w << (int)Active_clipped_relu << " " << 0.f << " " << coef << " " << 0 << " " << 0 << " " << 0 << "\n";
        } else {
            act_type = "Active_clipped_relu";
            code_w.feed("ParamBase* %s_param = new ActivationParam(%s, %f, %f, %d, %d, %d);\n",
                        node_name.c_str(),
                        act_type.c_str(), 0.f, coef, 0, 0, 0);

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
                       WeightsWritter& writter,
                       bool gen_param,
                       int lite_mode,
                       DataType op_precision) {
    // parsing parameter
    auto alpha = get_attr<float>("alpha", attr);

    std::string act_type("Active_relu");

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << (int)Active_relu << " " << alpha << " " << 0.f << " " << 0 << " " << 0 << " " << 0 << "\n";
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
                     bool gen_param,
                     int lite_mode,
                     DataType op_precision) {
    // parsing parameter
    auto axis = get_attr<int>("axis", attr);
    auto out_dim = get_attr<int>("out_dim", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);

    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    auto& weights_tensor = weights.h_tensor();
    int weights_size = weights_shape.count();

    // write weights
    trans_conv_weights_inplace(weights_tensor, op_precision, lite_mode);
    auto weights_dtype = weights_tensor.get_dtype();
    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count() << ", weights dtype: " << weights_dtype;

    // write scale
    PBlock<X86> scale_tensor(AK_FLOAT);
    std::vector<float> scale = weights_tensor.get_scale();
    Shape scale_shape({1, 1, 1, scale.size()});
    scale_tensor.h_tensor().reshape(scale_shape);
    float* scale_ptr = (float*)scale_tensor.h_tensor().mutable_data();
    for (int i = 0; i < scale.size(); ++i){
        scale_ptr[i] = scale[i];
    }
    writter.register_weights(node_name, scale_tensor);
    LOG(INFO) << node_name << "write scale: " << scale_ptr[0];

    // write bias
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        writter.register_weights(node_name, bias);
    }

    auto offset_info = writter.get_weights_by_name(node_name);

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d %d %d\n",
                    axis,
                    out_dim,
                    bias_term ? 1 : 0,
                    (int)(weights_dtype),
                    weights_size,
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    bias_term ? offset_info.weights[2].offset : 0,
                    0);
    } else {
        code_w.feed("ParamBase* %s_param = new FcParam(%d,%d,%s,%d,%d,%s+%d,%s+%d,%s+%d,%s);\n",
                    node_name.c_str(),
                    axis,
                    out_dim,
                    bias_term ? "true":"false",
                    (int)(weights_dtype),
                    weights_size,
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[2].offset : 0,
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
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
    // parsing parameter
    auto dims = get_attr<PTuple<int>>("dims", attr);

    CodeWritter dims_vec_code;
    dims_vec_code<<"{";
    for (int i=0; i<dims.size()-1; i++) {
        dims_vec_code<<dims.vector()[i]<<",";
    }
    if (dims.size() > 0) {
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
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
    // parsing parameter
    auto ceil_mode = !get_attr<bool>("cmp_out_shape_floor_as_conv", attr);
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
        code_w.feed("%d %d %d %d %d %d %d %d %d\n",
                    (int)pool_type,
                    global_pooling ? 1 : 0,
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    (int)ceil_mode);
    } else {
        code_w.feed("ParamBase* %s_param = new PoolParam(%s,%s,%d,%d,%d,%d,%d,%d,%d);\n",
                    node_name.c_str(),
                    str_pool_method.c_str(),
                    global_pooling ? "true" : "false",
                    pool_size[1],
                    pool_size[0],
                    pool_strides[1],
                    pool_strides[0],
                    pool_padding[1],
                    pool_padding[0],
                    (int)ceil_mode);
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
                           bool gen_param,
                           int lite_mode,
                           DataType op_precision) {
    // parsing parameter
    auto min_size  = get_attr<PTuple<float>>("min_size", attr);
    auto max_size  = get_attr<PTuple<float>>("max_size", attr);
    auto as_ratio  = get_attr<PTuple<float>>("aspect_ratio", attr);
    //add
    std::vector<float> fixed_size, fixed_ratio, density;
    if (find_attr("fixed_size", attr) == SaberSuccess) {
        auto fix_size  = get_attr<PTuple<float>>("fixed_size", attr);
        fixed_size = fix_size.vector();
    }

    if (find_attr("fixed_ratio", attr) == SaberSuccess) {
        auto fix_ratio  = get_attr<PTuple<float>>("fixed_ratio", attr);
        fixed_ratio = fix_ratio.vector();
    }

    if (find_attr("density", attr) == SaberSuccess) {
        auto den = get_attr<PTuple<float>>("density", attr);
        density = den.vector();
    }

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
        for (int i=0; i<ptuple.size()-1; i++) {
            dims_vec_code<<ptuple.vector()[i]<<",";
        }
        if (ptuple.size() > 0) {
            dims_vec_code<<ptuple.vector()[ptuple.size()-1] << "}";
        } else {
            dims_vec_code<< "}";
        }
        return dims_vec_code.get_code_string();
    };

    auto gen_vec_code = [](PTuple<float> ptuple) -> std::string {
        CodeWritter dims_vec_code;
        dims_vec_code<<"{";
        for (int i=0; i<ptuple.size()-1; i++) {
            dims_vec_code<<ptuple.vector()[i]<<",";
        }
        if (ptuple.size() > 0) {
            dims_vec_code<<ptuple.vector()[ptuple.size()-1] << "}";
        } else {
            dims_vec_code<< "}";
        }
        return dims_vec_code.get_code_string();
    };


    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << var.size() << " ";
        for (int i = 0; i < var.size(); ++i) {
            code_w << var[i] << " ";
        }
        code_w.feed("%d %d %d %d %f %f %f %d %d %d ",
                    flip_flag ? 1 : 0,
                    clip_flag ? 1 : 0,
                    image_w,
                    image_h,
                    step_w,
                    step_h,
                    offset,
                    (int)order_[0], (int)order_[1], (int)order_[2]);

        code_w << min_size.size() << " ";
        for (int i = 0; i < min_size.size(); ++i) {
            code_w << min_size[i] << " ";
        }
        code_w << max_size.size() << " ";
        for (int i = 0; i < max_size.size(); ++i) {
            code_w << max_size[i] << " ";
        }
        code_w << as_ratio.size() << " ";
        for (int i = 0; i < as_ratio.size(); ++i) {
            code_w << as_ratio[i] << " ";
        }
        code_w << fixed_size.size() << " ";
        for (int i = 0; i < fixed_size.size(); ++i) {
            code_w << fixed_size[i] << " ";
        }
        code_w << fixed_ratio.size() << " ";
        for (int i = 0; i < fixed_ratio.size(); ++i) {
            code_w << fixed_ratio[i] << " ";
        }
        code_w << density.size() << " ";
        for (int i = 0; i < density.size(); ++i) {
            code_w << density[i] << " ";
        }
        code_w << "\n";
    } else {
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
                        bool gen_param,
                        int lite_mode,
                        DataType op_precision) {
    // parsing parameter
    auto slice_dim = get_attr<int>("slice_dim", attr);
    auto slice_point = get_attr<PTuple<int>>("slice_point", attr);
    auto axis = get_attr<int>("axis", attr);

    CodeWritter slice_point_vec_code;
    slice_point_vec_code<<"{";
    for (int i=0; i<slice_point.size()-1; i++) {
        slice_point_vec_code<<slice_point.vector()[i]<<",";
    }
    if (slice_point.size() > 0) {
        slice_point_vec_code<<slice_point.vector()[slice_point.size()-1] << "}";
    } else {
        slice_point_vec_code<< "}";
    }

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << axis << " " << slice_point.size() << " ";
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

// SaberScale
std::string ParserScale(graph::AttrInfo& attr,
                        std::string& code_name,
                        std::string& op_class_name,
                        std::string& node_name,
                        std::string& weights_ptr_name,
                        WeightsWritter& writter,
                        bool gen_param,
                        int lite_mode,
                        DataType op_precision) {
    // parsing parameter
    auto num_axes = get_attr<int>("num_axes", attr);
    auto axis = get_attr<int>("axis", attr);
    auto bias_term = get_attr<bool>("bias_term", attr);
    auto weights = get_attr<PBlock<X86>>("weight_1", attr);
    auto weights_shape = weights.shape();
    int weights_size = weights_shape.count();

    writter.register_weights(node_name, weights);
    LOG(INFO) << node_name << " write weights: " << weights.count();

    int bias_size = 0;
    if (bias_term) {
        auto bias = get_attr<PBlock<X86>>("weight_2", attr);
        auto bias_shape = bias.shape();
        bias_size = bias_shape.count();
        writter.register_weights(node_name, bias);
        LOG(INFO) << node_name << " write bias: " << bias.count();
      }

      auto offset_info = writter.get_weights_by_name(node_name);

  // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d\n",
                    offset_info.weights[0].offset,
                    bias_term ? offset_info.weights[1].offset : 0,
                    weights_size,
                    bias_size,
                    bias_term ? 1 : 0,
                    axis,
                    num_axes);
    } else {
        code_w.feed("ParamBase* %s_param = new ScaleParam(%s+%d, %s+%d, %d, %d, %s, %d, %d);\n",
                    node_name.c_str(),
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    bias_term ? offset_info.weights[1].offset : 0,
                    weights_size,
                    bias_size,
                    bias_term ? "true":"false",
                    axis,
                    num_axes);

        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

// SaberScale
std::string ParserBatchNorm(graph::AttrInfo& attr,
                        std::string& code_name,
                        std::string& op_class_name,
                        std::string& node_name,
                        std::string& weights_ptr_name,
                        WeightsWritter& writter,
                        bool gen_param,
                        int lite_mode,
                        DataType op_precision) {

    // get batchnorm param
    auto eps = get_attr<float>("epsilon", attr);
    auto momentum = get_attr<float>("momentum", attr);
    auto mean = get_attr<PBlock<X86>>("weight_1", attr);
    auto mean_vec = mean.vector();
    auto var = get_attr<PBlock<X86>>("weight_2", attr);
    auto var_vec = var.vector();
    auto scale_factor = get_attr<PBlock<X86>>("weight_3", attr);
    auto scale_factor_vec = scale_factor.vector();

    std::vector<float> scale;
    std::vector<float> bias;
    scale.resize(mean.count());
    bias.resize(mean.count());
    auto scale_val = scale_factor_vec[0] == 0 ? 0 : 1 / scale_factor_vec[0];

    for (int i = 0; i < mean.count(); i++) {
        scale[i] = 1.0f / std::sqrt(var_vec[i] * scale_val + eps);
        bias[i] = - mean_vec[i] * scale_val / std::sqrt(var_vec[i] * scale_val + eps);
    }

    Shape sh1({1, 1, 1, scale.size()});
    Shape sh2({1, 1, 1, bias.size()});
    PBlock<X86> pscale(sh1);
    PBlock<X86> pbias(sh2);
    float* pscale_ptr = (float*)pscale.h_tensor().mutable_data();
    for (int j = 0; j < scale.size(); ++j) {
        pscale_ptr[j] = scale[j];
    }
    float* pbias_ptr = (float*)pbias.h_tensor().mutable_data();
    for (int j = 0; j < bias.size(); ++j) {
        pbias_ptr[j] = bias[j];
    }
    writter.register_weights(node_name, pscale);
    LOG(INFO) << node_name << " write weights: " << pscale.count();

    writter.register_weights(node_name, pbias);
    LOG(INFO) << node_name << " write bias: " << pbias.count();

    auto weights_shape = pscale.shape();
    int weights_size = weights_shape.count();

    auto bias_shape = pbias.shape();
    int bias_size = bias_shape.count();

    auto offset_info = writter.get_weights_by_name(node_name);

    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w.feed("%d %d %d %d %d %d %d\n",
                    offset_info.weights[0].offset,
                    offset_info.weights[1].offset,
                    weights_size,
                    bias_size,
                    1,
                    1,
                    1);
    } else {
        code_w.feed("ParamBase* %s_param = new ScaleParam(%s+%d, %s+%d, %d, %d, %s, %d, %d);\n",
                    node_name.c_str(),
                    weights_ptr_name.c_str(),
                    offset_info.weights[0].offset,
                    weights_ptr_name.c_str(),
                    offset_info.weights[1].offset,
                    weights_size,
                    bias_size,
                    "true",
                    1,
                    1);

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
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
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

// SaberShuffleChannel
std::string ParserShuffleChannel(graph::AttrInfo& attr,
                              std::string& code_name,
                              std::string& op_class_name,
                              std::string& node_name,
                              std::string& weights_ptr_name,
                              WeightsWritter& writter,
                              bool gen_param,
                              int lite_mode,
                              DataType op_precision) {
    // parsing parameter
    auto group = get_attr<int>("group", attr);

    // gen cpp code
    CodeWritter code_w;

    if (gen_param) {
        code_w << group;
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new ShuffleChannelParam(%d);\n",
                        node_name.c_str(),
                        group);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

// SaberCoord2Patch
std::string ParserCoord2Patch(graph::AttrInfo& attr,
                                 std::string& code_name,
                                 std::string& op_class_name,
                                 std::string& node_name,
                                 std::string& weights_ptr_name,
                                 WeightsWritter& writter,
                                 bool gen_param,
                                 int lite_mode,
                                 DataType op_precision) {
    // parsing parameter
    auto img_h = get_attr<int>("img_h", attr);
    auto output_h = get_attr<int>("output_h", attr);
    auto output_w = get_attr<int>("output_w", attr);

    // gen cpp code
    CodeWritter code_w;

    if (gen_param) {
        code_w << img_h << " " << output_h << " " << output_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new Coord2PatchParam(%d, %d, %d);\n",
                    node_name.c_str(),
                    img_h,
                    output_h,
                    output_w);
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
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
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
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
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
                               bool gen_param,
                               int lite_mode,
                               DataType op_precision) {
    // parsing parameter
    auto dims = get_attr<PTuple<int>>("dims", attr);
    std::vector<int> vdims = dims.vector();

    CodeWritter reshape_dims_vec_code;
    reshape_dims_vec_code << "{";
    for (int i = 0; i < vdims.size() - 1; i++) {
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

// SaberResize
std::string ParserResize(graph::AttrInfo& attr,
                          std::string& code_name,
                          std::string& op_class_name,
                          std::string& node_name,
                          std::string& weights_ptr_name,
                          WeightsWritter& writter,
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
    // parsing parameter
    float width_scale = 0.f;
    float height_scale = 0.f;
    if (find_attr("width_scale", attr) == SaberSuccess && find_attr("height_scale", attr) == SaberSuccess){
        width_scale = get_attr<float>("width_scale", attr);
        height_scale = get_attr<float>("height_scale", attr);
    }
    int out_w = -1;
    int out_h = -1;
    if (find_attr("out_width", attr) == SaberSuccess && find_attr("out_height", attr) == SaberSuccess){
        out_w = get_attr<int>("out_width", attr);
        out_h = get_attr<int>("out_height", attr);
    }
    auto resize_type = get_attr<std::string>("method", attr);
    ResizeType type;
    if (resize_type == "BILINEAR_ALIGN"){
        type = BILINEAR_ALIGN;
    } else if (resize_type == "BILINEAR_NO_ALIGN"){
        type = BILINEAR_NO_ALIGN;
    } else if (resize_type == "RESIZE_CUSTOM"){
        type = RESIZE_CUSTOM;
    } else {
        LOG(FATAL) << "Unsupport resize type: " << resize_type;
    }
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << (int)type << " " << width_scale << " " << height_scale << " "\
               << out_w << " " << out_h;
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new ResizeParam(%d, %f, %f, %d, %d);\n",
                    node_name.c_str(),
                    (int)type,
                    width_scale,
                    height_scale,
                    out_w,
                    out_h);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

// SaberNormalize
std::string ParserNormalize(graph::AttrInfo& attr,
                          std::string& code_name,
                          std::string& op_class_name,
                          std::string& node_name,
                          std::string& weights_ptr_name,
                          WeightsWritter& writter,
                          bool gen_param,
                          int lite_mode,
                          DataType op_precision) {
    // parsing parameter
    auto eps = get_attr<float>("eps", attr);
    // gen cpp code
    CodeWritter code_w;
    if (gen_param) {
        code_w << eps;
        code_w << "\n";
    } else {
        code_w.feed("ParamBase* %s_param = new NormalizeParam(%f);\n",
                    node_name.c_str(),
                    eps);
        code_w.feed("    %s_g_param.push_back(%s_param);\n", code_name.c_str(), node_name.c_str());
    }
    return code_w.get_code_string();
}

std::unordered_map<std::string, OpParser> OPERATION_MAP({
    {"Input", {"Input", not_impl_yet} },
    {"Convolution", {"SaberConv2D", ParserConvolution} }, // done
    {"Deconvolution", {"SaberDeconv2D", ParserDeconvolution}}, //done
    {"DeconvRelu", {"SaberDeconv2D", ParserDeConvolutionRelu}}, //done
    {"Activation", {"SaberActivation", ParserActivation} }, // done
    {"ReLU", {"SaberActivation",ParserRelu}}, // done
    {"ConvRelu", {"SaberConv2D", ParserConvolutionRelu} },  // done
    {"ConvAct", {"SaberConv2D", ParserConvAct} },  // done
    {"ConvReluPool", {"SaberConvPooling2D", ParserConvolutionReluPool} },  // done
    {"ConvBatchnormScaleRelu", {"SaberConv2D", ParserConvBatchnormScaleRelu}}, // done have question ??
    {"ConvBatchnormScaleReluPool", {"SaberConvPooling2D", ParserConvBatchnormScaleReluPool}}, // done have question ??
    {"ConvBatchnormScale", {"SaberConv2D", ParserConvBatchnormScale}}, //done
    {"ConvBatchnorm", {"SaberConv2D", ParserConvBatchnorm}}, //done
    {"Concat", {"SaberConcat", ParserConcat} },  // done
    {"DetectionOutput", {"SaberDetectionOutput", ParserDectionOutput} }, // done
    {"Eltwise", {"SaberEltwise", ParserEltwise} }, //done
    {"EltwiseRelu", {"SaberEltwiseAct", ParserEltwiseRelu}}, // done
    {"EltwiseActivation", {"SaberEltwiseAct", ParserEltwisePRelu}}, // done
    {"Dense", {"SaberFc", ParserFc} }, // done
    {"Permute", {"SaberPermute", ParserPermute} }, // done
    {"Pooling", {"SaberPooling", ParserPooling} }, // done
    {"PriorBox", {"SaberPriorBox", ParserPriorBox} }, // done
    {"Power", {"SaberPower", ParserPower} }, // done
    {"Scale", {"SaberScale", ParserScale} }, // done
    {"BatchNorm", {"SaberScale", ParserBatchNorm} }, // done
    {"Slice", {"SaberSlice", ParserSlice} }, // done
    {"Flatten", {"SaberFlatten", ParserFlatten}}, //done
    {"Reshape", {"SaberReshape", ParserReshape}}, //done
    {"Softmax", {"SaberSoftmax", ParserSoftmax}}, //done
    {"Split", {"SaberSplit", ParserSplit}}, // done
    {"ShuffleChannel", {"SaberShuffleChannel", ParserShuffleChannel}}, // done
    {"Coord2Patch", {"SaberCoord2Patch", ParserCoord2Patch}}, // done
    {"Resize", {"SaberResize", ParserResize}},  //done
    {"Normalize", {"SaberNormalize", ParserNormalize}} //done
});

} /* namespace lite */

} /* namespace anakin */

