#include "framework/operators/fusion_ops/conv_fusion.h"

namespace anakin {

namespace ops {

#define INSTANCE_CONVFUSION(Ttype, Ptype) \
template<> \
void ConvFusion<Ttype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) {\
    auto* impl = static_cast<ConvFusionHelper<Ttype, Ptype>*>(this->_helper);\
    auto& param_conv = static_cast<ConvFusionHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_conv_fusion;\
    auto& param_slice = static_cast<ConvFusionHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_slice;\
    auto mid_out = static_cast<ConvFusionHelper<Ttype, Ptype>*>\
                  (this->_helper)->_mid_out;\
    SABER_CHECK(impl->_funcs_conv_fusion(ins, mid_out, param_conv, ctx));\
    SABER_CHECK(impl->_funcs_slice(mid_out, outs, param_slice, ctx));\
}

template<typename Ttype, Precision Ptype>
Status ConvFusionHelper<Ttype, Ptype>::InitParam() {
    typedef typename target_host<Ttype>::type Ttype_H;

    std::vector<saber::ConvParam<Ttype_H>> param_vec;

    PTuple<int> slice_dims;
    if (CHECK_PARAMETER(slice_info)){
        slice_dims = GET_PARAMETER(PTuple<int>, slice_info);
    } else {
        slice_dims.push_back(1);
    }
    int count = 0;
    for (int i = 0; i < slice_dims.size(); ++i){
        count += slice_dims[i];
    }

    using pblock_type = PBlock<Ttype>;
    std::vector<pblock_type> weights_v;
    std::vector<pblock_type> bias_v;
    bool param_equal = true;
    for (int i = 0; i < count; ++i){
        //todo:make var names
        std::string base_str = "conv_";
        base_str += (char)(i + 0x30);
        base_str += "_";
        if (i == 0) {
            base_str = "";
        }
        auto t_group = GET_PARAMETER_BYSTR(int, base_str + "group");
        auto t_bias_term = GET_PARAMETER_BYSTR(bool, base_str + "bias_term");
        auto t_padding = GET_PARAMETER_BYSTR(PTuple<int>, base_str + "padding");
        auto t_strides = GET_PARAMETER_BYSTR(PTuple<int>, base_str + "strides");
        auto t_dilation_rate = GET_PARAMETER_BYSTR(PTuple<int>, base_str + "dilation_rate");
        auto t_filter_num = GET_PARAMETER_BYSTR(int, base_str + "filter_num");
        auto t_kernel_size = GET_PARAMETER_BYSTR(PTuple<int>, base_str + "kernel_size");
        auto t_axis = GET_PARAMETER_BYSTR(int, base_str + "axis");

        DLOG(INFO) << "conv group : " << t_group;
        DLOG(INFO) << "conv bias_term: " << t_bias_term;
        DLOG(INFO) << "conv padding : [" << t_padding[0] << " " << t_padding[1] << "]";
        DLOG(INFO) << "conv strides : [" << t_strides[0] << " " << t_strides[1] << "]";
        DLOG(INFO) << "conv dilation_rate : [" << t_dilation_rate[0] << " " << t_dilation_rate[1] << "]";
        DLOG(INFO) << "conv filter_num : " << t_filter_num;
        DLOG(INFO) << "conv kernel_size : [" << t_kernel_size[0] << " " << t_kernel_size[1] << "]";
        DLOG(INFO) << "conv axis : " << t_axis;

        auto t_weights = GET_PARAMETER_BYSTR(pblock_type, base_str + "weight_1");
        auto t_weights_shape = t_weights.shape();
        auto t_weights_dtype = t_weights.h_tensor().get_dtype();

        // get batchnorm param
        if (CHECK_PARAMETER_BYSTR(base_str + "batchnorm_0_epsilon")){
            DLOG(INFO) << "has bn ";
            auto t_epsilon = GET_PARAMETER_BYSTR(float, base_str + "batchnorm_0_epsilon");
            auto t_momentum = GET_PARAMETER_BYSTR(float, base_str + "batchnorm_0_momentum");
            auto t_batch_norm_weight_1 = GET_PARAMETER_BYSTR(pblock_type, base_str + "batchnorm_0_weight_1");
            auto t_batch_norm_weight_1_vector = t_batch_norm_weight_1.vector();
            auto t_batch_norm_weight_2 = GET_PARAMETER_BYSTR(pblock_type, base_str + "batchnorm_0_weight_2");
            auto t_batch_norm_weight_2_vector = t_batch_norm_weight_2.vector();
            auto t_batch_norm_weight_3 = GET_PARAMETER_BYSTR(pblock_type, base_str + "batchnorm_0_weight_3");
            auto t_batch_norm_weight_3_vector = t_batch_norm_weight_3.vector();

            if (CHECK_PARAMETER_BYSTR(base_str + "scale_0_num_axes")){
                DLOG(INFO) << "has scale ";
                auto t_scale_num_axes = GET_PARAMETER_BYSTR(int, base_str + "scale_0_num_axes");
                auto t_scale_bias_term = GET_PARAMETER_BYSTR(bool, base_str + "scale_0_bias_term");
                auto t_scale_axis = GET_PARAMETER_BYSTR(int, base_str + "scale_0_axis");
                auto t_scale_weight_1 = GET_PARAMETER_BYSTR(pblock_type, base_str + "scale_0_weight_1");
                auto t_scale_weight_1_vector = t_scale_weight_1.vector();
                auto t_scale_weight_2 = GET_PARAMETER_BYSTR(pblock_type, base_str + "scale_0_weight_2");
                auto t_scale_weight_2_vector = t_scale_weight_2.vector();
                //trans weights
                if (t_bias_term){
                    auto t_bias = GET_PARAMETER_BYSTR(pblock_type, base_str + "weight_2");
                    if (t_weights_dtype == AK_FLOAT) {
                        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<float, Ttype>::update_weights, t_weights, t_bias,
                        t_weights_shape[0], t_weights_shape[1], t_weights_shape[2], t_weights_shape[3],
                        true, t_batch_norm_weight_3_vector[0], t_epsilon,
                        t_batch_norm_weight_1_vector, t_batch_norm_weight_2_vector,
                        t_scale_weight_1_vector, t_scale_weight_2_vector,
                        t_scale_bias_term);
                    } else{
                        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<char, Ttype>::update_weights, t_weights, t_bias,
                        t_weights_shape[0], t_weights_shape[1],
                        t_weights_shape[2], t_weights_shape[3],
                        true, t_batch_norm_weight_3_vector[0], t_epsilon,
                        t_batch_norm_weight_1_vector, t_batch_norm_weight_2_vector,
                        t_scale_weight_1_vector, t_scale_weight_2_vector,
                        t_scale_bias_term);
                    }
                    //new param gen
                    param_vec.emplace_back(saber::ConvParam<Ttype_H>(t_group, t_padding[0], t_padding[1],
                                               t_strides[0], t_strides[1],
                                               t_dilation_rate[0], t_dilation_rate[1],
                                               &(t_weights.h_tensor()), &(t_bias.h_tensor())));

                } else {
                    pblock_type* t_bias = new pblock_type();
                    SET_PARAMETER_BYSTR(base_str + "bias_term", true, bool); // set attr bias_term true
                    SET_PARAMETER_BYSTR(base_str + "weight_2", *t_bias, pblock_type); // gen new bias
                    if (t_weights_dtype == AK_FLOAT) {
                        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<float, Ttype>::update_weights, t_weights, *t_bias,
                        t_weights_shape[0], t_weights_shape[1],
                        t_weights_shape[2], t_weights_shape[3],
                        false, t_batch_norm_weight_3_vector[0], t_epsilon,
                        t_batch_norm_weight_1_vector,
                        t_batch_norm_weight_2_vector,
                        t_scale_weight_1_vector,
                        t_scale_weight_2_vector,
                        t_scale_bias_term);
                    } else {
                        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<char, Ttype>::update_weights, t_weights, *t_bias,
                        t_weights_shape[0], t_weights_shape[1], t_weights_shape[2], t_weights_shape[3],
                        false, t_batch_norm_weight_3_vector[0], t_epsilon,
                        t_batch_norm_weight_1_vector,
                        t_batch_norm_weight_2_vector,
                        t_scale_weight_1_vector,
                        t_scale_weight_2_vector,
                        t_scale_bias_term);
                    }
                    //new param gen
                    param_vec.emplace_back(saber::ConvParam<Ttype_H>(t_group, t_padding[0], t_padding[1],
                    t_strides[0], t_strides[1], t_dilation_rate[0], t_dilation_rate[1],
                    &(t_weights.h_tensor()), &(t_bias->h_tensor())));
                }//bias_term
            //check scale
            } else {
                if (t_bias_term) {
                    auto t_bias = GET_PARAMETER_BYSTR(pblock_type, base_str + "weight_2");
                    if (t_weights_dtype == AK_FLOAT) {
                        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<float, Ttype>::update_weights_without_scale, t_weights, t_bias,
                        t_weights_shape[0], t_weights_shape[1], t_weights_shape[2], t_weights_shape[3],
                        true, t_batch_norm_weight_3_vector[0], t_epsilon,
                        t_batch_norm_weight_1_vector, t_batch_norm_weight_2_vector);
                    } else {
                        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<char, Ttype>::update_weights_without_scale, t_weights, t_bias,
                        t_weights_shape[0], t_weights_shape[1], t_weights_shape[2], t_weights_shape[3],
                        true, t_batch_norm_weight_3_vector[0], t_epsilon,
                        t_batch_norm_weight_1_vector, t_batch_norm_weight_2_vector);
                    }

                    param_vec.emplace_back(saber::ConvParam<Ttype_H>(t_group, t_padding[0], t_padding[1],
                                               t_strides[0], t_strides[1],
                                               t_dilation_rate[0], t_dilation_rate[1],
                                               &(t_weights.h_tensor()), &(t_bias.h_tensor())));
                } else {
                    pblock_type* t_bias = new pblock_type();
                    SET_PARAMETER_BYSTR(base_str + "bias_term", true, bool); // set attr bias_term true
                    SET_PARAMETER_BYSTR(base_str + "weight_2", *t_bias, pblock_type); // gen new bias
                    if (t_weights_dtype == AK_FLOAT){
                            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                            WeightsFusion<float, Ttype>::update_weights_without_scale, t_weights, *t_bias,
                            t_weights_shape[0], t_weights_shape[1], t_weights_shape[2], t_weights_shape[3],
                            false, t_batch_norm_weight_3_vector[0], t_epsilon,
                            t_batch_norm_weight_1_vector,
                            t_batch_norm_weight_2_vector);
                    } else {
                        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<char, Ttype>::update_weights_without_scale, t_weights, *t_bias,
                        t_weights_shape[0], t_weights_shape[1], t_weights_shape[2], t_weights_shape[3],
                        false, t_batch_norm_weight_3_vector[0], t_epsilon,
                        t_batch_norm_weight_1_vector,
                        t_batch_norm_weight_2_vector);
                    }

                param_vec.emplace_back(saber::ConvParam<Ttype_H>(t_group, t_padding[0], t_padding[1],
                    t_strides[0], t_strides[1], t_dilation_rate[0], t_dilation_rate[1],
                    &(t_weights.h_tensor()), &(t_bias->h_tensor())));

                }//if bias_term
            }//check scale
        //check bn
        } else {
            if (t_bias_term) {
                auto bias = GET_PARAMETER_BYSTR(pblock_type, base_str + "weight_2");
                param_vec.emplace_back(saber::ConvParam<Ttype_H>(t_group, t_padding[0], t_padding[1],
                t_strides[0], t_strides[1], t_dilation_rate[0], t_dilation_rate[1],
                &(t_weights.h_tensor()), &(bias.h_tensor())));
            } else {
                Tensor4d<Ttype_H>* bias = new Tensor4d<Ttype_H>();
                param_vec.emplace_back(saber::ConvParam<Ttype_H>(t_group, t_padding[0], t_padding[1],
                t_strides[0], t_strides[1], t_dilation_rate[0], t_dilation_rate[1],
                &(t_weights.h_tensor()), bias));
            }
        }//check bn

    }//for
    //if param equal, combine weights

    //LOG(ERROR) << "conv_fusion: " << param_equal;
    if (param_equal){
        //TODO:combine weight
        std::vector<int> out_channels;

        Shape new_w_sh = param_vec[0].weight()->valid_shape();
        Shape new_b_sh = param_vec[0].bias()->valid_shape();
        out_channels.push_back(param_vec[0].weight()->valid_shape()[0]);

        //set new weight and bias shape
        for (int i = 1; i < param_vec.size(); ++i){
            new_w_sh.set_num(new_w_sh.num() + param_vec[i].weight()->num());
            out_channels.push_back(param_vec[i].weight()->num());
        }
        new_b_sh = Shape({1, new_w_sh.num(), 1, 1});
        auto weights_dtype = param_vec[0].weight()->get_dtype();
        pblock_type* fusion_weight;
        if (weights_dtype == AK_FLOAT){
            fusion_weight = graph::GraphGlobalMem<Ttype>::Global().template new_block<AK_FLOAT>(new_w_sh);
        } else {
            fusion_weight = graph::GraphGlobalMem<Ttype>::Global().template new_block<AK_INT8>(new_w_sh);
        }
        pblock_type* fusion_bias = graph::GraphGlobalMem<Ttype>::Global().template new_block<AK_FLOAT>(new_b_sh);

        if (weights_dtype == AK_FLOAT){
            float* w_data = static_cast<float*>(fusion_weight->h_tensor().mutable_data());
            float* b_data = static_cast<float*>(fusion_bias->h_tensor().mutable_data());
            int w_offset = 0;
            int b_offset = 0;
            for (int i = 0; i < param_vec.size(); ++i){
                const float* cur_w_data = static_cast<const float*>(param_vec[i].weight()->data());
                const float* cur_b_data = static_cast<const float*>(param_vec[i].bias()->data());
                for (int j = 0; j < param_vec[i].weight()->valid_size(); ++j){
                    w_data[w_offset + j] = cur_w_data[j];
                    //printf("%f\n", cur_w_data[j]);
                }
                w_offset += param_vec[i].weight()->valid_size();
                //if param_i has bias, fill it, otherwise fill 0
                if (cur_b_data){
                    for (int j = 0; j < param_vec[i].bias()->valid_size(); ++j){
                        b_data[b_offset + j] = cur_b_data[j];
                    }
                    b_offset += param_vec[i].bias()->valid_size();
                } else {
                    for (int j = 0; j < param_vec[i].weight()->num(); ++j){
                        b_data[b_offset + j] = 0.f;
                    }
                    b_offset += param_vec[i].weight()->num();
                }
            }
        } else {
            char* w_data = static_cast<char*>(fusion_weight->h_tensor().mutable_data());
            float* b_data = static_cast<float*>(fusion_bias->h_tensor().mutable_data());
            int w_offset = 0;
            int b_offset = 0;
            for (int i = 0; i < param_vec.size(); ++i){
                const char* cur_w_data = static_cast<const char*>(param_vec[i].weight()->data());
                const float* cur_b_data = static_cast<const float*>(param_vec[i].bias()->data());
                for (int j = 0; j < param_vec[i].weight()->valid_size(); ++j){
                    w_data[w_offset + j] = cur_w_data[j];
                }

                //if param_i has bias, fill it, otherwise fill 0
                if (cur_b_data){
                    for (int j = 0; j < param_vec[i].bias()->valid_size(); ++j){
                        b_data[b_offset + j] = cur_b_data[j];
                    }
                    b_offset += param_vec[i].bias()->valid_size();
                } else {
                    for (int j = 0; j < param_vec[i].weight()->num(); ++j){
                        b_data[b_offset + j] = 0.f;
                    }
                    b_offset += param_vec[i].weight()->num();
                }
                w_offset += param_vec[i].weight()->valid_size();
            }
        }

        //fusion_weight->map_to_device();
        //fusion_bias->map_to_device();
        fusion_weight->d_tensor().copy_from(fusion_weight->h_tensor());
        fusion_bias->d_tensor().copy_from(fusion_bias->h_tensor());

        auto group = GET_PARAMETER(int, group);
        auto padding = GET_PARAMETER(PTuple<int>, padding);
        auto strides = GET_PARAMETER(PTuple<int>, strides);
        auto dilation_rate = GET_PARAMETER(PTuple<int>, dilation_rate);

        saber::ConvParam<Ttype> conv_fusion_param;
        if (CHECK_PARAMETER(relu_0_alpha)){
            DLOG(INFO) << "has relu ";
            auto alpha = GET_PARAMETER(float, relu_0_alpha);
            ActivationParam<Ttype> active_param(Active_relu, alpha); // TEMP
            conv_fusion_param = saber::ConvParam<Ttype>(group, padding[0], padding[1],
                    strides[0], strides[1], dilation_rate[0], dilation_rate[1],
                    &(fusion_weight->d_tensor()), &(fusion_bias->d_tensor()), active_param);
        } else {
            conv_fusion_param = saber::ConvParam<Ttype>(group, padding[0], padding[1],
                    strides[0], strides[1], dilation_rate[0], dilation_rate[1],
                    &(fusion_weight->d_tensor()), &(fusion_bias->d_tensor()));
        }
        _param_conv_fusion = conv_fusion_param;

        //trans output channel
        if (slice_dims.size() != 1){
            int s_ind = 0;
            for (int i = 0; i < slice_dims.size(); ++i){
                int channel = 0;
                for (int j = 0; j < slice_dims[i]; ++j){
                    CHECK_GT(out_channels.size(), j);
                    channel += out_channels[s_ind+j];
                }
                _slice_channels.push_back(channel);
                s_ind += slice_dims[i];
            }
        } else {
            int channel = 0;
            for (int i = 0; i < out_channels.size(); ++i){
                channel  += out_channels[i];
            }
            _slice_channels.push_back(channel);
        }
        PTuple<int> slice_point;
        slice_point.push_back(_slice_channels[0]);
        for (int i = 1; i < _slice_channels.size() - 1; ++i){
            slice_point.push_back(_slice_channels[i] + slice_point[i-1]);
        }

        SliceParam<Ttype> param_slice(1, slice_point.vector());
        _param_slice = param_slice;

    }
    _mid_out.push_back(new Tensor4d<Ttype>());
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvFusionHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    auto group = GET_PARAMETER(int, group);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto weights = _param_conv_fusion.mutable_weight();
    auto bias_term = true;

    //different device pleace change here..
    saber::ImplEnum impl_e = VENDER_IMPL;
    if (std::is_same<Ttype, X86>::value || std::is_same<Ttype, ARM>::value) {
        impl_e = SABER_IMPL;
    }
    if (std::is_same<Ttype, NV>::value && Ptype == Precision::INT8) {
        impl_e = SABER_IMPL;
    }
    bool use_k1s1p0 = (Ptype == Precision::FP32);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.weight()->height() == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.weight()->width() == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.pad_h == 0);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.pad_w == 0);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.stride_h == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.stride_w == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.dilation_h == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.dilation_w == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.group == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_fusion.bias()->valid_size() > 0);
    bool use_k3s1d1 = (Ptype == Precision::FP32);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_fusion.weight()->height() == 3);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_fusion.weight()->width() == 3);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_fusion.group == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_fusion.stride_h == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_fusion.stride_w == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_fusion.dilation_h == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_fusion.dilation_w == 1);
    bool use_depthwise = (Ptype == Precision::FP32);
    use_depthwise = use_depthwise && (_param_conv_fusion.group == ins[0]->channel());
    use_depthwise = use_depthwise && (_param_conv_fusion.group == outs[0]->channel());
    bool use_direct_k = (Ptype == Precision::FP32);
    use_direct_k = use_direct_k && (_param_conv_fusion.weight()->channel() >= 16);
    use_direct_k = use_direct_k && (_param_conv_fusion.group == 1);
    if (std::is_same<Ttype, NV>::value
        && (use_k1s1p0 || use_k3s1d1 || use_depthwise || use_direct_k)) {
        impl_e = SABER_IMPL;
    }
    //set midout scale
    float mid_scale = 0.f;
    for (int i = 0; i < outs.size(); ++i){
        if (outs[i]->get_scale().size() > 0){
            float t_scale = outs[i]->get_scale()[0];
            mid_scale = mid_scale >  t_scale ? mid_scale : t_scale;
        }
    } 
    std::vector<float> mid_scale_v{mid_scale};
    _mid_out[0] -> set_scale(mid_scale_v);
    SABER_CHECK(_funcs_conv_fusion.init(ins, _mid_out, _param_conv_fusion, SPECIFY, impl_e, ctx));
    SABER_CHECK(_funcs_slice.init(_mid_out, outs, _param_slice, SPECIFY, SABER_IMPL, ctx));

    // check if weights have been transposed
    auto is_weights_transed = CHECK_PARAMETER(is_weights_transed);
    if (!is_weights_transed) {
        LOG(ERROR) << "transing";
        SET_PARAMETER(is_weights_transed, true, bool);
        if (bias_term) {
            auto bias = _param_conv_fusion.mutable_bias();
            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                    std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                              &_funcs_conv_fusion, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                    *weights, *bias, _param_conv_fusion.pad_h, _param_conv_fusion.pad_w,
                    _param_conv_fusion.dilation_h, _param_conv_fusion.dilation_w,
                    strides[0], strides[1], group, impl_e);
            //bias->map_to_host();
        } else {
            PBlock<Ttype> bias_empty;
            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                    std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                              &_funcs_conv_fusion, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                    *weights, bias_empty.d_tensor(), _param_conv_fusion.pad_h,
                    _param_conv_fusion.pad_w, _param_conv_fusion.dilation_h, _param_conv_fusion.dilation_w,
                    strides[0], strides[1], group, impl_e);
        }
        //weights->map_to_host();
    } else {
        PBlock<Ttype> weight_empty;
        PBlock<Ttype> bias_empty;
        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                        &_funcs_conv_fusion, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                        weight_empty.d_tensor(), bias_empty.d_tensor(), _param_conv_fusion.pad_h, _param_conv_fusion.pad_w, _param_conv_fusion.dilation_h, _param_conv_fusion.dilation_w,
                        strides[0], strides[1], group, impl_e);
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvFusionHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    SABER_CHECK(_funcs_conv_fusion.compute_output_shape(ins, _mid_out, \
        _param_conv_fusion));
    _mid_out[0]->reshape(_mid_out[0]->valid_shape());
    SABER_CHECK(_funcs_slice.compute_output_shape(_mid_out, outs, _param_slice));
    return Status::OK();
}

#ifdef USE_ARM_PLACE
INSTANCE_CONVFUSION(ARM, Precision::FP32);
INSTANCE_CONVFUSION(ARM, Precision::INT8);
template class ConvFusionHelper<ARM, Precision::FP32>;
template class ConvFusionHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(ConvFusion, ConvFusionHelper, ARM, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ConvFusion, ConvFusionHelper, ARM, Precision::INT8);
#endif

#ifdef USE_CUDA
INSTANCE_CONVFUSION(NV, Precision::FP32);
INSTANCE_CONVFUSION(NV, Precision::INT8);
ANAKIN_REGISTER_OP_HELPER(ConvFusion, ConvFusionHelper, NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ConvFusion, ConvFusionHelper, NV, Precision::INT8);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONVFUSION(X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ConvFusion, ConvFusionHelper, X86, Precision::FP32);
#endif

#if defined BUILD_LITE
INSTANCE_CONVFUSION(X86, Precision::FP32);
template class ConvFusionHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvFusion, ConvFusionHelper, X86, Precision::FP32);
#endif


//! register op
ANAKIN_REGISTER_OP(ConvFusion)
.Doc("ConvFusion fusion operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("convolution_fusion")
.__alias__<NV, Precision::INT8>("convolution_fusion")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("convolution_fusion")
.__alias__<ARM, Precision::INT8>("convolution_fusion")
#endif
#if defined BUILD_LITE
.__alias__<X86, Precision::FP32>("convolution_fusion")
#endif
.num_in(1)
.num_out(1)
.Args<int>("group", " group of conv ")
.Args<bool>("bias_term", " whether conv weights have bias")
.Args<PTuple<int>>("padding", "padding of conv (x, y)")
.Args<PTuple<int>>("strides", "strides of conv (x)")
.Args<PTuple<int>>("dilation_rate", "dilation rate of conv (x)")
.Args<int>("filter_num", "filter(kernel) number of weights")
.Args<PTuple<int>>("kernel_size", "kernel size of kernel (x, y)")
.Args<int>("axis", "axis of conv")
.Args<float>("batchnorm_0_epsilon", "epsilon for batchnorm")
.Args<float>("batchnorm_0_momentum", "momentum for batchnorm");

} /* namespace ops */

} /* namespace anakin */


