#include "framework/operators/fusion_ops/conv_batchnorm.h"

namespace anakin {

namespace ops {

#define INSTANCE_CONVBATCHNORM(Ttype, Ptype) \
template<> \
void ConvBatchnorm<Ttype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) {\
    auto* impl = static_cast<ConvBatchnormHelper<Ttype, Ptype>*>(this->_helper);\
    auto& param = static_cast<ConvBatchnormHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_conv_batchnorm;\
    SABER_CHECK(impl->_funcs_conv_batchnorm(ins, outs, param, ctx));\
}

template<typename Ttype, Precision Ptype>
Status ConvBatchnormHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing ConvBatchnorm op parameter.";
    
    // get conv param
    auto group = GET_PARAMETER(int, group);
    auto bias_term = GET_PARAMETER(bool, bias_term);
    auto padding = GET_PARAMETER(PTuple<int>, padding);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto dilation_rate = GET_PARAMETER(PTuple<int>, dilation_rate);
    auto filter_num = GET_PARAMETER(int, filter_num);
    auto kernel_size = GET_PARAMETER(PTuple<int>, kernel_size);
    auto axis = GET_PARAMETER(int, axis);

    using pblock_type = PBlock<Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);
    auto weights_shape = weights.shape();
    auto weights_dtype = weights.h_tensor().get_dtype();
    // resize weights scale
    auto& w = weights.h_tensor();
    if (w.get_scale().size() == 1){
        float scale_tmp = w.get_scale()[0];
        std::vector<float> w_scale(filter_num, scale_tmp);
        w.set_scale(w_scale);
    }

    // get batchnorm param
    auto epsilon = GET_PARAMETER(float, batchnorm_0_epsilon);
    auto momentum = GET_PARAMETER(float, batchnorm_0_momentum);
    auto batch_norm_weight_1 = GET_PARAMETER(pblock_type, batchnorm_0_weight_1);
    auto batch_norm_weight_1_vector = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = GET_PARAMETER(pblock_type, batchnorm_0_weight_2);
    auto batch_norm_weight_2_vector = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = GET_PARAMETER(pblock_type, batchnorm_0_weight_3);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector();

    // check if batchnorm parameters have been optimized 
    auto is_param_updated = CHECK_PARAMETER(is_param_updated);
    if (!is_param_updated) {
        SET_PARAMETER(is_param_updated, true, bool);

        if (bias_term) {
            auto bias = GET_PARAMETER(pblock_type, weight_2);
            if (weights_dtype == AK_FLOAT) {
                graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<float, Ttype>::update_weights_without_scale, weights, bias,
                        weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                        true, batch_norm_weight_3_vector[0], epsilon,
                        batch_norm_weight_1_vector, batch_norm_weight_2_vector);
            } else {
                graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<char, Ttype>::update_weights_without_scale, weights, bias,
                        weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                        true, batch_norm_weight_3_vector[0], epsilon,
                        batch_norm_weight_1_vector, batch_norm_weight_2_vector);
            }

            saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                               strides[0], strides[1],
                                               dilation_rate[0], dilation_rate[1],
                                               &(weights.d_tensor()), &(bias.d_tensor()));
            _param_conv_batchnorm = conv_param;
        } else {
            pblock_type* bias = new pblock_type();
            SET_PARAMETER(bias_term, true, bool); // set attr bias_term true
            SET_PARAMETER(weight_2, *bias, pblock_type); // gen new bias
            if (weights_dtype == AK_FLOAT){
                    graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                            WeightsFusion<float, Ttype>::update_weights_without_scale, weights, *bias,
                            weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                            false, batch_norm_weight_3_vector[0], epsilon,
                            batch_norm_weight_1_vector,
                            batch_norm_weight_2_vector);
            } else {
                graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<char, Ttype>::update_weights_without_scale, weights, *bias,
                        weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                        false, batch_norm_weight_3_vector[0], epsilon,
                        batch_norm_weight_1_vector,
                        batch_norm_weight_2_vector);
            }

            saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                    strides[0], strides[1], dilation_rate[0], dilation_rate[1],
                    &(weights.d_tensor()), &(bias->d_tensor()));

            _param_conv_batchnorm = conv_param;
        }
    } else {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                strides[0], strides[1], dilation_rate[0], dilation_rate[1],
                &(weights.d_tensor()), &(bias.d_tensor()));

        _param_conv_batchnorm = conv_param;
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvBatchnormHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    auto group = GET_PARAMETER(int, group);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto weights = GET_PARAMETER(PBlock<Ttype>, weight_1);
    auto bias_term = GET_PARAMETER(bool, bias_term);

    //different device please change here!!!
    saber::ImplEnum impl_e = VENDER_IMPL;
    if (std::is_same<Ttype, X86>::value) {
        impl_e = SABER_IMPL;
    }
    bool use_k1s1p0 = (Ptype == Precision::FP32);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.weight()->height() == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.weight()->width() == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.pad_h == 0);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.pad_w == 0);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.stride_h == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.stride_w == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.dilation_h == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.dilation_w == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.group == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm.bias()->valid_size() > 0);
    bool use_k3s1d1 = (Ptype == Precision::FP32);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm.weight()->height() == 3);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm.weight()->width() == 3);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm.group == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm.stride_h == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm.stride_w == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm.dilation_h == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm.dilation_w == 1);
    bool use_depthwise = (Ptype == Precision::FP32);
    use_depthwise = use_depthwise && (_param_conv_batchnorm.group == ins[0]->channel());
    use_depthwise = use_depthwise && (_param_conv_batchnorm.group == outs[0]->channel());
    bool use_direct_k = (Ptype == Precision::FP32);
    use_direct_k = use_direct_k && (_param_conv_batchnorm.weight()->channel() >= 16);
    use_direct_k = use_direct_k && (_param_conv_batchnorm.group == 1);
    if (std::is_same<Ttype, NV>::value
        && (use_k1s1p0 || use_k3s1d1 || use_depthwise || use_direct_k)) {
        impl_e = SABER_IMPL;
    }
    if (std::is_same<Ttype, NV>::value && Ptype == Precision::INT8) {
        impl_e = SABER_IMPL;
    }
    SABER_CHECK(_funcs_conv_batchnorm.init(ins, outs, \
        _param_conv_batchnorm, SPECIFY, impl_e, ctx));

    // check if weights have been transposed
    auto is_weights_transed = CHECK_PARAMETER(is_weights_transed);
    if (!is_weights_transed) {
        SET_PARAMETER(is_weights_transed, true, bool);
        if (bias_term) {
            auto bias = GET_PARAMETER(PBlock<Ttype>, weight_2);
            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                    std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                            &_funcs_conv_batchnorm, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                    weights.d_tensor(), bias.d_tensor(), _param_conv_batchnorm.pad_h, _param_conv_batchnorm.pad_w, _param_conv_batchnorm.dilation_h, _param_conv_batchnorm.dilation_w,
                    strides[0], strides[1], group, impl_e);
            bias.map_to_host();
        } else {
            PBlock<Ttype> bias_empty;
            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                    std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                            &_funcs_conv_batchnorm, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                    weights.d_tensor(), bias_empty.d_tensor(), _param_conv_batchnorm.pad_h, _param_conv_batchnorm.pad_w, _param_conv_batchnorm.dilation_h, _param_conv_batchnorm.dilation_w,
                    strides[0], strides[1], group, impl_e);
        }
        weights.map_to_host();
    } else {
        PBlock<Ttype> weight_empty;
        PBlock<Ttype> bias_empty;
        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                        &_funcs_conv_batchnorm, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                        weight_empty.d_tensor(), bias_empty.d_tensor(), _param_conv_batchnorm.pad_h, _param_conv_batchnorm.pad_w, _param_conv_batchnorm.dilation_h, _param_conv_batchnorm.dilation_w,
                        strides[0], strides[1], group, impl_e);
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvBatchnormHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_conv_batchnorm.compute_output_shape(ins, outs, \
        _param_conv_batchnorm));
    return Status::OK();
}

#ifdef USE_ARM_PLACE
INSTANCE_CONVBATCHNORM(ARM, Precision::FP32);
template class ConvBatchnormHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvBatchnorm, ConvBatchnormHelper, ARM, Precision::FP32);
#endif

#ifdef USE_CUDA
INSTANCE_CONVBATCHNORM(NV, Precision::FP32);
INSTANCE_CONVBATCHNORM(NV, Precision::INT8);
ANAKIN_REGISTER_OP_HELPER(ConvBatchnorm, ConvBatchnormHelper, NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ConvBatchnorm, ConvBatchnormHelper, NV, Precision::INT8);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONVBATCHNORM(X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ConvBatchnorm, ConvBatchnormHelper, X86, Precision::FP32);
#endif

#if defined BUILD_LITE
INSTANCE_CONVBATCHNORM(X86, Precision::FP32);
template class ConvBatchnormHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvBatchnorm, ConvBatchnormHelper, X86, Precision::FP32);
#endif


//! register op
ANAKIN_REGISTER_OP(ConvBatchnorm)
.Doc("ConvBatchnorm fusion operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("convolution_batchnorm")
.__alias__<NV, Precision::INT8>("convolution_batchnorm")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("convolution_batchnorm")
#endif
#if defined BUILD_LITE
.__alias__<X86, Precision::FP32>("convolution_batchnorm")
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


