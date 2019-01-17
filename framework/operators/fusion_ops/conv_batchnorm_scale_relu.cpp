/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "framework/operators/fusion_ops/conv_batchnorm_scale_relu.h"

namespace anakin {

namespace ops {

#define INSTANCE_CONVBATCHNORMSCALERELU(Ttype, Ptype) \
template<> \
void ConvBatchnormScaleRelu<Ttype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) {\
    auto* impl = static_cast<ConvBatchnormScaleReluHelper<Ttype, Ptype>*>\
                 (this->_helper);\
    auto& param = static_cast<ConvBatchnormScaleReluHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_conv_batchnorm_scale_relu;\
    SABER_CHECK(impl->_funcs_conv_batchnorm_scale_relu(ins, outs, param, ctx));\
}

template<typename Ttype, Precision Ptype>
Status ConvBatchnormScaleReluHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ConvBatchnormScaleRelu op parameter.";
    
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

    // get scale param
    auto scale_num_axes = GET_PARAMETER(int, scale_0_num_axes);
    auto scale_bias_term = GET_PARAMETER(bool, scale_0_bias_term);
    auto scale_axis = GET_PARAMETER(int, scale_0_axis);
    auto scale_weight_1 = GET_PARAMETER(pblock_type, scale_0_weight_1);
    auto scale_weight_1_vector = scale_weight_1.vector();
    auto scale_weight_2 = GET_PARAMETER(pblock_type, scale_0_weight_2);
    auto  scale_weight_2_vector = scale_weight_2.vector();

    // get relu param
    auto alpha = GET_PARAMETER(float, relu_0_alpha);
    ActivationParam<Ttype> active_param(Active_relu, alpha); // TEMP

    // check if batchnorm parameters have been optimized 
    auto is_param_updated = CHECK_PARAMETER(is_param_updated);
    if (!is_param_updated) {
        SET_PARAMETER(is_param_updated, true, bool);

        if (bias_term) {
            auto bias = GET_PARAMETER(pblock_type, weight_2);
            if (weights_dtype == AK_FLOAT) {
                graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<float, Ttype>::update_weights,
                        weights, bias, weights_shape[0], weights_shape[1],
                        weights_shape[2], weights_shape[3],
                        true, batch_norm_weight_3_vector[0], epsilon,
                        batch_norm_weight_1_vector, batch_norm_weight_2_vector,
                        scale_weight_1_vector, scale_weight_2_vector, scale_bias_term);
            } else {
                graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<char, Ttype>::update_weights,
                        weights, bias, weights_shape[0], weights_shape[1],
                        weights_shape[2], weights_shape[3],
                        true, batch_norm_weight_3_vector[0], epsilon,
                        batch_norm_weight_1_vector, batch_norm_weight_2_vector,
                        scale_weight_1_vector, scale_weight_2_vector, scale_bias_term);
            }

            saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                               strides[0], strides[1],
                                               dilation_rate[0], dilation_rate[1],
                                               &(weights.d_tensor()), &(bias.d_tensor()),
                                               active_param);
            _param_conv_batchnorm_scale_relu = conv_param;
        } else {
            pblock_type* bias = new pblock_type();
            SET_PARAMETER(bias_term, true, bool); // set attr bias_term true
            SET_PARAMETER(weight_2, *bias, pblock_type); // gen new bias
            if (weights_dtype == AK_FLOAT) {
                graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<float, Ttype>::update_weights, weights, *bias,
                        weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                        false, batch_norm_weight_3_vector[0], epsilon,
                        batch_norm_weight_1_vector, batch_norm_weight_2_vector,
                        scale_weight_1_vector, scale_weight_2_vector, scale_bias_term);
            } else{
                graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                        WeightsFusion<char, Ttype>::update_weights, weights, *bias,
                        weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3],
                        false, batch_norm_weight_3_vector[0], epsilon,
                        batch_norm_weight_1_vector, batch_norm_weight_2_vector,
                        scale_weight_1_vector, scale_weight_2_vector, scale_bias_term);
            }

            saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                               strides[0], strides[1],
                                               dilation_rate[0], dilation_rate[1],
                                               &(weights.d_tensor()), &(bias->d_tensor()),
                                               active_param);
            _param_conv_batchnorm_scale_relu = conv_param;
        }
    } else {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                           strides[0], strides[1],
                                           dilation_rate[0], dilation_rate[1],
                                           &(weights.d_tensor()), &(bias.d_tensor()),
                                           active_param);
        _param_conv_batchnorm_scale_relu = conv_param;
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvBatchnormScaleReluHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    auto group = GET_PARAMETER(int, group);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto weights = GET_PARAMETER(PBlock<Ttype>, weight_1);
    auto bias_term = GET_PARAMETER(bool, bias_term);

    //different device please change here!!!
#ifdef AMD_GPU
    saber::ImplEnum impl_e = SABER_IMPL;
#else
    saber::ImplEnum impl_e = VENDER_IMPL;
    if (std::is_same<Ttype, X86>::value) {
        impl_e = SABER_IMPL;
    }
    if (std::is_same<Ttype, NV>::value && Ptype == Precision::INT8) {
        impl_e = SABER_IMPL;
    }
    bool use_k1s1p0 = (Ptype == Precision::FP32);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.weight()->height() == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.weight()->width() == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.pad_h == 0);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.pad_w == 0);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.stride_h == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.stride_w == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.dilation_h == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.dilation_w == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.group == 1);
    use_k1s1p0 = use_k1s1p0 && (_param_conv_batchnorm_scale_relu.bias()->valid_size() > 0);
    bool use_k3s1d1 = (Ptype == Precision::FP32);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm_scale_relu.weight()->height() == 3);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm_scale_relu.weight()->width() == 3);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm_scale_relu.group == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm_scale_relu.stride_h == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm_scale_relu.stride_w == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm_scale_relu.dilation_h == 1);
    use_k3s1d1 = use_k3s1d1 && (_param_conv_batchnorm_scale_relu.dilation_w == 1);
    bool use_depthwise = (Ptype == Precision::FP32);
    use_depthwise = use_depthwise && (_param_conv_batchnorm_scale_relu.group == ins[0]->channel());
    use_depthwise = use_depthwise && (_param_conv_batchnorm_scale_relu.group == outs[0]->channel());
    bool use_direct_k = (Ptype == Precision::FP32);
    use_direct_k = use_direct_k && (_param_conv_batchnorm_scale_relu.weight()->channel() >= 16);
    use_direct_k = use_direct_k && (_param_conv_batchnorm_scale_relu.group == 1);
    if (std::is_same<Ttype, NV>::value
        && (use_k1s1p0 || use_k3s1d1 || use_depthwise || use_direct_k)) {
        impl_e = SABER_IMPL;
    }
#endif

    SABER_CHECK(_funcs_conv_batchnorm_scale_relu.init(ins, outs,
            _param_conv_batchnorm_scale_relu, SPECIFY, impl_e, ctx));

    // check if weights have been transposed
    auto is_weights_transed = CHECK_PARAMETER(is_weights_transed);
    if (!is_weights_transed) {
        SET_PARAMETER(is_weights_transed, true, bool);
        if (bias_term) {
            auto bias = GET_PARAMETER(PBlock<Ttype>, weight_2);
            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                    std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                              &_funcs_conv_batchnorm_scale_relu, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                    weights.d_tensor(), bias.d_tensor(), _param_conv_batchnorm_scale_relu.pad_h, _param_conv_batchnorm_scale_relu.pad_w, _param_conv_batchnorm_scale_relu.dilation_h, _param_conv_batchnorm_scale_relu.dilation_w,
                    strides[0], strides[1], group, impl_e);
            bias.map_to_host();
        } else {
            PBlock<Ttype> bias_empty;
            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                    std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                              &_funcs_conv_batchnorm_scale_relu, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                    weights.d_tensor(), bias_empty.d_tensor(), _param_conv_batchnorm_scale_relu.pad_h, _param_conv_batchnorm_scale_relu.pad_w, _param_conv_batchnorm_scale_relu.dilation_h, _param_conv_batchnorm_scale_relu.dilation_w,
                    strides[0], strides[1], group, impl_e);
        }
        weights.map_to_host();
    } else {
        PBlock<Ttype> weight_empty;
        PBlock<Ttype> bias_empty;
        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_1>(
                std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                        &_funcs_conv_batchnorm_scale_relu, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                        weight_empty.d_tensor(), bias_empty.d_tensor(), _param_conv_batchnorm_scale_relu.pad_h, _param_conv_batchnorm_scale_relu.pad_w, _param_conv_batchnorm_scale_relu.dilation_h, _param_conv_batchnorm_scale_relu.dilation_w,
                        strides[0], strides[1], group, impl_e);
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvBatchnormScaleReluHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_conv_batchnorm_scale_relu.compute_output_shape(ins, outs,
            _param_conv_batchnorm_scale_relu));

    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_CONVBATCHNORMSCALERELU(NV, Precision::FP32);
INSTANCE_CONVBATCHNORMSCALERELU(NV, Precision::INT8);
ANAKIN_REGISTER_OP_HELPER(ConvBatchnormScaleRelu, ConvBatchnormScaleReluHelper, NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ConvBatchnormScaleRelu, ConvBatchnormScaleReluHelper, NV, Precision::INT8);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONVBATCHNORMSCALERELU(X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ConvBatchnormScaleRelu, ConvBatchnormScaleReluHelper, X86, Precision::FP32);
#endif

//#ifdef USE_X86_PLACE
//template class ConvBatchnormScaleReluHelper<X86, Precision::FP32>;
//INSTANCE_CONVBATCHNORMSCALERELU(X86, Precision::FP32);
//ANAKIN_REGISTER_OP_HELPER(ConvBatchnormScaleRelu, ConvBatchnormScaleReluHelper, X86, 
//                                  Precision::FP32);
//#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONVBATCHNORMSCALERELU(ARM, Precision::FP32);
template class ConvBatchnormScaleReluHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvBatchnormScaleRelu, ConvBatchnormScaleReluHelper, ARM, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_CONVBATCHNORMSCALERELU(AMD, Precision::FP32);
template class ConvBatchnormScaleReluHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvBatchnormScaleRelu, ConvBatchnormScaleReluHelper, AMD, Precision::FP32);
#endif

#if defined BUILD_LITE
template class ConvBatchnormScaleReluHelper<X86, Precision::FP32>;
INSTANCE_CONVBATCHNORMSCALERELU(X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ConvBatchnormScaleRelu, ConvBatchnormScaleReluHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(ConvBatchnormScaleRelu)
.Doc("ConvBatchnormScaleRelu fusion operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("convolution_batchnorm_scale_relu")
.__alias__<NV, Precision::INT8>("convolution_batchnorm_scale_relu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("convolution_batchnorm_scale_relu")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("convolution_batchnorm_scale_relu")
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
.Args<float>("relu_0_alpha", " alpha for relu")
.Args<int>("scale_0_num_axes", " num axes for scale")
.Args<bool>("scale_0_bias_term", "whether scale has bias")
.Args<int>("scale_0_axis", "axis for scale")
.Args<float>("batchnorm_0_epsilon", "epsilon for batchnorm")
.Args<float>("batchnorm_0_momentum", "momentum for batchnorm");

} /* namespace ops */

} /* namespace anakin */


