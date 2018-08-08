#include "framework/operators/fusion_ops/batchnorm_scale.h"

namespace anakin {

namespace ops {

#define INSTANCE_BATCHNORMSCALE(Ttype, Dtype, Ptype) \
template<> \
void BatchnormScale<Ttype, Dtype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,\
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {\
    auto* impl = static_cast<BatchnormScaleHelper<Ttype, Dtype, Ptype>*>(this->_helper);\
    auto& param = static_cast<BatchnormScaleHelper<Ttype, Dtype, Ptype>*>\
                  (this->_helper)->_param_scale;\
    SABER_CHECK(impl->_funcs_scale(ins, outs, param, ctx));\
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status BatchnormScaleHelper<Ttype, Dtype, Ptype>::InitParam() {
	using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;
    LOG(WARNING) << "Parsing BatchnormScale op parameter.";

    // get batchnorm param
    auto epsilon = GET_PARAMETER(float, batchnorm_0_epsilon);
    auto momentum = GET_PARAMETER(float, batchnorm_0_momentum);
    auto batch_norm_weight_1 = GET_PARAMETER(pblock_type, batchnorm_0_weight_1);
    auto mean = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = GET_PARAMETER(pblock_type, batchnorm_0_weight_2);
    auto var = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = GET_PARAMETER(pblock_type, batchnorm_0_weight_3);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector();

    // get scale param
    auto scale_num_axes = GET_PARAMETER(int, scale_0_num_axes);
    auto scale_bias_term = GET_PARAMETER(bool, scale_0_bias_term);
    auto scale_axis = GET_PARAMETER(int, scale_0_axis);
    auto scale_weight_1 = GET_PARAMETER(pblock_type, scale_0_weight_1);
    auto scale = scale_weight_1.vector();
    auto scale_weight_2 = GET_PARAMETER(pblock_type, scale_0_weight_2);
    auto shift = scale_weight_2.vector();

    CHECK_EQ(mean.size(), var.size());
    CHECK_EQ(mean.size(), scale.size());
    if (scale_bias_term){
        CHECK_EQ(mean.size(), shift.size());
    }

    auto new_scale = mean;
    auto new_shift = var;
    auto scale_factor = batch_norm_weight_3_vector[0];
    for (int i = 0; i < mean.size(); i++) {
        auto alpha = 1 / sqrtf(var[i] * scale_factor + epsilon);
        auto beta = -alpha * mean[i] * scale_factor;
        new_scale[i] = alpha * scale[i];
        new_shift[i] = beta * scale[i];

        if (scale_bias_term) {
             new_shift[i] += shift[i];
        }
    }
    
    saber::ScaleParam<Tensor4d<Ttype, Dtype>> scale_param(new_scale,  new_shift,
                                           scale_bias_term, scale_axis, scale_num_axes);

	_param_scale = scale_param;

	
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status BatchnormScaleHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_scale.init(ins, outs, \
        _param_scale, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status BatchnormScaleHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_scale.compute_output_shape(ins, outs, \
        _param_scale));
    return Status::OK();
}

#ifdef USE_ARM_PLACE
INSTANCE_BATCHNORMSCALE(ARM, AK_FLOAT, Precision::FP32);
template class BatchnormScaleHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchnormScale, BatchnormScaleHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_CUDA
INSTANCE_BATCHNORMSCALE(NV, AK_FLOAT, Precision::FP32);
template<>
Status BatchnormScaleHelper<NV, AK_FLOAT, Precision::FP32>::Init(OpContext<NV>& ctx, \
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins, \
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    _funcs_scale.init(ins, outs, _param_scale, SPECIFY, VENDER_IMPL, ctx);
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(BatchnormScale, BatchnormScaleHelper, NV, AK_FLOAT,
                          Precision::FP32);
#endif
//#ifdef USE_X86_PLACE
//INSTANCE_CONVBATCHNORMSCALE(X86, AK_FLOAT, Precision::FP32);
//template class BatchnormScaleHelper<X86, AK_FLOAT, Precision::FP32>;
//ANAKIN_REGISTER_OP_HELPER(ConvBatchnormScale, BatchnormScaleHelper, X86, AK_FLOAT,
//                          Precision::FP32);
//#endif


//! register op
ANAKIN_REGISTER_OP(BatchnormScale)
.Doc("BatchnormScale fusion operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("batchnorm_scale")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("batchnorm_scale")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", "axis of conv")
.Args<int>("scale_0_num_axes", " num axes for scale")
.Args<bool>("scale_0_bias_term", "whether scale has bias")
.Args<int>("scale_0_axis", "axis for scale")
.Args<float>("batchnorm_0_epsilon", "epsilon for batchnorm")
.Args<float>("batchnorm_0_momentum", "momentum for batchnorm");

} /* namespace ops */

} /* namespace anakin */


