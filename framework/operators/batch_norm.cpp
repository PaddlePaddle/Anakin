#include "framework/operators/batch_norm.h"

namespace anakin {

namespace ops {

#define INSTANCE_BATCHNORM(Ttype, Ptype) \
template<> \
void BatchNorm<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
        auto* impl = static_cast<BatchNormHelper<Ttype, Ptype>*>(this->_helper); \
        auto& param = static_cast<BatchNormHelper<Ttype, Ptype>*>(this->_helper)->_param_scale; \
        impl->_funcs_scale(ins, outs, param, ctx); \
}

#if 0//def USE_CUDA
template<>
void BatchNorm<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<BatchNormHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<BatchNormHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_scale;
    impl->_funcs_scale(ins, outs, param, ctx);
}
#endif

template<typename Ttype, Precision Ptype>
Status BatchNormHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Scale op parameter.";
    using pblock_type = PBlock<Ttype>;

    auto eps = GET_PARAMETER(float, epsilon);
    auto mean = GET_PARAMETER(pblock_type, weight_1);
    auto var = GET_PARAMETER(pblock_type, weight_2);
    auto scale_factor = GET_PARAMETER(pblock_type, weight_3);
    auto mean_vec = mean.vector();
    auto var_vec = var.vector();
    auto scale_factor_vec = scale_factor.vector();
    std::vector<typename PrecisionWrapper<Ptype>::type> scale;
    std::vector<typename PrecisionWrapper<Ptype>::type> bias;
    scale.resize(mean.count());
    bias.resize(mean.count());
    auto scale_val = scale_factor_vec[0] == 0 ? 0 : 1 / scale_factor_vec[0];
    for (int i = 0; i < mean.count(); i++) {
        scale[i] = 1.0f / std::sqrt(var_vec[i] * scale_val + eps);
        bias[i] = - mean_vec[i] * scale_val / std::sqrt(var_vec[i] * scale_val + eps); 
    }

    saber::ScaleParam<Ttype> param_scale(scale, bias, true, 1, 1);
    _param_scale = param_scale;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status BatchNormHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_scale.init(ins, outs, _param_scale, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status BatchNormHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_scale.compute_output_shape(ins, outs, _param_scale));
    return Status::OK();
}

// register helper
#ifdef USE_CUDA
INSTANCE_BATCHNORM(NV, Precision::FP32);
template class BatchNormHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_BATCHNORM(X86, Precision::FP32);
template class BatchNormHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_BATCHNORM(ARM, Precision::FP32);
template class BatchNormHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(BatchNorm)
.Doc("BatchNorm operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("eps")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("eps")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("eps")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


