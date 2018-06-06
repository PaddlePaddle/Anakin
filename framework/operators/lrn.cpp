#include "framework/operators/lrn.h"

namespace anakin {

namespace ops {

template<>
void Lrn<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<LrnHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = impl->_param_lrn;
    impl->_funcs_lrn(ins, outs, param, ctx);
}

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
LrnHelper<Ttype, Dtype, Ptype>::~LrnHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status LrnHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing Lrn op parameter.";

    auto local_size_in = GET_PARAMETER(int, local_size);
    auto alpha_in = GET_PARAMETER(float, alpha);
    auto beta_in = GET_PARAMETER(float, beta);
    auto norm_region_in = GET_PARAMETER(std::string, norm_region);
    auto k_in = GET_PARAMETER(float, k);

    if (norm_region_in == "ACROSS_CHANNELS") {
        LrnParam<Tensor4d<Ttype, Dtype>> param_lrn(local_size_in, alpha_in, beta_in, k_in, ACROSS_CHANNELS);
        _param_lrn = param_lrn;
    } else if (norm_region_in == "WITHIN_CHANNEL") {
        LrnParam<Tensor4d<Ttype, Dtype>> param_lrn(local_size_in, alpha_in, beta_in, k_in, WITHIN_CHANNEL);
        _param_lrn = param_lrn;
    } else {
        LOG(FATAL) << "Other Lrn norm_region" << norm_region_in << " should be replace by other ops.";
    }

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status LrnHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_lrn.init(ins, outs, _param_lrn, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status LrnHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_lrn.compute_output_shape(ins, outs, _param_lrn));
    return Status::OK();
}

#ifdef USE_CUDA
template class LrnHelper<NV, AK_FLOAT, Precision::FP32>;
template class LrnHelper<NV, AK_FLOAT, Precision::FP16>;
template class LrnHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class LrnHelper<ARM, AK_FLOAT, Precision::FP32>;
template class LrnHelper<ARM, AK_FLOAT, Precision::FP16>;
template class LrnHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Lrn, LrnHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Lrn, LrnHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Lrn)
.Doc("Lrn operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("lrn")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("lrn")
#endif
.num_in(3)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


