#include "framework/operators/lrn.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Lrn<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl =
        static_cast<LrnHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = impl->_param_lrn;
    impl->_funcs_lrn(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
LrnHelper<Ttype, Ptype>::~LrnHelper() {
}

template<typename Ttype, Precision Ptype>
Status LrnHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Lrn op parameter.";

    auto local_size_in = GET_PARAMETER(int, local_size);
    auto alpha_in = GET_PARAMETER(float, alpha);
    auto beta_in = GET_PARAMETER(float, beta);
    auto norm_region_in = GET_PARAMETER(std::string, norm_region);
    auto k_in = GET_PARAMETER(float, k);

    if (norm_region_in == "ACROSS_CHANNELS") {
        LrnParam<Ttype> param_lrn(local_size_in, alpha_in, beta_in, k_in, ACROSS_CHANNELS);
        _param_lrn = param_lrn;
    } else if (norm_region_in == "WITHIN_CHANNEL") {
        LrnParam<Ttype> param_lrn(local_size_in, alpha_in, beta_in, k_in, WITHIN_CHANNEL);
        _param_lrn = param_lrn;
    } else {
        LOG(FATAL) << "Other Lrn norm_region" << norm_region_in << " should be replace by other ops.";
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status LrnHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_lrn.init(ins, outs, _param_lrn, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status LrnHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_lrn.compute_output_shape(ins, outs, _param_lrn));
    return Status::OK();
}

#ifdef USE_CUDA
template class LrnHelper<NV, Precision::FP32>;
template class LrnHelper<NV, Precision::FP16>;
template class LrnHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class LrnHelper<ARM, Precision::FP32>;
template class LrnHelper<ARM, Precision::FP16>;
template class LrnHelper<ARM, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Lrn, LrnHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Lrn, LrnHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Lrn)
.Doc("LRN operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("LRN")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("LRN")
#endif
.num_in(3)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


