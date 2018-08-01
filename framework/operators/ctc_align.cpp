#include "framework/operators/ctc_align.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void CtcAlign<NV, AK_FLOAT, Precision::FP32>::operator() (OpContext<NV> &ctx, 
                          const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins, 
                          std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<CtcAlignHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<CtcAlignHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_ctc_align;
    impl->_funcs_ctc_align(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
CtcAlignHelper<Ttype, Dtype, Ptype>::~CtcAlignHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status CtcAlignHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing CtcAlign op parameter.";
    auto merge_repeated = GET_PARAMETER(bool, merge_repeated);
    auto blank = GET_PARAMETER(int, blank);

    CtcAlignParam<Tensor4d<Ttype, Dtype>> ctc_align_param(blank, merge_repeated);
    _param_ctc_align = ctc_align_param;

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status CtcAlignHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx, 
                                                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                                                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_ctc_align.init(ins, outs, _param_ctc_align, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status CtcAlignHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_ctc_align.compute_output_shape(ins, outs, _param_ctc_align));
    return Status::OK();
}

#ifdef USE_CUDA
template class CtcAlignHelper<NV, AK_FLOAT, Precision::FP32>;
template class CtcAlignHelper<NV, AK_FLOAT, Precision::FP16>;
template class CtcAlignHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class CtcAlignHelper<ARM, AK_FLOAT, Precision::FP32>;
template class CtcAlignHelper<ARM, AK_FLOAT, Precision::FP16>;
template class CtcAlignHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

//template class CtcAlignHelper<ARM, AK_FLOAT, Precision::FP32>;
//template class CtcAlignHelper<ARM, AK_FLOAT, Precision::FP16>;
//template class CtcAlignHelper<ARM, AK_FLOAT, Precision::INT8>;
// register helper 
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(CtcAlign, CtcAlignHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(CtcAlign, CtcAlignHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(CtcAlign)
    .Doc("CtcAlign operator")
#ifdef USE_CUDA
    .__alias__<NV, AK_FLOAT, Precision::FP32>("ctc_align")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, AK_FLOAT, Precision::FP32>("ctc_align")
#endif
    .num_in(1)
    .num_out(1)
    .Args<bool>("merge_repeated", " merge_repeated for ctc_align.")
    .Args<int>("blank",  "blank for ctc_align.");

} /* namespace ops */

} /* namespace anakin */


