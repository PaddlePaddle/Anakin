#include "framework/operators/pad.h"

namespace anakin {

namespace ops {

#define INSTANCE_PAD(Ttype, Dtype, Ptype) \
template<> \
void Pad<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<PadHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PadHelper<Ttype, Dtype, Ptype>*>\
                  (this->_helper)->_param_pad; \
    impl->_funcs_pad(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PadHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "!!!!!!!! Parsing Pad op parameter.";
    auto pad_c = GET_PARAMETER(PTuple<int>, pad_c);
    auto pad_h = GET_PARAMETER(PTuple<int>, pad_h);
    auto pad_w = GET_PARAMETER(PTuple<int>, pad_w);


    saber::PadParam<Tensor4d<Ttype, Dtype>> Pad_param(pad_c.vector(),pad_h.vector(),pad_w.vector());
    _param_pad = Pad_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PadHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_pad.init(ins, outs, _param_pad, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PadHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_pad.compute_output_shape(ins, outs, _param_pad));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_PAD(NV, AK_FLOAT, Precision::FP32);
template class PadHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad, PadHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_PAD(X86, AK_FLOAT, Precision::FP32);
template class PadHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad, PadHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PAD(ARM, AK_FLOAT, Precision::FP32);
template class PadHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad, PadHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Pad)
.Doc("Pad operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("Pad")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("Pad")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, AK_FLOAT, Precision::FP32>("Pad")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */


