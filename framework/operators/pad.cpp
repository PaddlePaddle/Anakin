#include "framework/operators/pad.h"

namespace anakin {

namespace ops {

#define INSTANCE_PAD(Ttype, Ptype) \
template<> \
void Pad<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<PadHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PadHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_pad; \
    impl->_funcs_pad(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status PadHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "!!!!!!!! Parsing Pad op parameter.";
    auto pad_c = GET_PARAMETER(PTuple<int>, pad_c);
    auto pad_h = GET_PARAMETER(PTuple<int>, pad_h);
    auto pad_w = GET_PARAMETER(PTuple<int>, pad_w);


    saber::PadParam<Ttype> Pad_param(pad_c.vector(),pad_h.vector(),pad_w.vector());
    _param_pad = Pad_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PadHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_pad.init(ins, outs, _param_pad, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PadHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_pad.compute_output_shape(ins, outs, _param_pad));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_PAD(NV, Precision::FP32);
template class PadHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad, PadHelper, NV, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_PAD(X86, Precision::FP32);
template class PadHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad, PadHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PAD(ARM, Precision::FP32);
template class PadHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pad, PadHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Pad)
.Doc("Pad operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Pad")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Pad")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("Pad")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */
