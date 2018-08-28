#include "framework/operators/im2sequence.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Im2Sequence<NV, Precision::FP32>::operator() (OpContext<NV> &ctx, 
                          const std::vector<Tensor4dPtr<NV> >& ins, 
                          std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<Im2SequenceHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<Im2SequenceHelper<NV, Precision::FP32>*>(this->_helper)->_param_im2sequence;
    impl->_funcs_im2sequence(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
Im2SequenceHelper<Ttype, Ptype>::~Im2SequenceHelper() {
}

template<typename Ttype, Precision Ptype>
Status Im2SequenceHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Im2Sequence op parameter.";
    auto paddings = GET_PARAMETER(PTuple<int>, paddings);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto window_size = GET_PARAMETER(PTuple<int>, window_size);
    auto dilations = GET_PARAMETER(PTuple<int>, dilations);

    Im2SequenceParam<Ttype> im2sequence_param(window_size[0], window_size[1],
                                                               paddings[0], paddings[1],
                                                               paddings[2], paddings[3],
                                                               strides[0], strides[1],
                                                               dilations[0], dilations[1]);

    _param_im2sequence = im2sequence_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Im2SequenceHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, 
                                                const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_im2sequence.init(ins, outs, _param_im2sequence, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status Im2SequenceHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                                                      std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_im2sequence.compute_output_shape(ins, outs, _param_im2sequence));
    return Status::OK();
}

#ifdef USE_CUDA
template class Im2SequenceHelper<NV, Precision::FP32>;
template class Im2SequenceHelper<NV, Precision::FP16>;
template class Im2SequenceHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class Im2SequenceHelper<ARM, Precision::FP32>;
template class Im2SequenceHelper<ARM, Precision::FP16>;
template class Im2SequenceHelper<ARM, Precision::INT8>;
#endif

//template class Im2SequenceHelper<ARM, Precision::FP32>;
//template class Im2SequenceHelper<ARM, Precision::FP16>;
//template class Im2SequenceHelper<ARM, Precision::INT8>;
// register helper 
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Im2Sequence, Im2SequenceHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Im2Sequence, Im2SequenceHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Im2Sequence)
    .Doc("Im2Sequence operator")
#ifdef USE_CUDA
    .__alias__<NV, Precision::FP32>("im2sequence")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, Precision::FP32>("im2sequence")
#endif
    .num_in(1)
    .num_out(1)
    .Args<PTuple<int>>("paddings", " paddings for im2sequence.")
    .Args<PTuple<int>>("strides",  "strides for im2sequence.")
    .Args<PTuple<int>>("window_size", "window_size for im2sequence.")
    .Args<PTuple<int>>("dilations", "dilations for im2sequence.");

} /* namespace ops */

} /* namespace anakin */


