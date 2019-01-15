#include "framework/operators/sequence_conv.h"

namespace anakin {

namespace ops {

#define INSTANCE_SEQUENCE_CONV(Ttype, Ptype) \
template<> \
void SequenceConv<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SequenceConvHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SequenceConvHelper<Ttype, Ptype>*>(this->_helper)->_param; \
    impl->_funcs(ins, outs, param, ctx); \
}

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
SequenceConvHelper<Ttype, Ptype>::~SequenceConvHelper() {
}

template<typename Ttype, Precision Ptype>
Status SequenceConvHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SequenceConv op parameter.";

    auto context_length=GET_PARAMETER(int, context_length);
    auto context_start=GET_PARAMETER(int, context_start);
    auto context_stride=GET_PARAMETER(int, context_stride);
    auto padding_trainable=GET_PARAMETER(bool, padding_trainable);
    //auto filter_tensor=GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type> , filter_tensor);
    //auto padding_tensor=GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type> , padding_tensor);
    using pblock_type = PBlock<Ttype>;
    auto filter_tensor = GET_PARAMETER(pblock_type, filter_tensor);
    auto padding_tensor = GET_PARAMETER(pblock_type, padding_tensor);


    if(padding_tensor.d_tensor().valid_size()>0) {
        SequenceConvParam<Ttype> param(&(filter_tensor.d_tensor()), context_length, context_start,
                                                        context_stride, padding_trainable, &(padding_tensor.d_tensor()));
        _param = param;
    }else{
        SequenceConvParam<Ttype> param(&(filter_tensor.d_tensor()), context_length, context_start,
                                                        context_stride, padding_trainable);
        _param = param;
    }

    return Status::OK();
}

template<>
Status SequenceConvHelper<X86, Precision::FP32>::Init(OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs.init(ins, outs, _param, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
template<>
Status SequenceConvHelper<X86, Precision::FP16>::Init(OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs.init(ins, outs, _param, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<>
Status SequenceConvHelper<X86, Precision::INT8>::Init(OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs.init(ins, outs, _param, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequenceConvHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs.init(ins, outs, _param, STATIC, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequenceConvHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs.compute_output_shape(ins, outs, _param));
    return Status::OK();
}
#ifdef AMD_GPU
INSTANCE_SEQUENCE_CONV(AMD, Precision::FP32);
template class SequenceConvHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceConv, SequenceConvHelper, AMD, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
INSTANCE_SEQUENCE_CONV(X86, Precision::FP32);
template class SequenceConvHelper<X86, Precision::FP32>;
template class SequenceConvHelper<X86, Precision::FP16>;
template class SequenceConvHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(SequenceConv, SequenceConvHelper, X86, Precision::FP32);
#endif
#ifdef USE_CUDA
INSTANCE_SEQUENCE_CONV(NV, Precision::FP32);
template class SequenceConvHelper<NV, Precision::FP32>;
template class SequenceConvHelper<NV, Precision::FP16>;
template class SequenceConvHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(SequenceConv, SequenceConvHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
INSTANCE_SEQUENCE_CONV(ARM, Precision::FP32);
template class SequenceConvHelper<ARM, Precision::FP32>;
template class SequenceConvHelper<ARM, Precision::FP16>;
template class SequenceConvHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(SequenceConv, SequenceConvHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(SequenceConv)
.Doc("SequenceConv operator")
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("SequenceConv")
#endif
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("SequenceConv")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("SequenceConv")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("SequenceConv")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", " axis ");

} /* namespace ops */

} /* namespace anakin */


