#include "framework/operators/shuffle_channel.h"

namespace anakin {

namespace ops {

#define INSTANCE_SHUFFLE_CHANNEL(Ttype, Ptype) \
template<> \
void ShuffleChannel<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<ShuffleChannelHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<ShuffleChannelHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_shuffle_channel; \
    impl->_funcs_shuffle_channel(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status ShuffleChannelHelper<Ttype, Ptype>::InitParam() {
    auto group = GET_PARAMETER(int, group);
    saber::ShuffleChannelParam<Ttype> param(group);
    _param_shuffle_channel = param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ShuffleChannelHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_shuffle_channel.init(ins, outs, _param_shuffle_channel, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ShuffleChannelHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_shuffle_channel.compute_output_shape(ins, outs, _param_shuffle_channel));
    return Status::OK();
}

#ifdef AMD_GPU
INSTANCE_SHUFFLE_CHANNEL(AMD, Precision::FP32);
template class ShuffleChannelHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, AMD, Precision::FP32);
#endif

#ifdef USE_CUDA
INSTANCE_SHUFFLE_CHANNEL(NV, Precision::FP32);
INSTANCE_SHUFFLE_CHANNEL(NV, Precision::INT8);
template class ShuffleChannelHelper<NV, Precision::FP32>;
template class ShuffleChannelHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, NV, Precision::INT8);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_SHUFFLE_CHANNEL(X86, Precision::FP32);
INSTANCE_SHUFFLE_CHANNEL(X86, Precision::INT8);
template class ShuffleChannelHelper<X86, Precision::FP32>;
template class ShuffleChannelHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, X86, Precision::INT8);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SHUFFLE_CHANNEL(ARM, Precision::FP32);
INSTANCE_SHUFFLE_CHANNEL(ARM, Precision::INT8);
template class ShuffleChannelHelper<ARM, Precision::FP32>;
template class ShuffleChannelHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, ARM, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, ARM, Precision::INT8);
#endif

//! register op
ANAKIN_REGISTER_OP(ShuffleChannel)
.Doc("ShuffleChannel operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("shufflechannel")
.__alias__<NV, Precision::INT8>("shufflechannel")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("shufflechannel")
.__alias__<ARM, Precision::INT8>("shufflechannel")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("shufflechannel")
.__alias__<X86, Precision::INT8>("shufflechannel")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("shufflechannel")
#endif
.num_in(1)
.num_out(1)
.Args<int>("group", " group number for shuffle ");

} /* namespace ops */

} /* namespace anakin */


