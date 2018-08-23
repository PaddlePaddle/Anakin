#include "framework/operators/shuffle_channel.h"

namespace anakin {

namespace ops {

#define INSTANCE_SHUFFLE_CHANNEL(Ttype, Dtype, Ptype) \
template<> \
void ShuffleChannel<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<ShuffleChannelHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<ShuffleChannelHelper<Ttype, Dtype, Ptype>*>\
                  (this->_helper)->_param_shuffle_channel; \
    impl->_funcs_shuffle_channel(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ShuffleChannelHelper<Ttype, Dtype, Ptype>::InitParam() {
    auto group = GET_PARAMETER(int, group);
    saber::ShuffleChannelParam<Tensor4d<Ttype, Dtype>> param(group);
    _param_shuffle_channel = param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ShuffleChannelHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_shuffle_channel.init(ins, outs, _param_shuffle_channel, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ShuffleChannelHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_shuffle_channel.compute_output_shape(ins, outs, _param_shuffle_channel));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SHUFFLE_CHANNEL(NV, AK_FLOAT, Precision::FP32);
template class ShuffleChannelHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_SHUFFLE_CHANNEL(X86, AK_FLOAT, Precision::FP32);
template class ShuffleChannelHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SHUFFLE_CHANNEL(ARM, AK_FLOAT, Precision::FP32);
template class ShuffleChannelHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ShuffleChannel, ShuffleChannelHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(ShuffleChannel)
.Doc("ShuffleChannel operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("shufflechannel")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("shufflechannel")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, AK_FLOAT, Precision::FP32>("shufflechannel")
#endif
.num_in(1)
.num_out(1)
.Args<int>("group", " group number for shuffle ");

} /* namespace ops */

} /* namespace anakin */


