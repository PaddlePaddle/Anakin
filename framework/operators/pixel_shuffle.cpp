#include "framework/operators/pixel_shuffle.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
void PixelShuffle<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    auto* impl = static_cast<PixelShuffleHelper<Ttype, Ptype>*>(this->_helper);
    auto& param = static_cast<PixelShuffleHelper<Ttype, Ptype>*>
                  (this->_helper)->_param_pixel_shuffle;
    impl->_funcs_pixel_shuffle(ins, outs, param, ctx);
}

template<typename Ttype, Precision Ptype>
Status PixelShuffleHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << " Parsing PixelShuffle op parameter.";
    auto rw = GET_PARAMETER(int, rw);
    auto rh = GET_PARAMETER(int, rh);
    auto channel_first = GET_PARAMETER(bool, channel_first);

    saber::PixelShuffleParam<Ttype> pixel_shuffle_param(rh, rw, channel_first);
    _param_pixel_shuffle = pixel_shuffle_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PixelShuffleHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_pixel_shuffle.init(ins, outs, _param_pixel_shuffle, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PixelShuffleHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_pixel_shuffle.compute_output_shape(ins, outs, _param_pixel_shuffle));
    return Status::OK();
}

#ifdef USE_CUDA
template class PixelShuffleHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PixelShuffle, PixelShuffleHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
template class PixelShuffleHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PixelShuffle, PixelShuffleHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
template class PixelShuffleHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PixelShuffle, PixelShuffleHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(PixelShuffle)
.Doc("PixelShuffle operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("PixelShuffle")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("PixelShuffle")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("PixelShuffle")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


