#include "saber/funcs/impl/arm/saber_pooling.h"

#ifdef USE_ARM_PLACE

#include "saber/funcs/impl/arm/impl/pooling_arm_impl.h"

namespace anakin {

namespace saber {
template <>
SaberStatus SaberPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::create(\
    const std::vector<DataTensor_in*>& inputs, \
    std::vector<DataTensor_out*>& outputs, \
    PoolingParam<OpTensor> &param, Context<ARM> &ctx) {

    this->_ctx = &ctx;

    if (param.global_pooling) {
        _impl = pooling_global;
        return SaberSuccess;
    }

    if (param.window_w != param.window_h || param.stride_w != param.stride_h \
        || param.stride_w != 2 || param.pad_w != param.pad_h || param.pad_w > 1) {
        _impl = pooling_basic;
        return SaberSuccess;
    }

    if (param.window_w == 2) {
        if (param.pooling_type == Pooling_max) {
            _impl = pooling2x2s2_max;
        } else {
            _impl = pooling2x2s2_ave;
        }
        return SaberSuccess;
    }

    if (param.window_w == 3) {
        if (param.pooling_type == Pooling_max) {
            _impl = pooling3x3s2_max;
        } else {
            _impl = pooling3x3s2_ave;
        }
        return SaberSuccess;
    }

    _impl = pooling_basic;
    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in *>& inputs,
    std::vector<DataTensor_out *>& outputs, PoolingParam<OpTensor> &param) {
    _impl(*outputs[0], *inputs[0], param.pooling_type, param.global_pooling, \
            param.window_w, param.window_h, \
            param.stride_w, param.stride_h, \
            param.pad_w, param.pad_h);
    return SaberSuccess;
}

template class SaberPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE