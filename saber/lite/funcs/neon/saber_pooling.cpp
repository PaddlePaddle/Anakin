#include "saber/lite/funcs/saber_pooling.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/pooling_arm_impl.h"

namespace anakin {

namespace saber {

namespace lite{
template <>
SaberStatus SaberPooling<float>::create(\
    const std::vector<Tensor<float>*>& inputs, \
    std::vector<Tensor<float>*>& outputs, \
    PoolingParam<Tensor<float>> &param, Context &ctx) {

    _ctx = ctx;

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
SaberStatus SaberPooling<float>::dispatch(\
    const std::vector<Tensor<float> *>& inputs,
    std::vector<Tensor<float> *>& outputs, PoolingParam<Tensor<float>> &param) {
    _impl(*outputs[0], *inputs[0], param.pooling_type, param.global_pooling, \
            param.window_w, param.window_h, \
            param.stride_w, param.stride_h, \
            param.pad_w, param.pad_h);
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE