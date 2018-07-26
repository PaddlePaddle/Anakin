#ifndef ANAKIN_SABER_FUNCS_CALIBRATE_H
#define ANAKIN_SABER_FUNCS_CALIBRATE_H

#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include <vector>

namespace anakin {
namespace saber {

SaberStatus conv_calibrate_fp32_int8(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor, float in_scale, Context<NV> ctx);

SaberStatus conv_calibrate_int32_fp32(
        Tensor<NV> &out_tensor, const Tensor<NV> &in_tensor, float* weight_scale, Context<NV> ctx);
}
}
#endif