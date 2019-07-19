#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_REORDER_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_REORDER_H

#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"

namespace anakin {
namespace saber {

template<typename TargetType>
SaberStatus convert_nchw_to_nchwc4(
        Tensor<TargetType> &out_tensor,
        const Tensor<TargetType> &in_tensor,
        Context<TargetType> ctx);

template<typename TargetType>
SaberStatus convert_nchwc4_to_nchw(
        Tensor<TargetType> &out_tensor,
        const Tensor<TargetType> &in_tensor,
        Context<TargetType> ctx);

}
}

#endif