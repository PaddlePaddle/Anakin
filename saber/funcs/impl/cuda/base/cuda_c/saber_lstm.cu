#include "saber/funcs/impl/cuda/saber_lstm.h"
#include "saber/core/tensor_op.h"
#include "cuda_fp16.h"
namespace anakin {

namespace saber {
template<>
SaberStatus
SaberLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(
    const std::vector < DataTensor_in* >& inputs,
    std::vector < DataTensor_out* >& outputs,
    LstmParam < OpTensor >& param) {

}

template class SaberLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
}

