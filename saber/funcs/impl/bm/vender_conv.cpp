
#include "saber/funcs/impl/bm/vender_conv.h"
#include "bmkernel_base.h"
#include <string.h>
#include <stdio.h>
#include <iostream>

namespace anakin
{
namespace saber
{

// FP32 part
template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    create(const std::vector<Tensor<BM> *>& inputs,
            std::vector<Tensor<BM> *>& outputs,
            ConvParam<BM>& param, Context<BM>& ctx)
{
}

template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    init(const std::vector<Tensor<BM> *> &inputs,
         std::vector<Tensor<BM> *> &outputs,
         ConvParam<BM> &param, Context<BM> &ctx)
{

    _handle = ctx.get_handle();
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderConv2D<BM, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<BM>*>& inputs,
                std::vector<Tensor<BM>*>& outputs,
                ConvParam<BM>& param)
{
    enum BmOpType op = CONV;
    bmkernel_api_base api = { op };

    //TODO: pass conv args into BM Kernel

    bm_status_t bm_stat = bmkernel_launch(_handle, "/usr/local/include/bm/bmkernel_bin.bin");
    CHECK_EQ(BM_SUCCESS, bm_stat) << "bmkernel_launch failed.";
    
    /* Send arguments. */
    BM_CHECK(bmkernel_send_args(_handle, reinterpret_cast<void *>(&api), sizeof(api)));

    LOG(INFO) << "BM conv done!";

    return SaberSuccess;
}

// INT8 part
// Not supported yet

template class VenderConv2D<BM, AK_FLOAT>;
} // namespace saber
} // namespace anakin