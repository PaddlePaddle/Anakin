/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "saber/funcs/impl/amd/include/saber_pooling.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberPooling<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPooling<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    KernelInfo kernelInfo;
    int pooling_type = 0;
    int average_include = 0;

#ifdef ENABLE_DEBUG
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "param.pooling_type=" << param.pooling_type <<
                                         " param.window_h=" << param.window_h
                                         << " param.window_w=" << param.window_w
                                         << " param.pad_h=" << param.pad_h << " param.pad_w=" << param.pad_w
                                         << " param.stride_h=" << param.stride_h
                                         << " param.stride_w=" << param.stride_w
                                         << " param.global_pooling=" << param.global_pooling;
#endif

    switch (param.pooling_type) {
        case Pooling_max:
        {
            pooling_type = (PoolingType)MLO_POOLING_OP_MAX;
        } break;

        case Pooling_average_exclude_padding:
        {
            pooling_type = (PoolingType)MLO_POOLING_OP_AVE;
        } break;

        case Pooling_average_include_padding:
        {
            pooling_type = (PoolingType)MLO_POOLING_OP_AVE;
            average_include = 1;
        } break;

        default:
        {
            LOG(ERROR) << "Unknown polling type: " << param.pooling_type;
        } break;
    }

    int _grp_tile0     = 8;
    int _grp_tile1     = 8;
    int _out_width     = (inputs[0]->width() + param.pad_w * 2 - param.window_w + param.stride_w - 1) / param.stride_w + 1;
    int _out_height    = (inputs[0]->height() + param.pad_h * 2 - param.window_h + param.stride_h - 1) / param.stride_h + 1;
    int _out_pix_tile0 = std::max(1, 8 / param.stride_w);
    int _out_pix_tile1 = std::max(1, 8 / param.stride_h);

    while (_out_pix_tile0 * _grp_tile0 > _out_width * 2 && _out_pix_tile0 > 1) {
        _out_pix_tile0 >>= 1;
    }

    while (_out_pix_tile1 * _grp_tile1 > _out_height * 2 && _out_pix_tile1 > 1) {
        _out_pix_tile1 >>= 1;
    }

    kernelInfo.wk_dim = 3;

    if (param.window_h == 2
            && param.window_w == 2
            && param.pad_w == 0
            && param.pad_h == 0)
    {
        kernelInfo.l_wk        = {256, 1, 1};
        kernelInfo.g_wk        = {64 * 64 * 40, 1, 1};
        kernelInfo.kernel_file = "Pooling.cl";
        kernelInfo.kernel_name = "mloPooling";
    }
    else if (param.window_h == inputs[0]->height()
            && param.window_w == inputs[0]->width()
            && param.pad_w == 0
            && param.pad_h == 0
            && (inputs[0]->channel()*inputs[0]->num() % 256) == 0)
    {
        int g_wk_width  = 1;
        int g_wk_height = 1;
        kernelInfo.l_wk = {256, 1, 1};
        kernelInfo.g_wk = {inputs[0]->channel() * inputs[0]->num(), 1, 1};
        kernelInfo.kernel_file = "Pooling7x7_7_7_2048.cl";
        kernelInfo.kernel_name = "mloPoolingG";
    }
    else
    {
        int g_wk_width  = ((_out_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
        int g_wk_height = ((_out_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));
        kernelInfo.l_wk = {_grp_tile0, _grp_tile1, 1};
        kernelInfo.g_wk = {g_wk_width * _grp_tile0,
                           g_wk_height * _grp_tile1,
                           inputs[0]->channel()* inputs[0]->num()
                          };
        kernelInfo.kernel_file = "MIOpenPooling.cl";
        kernelInfo.kernel_name = "mloPoolingG";
        kernelInfo.kernel_type = MIOPEN;
    }

    int bot_batch_stride   = inputs[0]->width() * inputs[0]->height() * outputs[0]->channel();
    int bot_channel_stride = inputs[0]->width() * inputs[0]->height();

    int top_batch_stride   = outputs[0]->width() * outputs[0]->height() * outputs[0]->channel();
    int top_channel_stride = outputs[0]->width() * outputs[0]->height();

    // set comp_options...
    kernelInfo.comp_options =
            std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(pooling_type)
            + std::string(" -DMLO_POOLING_KERNEL_SZ0=") + std::to_string(param.window_w)
            + std::string(" -DMLO_POOLING_KERNEL_SZ1=") + std::to_string(param.window_h)
            + std::string(" -DMLO_POOLING_PAD0=") + std::to_string(param.pad_w)
            + std::string(" -DMLO_POOLING_PAD1=") + std::to_string(param.pad_h)
            + std::string(" -DMLO_POOLING_STRIDE0=") + std::to_string(param.stride_w)
            + std::string(" -DMLO_POOLING_STRIDE1=") + std::to_string(param.stride_h)
            + std::string(" -DMLO_POOLING_N_OUTPUTS=") + std::to_string(inputs[0]->channel())
            + std::string(" -DMLO_POOLING_N_CHANNELS=") + std::to_string(outputs[0]->channel())
            + std::string(" -DMLO_POOLING_N_HORIZ_OUT_PIX=") + std::to_string(_out_pix_tile0)
            + std::string(" -DMLO_POOLING_N_VERT_OUT_PIX=") + std::to_string(_out_pix_tile1)
            + std::string(" -DMLO_POOLING_GROUP_SZ0=8")
            + std::string(" -DMLO_POOLING_GROUP_SZ1=8")
            + std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") + std::to_string(bot_batch_stride)
            + std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(bot_channel_stride)
            + std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(inputs[0]->width())
            + std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") + std::to_string(top_batch_stride)
            + std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string(top_channel_stride)
            + std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string(outputs[0]->width())
            + std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(inputs[0]->width())
            + std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(inputs[0]->height())
            + std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string(outputs[0]->width())
            + std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string(outputs[0]->height())
            + std::string(" -DBATCH_NUM=") + std::to_string(inputs[0]->num())
            + std::string(" -DAVERAGE_INCLUDE=") + std::to_string(average_include)
            + std::string(" -DCU_NUM=64")
            + std::string(" -DMIOPEN_USE_FP32=1")
            + std::string(" -DMIOPEN_USE_FP16=0");

    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernel_ptr = kptr;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPooling<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param) {
#ifdef ENABLE_DEBUG
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch";
#endif

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    bool err = false;

    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    err = kernel->SetKernelArgs(
              (PtrDtype)inputs[0]->data(), (PtrDtype)outputs[0]->mutable_data(), (PtrDtype)0);

    amd_kernel_list list;
    list.push_back(_kernel_ptr);
    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}
template class SaberPooling<AMD, AK_FLOAT>;
} // namespace saber
} // namespace anakin
