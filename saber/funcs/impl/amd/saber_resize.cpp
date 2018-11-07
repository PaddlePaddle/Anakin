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
#include "include/saber_resize.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberResize<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ResizeParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberResize<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ResizeParam<AMD>& param,
    Context<AMD>& ctx) {

    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 2;
    kernelInfo.l_wk        = {8, 8};
    kernelInfo.g_wk        = {(outputs[0]->width() + 8 - 1) / 8 * 8,
                              (outputs[0]->height() + 8 - 1) / 8 * 8
                             };
    kernelInfo.kernel_file = "Resize.cl";
    kernelInfo.kernel_name = "resize_2d_kernel";

    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        ALOGE("Failed to load program");
        return SaberInvalidValue;
    }

    _kernel_ptr = kptr;

    ALOGD("COMPLETE CREATE KERNEL");

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberResize<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ResizeParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        ALOGE("Kernel is not exist");
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    int w_out = outputs[0]->width();
    int h_out = outputs[0]->height();
    int c_out = outputs[0]->channel();
    int n_out = outputs[0]->num();

    int w_in = inputs[0]->width();
    int h_in = inputs[0]->height();
    int c_in = inputs[0]->channel();
    int n_in = inputs[0]->num();

    int num_idx     = inputs[0]->num_index();
    int channel_idx = inputs[0]->channel_index();
    int height_idx  = inputs[0]->height_index();
    int width_idx   = inputs[0]->width_index();

    CHECK_EQ(c_in, c_out) << "input channel should = output channel";
    CHECK_EQ(c_in, c_out) << "input batch size should = output batch size";

    Shape src_real_shape;
    Shape dst_real_shape;

    if (inputs[0]->is_continue_mem()) {
        src_real_shape = inputs[0]->valid_shape();
    } else {
        src_real_shape = inputs[0]->shape();
    }

    if (outputs[0]->is_continue_mem()) {
        dst_real_shape = outputs[0]->valid_shape();
    } else {
        dst_real_shape = outputs[0]->shape();
    }

    int src_stride_w = src_real_shape.count(width_idx + 1); // inputs[0]->count(width_idx + 1,
    // dims);
    int src_stride_h =
        src_real_shape.count(height_idx + 1); // inputs[0]->count(height_idx + 1, dims);
    int src_stride_channel =
        src_real_shape.count(channel_idx + 1); // inputs[0]->count(channel_idx + 1, dims);
    int src_stride_batch = src_real_shape.count(num_idx + 1); // inputs[0]->count(num_idx + 1,
    // dims);
    int dst_stride_w =
        dst_real_shape.count(width_idx + 1); // outputs[0]->count(width_idx + 1, dims);
    int dst_stride_h =
        dst_real_shape.count(height_idx + 1); // outputs[0]->count(height_idx + 1, dims);
    int dst_stride_channel =
        dst_real_shape.count(channel_idx + 1); // outputs[0]->count(channel_idx + 1, dims);
    int dst_stride_batch =
        dst_real_shape.count(num_idx + 1); // outputs[0]->count(num_idx + 1, dims);

    printf("w_out:%d, h_out:%d, c_out:%d, n_out:%d\n", w_out, h_out, c_out, n_out);
    printf("w_in:%d, h_in:%d, c_in:%d, n_in:%d\n", w_in, h_in, c_in, n_in);

    bool err = false;

    for (int i = 0; i < n_out; ++i) {
        err = kernel->SetKernelArgs(
                  (int)w_out,
                  (int)h_out,
                  (int)n_out,
                  (int)c_out,
                  (int)dst_stride_w,
                  (int)dst_stride_h,
                  (int)dst_stride_channel,
                  (int)dst_stride_batch,
                  (int)w_in,
                  (int)h_in,
                  (int)src_stride_w,
                  (int)src_stride_h,
                  (int)src_stride_channel,
                  (int)src_stride_batch,
                  (float)(1 / param.width_scale),
                  (float)(1 / param.height_scale),
                  (PtrDtype)inputs[0]->data(),
                  (PtrDtype)outputs[0]->mutable_data());

        if (!err) {
            ALOGE("Failed to set kernel args");
            return SaberInvalidValue;
        }

        amd_kernel_list list;
        list.push_back(_kernel_ptr);
        err = LaunchKernel(cm, list);

        if (!err) {
            ALOGE("Failed to set execution");
            return SaberInvalidValue;
        }

        ALOGD("COMPLETE EXECUTION");
    }

    return SaberSuccess;
}
template class SaberResize<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberResize, ResizeParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberResize, ResizeParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
