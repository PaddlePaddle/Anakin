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
#include "include/saber_im2sequence.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberIm2Sequence<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    Im2SequenceParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberIm2Sequence<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    Im2SequenceParam<AMD>& param,
    Context<AMD>& ctx) {
    const int count  = outputs[0]->valid_size();
    int input_height = inputs[0]->height(); // P
    _kernel_exten_h  = param.dilation_h * (param.window_h - 1) + 1;
    _output_height =
        (input_height + param.pad_up + param.pad_down - _kernel_exten_h) / param.stride_h + 1;

    int input_width = inputs[0]->width(); // Q
    _kernel_exten_w = param.dilation_w * (param.window_w - 1) + 1;
    _output_width =
        (input_width + param.pad_left + param.pad_right - _kernel_exten_w) / param.stride_w + 1;

    int out_n       = outputs[0]->num();
    int c           = inputs[0]->channel();
    int num_threads = out_n * c;

    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {256};
    kernelInfo.g_wk        = {(num_threads + 256 - 1) / 256 * 256};
    kernelInfo.kernel_file = "Im2sequence.cl";
    kernelInfo.kernel_name = "ker_im2sequence_fwd_shared";

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
SaberStatus SaberIm2Sequence<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    Im2SequenceParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int count       = outputs[0]->valid_size();
    int out_n       = outputs[0]->num();
    int out_c       = outputs[0]->channel();
    int n           = inputs[0]->num();
    int c           = inputs[0]->channel();
    int in_h        = inputs[0]->height();
    int in_w        = inputs[0]->width();
    int num_threads = out_n * c;
    std::vector<int> offset(n + 1);
    std::vector<std::vector<int>> seq_offset;
    seq_offset.push_back(offset);
    int per_seq_len = out_n / n;

    for (int i = 0; i < n; i++) {
        seq_offset[0].push_back(i * per_seq_len);
    }

    seq_offset[0].push_back(n * per_seq_len);
    outputs[0]->set_seq_offset(seq_offset);

    bool err = false;

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        ALOGE("Kernel is not exist");
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {

        // To set the argument
        err = kernel->SetKernelArgs(
                  (PtrDtype)outputs[0]->mutable_data(),
                  (PtrDtype)inputs[0]->data(),
                  (int)n,
                  (int)c,
                  (int)in_h,
                  (int)in_w,
                  (int)_output_height,
                  (int)_output_width,
                  (int)out_n,
                  (int)out_c,
                  (int)param.window_h,
                  (int)param.window_w,
                  (int)param.pad_up,
                  (int)param.pad_down,
                  (int)param.stride_h,
                  (int)param.stride_w,
                  (int)param.dilation_h,
                  (int)param.dilation_w,
                  (int)_kernel_exten_h,
                  (int)_kernel_exten_w,
                  (int)num_threads);

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
    }

    ALOGD("COMPLETE EXECUTION");
    return SaberSuccess;
}
template class SaberIm2Sequence<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberIm2Sequence, Im2SequenceParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberIm2Sequence, Im2SequenceParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
