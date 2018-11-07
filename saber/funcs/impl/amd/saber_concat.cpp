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
#include "include/saber_concat.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberConcat<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConcatParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberConcat<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConcatParam<AMD>& param,
    Context<AMD>& ctx) {
    KernelInfo kernelInfo;
    int input_size = inputs.size();
    // input_size = 1;
    _num_concats       = inputs[0]->count_valid(0, param.axis);
    _concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());

    bool out_cont_flag = outputs[0]->is_continue_mem();
    bool in_cont_flag  = inputs[0]->is_continue_mem();

    int offset_concat_axis    = 0;
    Shape out_shape           = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];

    for (int i = 1; i < input_size; ++i) {
        in_cont_flag &= inputs[i]->is_continue_mem();
    }

    kernelInfo.kernel_file = "Concat.cl";
    int out_size           = outputs[0]->size();

    if (in_cont_flag && out_cont_flag) {
        kernelInfo.kernel_name = "Concat_normal";

        kernelInfo.l_wk       = {AMD_NUM_THREADS, 1, 1};
        kernelInfo.g_wk       = {(out_size + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]
                                 * kernelInfo.l_wk[0]
                                };
        _kernel_concat_normal = CreateKernel(inputs[0]->device_id(), &kernelInfo);
    }

    ALOGD("COMPLETE CREATE KERNEL");
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberConcat<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConcatParam<AMD>& param) {
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    int input_size       = inputs.size();
    bool err             = false;

    //! get output data, valid shape and stride shape
    OpDataType* out_data      = (OpDataType*)outputs[0]->mutable_data();
    int offset_concat_axis    = 0;
    Shape out_shape           = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];
    bool out_cont_flag        = outputs[0]->is_continue_mem();
    bool in_cont_flag         = inputs[0]->is_continue_mem();

    for (int i = 1; i < input_size; ++i) {
        in_cont_flag &= inputs[i]->is_continue_mem();
    }

    //! inputs and outputs are all with continuous memory
    amd_kernel_list list;

    if (in_cont_flag && out_cont_flag) {
        for (int i = 0; i < input_size; ++i) {
            Shape in_shape = inputs[i]->valid_shape();
            // std::vector<int> bottom_shape = {tmp[3], tmp[2], tmp[1], tmp[0]};
            const OpDataType* in_data = (const OpDataType*)inputs[i]->data();
            const int in_concat_axis  = in_shape[param.axis];
            const int in_concat_size  = in_concat_axis * _concat_input_size;
            const int nthreads        = in_concat_size * _num_concats;
            AMDKernel* kernel         = _kernel_concat_normal.get();
            err                       = kernel->SetKernelArgs(
                                            (int)nthreads,
                                            (PtrDtype)inputs[i]->data(),
                                            (int)_num_concats,
                                            (int)_concat_input_size,
                                            (int)out_concat_axis,
                                            (int)in_concat_axis,
                                            (int)offset_concat_axis,
                                            (PtrDtype)out_data);
            list.push_back(_kernel_concat_normal);

            if (!err) {
                ALOGE("Fail to set execution SetKernelArgs is_balance == false ");
                return SaberInvalidValue;
            }

            offset_concat_axis += in_concat_axis;
            err = LaunchKernel(cm, list);
            list.clear();

            if (!err) {
                ALOGE("Fail to set execution kernels ");
                return SaberInvalidValue;
            }
        }
    } else {
        Shape offset_out = outputs[0]->offset();
        Tensor<AMD> tsub;

        for (int i = 0; i < input_size; ++i) {
            Shape in_shape = inputs[i]->valid_shape();
            tsub.share_sub_buffer(*outputs[0], in_shape, offset_out);
            offset_out[param.axis] += in_shape[param.axis];
            tsub.async_copy_from(*inputs[i], cm);
        }
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, AMD, AK_INT8);
} // namespace saber
} // namespace anakin
