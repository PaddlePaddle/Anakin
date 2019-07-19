
/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

#include "include/saber_ctc_align.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberCtcAlign<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CtcAlignParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx         = &ctx;
    Shape offset_shape = Shape({inputs[0]->num(), 1, 1, 1});
    _in_offset.re_alloc(offset_shape);
    _out_offset.re_alloc(offset_shape);
    // following 3 lines code are in "create" funcion
    Shape offset_shape2 = Shape({inputs[0]->get_seq_offset().size(), 1, 1, 1});
    _in_offset.reshape(offset_shape2);
    _out_offset.reshape(offset_shape2);
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberCtcAlign<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CtcAlignParam<AMD>& param,
    Context<AMD>& ctx) {
    const int count = outputs[0]->size();
    int globalsize  = outputs[0]->size();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "CtcAlign.cl";
    kernelInfo.kernel_name = "CtcAlign";
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {AMD_NUM_THREADS};
    kernelInfo.g_wk = {(count + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};

    _kernel_ctc_align = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!_kernel_ctc_align.get()->isInit()) {
        LOG(ERROR) << "Failed to load program _kernel_Scale_multiBias";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberCtcAlign<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CtcAlignParam<AMD>& param) {
    bool err             = false;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    amd_kernel_list list;

    int out_n = outputs[0]->num();
    // To set the argument
    cl_mem in_data, out_data;
    in_data  = (cl_mem)inputs[0]->data();          // in_data
    out_data = (cl_mem)outputs[0]->mutable_data(); // out_data

    cl_mem in_offset  = (cl_mem)_in_offset.mutable_data();
    cl_mem out_offset = (cl_mem)_out_offset.mutable_data();
    int seq_num       = (inputs[0]->get_seq_offset()).size() - 1;
    cl_int status;
    int num_threads    = 1;
    int blank          = param.blank;
    int merge_repeated = param.merge_repeated;

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        AMD_API::async_memcpy(
            (cl_mem)in_offset, //  TPtr dst,
            0,
            (int)inputs[0]->device_id(),
            (void*) & (inputs[0]->get_seq_offset())[0], // const void* src,
            0,
            (int)inputs[0]->device_id(),
            sizeof(int) * (seq_num + 1), //  size_t count,
            cm,
            __HtoD());

        AMDKernel* kernel = _kernel_ctc_align.get();
        err               = kernel->SetKernelArgs(
                                out_data,
                                out_offset,
                                in_data,
                                in_offset,
                                seq_num,
                                blank,
                                merge_repeated,
                                num_threads);

        if (!err) {
            LOG(ERROR) << "Fail to s _kernel_ctc_align->SetKernelArgs";
            return SaberInvalidValue;
        }

        list.push_back(_kernel_ctc_align);

        err = LaunchKernel(cm, list);

        if (!err) {
            LOG(ERROR) << "Fail to set execution";
            return SaberInvalidValue;
        }

        std::vector<std::vector<int>> seq_offset;
        seq_offset.resize((inputs[0]->get_seq_offset()).size());

        AMD_API::async_memcpy(
            (void*)&seq_offset[0], // cpu_data
            0,
            0,
            (cl_mem)out_offset, // device_data
            0,
            inputs[0]->device_id(),
            sizeof(int) * (seq_num + 1),
            cm,
            __DtoH());

        outputs[0]->set_seq_offset(seq_offset);

        return SaberSuccess;
    }
}

template class SaberCtcAlign<AMD, AK_FLOAT>;

} // namespace saber
} // namespace anakin
