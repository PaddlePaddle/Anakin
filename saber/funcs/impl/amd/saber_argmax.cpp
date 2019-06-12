/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#include "include/saber_argmax.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

#define LDS_MAX_FLOAT4_NUM 15360

template <DataType OpDtype>
SaberStatus SaberArgmax<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ArgmaxParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberArgmax<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ArgmaxParam<AMD>& param,
    Context<AMD>& ctx) {
    int localSize  = 256;
    int globalSize = 0;
    // _localWorkSize = localSize;

    if (!param.has_axis) {
        int inner_dim = inputs[0]->count(1, inputs[0]->dims());
        int outer_dim = inputs[0]->num();
        int group_num = (inner_dim + localSize - 1) / localSize;
        _group_max_value.re_alloc(Shape({outer_dim, group_num, 1, 1}, Layout_NCHW));
        _group_max_index.re_alloc(Shape({outer_dim, group_num, 1, 1}, Layout_NCHW));
    }

    while (localSize > 1) {
        if (2 * localSize * param.top_k > LDS_MAX_FLOAT4_NUM) {
            localSize >>= 1;
        } else {
            break;
        }
    }

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "Argmax.cl";
    kernelInfo.l_wk        = {localSize};
    kernelInfo.wk_dim      = 1;

    std::string strLocalSize = std::to_string(kernelInfo.l_wk[0]);
    std::string strTreeMemSize =
        std::to_string(2 * kernelInfo.l_wk[0] * param.top_k);
    kernelInfo.comp_options = std::string(" -DLOCAL_WORK_SIZE=") + strLocalSize
                              + std::string(" -DTREE_MEM_SIZE=") + strTreeMemSize;

    AMDKernelPtr kptr = NULL;

    if (param.has_axis) {
        int count         = inputs[0]->count(0, inputs[0]->dims());
        int dim           = inputs[0]->shape()[param.axis];
        int inner_dim     = inputs[0]->count(param.axis + 1, inputs[0]->dims());
        int total_threads = count / dim;
        globalSize        = (total_threads + localSize - 1) / localSize * localSize;
        kernelInfo.g_wk   = {globalSize};

        if (param.top_k == 1) {
            kernelInfo.kernel_name = "top1_channel";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_map[TOP_1_CHANNEL]     = kptr;
        } else {
            kernelInfo.kernel_name = "topk_channel";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_map[TOPK_CHANNEL]     = kptr;
        }
    } else {
        int inner_dim = inputs[0]->count(1, inputs[0]->dims());
        int outer_dim = inputs[0]->num();

        if (param.top_k == 1) {
            if (inner_dim / kernelInfo.l_wk[0] < 10) {
                globalSize             = outer_dim * kernelInfo.l_wk[0];
                kernelInfo.g_wk        = {globalSize};
                kernelInfo.kernel_name = "top1";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (NULL == kptr) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernel_map[TOP_1]     = kptr;
            } else {
                kernelInfo.kernel_name = "block_top1";
                int inner_group_num    = (inner_dim + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0];
                globalSize             = inner_group_num * outer_dim * kernelInfo.l_wk[0];
                kernelInfo.g_wk        = {globalSize};
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (NULL == kptr) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernel_map[BLOCK_TOP_1]     = kptr;

                kernelInfo.kernel_name = "top1_big";
                globalSize             = outer_dim * kernelInfo.l_wk[0];
                kernelInfo.g_wk        = {globalSize};
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (NULL == kptr) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernel_map[TOP_1_BIG]     = kptr;
            }
        } else if (param.top_k >= 13 && outer_dim == 1 && inner_dim > 100000) {
            unsigned int size = inner_dim;
            unsigned int begin_bit = 0;
            unsigned int end_bit = 8 * type_length(OpDtype);
            radix_sort_genIndexBuffer<AK_INT32>(inputs[0], _values_input);
            radix_sort_genRadixSortBuffer<AK_FLOAT, AK_INT32>(size, _keys_output, _values_output,
                    _batch_digit_counts, _digit_counts, _keys_tmp, _values_tmp, _radix_params, begin_bit, end_bit);
            radix_sort_gen_kernel<7>(_radix_sort_kernel_map, 0, inputs[0]->device_id(), _radix_params);
            radix_sort_gen_kernel<7>(_radix_sort_kernel_map, 1, inputs[0]->device_id(), _radix_params);
            radix_sort_gen_kernel<7>(_radix_sort_kernel_map, 2, inputs[0]->device_id(), _radix_params);
            radix_sort_gen_kernel<7>(_radix_sort_kernel_map, 3, inputs[0]->device_id(), _radix_params);

            radix_sort_gen_kernel<6>(_radix_sort_kernel_map, 0, inputs[0]->device_id(), _radix_params);
            radix_sort_gen_kernel<6>(_radix_sort_kernel_map, 1, inputs[0]->device_id(), _radix_params);
            radix_sort_gen_kernel<6>(_radix_sort_kernel_map, 2, inputs[0]->device_id(), _radix_params);
            radix_sort_gen_kernel<6>(_radix_sort_kernel_map, 3, inputs[0]->device_id(), _radix_params);

            KernelInfo kernelInfo;
            kernelInfo.kernel_file = "Argmax_radix_sort.cl";
            kernelInfo.kernel_name = "topk_radix_sort";
            kernelInfo.wk_dim      = 1;
            kernelInfo.l_wk        = {localSize};
            kernelInfo.g_wk        = {param.top_k};

            kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _radix_sort_kernel_map[RADIX_SORT_OUTPUT_COMBINE] = kptr;
        } else {
#if 0
            globalSize             = outer_dim * kernelInfo.l_wk[0];
            kernelInfo.g_wk        = {globalSize};
            kernelInfo.kernel_name = "topk_heap_shared";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (NULL == kptr) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_map[TOPK_HEAP_SHARED]     = kptr;
#else
            int localSize_t = 512;

            while (localSize_t > 1) {
                if (2 * localSize_t * param.top_k > LDS_MAX_FLOAT4_NUM) {
                    localSize_t >>= 1;
                } else {
                    break;
                }
            }

            if (inputs[0]->num() < 64) {
                _localWorkSize = localSize_t ;
            } else if (inputs[0]->num() > 64 && (2 * (2 * localSize_t * param.top_k) < LDS_MAX_FLOAT4_NUM)) {
                _localWorkSize = localSize_t ;
            } else {
                _localWorkSize = localSize >>= 1 ;
            }

            if (_localWorkSize == 512) {
                kernelInfo.l_wk        = {512};
                globalSize             = outer_dim * kernelInfo.l_wk[0];
                kernelInfo.g_wk        = {globalSize};
                kernelInfo.kernel_name = "topk_heap_shared_512";

                std::string strLocalSize = std::to_string(kernelInfo.l_wk[0]);
                std::string strTreeMemSize =
                    std::to_string(2 * kernelInfo.l_wk[0] * param.top_k);
                kernelInfo.comp_options = std::string(" -DLOCAL_WORK_SIZE=") + strLocalSize
                                          + std::string(" -DTREE_MEM_SIZE=") + strTreeMemSize;
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (NULL == kptr) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernel_map[TOPK_HEAP_SHARED_512]     = kptr;
                _globalWorkSize[TOPK_HEAP_SHARED_512] = kernelInfo.g_wk[0];
                _localWorkSize_map[TOPK_HEAP_SHARED_512] = kernelInfo.l_wk[0];
            } else {
                kernelInfo.l_wk        = {_localWorkSize};
                globalSize             = outer_dim * kernelInfo.l_wk[0];
                kernelInfo.g_wk        = {globalSize};
                kernelInfo.kernel_name = "topk_heap_shared";

                std::string strLocalSize = std::to_string(kernelInfo.l_wk[0]);
                std::string strTreeMemSize = std::to_string(2 * kernelInfo.l_wk[0] * param.top_k);
                kernelInfo.comp_options = std::string(" -DLOCAL_WORK_SIZE=") + strLocalSize +
                                          std::string(" -DTREE_MEM_SIZE=") + strTreeMemSize;
                kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (NULL == kptr) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernel_map[TOPK_HEAP_SHARED]     = kptr;
                _globalWorkSize[TOPK_HEAP_SHARED] = kernelInfo.g_wk[0];
                _localWorkSize_map[TOPK_HEAP_SHARED] = kernelInfo.l_wk[0];
            }

#endif
        }
    }

    LOG(ERROR) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberArgmax<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ArgmaxParam<AMD>& param) {
    // to get the compute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    // to get the argument
    int outer_dim     = inputs[0]->count(0, param.axis);
    bool err          = false;
    AMDKernel* kernel = NULL;

    amd_kernel_list list;

    if (param.has_axis) {
        int count         = inputs[0]->count(0, inputs[0]->dims());
        int dim           = inputs[0]->shape()[param.axis];
        int inner_dim     = inputs[0]->count(param.axis + 1, inputs[0]->dims());
        int total_threads = count / dim;

        if (param.top_k == 1) {
            kernel = _kernel_map[TOP_1_CHANNEL].get();
            err    = kernel->SetKernelArgs(
                         (PtrDtype)inputs[0]->data(),
                         (int)outer_dim,
                         (int)dim,
                         (int)inner_dim,
                         (int)param.out_max_val,
                         (PtrDtype)outputs[0]->mutable_data());

            if (!err) {
                LOG(ERROR) << "Failed to set kernel args";
                return SaberInvalidValue;
            }

            list.push_back(_kernel_map[TOP_1_CHANNEL]);
        } else {
            kernel = _kernel_map[TOPK_CHANNEL].get();
            err    = kernel->SetKernelArgs(
                         (PtrDtype)inputs[0]->data(),
                         (int)outer_dim,
                         (int)dim,
                         (int)inner_dim,
                         (int)param.top_k,
                         (int)param.out_max_val,
                         (PtrDtype)outputs[0]->mutable_data());

            if (!err) {
                LOG(ERROR) << "Failed to set kernel args";
                return SaberInvalidValue;
            }

            list.push_back(_kernel_map[TOPK_CHANNEL]);
        }
    } else {
        int inner_dim = inputs[0]->count(1, inputs[0]->dims());
        int outer_dim = inputs[0]->num();

        if (param.top_k == 1) {
            if (inner_dim / _localWorkSize < 10) {
                kernel = _kernel_map[TOP_1].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)inputs[0]->data(),
                             (int)outer_dim,
                             (int)inner_dim,
                             (int)param.out_max_val,
                             (PtrDtype)outputs[0]->mutable_data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_kernel_map[TOP_1]);
            } else {
                int inner_group_num = (inner_dim + _localWorkSize - 1) / _localWorkSize;
                kernel = _kernel_map[BLOCK_TOP_1].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)inputs[0]->data(),
                             (int)outer_dim,
                             (int)inner_dim,
                             (int)inner_group_num,
                             (PtrDtype)_group_max_value.mutable_data(),
                             (PtrDtype)_group_max_index.mutable_data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_kernel_map[BLOCK_TOP_1]);

                kernel = _kernel_map[TOP_1_BIG].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)_group_max_value.data(),
                             (PtrDtype)_group_max_index.data(),
                             (int)outer_dim,
                             (int)inner_group_num,
                             (int)param.out_max_val,
                             (PtrDtype)outputs[0]->mutable_data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_kernel_map[TOP_1_BIG]);
            }
        } else if (param.top_k >= 13 && outer_dim == 1 && inner_dim > 100000) {
            for (int i = 0; i < outer_dim; i++) {
                unsigned int bit = 0;
                unsigned int begin_bit = 0;
                unsigned int end_bit = 8 * type_length(OpDtype);

                unsigned int iterations = _radix_params.long_iterations + _radix_params.short_iterations;
                bool to_output = (iterations - 1) % 2 == 0;

                Tensor<AMD> sub_input;
                int device_id = inputs[0]->device_id();
                size_t count = inner_dim * type_length(OpDtype);
                size_t offset = i * count;
                sub_input.re_alloc(Shape({1, inner_dim, 1, 1}), OpDtype);

                AMD_API::sync_memcpy(sub_input.mutable_data(), 0, device_id,
                                     inputs[0]->data(), offset, device_id, count, __DtoD());


                for (unsigned int i = 0; i < _radix_params.long_iterations; i++) {
                    err = radix_sort_launch_kernel<7>(_radix_sort_kernel_map,
                                                      (cl_mem)sub_input.data(),
                                                      (cl_mem)_keys_tmp.mutable_data(),
                                                      (cl_mem)_keys_output.mutable_data(),
                                                      (cl_mem)_values_input.data(),
                                                      (cl_mem)_values_tmp.mutable_data(),
                                                      (cl_mem)_values_output.mutable_data(),
                                                      (cl_mem)_batch_digit_counts.mutable_data(),
                                                      (cl_mem)_digit_counts.mutable_data(),
                                                      inner_dim,
                                                      to_output,
                                                      bit,
                                                      begin_bit,
                                                      end_bit,
                                                      _radix_params,
                                                      cm);

                    if (!err) {
                        LOG(ERROR) << "Fail to set execution";
                        return SaberInvalidValue;
                    }

                    to_output = !to_output;
                    bit += 7;
                }

                for (unsigned int i = 0; i < _radix_params.short_iterations; i++) {
                    err = radix_sort_launch_kernel<6>(_radix_sort_kernel_map,
                                                      (PtrDtype)sub_input.data(),
                                                      (PtrDtype)_keys_tmp.mutable_data(),
                                                      (PtrDtype)_keys_output.mutable_data(),
                                                      (PtrDtype)_values_input.data(),
                                                      (PtrDtype)_values_tmp.mutable_data(),
                                                      (PtrDtype)_values_output.mutable_data(),
                                                      (PtrDtype)_batch_digit_counts.mutable_data(),
                                                      (PtrDtype)_digit_counts.mutable_data(),
                                                      inner_dim,
                                                      to_output,
                                                      bit,
                                                      begin_bit,
                                                      end_bit,
                                                      _radix_params,
                                                      cm);

                    if (!err) {
                        LOG(ERROR) << "Fail to set execution";
                        return SaberInvalidValue;
                    }

                    to_output = !to_output;
                    bit += 6;
                }

                kernel = _radix_sort_kernel_map[RADIX_SORT_OUTPUT_COMBINE].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)outputs[0]->mutable_data(),
                             (int)i,
                             (int)inner_dim,
                             (int)param.top_k,
                             (int)param.out_max_val,
                             (PtrDtype)_keys_output.data(),
                             (PtrDtype)_values_output.data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_radix_sort_kernel_map[RADIX_SORT_OUTPUT_COMBINE]);
                err = LaunchKernel(cm, list);

                if (!err) {
                    LOG(ERROR) << "Fail to set execution";
                    return SaberInvalidValue;
                }
            }

            LOG(INFO) << "COMPLETE EXECUTION";
            return SaberSuccess;
        } else {
#if 0
            kernel = _kernel_map[TOPK_HEAP_SHARED].get();
            err    = kernel->SetKernelArgs(
                         (PtrDtype)outputs[0]->mutable_data(),
                         (int)outer_dim,
                         (int)inner_dim,
                         (int)param.top_k,
                         (int)param.out_max_val,
                         (PtrDtype)inputs[0]->data());

            if (!err) {
                LOG(ERROR) << "Failed to set kernel args";
                return SaberInvalidValue;
            }

            list.push_back(_kernel_map[TOPK_HEAP_SHARED]);
#else

            if (_localWorkSize == 512) {
                kernel = _kernel_map[TOPK_HEAP_SHARED_512].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)outputs[0]->mutable_data(),
                             (int)outer_dim,
                             (int)inner_dim,
                             (int)param.top_k,
                             (int)param.out_max_val,
                             (PtrDtype)inputs[0]->data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_kernel_map[TOPK_HEAP_SHARED_512]);
            } else {
                kernel = _kernel_map[TOPK_HEAP_SHARED].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)outputs[0]->mutable_data(),
                             (int)outer_dim,
                             (int)inner_dim,
                             (int)param.top_k,
                             (int)param.out_max_val,
                             (PtrDtype)inputs[0]->data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_kernel_map[TOPK_HEAP_SHARED]);
            }

#endif
        }
    }

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    LOG(INFO) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class SaberArgmax<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, AMD, AK_HALF);
} // namespace saber
} // namespace anakin
