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
#include "include/vender_deformable_conv.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"
namespace anakin {

namespace saber {

typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;
typedef Tensor<AMD> TensorDf4;

template <DataType OpDtype>
SaberStatus VenderDeformableConv2D<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DeformableConvParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;

    _kernel_dim = param.weight()->channel() * param.weight()->height() * param.weight()->width();

    _bottom_dim = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();

    _offset_dim = inputs[1]->channel() * inputs[1]->height() * inputs[1]->width();

    // Shape deform_col_buffer_shape = {1, _kernel_dim, outputs[0]->height(), outputs[0]->width()};
    _deform_col_buffer.re_alloc(Shape({1, _kernel_dim, outputs[0]->height(), outputs[0]->width()}));
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus VenderDeformableConv2D<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DeformableConvParam<AMD>& param,
    Context<AMD>& ctx) {

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "create";

    const int count = outputs[0]->valid_size();
    KernelInfo kernelInfo;
    AMDKernelPtr kptr;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int in_channel        = inputs[0]->channel();
    int conv_out_channel  = outputs[0]->channel();
    _conv_out_spatial_dim = outputs[0]->height() * outputs[0]->width();

    _kernel_dim = param.weight()->channel() * param.weight()->height() * param.weight()->width();

    _offset_dim = inputs[1]->channel() * inputs[1]->height() * inputs[1]->width();

    _col_offset    = (_kernel_dim / param.group) * _conv_out_spatial_dim;
    _output_offset = (conv_out_channel / param.group) * _conv_out_spatial_dim;
    _kernel_offset = (_kernel_dim / param.group) * (conv_out_channel / param.group);

    if ((outputs[0]->height() != _deform_col_buffer.height())
            || (outputs[0]->width() != _deform_col_buffer.width())) {

        _deform_col_buffer.reshape(
            Shape({1, _kernel_dim, outputs[0]->height(), outputs[0]->width()}));
    }

    for (int n = 0; n < inputs[0]->num(); ++n) {
        for (int g = 0; g < param.group; ++g) {
            // transform image to col_buffer in order to use gemm

            int channel_per_group = in_channel / param.group;
            int num_kernels = in_channel / param.group * _deform_col_buffer.height() *
                              _deform_col_buffer.width();

            kernelInfo.kernel_file = "Deformableconv.cl";
            kernelInfo.kernel_name = "deformable_im2col_gpu_kernel";
            kernelInfo.kernel_type = SABER;
            kernelInfo.wk_dim      = 1;
            kernelInfo.l_wk        = {256};
            kernelInfo.g_wk        = {(num_kernels + 256 - 1) / 256 * 256};

            kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to create kernel";
                return SaberInvalidValue;
            }

            _kernels_ptr.push_back(kptr);

            //for (int g = 0; g < param.group; ++g) {

            //            _outGemmWorkspace = new Tensor<AMD>;

            //            _outGemmWorkspace->re_alloc(
            //                Shape({(inputs[0]->num()),
            //                       std::max((inputs[0]->channel()), (param.weight()->num())),
            //                       (inputs[0]->height()),
            //                       (inputs[0]->width())
            //                      }));

            int K       = _kernel_dim / param.group;
            int M       = _conv_out_spatial_dim;
            int N       = conv_out_channel / param.group;
            float alpha = 1.0;
            float beta  = 0.0;
            bool transA     = false;
            bool transB     = false;
            bool transC     = false;
            int leadingd_A     = _conv_out_spatial_dim;
            int leadingd_B     = _kernel_dim / param.group;
            int leadingd_C     = _conv_out_spatial_dim;

            MIOpenGEMM::Geometry tgg {};
            tgg = MIOpenGEMM::Geometry(true, transA, transB, transC, leadingd_A, leadingd_B, leadingd_C, M, N,
                                       K, 0, 'f');

            /////////////////////////////////////////////////////////////
            // gemm kernel
            // jn : print search results to terminal
            bool miopengemm_verbose = false;

            // jn : print warning messages when the returned kernel(s) might be sub-optimal
            bool miopengemm_warnings = false;

            Tensor<AMD> _outGemmWorkspace;
            _outGemmWorkspace.reshape(outputs[0]->shape());

            // jn : find with no workspace
            MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                            0.003f,
                                            cm,
                                            (cl_mem)(_deform_col_buffer.data()),
                                            (cl_mem)param.weight()->data(),
                                            (cl_mem)(_outGemmWorkspace.mutable_data()),
                                            false,
                                            tgg,
                                            miopengemm_verbose,
                                            miopengemm_warnings);

            std::string kernel_clstring;
            size_t local_work_size;
            size_t global_work_size;

            int i = 0;

            if (soln.v_tgks.size() == 2) {
                _multikernel = true;
                // jn : the main kernel is at the back of the solution vector
                kernel_clstring = soln.v_tgks[i].kernstr;
                tempfix::set_offsets_to_uint(kernel_clstring, 1);

                kernelInfo.kernel_type = SOURCE;
                kernelInfo.kernel_name = soln.v_tgks[i].fname;
                local_work_size        = soln.v_tgks[i].local_work_size;
                global_work_size       = soln.v_tgks[i].global_work_size;

                kernelInfo.kernel_file = kernel_clstring;
                kernelInfo.wk_dim      = 1;
                kernelInfo.l_wk        = {local_work_size, 1, 1};
                kernelInfo.g_wk        = {global_work_size, 1, 1};

                kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to create kernel";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);
                i++;
            }

            // jn : the main kernel is at the back of the solution vector
            kernel_clstring = soln.v_tgks[i].kernstr;
            tempfix::set_offsets_to_uint(kernel_clstring, 3);

            kernelInfo.kernel_name = soln.v_tgks[i].fname;
            local_work_size        = soln.v_tgks[i].local_work_size;
            global_work_size       = soln.v_tgks[i].global_work_size;

            kernelInfo.kernel_file = kernel_clstring;
            kernelInfo.wk_dim      = 1;
            kernelInfo.l_wk        = {local_work_size};
            kernelInfo.g_wk        = {global_work_size};
            kernelInfo.kernel_type = SOURCE;

            kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to create kernel";
                return SaberInvalidValue;
            }

            _kernels_ptr.push_back(kptr);
        }
    }

    if (param.bias()->size() > 0) {
        int out_count = outputs[0]->valid_size();

        kernelInfo.kernel_file = "Deformableconv.cl";
        kernelInfo.kernel_name = "gpu_add_bias";
        kernelInfo.kernel_type = SABER;
        kernelInfo.wk_dim      = 1;
        kernelInfo.l_wk        = {256};
        kernelInfo.g_wk        = {(out_count + 256 - 1) / 256 * 256};

        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);
    }


    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}
template <DataType OpDtype>

SaberStatus VenderDeformableConv2D<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DeformableConvParam<AMD>& param) {
    bool err;
    amd_kernel_list list;

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int in_channel       = inputs[0]->channel();
    int conv_out_channel = outputs[0]->channel();

    int j = 0; // kernel index

    for (int n = 0; n < inputs[0]->num(); ++n) {

        // transform image to col_buffer in order to use gemm
        for (int g = 0; g < param.group; ++g) {
            int channel_per_group = in_channel / param.group;
            int num_kernels = in_channel / param.group * _deform_col_buffer.height() *
                              _deform_col_buffer.width();

            if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            err = _kernels_ptr[j].get()->SetKernelArgs(
                      (int)num_kernels,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)inputs[1]->data(),
                      (int)(n * _bottom_dim + g * _bottom_dim / param.group),
                      (int)(n * _offset_dim + g * _offset_dim / param.group),
                      (int)inputs[0]->height(),
                      (int)inputs[0]->width(),
                      (int)param.weight()->height(),
                      (int)param.weight()->width(),
                      (int)param.pad_h,
                      (int)param.pad_w,
                      (int)param.stride_h,
                      (int)param.stride_w,
                      (int)param.dilation_h,
                      (int)param.dilation_w,
                      (int)channel_per_group,
                      (int)_deform_col_buffer.height(),
                      (int)_deform_col_buffer.width(),
                      (PtrDtype)_deform_col_buffer.mutable_data());

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[j]);
            j++;

            //for (int g = 0; g < param.group; ++g) {
            cl_float floatObjects[2] = {1.0f, 0.0f};
            cl_uint offsetObjects[3] = {0, 0, 0};
            offsetObjects[0] = (param.weight()->num() / param.group) * (param.weight()->channel() / param.group)
                               * param.weight()->height() * param.weight()->width() * g;
            //offsetObjects[1] = _col_offset * g;
            offsetObjects[1] = 0;
            offsetObjects[2] = _output_offset * g + n * param.weight()->num() * outputs[0]->height()
                               * outputs[0]->width();

            if (_multikernel) {
                if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[j].get()->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)offsetObjects[2],
                          (float)floatObjects[1]);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[j]);
                j++;

                err = _kernels_ptr[j].get()->SetKernelArgs(
                          (PtrDtype)_deform_col_buffer.data(),
                          (cl_uint)offsetObjects[1],
                          (PtrDtype)param.weight()->data(),
                          (cl_uint)offsetObjects[0],
                          (PtrDtype)outputs[0]->mutable_data(),
                          (cl_uint)offsetObjects[2],
                          (float)floatObjects[0]);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[j]);
                j++;
            } else {
                if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[j].get()->SetKernelArgs(
                          (PtrDtype)_deform_col_buffer.data(),
                          (cl_uint)offsetObjects[1],
                          (PtrDtype)param.weight()->data(),
                          (cl_uint)offsetObjects[0],
                          (PtrDtype)outputs[0]->mutable_data(),
                          (cl_uint)offsetObjects[2],
                          (float)floatObjects[0],
                          (float)floatObjects[1]);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[j]);
                j++;
            }
        }
    }

    if (param.bias()->size() > 0) {
        Shape out_shape  = outputs[0]->valid_shape();
        Shape out_stride = outputs[0]->get_stride();
        int out_count    = outputs[0]->valid_size();

        if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        err = _kernels_ptr[j].get()->SetKernelArgs(
                  (PtrDtype)outputs[0]->mutable_data(),
                  (int)out_count,
                  (int)out_shape[0],
                  (int)out_shape[1],
                  (int)out_shape[2],
                  (int)out_shape[3],
                  (int)out_stride[0],
                  (int)out_stride[1],
                  (int)out_stride[2],
                  (int)out_stride[3],
                  (PtrDtype)param.bias()->data());

        if (!err) {
            LOG(ERROR) << "Fail to set kernel args :" << err;
            return SaberInvalidValue;
        }

        list.push_back(_kernels_ptr[j]);
        j++;
    }


    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE SET ARGUMENT";
    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fialed to set execution.";
        return SaberInvalidValue;
    }

    list.clear();

    return SaberSuccess;
}

template class SaberDeformableConv2D<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderDeformableConv2D, DeformableConvParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(VenderDeformableConv2D, DeformableConvParam, AMD, AK_HALF);
} // namespace saber

} // namespace anakin

