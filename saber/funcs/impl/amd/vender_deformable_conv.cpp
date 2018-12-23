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
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;

    _kernel_dim = param.weight()->channel() * param.weight()->height() * param.weight()->width();

    _bottom_dim = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();

    _offset_dim = inputs[1]->channel() * inputs[1]->height() * inputs[1]->width();

    // Shape deform_col_buffer_shape = {1, _kernel_dim, outputs[0]->height(), outputs[0]->width()};
    _deform_col_buffer.re_alloc(Shape({1, _kernel_dim, outputs[0]->height(), outputs[0]->width()}));
    // outputs[0]->reshape(Shape({1, 1, 1, 7}));
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus VenderDeformableConv2D<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {

    const int count = outputs[0]->valid_size();
    KernelInfo kernelInfo;
    AMDKernelPtr kptr;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int in_channel        = inputs[0]->channel();
    int conv_out_channel  = outputs[0]->channel();
    _conv_out_spatial_dim = outputs[0]->height() * outputs[0]->width();

    _kernel_dim = param.weight()->channel() * param.weight()->height() * param.weight()->width();

    //_bottom_dim = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();

    _offset_dim = inputs[1]->channel() * inputs[1]->height() * inputs[1]->width();

    _col_offset    = _kernel_dim * _conv_out_spatial_dim;
    _output_offset = conv_out_channel * _conv_out_spatial_dim;
    _kernel_offset = _kernel_dim * conv_out_channel;

    if ((outputs[0]->height() != _deform_col_buffer.height())
            || (outputs[0]->width() != _deform_col_buffer.width())) {

        _deform_col_buffer.reshape(
            Shape({1, _kernel_dim, outputs[0]->height(), outputs[0]->width()}));
    }

    for (int n = 0; n < inputs[0]->num(); ++n) {

        // transform image to col_buffer in order to use gemm

        int channel_per_group = in_channel / param.group;
        int num_kernels = in_channel * _deform_col_buffer.height() * _deform_col_buffer.width();

        kernelInfo.kernel_file = "Deformableconv.cl";
        kernelInfo.kernel_name = "deformable_im2col_gpu_kernel";
        kernelInfo.wk_dim      = 1;
        kernelInfo.l_wk        = {256};
        kernelInfo.g_wk        = {(num_kernels + 256 - 1) / 256 * 256};

        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);

        for (int g = 0; g < param.group; ++g) {

            _outGemmWorkspace = new Tensor<AMD>;

            _outGemmWorkspace->re_alloc(
                Shape({(inputs[0]->num()),
                       std::max((inputs[0]->channel()), (param.weight()->num())),
                       (inputs[0]->height()),
                       (inputs[0]->width())
                      }));

            int K       = _kernel_dim / param.group;
            int M       = _conv_out_spatial_dim;
            int N       = conv_out_channel / param.group;
            float alpha = 1.0;
            float beta  = 0.0;
            bool tA     = false;
            bool tB     = false;
            bool tC     = false;
            int lda     = K; //_conv_out_spatial_dim;
            int ldb     = N; //_kernel_dim / param.group;
            int ldc     = N; //_conv_out_spatial_dim;

            MIOpenGEMM::Geometry tgg {};
            tgg = MIOpenGEMM::Geometry(true, tB, tA, false, ldb, lda, ldc, N, M, K, 0, 'f');

            /////////////////////////////////////////////////////////////
            // gemm kernel
            // jn : print search results to terminal
            bool miopengemm_verbose = false;

            // jn : print warning messages when the returned kernel(s) might be sub-optimal
            bool miopengemm_warnings = false;

            // jn : find with no workspace
            MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                            0.003f,
                                            cm,
                                            //(cl_mem)(deform_col_buffer_data_const + _col_offset * g),
                                            //(cl_mem)(param.weight()->data() + _kernel_offset * g),
                                            (cl_mem)param.weight()->data(),
                                            //(cl_mem)(_deform_col_buffer.data() + _col_offset * g),
                                            (cl_mem)param.weight()->data(),
                                            //(cl_mem)(outputs[0]->mutable_data() + _output_offset * g),
                                            (cl_mem)param.weight()->data(),
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

        if (param.bias()->size() > 0) {
            int out_count = outputs[0]->valid_size();

            kernelInfo.kernel_file = "Deformableconv.cl";
            kernelInfo.kernel_name = "gpu_add_bias";
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
    }

    return SaberSuccess;
}
template <DataType OpDtype>

SaberStatus VenderDeformableConv2D<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param) {
    bool err;
    amd_kernel_list list;

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int in_channel       = inputs[0]->channel();
    int conv_out_channel = outputs[0]->channel();

    int j = 0; // kernel index

    for (int n = 0; n < inputs[0]->num(); ++n) {

        // transform image to col_buffer in order to use gemm

        int channel_per_group = in_channel / param.group;
        int num_kernels = in_channel * _deform_col_buffer.height() * _deform_col_buffer.width();

        if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        err = _kernels_ptr[j].get()->SetKernelArgs(
                  (int)num_kernels,
                  //(PtrDtype)(inputs[0]->data() + n * _bottom_dim),
                  (PtrDtype)inputs[0]->data(),
                  //(PtrDtype)(inputs[1]->data() + n * _offset_dim),
                  (PtrDtype)inputs[1]->data(),
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

        for (int g = 0; g < param.group; ++g) {
            cl_float floatObjects[2] = {1.0f, 0.0f};
            cl_uint offsetObjects[3] = {0, 0, 0};

            if (_multikernel) {
                if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[j].get()->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)offsetObjects[2],
                          (float)floatObjects[1]);
                j++;
            }

            if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            err = _kernels_ptr[j].get()->SetKernelArgs(
                      //(PtrDtype)(param.weight()->data() + _kernel_offset * g),
                      (PtrDtype)param.weight()->data(),
                      (int)offsetObjects[0],
                      //(PtrDtype)(_deform_col_buffer.data() + _col_offset * g),
                      (PtrDtype)_deform_col_buffer.data(),
                      (int)offsetObjects[1],
                      //(PtrDtype)(outputs[0]->mutable_data() + _output_offset * g),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (int)offsetObjects[2],
                      (float)floatObjects[0],
                      (float)floatObjects[1]);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[j]);
            j++;
        }

        if (param.bias()->size() > 0) {
            Shape out_shape  = outputs[0]->valid_shape();
            Shape out_stride = outputs[0]->get_stride();
            int out_count    = outputs[0]->size();

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
    }

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fialed to set execution.";
        return SaberInvalidValue;
    }

    list.clear();

    return SaberSuccess;
}

template class SaberDeformableConv2D<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderDeformableConv2D, ConvParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(VenderDeformableConv2D, ConvParam, AMD, AK_HALF);
} // namespace saber

} // namespace anakin

