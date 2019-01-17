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
#include "include/saber_normalize.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <typename TargetType>
void print_tensor(
    Tensor<TargetType>& tensor,
    typename Tensor<TargetType>::API::stream_t stream = nullptr) {

    typedef typename DataTrait<AMD, AK_FLOAT>::Dtype Dtype;
    typedef typename DataTrait<AMD, AK_FLOAT>::PtrDtype PtrDtype;

    PtrDtype ptr = (PtrDtype)tensor.mutable_data();
    cl_mem mem   = (cl_mem)ptr;

    LOG(INFO) << "device tensor size: " << tensor.size() << " type size: " << sizeof(Dtype);
    int size = tensor.size();

    Device<AMD> dev = Env<AMD>::cur_env()[tensor.device_id()];

    if (stream == nullptr) {
        LOG(INFO) << "stream is empty, use default stream";
        stream = dev._data_stream[0];
    }

    cl_int err;
    Dtype* data_ptr = (Dtype*)clEnqueueMapBuffer(
                          stream, mem, CL_TRUE, CL_MAP_READ, 0, size * sizeof(Dtype), 0, NULL, NULL, &err);

    if (err != CL_SUCCESS) {
        LOG(ERROR) << "Can't map buffer to host, err=" << err;
        return;
    }

    for (int i = 0; i < size; ++i) {
        printf("%.5f ", static_cast<float>(data_ptr[i]));

        if ((i + 1) % tensor.width() == 0) {
            printf("\n");
        }
    }

    printf("\n");

    clEnqueueUnmapMemObject(stream, mem, data_ptr, 0, NULL, NULL);
    // clFinish(stream);
};

template <DataType OpDtype>
SaberStatus SaberNormalize<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    NormalizeParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberNormalize<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    NormalizeParam<AMD>& param,
    Context<AMD>& ctx) {

    // compute norm size
    int channel_index = inputs[0]->channel_index();
    _dims             = inputs[0]->dims();
    _size             = inputs[0]->valid_size();
    _channels         = inputs[0]->channel();
    _batchs           = inputs[0]->num();

    //! check the scale size
    if (param.has_scale) {
        if (!param.channel_shared) {
            CHECK_EQ(_channels, param.scale->valid_size()) << "scale data size must = channels";
        }
    }

    //! size of data to compute square root sum (eg. H * W for channel, C * H * W for batch)
    if (param.across_spatial) {
        _norm_size = _batchs;
    } else {
        _norm_size = _channels * _batchs;
    }

    _channel_stride = inputs[0]->count_valid(channel_index + 1, _dims);
    _compute_size   = _size / _norm_size;
    Shape sh_norm({1, 1, 1, _norm_size}, Layout_NCHW);
    _norm_reduce.reshape(sh_norm);

    _is_continue_buf = outputs[0]->is_continue_mem() && inputs[0]->is_continue_mem();

    if (!_is_continue_buf) {
        Shape sh_input_real_stride  = inputs[0]->get_stride();
        Shape sh_output_real_stride = outputs[0]->get_stride();

        //! re_alloc device memory
        Shape sh({1, 1, 1, _dims}, Layout_NCHW);
        _valid_shape.reshape(sh);
        _input_stride.reshape(sh);
        _output_stride.reshape(sh);

        AMD_API::sync_memcpy(
            _valid_shape.mutable_data(),
            0,
            inputs[0]->device_id(),
            inputs[0]->valid_shape().data(),
            0,
            0,
            sizeof(int) * _dims,
            __HtoD());

        AMD_API::sync_memcpy(
            _input_stride.mutable_data(),
            0,
            inputs[0]->device_id(),
            sh_input_real_stride.data(),
            0,
            0,
            sizeof(int) * _dims,
            __HtoD());

        AMD_API::sync_memcpy(
            _output_stride.mutable_data(),
            0,
            inputs[0]->device_id(),
            sh_output_real_stride.data(),
            0,
            0,
            sizeof(int) * _dims,
            __HtoD());
    }

    int globalSize = 0;
    int localSize  = 256;
    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {localSize};
    kernelInfo.kernel_file = "Normalize.cl";

    std::string strLocalSize = std::to_string(kernelInfo.l_wk[0]);
    kernelInfo.comp_options  = std::string("-DSHARE_MEMORY_DIM=") + strLocalSize;

    AMDKernelPtr kptr = NULL;

    if (!param.across_spatial) {
        globalSize      = inputs[0]->width() * inputs[0]->height() * inputs[0]->num();
        kernelInfo.g_wk = {(globalSize + localSize - 1) / localSize * localSize};

        kernelInfo.kernel_name = "NormalizeNoAcrossSpatial";
        kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        _kernel_map[NORM_NO_ACROSS_SPATIAL] = kptr;
    } else {
        globalSize             = _size;
        kernelInfo.g_wk        = {(globalSize + localSize - 1) / localSize * localSize};
        kernelInfo.kernel_name = "ReduceAddAtomic";
        kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        _kernel_map[REDUCE_ADD_ATOMIC] = kptr;

        KernelInfo kernelInfoGpuPowReverse;
        kernelInfoGpuPowReverse.kernel_file = "Normalize.cl";
        kernelInfoGpuPowReverse.kernel_name = "GpuPowReverse";
        kernelInfoGpuPowReverse.wk_dim      = 1;
        kernelInfoGpuPowReverse.l_wk        = {1};
        kernelInfoGpuPowReverse.g_wk        = {_norm_size};
        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfoGpuPowReverse);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        _kernel_map[GPU_POW_REVERSE] = kptr;

        if (param.has_scale) {
            kernelInfo.kernel_name = "NormalizeWithScale";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_map[NORM_WITH_SCALE] = kptr;
        } else {
            kernelInfo.kernel_name = "Normalize";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_map[NORM] = kptr;
        }
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberNormalize<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    NormalizeParam<AMD>& param) {

    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    amd_kernel_list list;
    bool err          = false;
    AMDKernelPtr kptr = NULL;
    AMDKernel* kernel = NULL;

    if (!param.across_spatial) {
        int size_in_channel = inputs[0]->width() * inputs[0]->height();
        int channel         = inputs[0]->channel();
        int num             = inputs[0]->num();
        kptr                = _kernel_map[NORM_NO_ACROSS_SPATIAL];

        if (kptr == NULL || kptr.get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        kernel = kptr.get();

        if (param.has_scale) {
            err = kernel->SetKernelArgs(
                      (int)size_in_channel,
                      (int)num,
                      (int)channel,
                      (PtrDtype)param.scale->data(),
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (float)param.eps,
                      (int)param.p,
                      (int)param.has_scale,
                      (int)param.channel_shared);

            if (!err) {
                LOG(ERROR) << "Failed to set kernel args";
                return SaberInvalidValue;
            }
        } else {
            err = kernel->SetKernelArgs(
                      (int)size_in_channel,
                      (int)num,
                      (int)channel,
                      NULL,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (float)param.eps,
                      (int)param.p,
                      (int)param.has_scale,
                      (int)param.channel_shared);

            if (!err) {
                LOG(ERROR) << "Failed to set kernel args";
                return SaberInvalidValue;
            }
        }

        list.push_back(kptr);
    } else {
        kptr = _kernel_map[REDUCE_ADD_ATOMIC];

        if (kptr == NULL || kptr.get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        kernel = kptr.get();
        err    = kernel->SetKernelArgs(
                     (int)_size,
                     (int)param.p,
                     (int)_compute_size,
                     (PtrDtype)inputs[0]->data(),
                     (PtrDtype)_norm_reduce.mutable_data());

        if (!err) {
            LOG(ERROR) << "Failed to set kernel Args";
            return SaberInvalidValue;
        }

        list.push_back(kptr);

        float pw = 0.5f;

        if (param.p == 1) {
            pw = 1.f;
        }

        kptr = _kernel_map[GPU_POW_REVERSE];

        if (kptr == NULL || kptr.get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        kernel = kptr.get();
        err    = kernel->SetKernelArgs(
                     (int)_norm_size,
                     (PtrDtype)_norm_reduce.data(),
                     (PtrDtype)_norm_reduce.mutable_data(),
                     (float)pw,
                     (float)param.eps);

        if (!err) {
            LOG(ERROR) << "Failed to set kernel Args";
            return SaberInvalidValue;
        }

        list.push_back(kptr);

        if (param.has_scale) {
            kptr = _kernel_map[NORM_WITH_SCALE];

            if (kptr == NULL || kptr.get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            kernel = kptr.get();
            err    = kernel->SetKernelArgs(
                         (int)_size,
                         (int)_compute_size,
                         (int)_channel_stride,
                         (int)_channels,
                         (PtrDtype)_norm_reduce.data(),
                         (PtrDtype)param.scale->data(),
                         (PtrDtype)inputs[0]->data(),
                         (PtrDtype)outputs[0]->mutable_data(),
                         (int)param.channel_shared);

            if (!err) {
                LOG(ERROR) << "Failed to set kernel Args";
                return SaberInvalidValue;
            }

            list.push_back(kptr);
        } else {
            kptr = _kernel_map[NORM];

            if (kptr == NULL || kptr.get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            kernel = kptr.get();
            err    = kernel->SetKernelArgs(
                         (int)_size,
                         (int)_compute_size,
                         (PtrDtype)_norm_reduce.data(),
                         (PtrDtype)inputs[0]->data(),
                         (PtrDtype)outputs[0]->mutable_data());

            if (!err) {
                LOG(ERROR) << "Failed to set kernel Args";
                return SaberInvalidValue;
            }

            list.push_back(kptr);
        }
    }

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fialed to set execution.";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class SaberNormalize<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberNormalize, NormalizeParam, AMD, AK_HALF);
} // namespace saber
} // namespace anakin
