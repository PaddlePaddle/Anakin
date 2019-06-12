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
#include "include/saber_eltwise_act.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberEltwiseActive<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    EltwiseActiveParam<AMD>& param,
    Context<AMD>& ctx) {

    EltwiseParam<AMD> eltwise_param = param.eltwise_param;

    this->_ctx = &ctx;
    CHECK_GE(outputs.size(), 1) << "outputs size has to == 1";
    CHECK_GE(inputs.size(), 2) << "input size has to >= 2";
    CHECK(!(inputs.size() > 2 && eltwise_param.operation == Eltwise_sum))
            << "not support input size>2 and operation==Eltwise_sum, size = " << inputs.size()
            << ",activation = " << eltwise_param.operation;
    _with_relu        = eltwise_param.has_eltwise && param.activation_param.active == Active_relu;
    _other_activation = eltwise_param.has_eltwise && param.activation_param.active != Active_relu
                        && param.activation_param.active != Active_unknow;

    if (_other_activation) {
        SABER_CHECK(_saber_activation.init(inputs, outputs, param.activation_param, ctx));
    }

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberEltwiseActive<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    EltwiseActiveParam<AMD>& param,
    Context<AMD>& ctx) {

    EltwiseParam<AMD> eltwise_param = param.eltwise_param;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "create";

    if (_other_activation) {
        _out_eltwise = new Tensor<AMD>();
        _out_eltwise->re_alloc(outputs[0]->shape());
        SABER_CHECK(_saber_activation.create(inputs, outputs, param.activation_param, ctx));
    }

    const int count = outputs[0]->size();

    int global_size = count;
    int local_size  = 256;
    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {local_size};
    kernelInfo.g_wk        = {(global_size + local_size - 1) / local_size * local_size};
    kernelInfo.kernel_file = "Eltwise.cl";

    AMDKernelPtr kptr      = NULL;
    _kernels_ptr.clear();

    switch (eltwise_param.operation) {
    case Eltwise_prod:
        if (_with_relu) {
            if (inputs.size() <= 2) {
                kernelInfo.kernel_name = "ker_elt_production";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);

            } else {
                kernelInfo.kernel_name = "ker_elt_production";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);

                for (int i = 2; i < inputs.size() - 1; i++) {
                    kernelInfo.kernel_name = "ker_elt_production";
                    kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                    if (!kptr.get()->isInit()) {
                        LOG(ERROR) << "Failed to load program";
                        return SaberInvalidValue;
                    }

                    _kernels_ptr.push_back(kptr);
                }

                kernelInfo.kernel_name = "ker_elt_production";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);
            }

        } else {

            kernelInfo.kernel_name = "ker_elt_production";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernels_ptr.push_back(kptr);

            for (int i = 2; i < inputs.size(); i++) {

                kernelInfo.kernel_name = "ker_elt_production";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);
            }
        }

        break;

    case Eltwise_sum:
        if (_with_relu) {
            kernelInfo.kernel_name = "ker_elt_sum";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernels_ptr.push_back(kptr);

        } else {
            kernelInfo.kernel_name = "ker_elt_sum";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernels_ptr.push_back(kptr);
        }

        break;

    case Eltwise_max:

        if (_with_relu) {
            if (inputs.size() <= 2) {
                kernelInfo.kernel_name = "ker_elt_max";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);

            } else {
                kernelInfo.kernel_name = "ker_elt_max";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);

                for (int i = 2; i < inputs.size() - 1; i++) {
                    kernelInfo.kernel_name = "ker_elt_max";
                    kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                    if (!kptr.get()->isInit()) {
                        LOG(ERROR) << "Failed to load program";
                        return SaberInvalidValue;
                    }

                    _kernels_ptr.push_back(kptr);
                }

                kernelInfo.kernel_name = "ker_elt_max";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);
            }
        } else {

            kernelInfo.kernel_name = "ker_elt_max";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernels_ptr.push_back(kptr);

            for (int i = 2; i < inputs.size(); i++) {
                kernelInfo.kernel_name = "ker_elt_max";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernels_ptr.push_back(kptr);
            }
        }

        break;

    default:
        LOG(FATAL) << "unknown elementwise operation. ";
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberEltwiseActive<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    EltwiseActiveParam<AMD>& param) {

    EltwiseParam<AMD> eltwise_param = param.eltwise_param;
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    amd_kernel_list list;
    bool err      = false;
    int with_relu = 0;

    const int count  = outputs[0]->valid_size();
    int kernel_index = 0;

    PtrDtype outputptr = (PtrDtype)outputs[0]->mutable_data();
    if (_other_activation)
    {
        outputptr = (PtrDtype)_out_eltwise->mutable_data();
    }

    switch (eltwise_param.operation) {
    case Eltwise_prod:
        if (_with_relu) {
            if (inputs.size() <= 2) {
                with_relu = 1;

                if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[0].get()->SetKernelArgs(
                          outputptr,
                          (int)with_relu,
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)inputs[1]->data(),
                          (int)count);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[0]);
            } else {
                with_relu = 1;

                if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[0].get()->SetKernelArgs(
                          outputptr,
                          (int)with_relu,
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)inputs[1]->data(),
                          (int)count);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[0]);

                for (int i = 2; i < inputs.size() - 1; i++) {
                    if (_kernels_ptr[i - 1] == NULL || _kernels_ptr[i - 1].get() == NULL) {
                        LOG(ERROR) << "Kernel is not exist";
                        return SaberInvalidValue;
                    }

                    err = _kernels_ptr[i - 1].get()->SetKernelArgs(
                              outputptr,
                              (int)with_relu,
                              outputptr,
                              (PtrDtype)inputs[i]->data(),
                              (int)count);

                    if (!err) {
                        LOG(ERROR) << "Fail to set kernel args :" << err;
                        return SaberInvalidValue;
                    }

                    list.push_back(_kernels_ptr[i - 1]);
                }

                with_relu = 1;

                if (_kernels_ptr[inputs.size() - 1 - 1] == NULL
                        || _kernels_ptr[inputs.size() - 1 - 1].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[inputs.size() - 1 - 1].get()->SetKernelArgs(
                          outputptr,
                          (int)with_relu,
                          outputptr,
                          (PtrDtype)inputs[inputs.size() - 1]->data(),
                          (int)count);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[inputs.size() - 1 - 1]);
            }

        } else {

            with_relu = 0;

            if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            err = _kernels_ptr[0].get()->SetKernelArgs(
                      outputptr,
                      (int)with_relu,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)inputs[1]->data(),
                      (int)count);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[0]);

            for (int i = 2; i < inputs.size(); i++) {
                if (_kernels_ptr[i - 1] == NULL || _kernels_ptr[i - 1].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[i - 1].get()->SetKernelArgs(
                          outputptr,
                          (int)with_relu,
                          outputptr,
                          (PtrDtype)inputs[i]->data(),
                          (int)count);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i - 1]);
            }
        }

        break;

    case Eltwise_sum:
        if (_with_relu) {
            with_relu = 1;

            if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            err = _kernels_ptr[0].get()->SetKernelArgs(
                      (PtrDtype)outputptr,
                      (int)with_relu,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)inputs[1]->data(),
                      (float)eltwise_param.coeff[0],
                      (float)eltwise_param.coeff[1],
                      (int)count);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[0]);

        } else {
            with_relu = 0;

            if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            err = _kernels_ptr[0].get()->SetKernelArgs(
                      (PtrDtype)outputptr,
                      (int)with_relu,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)inputs[1]->data(),
                      (float)eltwise_param.coeff[0],
                      (float)eltwise_param.coeff[1],
                      (int)count);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[0]);
        }

        break;

    case Eltwise_max:

        if (_with_relu) {
            if (inputs.size() <= 2) {
                with_relu = 1;

                if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[0].get()->SetKernelArgs(
                          outputptr,
                          (int)with_relu,
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)inputs[1]->data(),
                          (int)count);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[0]);
            } else {
                with_relu = 1;

                if (_kernels_ptr[0].get() == NULL || _kernels_ptr[0].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[0].get()->SetKernelArgs(
                          outputptr,
                          (int)with_relu,
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)inputs[1]->data(),
                          (int)count);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[0]);

                for (int i = 2; i < inputs.size() - 1; i++) {
                    if (_kernels_ptr[i - 1] == NULL || _kernels_ptr[i - 1].get() == NULL) {
                        LOG(ERROR) << "Kernel is not exist";
                        return SaberInvalidValue;
                    }

                    err = _kernels_ptr[i - 1].get()->SetKernelArgs(
                              outputptr,
                              (int)with_relu,
                              outputptr,
                              (PtrDtype)inputs[i]->data(),
                              (int)count);

                    if (!err) {
                        LOG(ERROR) << "Fail to set kernel args :" << err;
                        return SaberInvalidValue;
                    }

                    list.push_back(_kernels_ptr[i - 1]);
                }

                with_relu = 1;

                if (_kernels_ptr[inputs.size() - 1 - 1] == NULL
                        || _kernels_ptr[inputs.size() - 1 - 1].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[inputs.size() - 1 - 1].get()->SetKernelArgs(
                          outputptr,
                          (int)with_relu,
                          outputptr,
                          (PtrDtype)inputs[inputs.size() - 1]->data(),
                          (int)count);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[inputs.size() - 1 - 1]);
            }
        } else {

            with_relu = 0;

            if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            err = _kernels_ptr[0].get()->SetKernelArgs(
                      outputptr,
                      (int)with_relu,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)inputs[1]->data(),
                      (int)count);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[0]);

            for (int i = 2; i < inputs.size(); i++) {
                if (_kernels_ptr[i - 1] == NULL || _kernels_ptr[i - 1].get() == NULL) {
                    LOG(ERROR) << "Kernel is not exist";
                    return SaberInvalidValue;
                }

                err = _kernels_ptr[i - 1].get()->SetKernelArgs(
                          outputptr,
                          (int)with_relu,
                          outputptr,
                          (PtrDtype)inputs[i]->data(),
                          (int)count);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i - 1]);
            }
        }

        break;

    default:
        LOG(FATAL) << "unknown elementwise operation. ";
    }

    if (!err) {
        LOG(ERROR) << "Failed to set kernel args";
        return SaberInvalidValue;
    }

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Failed to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";

    if (_other_activation) {
        std::vector<Tensor<AMD>*> eltwise_outputs;
        eltwise_outputs.push_back(_out_eltwise);
        SABER_CHECK(_saber_activation.dispatch(eltwise_outputs, outputs, param.activation_param));
    }

    return SaberSuccess;
}
template class SaberEltwiseActive<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberEltwiseActive, EltwiseActiveParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberEltwiseActive, EltwiseActiveParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
