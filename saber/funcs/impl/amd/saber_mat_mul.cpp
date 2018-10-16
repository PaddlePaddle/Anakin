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
#include "include/saber_mat_mul.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberMatMul<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    MatMulParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberMatMul<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    MatMulParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx   = &ctx;
    this->_param = &param;
    KernelInfo kernelInfo;
    AMDKernelPtr kptr;
    _kernels_ptr.clear();
    std::string kernel_clstring;
    size_t local_work_size;
    size_t global_work_size;
    int errCode;

    int M       = param._m;
    int N       = param._n;
    int K       = param._k;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = false;
    bool tB     = false;
    bool tC     = false;
    int lda     = 0;
    int ldb     = 0;
    int ldc     = 0;

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    PtrDtype X           = (PtrDtype)inputs[0]->data();
    PtrDtype Y           = (PtrDtype)inputs[1]->data();

    MIOpenGEMM::Geometry tgg {};
    // jn : print search results to terminal
    bool miopengemm_verbose = false;
    // jn : print warning messages when the returned kernel(s) might be sub-optimal
    bool miopengemm_warnings = false;

    _outGemmWorkspace = new Tensor<AMD>();
    _outGemmWorkspace->re_alloc(outputs[0]->shape());

    if (!param._is_transpose_X && !param._is_transpose_Y) {
        tA  = false;
        tB  = false;
        tC  = false;
        lda = K;
        ldb = N;
        ldc = N;
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
    } else if (!param._is_transpose_X && param._is_transpose_Y) {
        tA  = false;
        tB  = true;
        tC  = false;
        lda = K;
        ldb = K;
        ldc = N;
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
    } else if (param._is_transpose_X && !param._is_transpose_Y) {
        tA  = true;
        tB  = false;
        tC  = false;
        lda = M;
        ldb = N;
        ldc = N;
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
    } else {
        tA  = true;
        tB  = true;
        tC  = false;
        lda = M;
        ldb = K;
        ldc = N;
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
    }

    // gemm kernel
    // jn : find with no workspace
    MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                    0.003f,
                                    cm,
                                    (PtrDtype)(X),
                                    (PtrDtype)(Y),
                                    (PtrDtype)(_outGemmWorkspace->mutable_data()),
                                    false,
                                    tgg,
                                    miopengemm_verbose,
                                    miopengemm_warnings);

    int i = 0;

    if (soln.v_tgks.size() == 2) {
        _multikernel = true;

        // jn : the main kernel is at the back of the solution vector
        kernel_clstring = soln.v_tgks[i].kernstr;
        tempfix::set_offsets_to_uint(kernel_clstring, 1);

        kernelInfo.kernel_name = soln.v_tgks[i].fname;
        local_work_size        = soln.v_tgks[i].local_work_size;
        global_work_size       = soln.v_tgks[i].global_work_size;

        kernelInfo.kernel_file = kernel_clstring;
        kernelInfo.l_wk        = {local_work_size * param._b, 1, 1};
        kernelInfo.g_wk        = {global_work_size * param._b, 1, 1};
        kernelInfo.kernel_type = SOURCE;

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
    kernelInfo.l_wk        = {local_work_size, 1, 1};
    kernelInfo.g_wk        = {global_work_size, 1, 1};
    kernelInfo.kernel_type = SOURCE;

    kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to create kernel";
        return SaberInvalidValue;
    }

    _kernels_ptr.push_back(kptr);

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberMatMul<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    MatMulParam<AMD>& param) {

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    bool err = false;
    amd_kernel_list list;
    // To set the argument
    PtrDtype memObjects[3]   = {0, 0, 0};
    cl_float floatObjects[2] = {1.0f, 0.0f};
    cl_uint offsetObjects[3] = {0, 0, 0};
    cl_event event;
    int M = param._m;
    int N = param._n;
    int K = param._k;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch"
                                         << "in0 num=" << inputs[0]->num() << " channel=" << inputs[0]->channel()
                                         << " height=" << inputs[0]->height() << " width=" << inputs[0]->width()
                                         << "in1 num=" << inputs[1]->num() << " channel=" << inputs[1]->channel()
                                         << " height=" << inputs[1]->height() << " width=" << inputs[1]->width() << " M=" << M
                                         << " N=" << N << " K=" << K << " batch=" << param._b;

    for (int b = 0; b < param._b; b++) {
        memObjects[0]    = (PtrDtype)(inputs[0]->data());
        memObjects[1]    = (PtrDtype)(inputs[1]->data());
        memObjects[2]    = (PtrDtype)(outputs[0]->mutable_data());
        offsetObjects[0] = b * M * K;
        offsetObjects[1] = b * K * N;
        offsetObjects[2] = b * M * N;
        int j            = 0;

        if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        if (_multikernel) {
            err = _kernels_ptr[j++].get()->SetKernelArgs(
                      memObjects[2], offsetObjects[2], floatObjects[1]);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[j++]);

            err = _kernels_ptr[j].get()->SetKernelArgs(
                      memObjects[0],
                      offsetObjects[0],
                      memObjects[1],
                      offsetObjects[1],
                      memObjects[2],
                      offsetObjects[2],
                      floatObjects[0]);
        } else {
            err = _kernels_ptr[j].get()->SetKernelArgs(
                      memObjects[0],
                      offsetObjects[0],
                      memObjects[1],
                      offsetObjects[1],
                      memObjects[2],
                      offsetObjects[2],
                      floatObjects[0],
                      floatObjects[1]);
        }

        if (!err) {
            LOG(ERROR) << "Fail to set kernel args :" << err;
            return SaberInvalidValue;
        }

        list.push_back(_kernels_ptr[j]);
        err = LaunchKernel(cm, list);

        if (!err) {
            LOG(ERROR) << "Fail to set execution :" << err;
            return SaberInvalidValue;
        }
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";

    return SaberSuccess;
}

template class SaberMatMul<AMD, AK_FLOAT>;

} // namespace saber

} // namespace anakin
