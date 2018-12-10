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

#include "include/saber_fc.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"

namespace anakin {
namespace saber {
#if 1
typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;
typedef Tensor<AMD> TensorDf4;
template <DataType OpDtype>
SaberStatus SaberFc<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    FcParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS1 (128)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS1 (8192 * 8)
#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS2 (64)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS2 (8192 * 4)
#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS4 (64)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS4 (8192 * 8)
#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS8 (64)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS8 (8192 * 4)
#define VGG16_FC6_NT_LOCAL_WORK_SIZE_BS32 (256)
#define VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS32 (4096 * 16)

#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS1 (128)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS1 (8192 * 8)
#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS2 (64)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS2 (8192 * 4)
#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS4 (64)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS4 (8192 * 4)
#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS8 (64)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS8 (8192 * 4)
#define VGG16_FC7_NT_LOCAL_WORK_SIZE_BS32 (256)
#define VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS32 (4096 * 16)

#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS1 (64)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS1 (64 * 64 * 16)
#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS2 (64)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS2 (64 * 64 * 16)
#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS4 (64)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS4 (64 * 64 * 16)
#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS8 (64)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS8 (64 * 64 * 16)
#define VGG16_FC8_NT_LOCAL_WORK_SIZE_BS32 (64)
#define VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS32 (64 * 64 * 16)

#define RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS1 (64)
#define RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS1 (64 * 64 * 16)
#define RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS2 (64)
#define RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS2 (64 * 64 * 16)
#define RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS4 (64)
#define RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS4 (64 * 64 * 16)
#define RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS8 (64)
#define RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS8 (64 * 64 * 16)
#define RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS32 (64)
#define RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS32 (8192 * 8)

#define YOLO_FC25_NT_LOCAL_WORK_SIZE_BS1 (128)
#define YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS1 (8192 * 8)
#define YOLO_FC25_NT_LOCAL_WORK_SIZE_BS2 (64)
#define YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS2 (8192 * 4)
#define YOLO_FC25_NT_LOCAL_WORK_SIZE_BS4 (64)
#define YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS4 (8192 * 8)
#define YOLO_FC25_NT_LOCAL_WORK_SIZE_BS8 (64)
#define YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS8 (8192 * 4)
#define YOLO_FC25_NT_LOCAL_WORK_SIZE_BS32 (256)
#define YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS32 (4096 * 16)

#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS1 (256)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS1 (4096 * 6)
#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS2 (128)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS2 (4096 * 12)
#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS4 (64)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS4 (4096 * 12)
#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS8 (64)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS8 (4096 * 24)
#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS32 (64)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS32 (4096 * 24)

#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS1 "InnerProductBNTFC6M1.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS2 "InnerProductBNTFC6M2.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS4 "InnerProductBNTFC6M4.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS8 "InnerProductBNTFC6M8.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC6M32.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS1 "InnerProductBNTFC7M1.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS2 "InnerProductBNTFC7M2.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS4 "InnerProductBNTFC7M4.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS8 "InnerProductBNTFC7M8.cl"
#define VGG16_FC7_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC7M32.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS1 "Conv1x1FC7.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS2 "Conv1x1FC7.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS4 "Conv1x1FC7.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS8 "Conv1x1FC7.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC8M32.cl"

#define RESNET_FC1000_NT_KERNEL_FILE_NAME_BS1 "Conv1x1FC7.cl"
#define RESNET_FC1000_NT_KERNEL_FILE_NAME_BS2 "Conv1x1FC7.cl"
#define RESNET_FC1000_NT_KERNEL_FILE_NAME_BS4 "Conv1x1FC7.cl"
#define RESNET_FC1000_NT_KERNEL_FILE_NAME_BS8 "Conv1x1FC7.cl"
#define RESNET_FC1000_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC1000M32.cl"

#define YOLO_FC25_NT_KERNEL_FILE_NAME_BS1 "InnerProductBNTFC25M1.cl"
#define YOLO_FC25_NT_KERNEL_FILE_NAME_BS2 "InnerProductBNTFC25M2.cl"
#define YOLO_FC25_NT_KERNEL_FILE_NAME_BS4 "InnerProductBNTFC25M4.cl"
#define YOLO_FC25_NT_KERNEL_FILE_NAME_BS8 "InnerProductBNTFC25M8.cl"
#define YOLO_FC25_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC25M32.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS1 "InnerProductBNTFC26M1.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS2 "InnerProductBNTFC26M2.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS4 "InnerProductBNTFC26M4.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS8 "InnerProductBNTFC26M8.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS32 "InnerProductBNTFC26M32.cl"

#define BATCH_SIZE_1_INDEX 0
#define BATCH_SIZE_2_INDEX 1
#define BATCH_SIZE_4_INDEX 2
#define BATCH_SIZE_8_INDEX 3
#define BATCH_SIZE_32_INDEX 4

#define FC6_INDEX 0
#define FC7_INDEX 1
#define FC8_INDEX 2
#define FC1000_INDEX 3
#define FC25_INDEX 4
#define FC26_INDEX 5

template <DataType OpDtype>
SaberStatus SaberFc<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    FcParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx   = &ctx;
    this->_param = &param;

    KernelInfo kernelInfo;
    int batch_size_index = 0;
    int fc_index         = 0;
    AMDKernelPtr kptr;
    _kernels_ptr.clear();

    int M = inputs[0]->num();
    int K = inputs[0]->count_valid(param.axis, inputs[0]->dims());
    int N = param.num_output;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "M: " << M << " N: " << N << " K: " << K;

    if (N <= 0) {
        int weight_size = param.weights->valid_size();
        N               = weight_size / K;
    }

    const int gwk[5][6] = {{
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS1,
            VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS1,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS1,
            RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS1,
            YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS1,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS1
        },
        {
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS2,
            VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS2,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS2,
            RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS2,
            YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS2,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS2
        },
        {
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS4,
            VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS4,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS4,
            RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS4,
            YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS4,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS4
        },
        {
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS8,
            VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS8,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS8,
            RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS8,
            YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS8,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS8
        },
        {
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS32,
            VGG16_FC7_NT_GLOBAL_WORK_SIZE_BS32,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS32,
            RESNET_FC1000_NT_GLOBAL_WORK_SIZE_BS32,
            YOLO_FC25_NT_GLOBAL_WORK_SIZE_BS32,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS32
        }
    };

    const int lwk[5][6] = {{
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS1,
            VGG16_FC7_NT_LOCAL_WORK_SIZE_BS1,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS1,
            RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS1,
            YOLO_FC25_NT_LOCAL_WORK_SIZE_BS1,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS1
        },
        {
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS2,
            VGG16_FC7_NT_LOCAL_WORK_SIZE_BS2,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS2,
            RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS2,
            YOLO_FC25_NT_LOCAL_WORK_SIZE_BS2,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS2
        },
        {
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS4,
            VGG16_FC7_NT_LOCAL_WORK_SIZE_BS4,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS4,
            RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS4,
            YOLO_FC25_NT_LOCAL_WORK_SIZE_BS4,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS4
        },
        {
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS8,
            VGG16_FC7_NT_LOCAL_WORK_SIZE_BS8,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS8,
            RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS8,
            YOLO_FC25_NT_LOCAL_WORK_SIZE_BS8,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS8
        },
        {
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS32,
            VGG16_FC7_NT_LOCAL_WORK_SIZE_BS32,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS32,
            RESNET_FC1000_NT_LOCAL_WORK_SIZE_BS32,
            YOLO_FC25_NT_LOCAL_WORK_SIZE_BS32,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS32
        }
    };

    const std::string kfn[5][6] = {{
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS1,
            VGG16_FC7_NT_KERNEL_FILE_NAME_BS1,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS1,
            RESNET_FC1000_NT_KERNEL_FILE_NAME_BS1,
            YOLO_FC25_NT_KERNEL_FILE_NAME_BS1,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS1
        },
        {
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS2,
            VGG16_FC7_NT_KERNEL_FILE_NAME_BS2,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS2,
            RESNET_FC1000_NT_KERNEL_FILE_NAME_BS2,
            YOLO_FC25_NT_KERNEL_FILE_NAME_BS2,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS2
        },
        {
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS4,
            VGG16_FC7_NT_KERNEL_FILE_NAME_BS4,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS4,
            RESNET_FC1000_NT_KERNEL_FILE_NAME_BS4,
            YOLO_FC25_NT_KERNEL_FILE_NAME_BS4,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS4
        },
        {
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS8,
            VGG16_FC7_NT_KERNEL_FILE_NAME_BS8,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS8,
            RESNET_FC1000_NT_KERNEL_FILE_NAME_BS8,
            YOLO_FC25_NT_KERNEL_FILE_NAME_BS8,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS8
        },
        {
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS32,
            VGG16_FC7_NT_KERNEL_FILE_NAME_BS32,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS32,
            RESNET_FC1000_NT_KERNEL_FILE_NAME_BS32,
            YOLO_FC25_NT_KERNEL_FILE_NAME_BS32,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS32
        }
    };

    switch (inputs[0]->num()) {
    case 1:
        batch_size_index = BATCH_SIZE_1_INDEX;
        break;

    case 2:
        batch_size_index = BATCH_SIZE_2_INDEX;
        break;

    case 4:
        batch_size_index = BATCH_SIZE_4_INDEX;
        break;

    case 8:
        batch_size_index = BATCH_SIZE_8_INDEX;
        break;

    case 32:
        batch_size_index = BATCH_SIZE_32_INDEX;
        break;
    }

    switch (param.weights->width() * param.weights->height()) {
    case 25088 * 4096:
        fc_index = FC6_INDEX;
        break;

    case 4096 * 4096:
        fc_index = FC7_INDEX;
        break;

    case 4096 * 1000:
        fc_index = FC8_INDEX;
        break;

    case 2048 * 1000:
        fc_index = FC1000_INDEX;
        break;

    case 50176 * 4096:
        fc_index = FC25_INDEX;
        break;

    case 4096 * 1470:
        fc_index = FC26_INDEX;
        break;

    default:
        optmized = false;
        break;
    }

    if (!param.is_transpose_weights) {
        if (optmized) {
            kernelInfo.l_wk        = {lwk[batch_size_index][fc_index], 1, 1};
            kernelInfo.g_wk        = {gwk[batch_size_index][fc_index], 1, 1};
            kernelInfo.kernel_file = kfn[batch_size_index][fc_index];
            kernelInfo.kernel_name = "InnerProduct";
        } else { // gemm
            _outGemmWorkspace = new Tensor<AMD>();
            _outGemmWorkspace->re_alloc(outputs[0]->shape());

            float alpha = 1.0;
            float beta  = 0.0;
            bool tA     = false;
            bool tB     = true;
            bool tC     = false;
            int lda     = K;
            int ldb     = K;
            int ldc     = N;

            MIOpenGEMM::Geometry tgg {};
            tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
            AMD_API::stream_t cm = this->_ctx->get_compute_stream();

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
                                            (PtrDtype)inputs[0]->data(),
                                            (PtrDtype)param.weights->data(),
                                            (PtrDtype)_outGemmWorkspace->mutable_data(),
                                            false,
                                            tgg,
                                            miopengemm_verbose,
                                            miopengemm_warnings);

            std::string kernel_clstring;
            size_t local_work_size;
            size_t global_work_size;
            int errCode;

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
                kernelInfo.l_wk        = {local_work_size, 1, 1};
                kernelInfo.g_wk        = {global_work_size, 1, 1};
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
        }
    } else {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "Transpose Weights!";
        // gemm
        optmized          = false;
        _outGemmWorkspace = new Tensor<AMD>();
        _outGemmWorkspace->re_alloc(outputs[0]->shape());

        float alpha = 1.0;
        float beta  = 0.0;
        bool tA     = false;
        bool tB     = false;
        bool tC     = false;
        int lda     = K;
        int ldb     = N;
        int ldc     = N;

        MIOpenGEMM::Geometry tgg {};
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        AMD_API::stream_t cm = this->_ctx->get_compute_stream();

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
                                        (PtrDtype)inputs[0]->data(),
                                        (PtrDtype)param.weights->data(),
                                        (PtrDtype)_outGemmWorkspace->mutable_data(),
                                        false,
                                        tgg,
                                        miopengemm_verbose,
                                        miopengemm_warnings);

        std::string kernel_clstring;
        size_t local_work_size;
        size_t global_work_size;
        int errCode;

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
            kernelInfo.l_wk        = {local_work_size, 1, 1};
            kernelInfo.g_wk        = {global_work_size, 1, 1};
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
    }

    if (optmized) {
        // set comp_options...
        kernelInfo.comp_options = std::string(" -DSTRIDE=") + std::to_string(inputs[0]->channel()) +
                                  std::string(" -DN=") + std::to_string(inputs[0]->num());

        if (param.bias != nullptr && param.bias->valid_size() > 0) {
            kernelInfo.comp_options += std::string(" -DBIAS");
        }

        kernelInfo.kernel_type = SABER;

        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);
    } else {
        if (param.bias != nullptr && param.bias->valid_size() > 0) {
            kernelInfo.kernel_file = "MIOpenBiasReLuUni.cl";
            kernelInfo.kernel_name = "MIOpenReLu";

            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {(inputs[0]->num())* (outputs[0]->channel())* (outputs[0]->height())
                               * (outputs[0]->width()),
                               1,
                               1
                              };

            kernelInfo.kernel_type = SABER;

            kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to create kernel";
                return SaberInvalidValue;
            }

            _kernels_ptr.push_back(kptr);
        }
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberFc<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    FcParam<AMD>& param) {

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch\n";
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "num_output= " << param.num_output << " axis=" <<
                                         param.axis;
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "in num= " << inputs[0]->num() << " channel= " <<
                                         inputs[0]->channel()
                                         << " height= " << inputs[0]->height() << " width= " << inputs[0]->width();
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "weight num= " << param.weights->num() << " channel= " <<
                                         param.weights->channel()
                                         << " height= " << param.weights->height()
                                         << " width= " << param.weights->width();
    {
        bool err = false;
        amd_kernel_list list;

        // To get the commpute command queue
        AMD_API::stream_t cm = this->_ctx->get_compute_stream();

        // To set the argument
        PtrDtype memObjects[4] = {0, 0, 0, 0};
        cl_event event;

        if (optmized) {
            memObjects[0] = (PtrDtype)inputs[0]->data();
            memObjects[1] = (PtrDtype)param.weights->data();
            memObjects[2] = (param.bias != nullptr) ? (PtrDtype)param.bias->data() : nullptr;
            memObjects[3] = (PtrDtype)outputs[0]->mutable_data();

            if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            if (param.bias != nullptr && param.bias->valid_size() > 0) {
                err = _kernels_ptr[0].get()->SetKernelArgs(
                          (PtrDtype)memObjects[0],
                          (PtrDtype)memObjects[1],
                          (PtrDtype)memObjects[2],
                          (PtrDtype)memObjects[3]);
            } else {
                err = _kernels_ptr[0].get()->SetKernelArgs(
                          (PtrDtype)memObjects[0],
                          (PtrDtype)memObjects[1],
                          (PtrDtype)memObjects[3]);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[0]);
        } else {
            memObjects[0] = (PtrDtype)inputs[0]->data();
            memObjects[1] = (PtrDtype)param.weights->data();
            memObjects[2] = (param.bias != nullptr) ? (PtrDtype)param.bias->data() : nullptr;
            memObjects[3] = (PtrDtype)outputs[0]->mutable_data();

            if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                LOG(ERROR) << "Kernel is not exist";
                return SaberInvalidValue;
            }

            int j = 0;

            if (_multikernel) {
                err = _kernels_ptr[j].get()->SetKernelArgs(memObjects[3], 0, 0.0f);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[j++]);

                err = _kernels_ptr[j].get()->SetKernelArgs(
                          memObjects[0], 0, memObjects[1], 0, memObjects[3], 0, 1.0f);
            } else {
                err = _kernels_ptr[j].get()->SetKernelArgs(
                          memObjects[0], 0, memObjects[1], 0, memObjects[3], 0, 1.0f, 0.0f);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[j++]);

            if (param.bias != nullptr && param.bias->valid_size() > 0) {
                err = _kernels_ptr[j].get()->SetKernelArgs(
                          memObjects[3],
                          memObjects[3],
                          memObjects[2],
                          1.0f,
                          (inputs[0]->num()),
                          (outputs[0]->channel()),
                          (outputs[0]->height()),
                          (outputs[0]->width()));

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[j]);
            }
        }

        err = LaunchKernel(cm, list);

        if (!err) {
            LOG(ERROR) << "Fail to set execution :" << err;
            return SaberInvalidValue;
        }

        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    }
    return SaberSuccess;
}

template class SaberFc<AMD, AK_FLOAT>;
#endif
} // namespace saber
} // namespace anakin
