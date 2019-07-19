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
/*
   MIT License

   Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#include "include/vender_fc.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"

namespace anakin {
namespace saber {
#if 1
typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;
typedef Tensor<AMD> TensorDf4;
template <DataType OpDtype>
SaberStatus VenderFc<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    FcParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

enum FCType {
    FC6,
    FC7,
    FC,
    FC_NUM
};

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



#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS1 (64)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS1 (64 * ((param.weights->height() + 63) / 64 * 64))
#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS2 (64)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS2 (64 * ((param.weights->height() + 63) / 64 * 64))
#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS4 (64)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS4 (64 * ((param.weights->height() + 63) / 64 * 64))
#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS8 (64)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS8 (64 * ((param.weights->height() + 63) / 64 * 64))
#define YOLO_FC26_NT_LOCAL_WORK_SIZE_BS32 (64)
#define YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS32 (4096 * 24)

#define WAVE_SIZE   64



#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS1 "Conv1x1FC.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS2 "Conv1x1FC.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS4 "Conv1x1FC.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS8 "Conv1x1FC.cl"
#define VGG16_FC6_NT_KERNEL_FILE_NAME_BS32 "Conv1x1FC.cl"

#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS1 "Conv1x1FC.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS2 "Conv1x1FC.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS4 "Conv1x1FC.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS8 "Conv1x1FC.cl"
#define VGG16_FC8_NT_KERNEL_FILE_NAME_BS32 "Conv1x1FC.cl"

#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS1 "Conv1x1FC.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS2 "Conv1x1FC.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS4 "Conv1x1FC.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS8 "Conv1x1FC.cl"
#define YOLO_FC26_NT_KERNEL_FILE_NAME_BS32 "Conv1x1FC.cl"


#define BATCH_SIZE_1_INDEX 0
#define BATCH_SIZE_2_INDEX 1
#define BATCH_SIZE_4_INDEX 2
#define BATCH_SIZE_8_INDEX 3
#define BATCH_SIZE_32_INDEX 4
#define BATCH_SIZE_GE_INDEX 5

#define FC6_INDEX 0
#define FC7_INDEX 1
#define FC1_INDEX 2

template <DataType OpDtype>
SaberStatus VenderFc<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    FcParam<AMD>& param,
    Context<AMD>& ctx) {

    bool is_weights_trans = param.is_transpose_weights;

    if (param.num_output == param.weights->width()) {
        is_weights_trans = true;
    }

    this->_ctx   = &ctx;
    this->_param = &param;

    int atomic = 0;
    int local_size = 64;

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

    const int gwk[5][FC_NUM] = {{
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS1,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS1,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS1
        },
        {
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS2,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS2,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS2
        },
        {
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS4,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS4,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS4
        },
        {
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS8,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS8,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS8
        },
        {
            VGG16_FC6_NT_GLOBAL_WORK_SIZE_BS32,
            VGG16_FC8_NT_GLOBAL_WORK_SIZE_BS32,
            YOLO_FC26_NT_GLOBAL_WORK_SIZE_BS32
        }
    };

    const int lwk[5][FC_NUM] = {{
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS1,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS1,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS1
        },
        {
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS2,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS2,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS2
        },
        {
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS4,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS4,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS4
        },
        {
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS8,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS8,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS8
        },
        {
            VGG16_FC6_NT_LOCAL_WORK_SIZE_BS32,
            VGG16_FC8_NT_LOCAL_WORK_SIZE_BS32,
            YOLO_FC26_NT_LOCAL_WORK_SIZE_BS32
        }
    };

    const std::string kfn[5][FC_NUM] = {{
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS1,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS1,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS1
        },
        {
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS2,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS2,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS2
        },
        {
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS4,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS4,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS4
        },
        {
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS8,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS8,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS8
        },
        {
            VGG16_FC6_NT_KERNEL_FILE_NAME_BS32,
            VGG16_FC8_NT_KERNEL_FILE_NAME_BS32,
            YOLO_FC26_NT_KERNEL_FILE_NAME_BS32
        }
    };

    switch (inputs[0]->num()) {
    case 1:
        batch_size_index = BATCH_SIZE_1_INDEX;
        _branch = 3;
        break;

    case 2:
        batch_size_index = BATCH_SIZE_2_INDEX;
        _branch = 4;
        break;

    case 4:
        batch_size_index = BATCH_SIZE_4_INDEX;
        _branch = 5;
        break;

    case 8:
        batch_size_index = BATCH_SIZE_8_INDEX;
        _branch = 6;
        break;

    case 32:
        batch_size_index = BATCH_SIZE_32_INDEX;
        break;

    default:
        batch_size_index = BATCH_SIZE_GE_INDEX;
        _branch = 1;
        break;
    }

    if (inputs[0]->num() <= 8) {
        if ((param.weights->width() == 25088 && param.weights->height() == 4096)
                || (param.weights->width() == 4096 && param.weights->height() == 4096)
                || (param.weights->width() == 50176 && param.weights->height() == 4096)) {
            fc_index = FC6;
        } else if ((param.weights->width() == 4096 && param.weights->height() == 1000)
                   || (param.weights->width() == 2048 && param.weights->height() == 1000)) {
            fc_index = FC7;
            _branch = 2;
        } else {
            fc_index = FC;

            if (param.weights->width() % 64 == 0) {
                if (outputs[0]->channel() <= 32) {
                    int align = 1;

                    for (; outputs[0]->channel() > align; align *= 2) {
                    }

                    atomic = 64 / align;

                    for (; param.weights->width() > local_size && local_size < 1024; local_size *= 2) {
                    }

                    _branch = 10;
                } else if (outputs[0]->channel() <= 1024) {
                    for (; param.weights->width() > local_size && local_size < 1024; local_size *= 2) {
                    }

                    _branch = 11;
                } else {
                    local_size = 64;
                    _branch = 1;
                }
            } else {
                if (outputs[0]->channel() <= 32) {
                    int align = 1;

                    for (; outputs[0]->channel() > align; align *= 2) {
                    }

                    atomic = 64 / align;

                    for (; param.weights->width() > local_size && local_size < 1024; local_size *= 2) {
                    }

                    _branch = 10;
                } else {
                    for (; param.weights->width() > local_size && local_size < 1024; local_size *= 2) {
                    }

                    _branch = 11;
                }
            }
        }
    } else {
        _branch = 0;
    }

    if (!is_weights_trans) {
        if (_branch) {
            if (batch_size_index == BATCH_SIZE_GE_INDEX) {
                kernelInfo.l_wk        = {64, 1, 1};
                kernelInfo.g_wk        = {(64 * ((outputs[0]->channel() + 63) / 64 * 64)), 1, 1};
                kernelInfo.kernel_file = "Conv1x1FC.cl";
                kernelInfo.kernel_name = "InnerProduct";
            } else {
                if (_branch < 10) {
                    kernelInfo.l_wk        = {lwk[batch_size_index][fc_index], 1, 1};
                    kernelInfo.g_wk        = {gwk[batch_size_index][fc_index], 1, 1};
                    kernelInfo.kernel_file = kfn[batch_size_index][fc_index];
                    kernelInfo.kernel_name = "InnerProduct";
                } else {
                    if (outputs[0]->channel() <= 32) {
                        int align = (WAVE_SIZE / atomic);
                        kernelInfo.l_wk        = {local_size, 1, 1};
                        kernelInfo.g_wk        = {(local_size* atomic * ((param.weights->height() + align - 1) / align * align)), 1, 1};
                        kernelInfo.kernel_file = "Conv1x1FC.cl";
                        kernelInfo.kernel_name = "InnerProduct";
                    } else {
                        int multiple = 2048 / local_size;
                        kernelInfo.l_wk        = {local_size, 1, 1};
                        kernelInfo.g_wk        = {(local_size* multiple * ((param.weights->height() + WAVE_SIZE - 1) / WAVE_SIZE * WAVE_SIZE)), 1, 1};
                        kernelInfo.kernel_file = "Conv1x1FC.cl";
                        kernelInfo.kernel_name = "InnerProduct";
                    }
                }
            }
        } else { // gemm
            //The below section of code are as MIT license, the permission notice is from above (line 375 to 407)
            float alpha = 1.0;
            float beta  = 0.0;
            bool transA     = false;
            bool transB     = true;
            bool transC     = false;
            int leadingd_A     = K;
            int leadingd_B     = K;
            int leadingd_C     = N;

            MIOpenGEMM::Geometry tgg {};
            tgg = MIOpenGEMM::Geometry(false, transA, transB, transC, leadingd_A, leadingd_B, leadingd_C, M, N,
                                       K, 0, 'f');
            AMD_API::stream_t cm = this->_ctx->get_compute_stream();

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
                                            (PtrDtype)inputs[0]->data(),
                                            (PtrDtype)param.weights->data(),
                                            (PtrDtype)_outGemmWorkspace.mutable_data(),
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
        //The below section of code are as MIT license, the permission notice is from above (line 471 to 503)
        // gemm
        _branch          = 0;

        float alpha = 1.0;
        float beta  = 0.0;
        bool transA     = false;
        bool transB     = true;
        bool transC     = false;
        int leadingd_A     = K;
        int leadingd_B     = K;
        int leadingd_C     = N;

        MIOpenGEMM::Geometry tgg {};
        tgg = MIOpenGEMM::Geometry(false, transA, transB, transC, leadingd_A, leadingd_B, leadingd_C, M, N,
                                   K, 0, 'f');
        AMD_API::stream_t cm = this->_ctx->get_compute_stream();

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
                                        (PtrDtype)inputs[0]->data(),
                                        (PtrDtype)param.weights->data(),
                                        (PtrDtype)_outGemmWorkspace.mutable_data(),
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

    if (_branch) {
        // set comp_options...
        if (_usemacro) {
            kernelInfo.comp_options = std::string(" -DSTRIDE=") + std::to_string(inputs[0]->channel()) +
                                      std::string(" -DMACRO") +
                                      std::string(" -DATOMIC=") + std::to_string(atomic) +
                                      std::string(" -DLOCAL_SIZE=") + std::to_string(local_size) +
                                      std::string(" -DN=") + std::to_string(inputs[0]->num()) +
                                      std::string(" -DWIDTH=") + std::to_string(param.weights->width()) +
                                      std::string(" -DOUTPUT=") + std::to_string(outputs[0]->channel()) +
                                      std::string(" -DKERNEL_METHOD=") + std::to_string(_branch) +
                                      std::string(" -DNO_SLOPE");
        } else {
            kernelInfo.comp_options = std::string(" -DSTRIDE=") + std::to_string(inputs[0]->channel()) +
                                      std::string(" -DATOMIC=") + std::to_string(atomic) +
                                      std::string(" -DLOCAL_SIZE=") + std::to_string(local_size) +
                                      std::string(" -DKERNEL_METHOD=") + std::to_string(_branch) +
                                      std::string(" -DNO_SLOPE");
        }

        if (param.bias != nullptr && param.bias->valid_size() > 0) {
            kernelInfo.comp_options += std::string(" -DBIAS");
        }

        kernelInfo.kernel_type = MIOPEN;

        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        atomic = (atomic == 0) ? 32 : atomic;
        Shape pCounterShape({1, M * 64 / atomic, 1, 1},
                            Layout_NCHW);
        _pCounterForConv1x1FC = new Tensor<AMD>();
        _pCounterForConv1x1FC->re_alloc(pCounterShape);
        _kernels_ptr.push_back(kptr);
    } else {
        if (param.bias != nullptr && param.bias->valid_size() > 0) {
            kernelInfo.kernel_file = "BiasReLuUni.cl";
            kernelInfo.kernel_name = "BiasOnly";

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
SaberStatus VenderFc<AMD, OpDtype>::dispatch(
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
        PtrDtype memObjects[5] = {0, 0, 0, 0, 0};
        cl_event event;

        if (_branch) {
            memObjects[0] = (PtrDtype)inputs[0]->data();
            memObjects[1] = (PtrDtype)param.weights->data();
            memObjects[2] = (param.bias != nullptr) ? (PtrDtype)param.bias->data() : nullptr;
            memObjects[3] = (PtrDtype)outputs[0]->mutable_data();
            memObjects[4] = (PtrDtype)_pCounterForConv1x1FC->mutable_data();

            if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
                return SaberInvalidValue;
            }

            if (param.bias != nullptr && param.bias->valid_size() > 0) {
                if (_usemacro) {
                    err = _kernels_ptr[0].get()->SetKernelArgs(
                              (PtrDtype)memObjects[0],
                              (PtrDtype)memObjects[1],
                              (PtrDtype)memObjects[2],
                              (PtrDtype)memObjects[3]);
                } else {
                    if (_branch == 10) {
                        err = _kernels_ptr[0].get()->SetKernelArgs(
                                  (PtrDtype)memObjects[0],
                                  (PtrDtype)memObjects[1],
                                  (PtrDtype)memObjects[2],
                                  (PtrDtype)memObjects[3],
                                  (PtrDtype)memObjects[4],
                                  inputs[0]->num(),
                                  param.weights->width(),
                                  outputs[0]->channel()
                              );
                    } else {
                        err = _kernels_ptr[0].get()->SetKernelArgs(
                                  (PtrDtype)memObjects[0],
                                  (PtrDtype)memObjects[1],
                                  (PtrDtype)memObjects[2],
                                  (PtrDtype)memObjects[3],
                                  inputs[0]->num(),
                                  param.weights->width(),
                                  outputs[0]->channel()
                              );
                    }
                }
            } else {
                if (_usemacro) {
                    err = _kernels_ptr[0].get()->SetKernelArgs(
                              (PtrDtype)memObjects[0],
                              (PtrDtype)memObjects[1],
                              (PtrDtype)memObjects[3]);
                } else {
                    if (_branch == 10) {
                        err = _kernels_ptr[0].get()->SetKernelArgs(
                                  (PtrDtype)memObjects[0],
                                  (PtrDtype)memObjects[1],
                                  (PtrDtype)memObjects[3],
                                  (PtrDtype)memObjects[4],
                                  inputs[0]->num(),
                                  param.weights->width(),
                                  outputs[0]->channel()
                              );
                    } else {
                        err = _kernels_ptr[0].get()->SetKernelArgs(
                                  (PtrDtype)memObjects[0],
                                  (PtrDtype)memObjects[1],
                                  (PtrDtype)memObjects[3],
                                  inputs[0]->num(),
                                  param.weights->width(),
                                  outputs[0]->channel()
                              );
                    }
                }
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

template class VenderFc<AMD, AK_FLOAT>;
#endif
} // namespace saber
} // namespace anakin
