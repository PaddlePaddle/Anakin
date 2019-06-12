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
#include "include/vender_deconv.h"
namespace anakin {

namespace saber {

typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;
typedef Tensor<AMD> TensorDf4;

void set_offsets_to_uint(std::string& clstr, int times) {
    for (int i = 0; i < times; i++) {
        clstr = clstr.replace(clstr.find("const ulong"), 11, "const uint");
    }
}
void set_offsets_to_uint(std::string& clstr) {
    auto get_target = [](std::string inttype, char x) {
        std::stringstream ss;
        ss << "const " << inttype << ' ' << std::string(1, x) << "_offset";
        return std::regex(ss.str());
    };

    for (char x : {
                'a', 'b', 'c'
            }) {
        std::string replacement = "const unsigned " + std::string(1, x) + "_offset";

        for (auto inttype : {
                    "size_t", "ulong"
                }) {
            clstr = std::regex_replace(clstr, get_target(inttype, x), replacement);
        }
    }
}

template <DataType OpDtype>
void VenderDeconv2D<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return;
    }

    _kernels_ptr.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus VenderDeconv2D<AMD, OpDtype>::create_gemm(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {

    _use_gemm = true;
    _kernel_atomic = NULL;
    _kernel_normal = NULL;
    _kernel_col2Im = NULL;
    _kernel_col2Im_bias_relu = NULL;
    _kernel_isBias = NULL;

    KernelInfo kernelInfo;

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    if (param.weight()->height() == 1 && param.weight()->width() == 1 && param.stride_h == 1
            && param.stride_w == 1 && param.pad_h == 0 && param.pad_w == 0 && param.group == 1) {

        //The below section of code are as MIT license, the permission notice is from above (line 16 to 36)
        int K       = (param.weight()->channel());
        int N       = (inputs[0]->height()) * (inputs[0]->width());
        int M       = (outputs[0]->channel()) * (param.weight()->height()) * (param.weight()->width());
        bool transA     = true;
        bool transB     = false;
        bool transC     = false;
        int leadingd_A     = M;
        int leadingd_B     = N;
        int leadingd_C     = N;

        MIOpenGEMM::Geometry tgg {};
        tgg = MIOpenGEMM::Geometry(false, transA, transB, transC, leadingd_A, leadingd_B, leadingd_C, M, N,
                                   K, 0, 'f');
        _outGemmWorkspace = new Tensor<AMD>;
        Shape sh({inputs[0]->num(), inputs[0]->channel(), inputs[0]->height(), inputs[0]->width()});

        _outGemmWorkspace->re_alloc(sh);
        bool miopengemm_verbose = false;
        // jn : print warning messages when the returned kernel(s) might be sub-optimal
        bool miopengemm_warnings = false;
        // jn : find with no workspace
        MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                        0.003f,
                                        cm,
                                        (cl_mem)param.weight()->data(),
                                        (cl_mem)inputs[0]->data(),
                                        (cl_mem)_outGemmWorkspace->mutable_data(),
                                        false,
                                        tgg,
                                        miopengemm_verbose,
                                        miopengemm_warnings);
        std::string kernel_clstring;
        size_t local_work_size;
        size_t global_work_size;
        cl_int errCode;
        int soln_tgks_indx = 0;

        if (soln.v_tgks.size() == 2) {
            _multikernel = true;

            // jn : the main kernel is at the back of the solution vector
            kernel_clstring = soln.v_tgks[soln_tgks_indx].kernstr;
            set_offsets_to_uint(kernel_clstring, 1);

            kernelInfo.kernel_type = SOURCE;
            kernelInfo.kernel_name = soln.v_tgks[soln_tgks_indx].fname;
            local_work_size        = soln.v_tgks[soln_tgks_indx].local_work_size;
            global_work_size       = soln.v_tgks[soln_tgks_indx].global_work_size;

            kernelInfo.kernel_file   = kernel_clstring;
            kernelInfo.wk_dim        = 1;
            kernelInfo.l_wk          = {local_work_size, 1, 1};
            kernelInfo.g_wk          = {global_work_size, 1, 1};
            AMDKernelPtr kptr_atomic = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr_atomic.get()->isInit()) {
                LOG(ERROR) << "Failed to create kernel";
                return SaberInvalidValue;
            }

            _kernel_atomic = kptr_atomic;

            soln_tgks_indx++;
        }

        // jn : the main kernel is at the back of the solution vector
        kernel_clstring = soln.v_tgks[soln_tgks_indx].kernstr;
        set_offsets_to_uint(kernel_clstring, 3);

        kernelInfo.kernel_name = soln.v_tgks[soln_tgks_indx].fname;
        local_work_size        = soln.v_tgks[soln_tgks_indx].local_work_size;
        global_work_size       = soln.v_tgks[soln_tgks_indx].global_work_size;

        kernelInfo.kernel_file = kernel_clstring;
        kernelInfo.wk_dim      = 1;
        kernelInfo.l_wk        = {local_work_size, 1, 1};
        kernelInfo.g_wk        = {global_work_size, 1, 1};
        kernelInfo.kernel_type = SOURCE;

        AMDKernelPtr kptr_normal = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr_normal.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernel_normal = kptr_normal;

        kernelInfo.kernel_file = "BiasReLuUni.cl";
        kernelInfo.kernel_name = "BiasReluBoth";
        kernelInfo.kernel_type = SABER;

        kernelInfo.l_wk = {256, 1, 1};
        kernelInfo.g_wk = {(outputs[0]->num())* (param.weight()->num())* (outputs[0]->height())
                           * (outputs[0]->width()),
                           1,
                           1
                          };

        AMDKernelPtr kptr_isBias = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr_isBias.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        _kernel_isBias = kptr_isBias;
    } else {
        //The below section of code are as MIT license, the permission notice is from above (line 16 to 36)
        int K       = (param.weight()->channel() / param.group);
        int N       = (inputs[0]->height()) * (inputs[0]->width());
        int M       = (outputs[0]->channel() / param.group) * (param.weight()->height()) *
                      (param.weight()->width());
        bool transA     = true;
        bool transB     = false;
        bool transC     = false;
        int leadingd_A     = M;
        int leadingd_B     = N;
        int leadingd_C     = N;

        MIOpenGEMM::Geometry tgg {};
        tgg = MIOpenGEMM::Geometry(false, transA, transB, transC, leadingd_A, leadingd_B, leadingd_C, M, N,
                                   K, 0, 'f');

        _outGemmWorkspace = new Tensor<AMD>();
        Shape sh({param.weight()->channel(),
                  param.weight()->height(),
                  param.weight()->width(),
                  inputs[0]->height() * inputs[0]->width() * param.group
                 });
        _outGemmWorkspace->re_alloc(sh);
        bool miopengemm_verbose = false;

        bool miopengemm_warnings = false;

        MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                        0.003f,
                                        cm,
                                        (cl_mem)param.weight()->data(),
                                        (cl_mem)inputs[0]->data(),
                                        (cl_mem)_outGemmWorkspace->mutable_data(),
                                        false,
                                        tgg,
                                        miopengemm_verbose,
                                        miopengemm_warnings);
        std::string kernel_clstring;
        size_t local_work_size;
        size_t global_work_size;
        cl_int errCode;
        int soln_tgks_indx = 0;

        if (soln.v_tgks.size() == 2) {
            _multikernel = true;

            kernel_clstring = soln.v_tgks[soln_tgks_indx].kernstr;
            set_offsets_to_uint(kernel_clstring, 1);

            kernelInfo.kernel_name = soln.v_tgks[soln_tgks_indx].fname;
            local_work_size        = soln.v_tgks[soln_tgks_indx].local_work_size;
            global_work_size       = soln.v_tgks[soln_tgks_indx].global_work_size;

            kernelInfo.kernel_file = kernel_clstring;
            kernelInfo.wk_dim      = 1;
            kernelInfo.l_wk        = {local_work_size, 1, 1};
            kernelInfo.g_wk        = {global_work_size, 1, 1};
            kernelInfo.kernel_type = SOURCE;

            AMDKernelPtr kptr_atomic = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr_atomic.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_atomic = kptr_atomic;
            soln_tgks_indx++;
        }

        // jn : the main kernel is at the back of the solution vector
        kernel_clstring = soln.v_tgks[soln_tgks_indx].kernstr;
        set_offsets_to_uint(kernel_clstring, 3);

        kernelInfo.kernel_name = soln.v_tgks[soln_tgks_indx].fname;
        local_work_size        = soln.v_tgks[soln_tgks_indx].local_work_size;
        global_work_size       = soln.v_tgks[soln_tgks_indx].global_work_size;

        kernelInfo.kernel_file = kernel_clstring;

        kernelInfo.l_wk        = {local_work_size, 1, 1};
        kernelInfo.g_wk        = {global_work_size, 1, 1};
        kernelInfo.wk_dim      = 1;
        kernelInfo.kernel_type = SOURCE;

        AMDKernelPtr kptr_normal = CreateKernel(inputs[0]->device_id(), &kernelInfo);
        _kernel_normal           = kptr_normal;

        kernelInfo.kernel_file = "MIOpenUtilKernels2.cl";
        kernelInfo.kernel_name = "Col2ImBiasRelu";
        kernelInfo.kernel_type = MIOPEN;
        kernelInfo.wk_dim      = 1;
        kernelInfo.l_wk        = {256, 1, 1};
        kernelInfo.g_wk        = {
            outputs[0]->channel()* outputs[0]->height()* outputs[0]->width(), 1, 1
        };

        AMDKernelPtr kptr_col2Im_bias_relu = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr_col2Im_bias_relu.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernel_col2Im_bias_relu = kptr_col2Im_bias_relu;

    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus VenderDeconv2D<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "create";

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG)
            << "AMD Summary: input size N " << inputs[0]->num()
            << " C " << inputs[0]->channel()
            << " H " << inputs[0]->height()
            << " W " << inputs[0]->width();

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG)
            << "AMD Summary: op param K " << param.weight()->num()
            << " Y " << param.weight()->height() << " X " << param.weight()->width()
            << " SH " << param.stride_h << " SW " << param.stride_w
            << " PH " << param.pad_h << " PW " << param.pad_w
            << " DH " << param.dilation_h << " DW " << param.dilation_w
            << " Alpha " << param.alpha << " Beta " << param.beta << " GP " << param.group
            << " hasAct " << param.activation_param.has_active
            << " ActType " << param.activation_param.active
            << " slop " << param.activation_param.negative_slope
            << " coef " << param.activation_param.coef;

    // Clear the kernel object ptr
    _kernels_ptr.clear();

    std::vector<KernelInfo> solution = FindDeconvSolution(inputs, outputs, param);

    if (!solution.empty()) {
        _use_gemm = false;

        for (auto s : solution) {
            CreateKernelList(inputs[0]->device_id(), s);
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "s.kernel_name=" << s.kernel_name;
        }
    } else {
#if NOT_USE_GEMM
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "No solution found!!!";
        return SaberInvalidValue;
#else
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "No solution found!!!Trying to use GEMM";
        create_gemm(inputs, outputs, param, ctx);
#endif
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus VenderDeconv2D<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus VenderDeconv2D<AMD, OpDtype>::dispatch_gemm(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    amd_kernel_list& list) {

    bool err;
    AMD_API::stream_t cm    = this->_ctx->get_compute_stream();
    bool isBias             = (param.bias()->size() > 0) ? 1 : 0;
    int relu_flag           = (param.activation_param.active == Active_relu) ? 1 : 0;

    if (param.weight()->height() == 1 && param.weight()->width() == 1 && param.stride_h == 1
            && param.stride_w == 1 && param.pad_h == 0 && param.pad_w == 0 && param.group == 1) {
        cl_uint uintObjects[3]   = {0, 0, 0};
        cl_float floatObjects[2] = {1.0f, 0.0f};
        cl_uint im_offset        = 0;
        cl_uint out_offset       = 0;
        cl_mem memObjects[4]     = {(cl_mem)inputs[0]->data(),
                                    (cl_mem)param.weight()->data(),
                                    (cl_mem)outputs[0]->mutable_data(),
                                    (cl_mem)param.bias()->data()
                                   };

        for (int i = 0; i < (inputs[0]->num()); i++) {
            uintObjects[0] =
                i * (inputs[0]->channel()) * (inputs[0]->height()) * (inputs[0]->width());
            uintObjects[2] =
                i * (param.weight()->num()) * outputs[0]->height() * outputs[0]->width();

            if (_multikernel) {
                AMDKernel* kernel = _kernel_atomic.get();
                kernel->SetKernelArgs(
                    (PtrDtype)memObjects[2], (int)uintObjects[2], (float)floatObjects[1]);

                list.push_back(_kernel_atomic);
            }

            AMDKernel* kernel = _kernel_normal.get();

            kernel->SetKernelArgs(
                (PtrDtype)memObjects[1],
                (int)uintObjects[1],
                (PtrDtype)memObjects[0],
                (int)uintObjects[0],
                (PtrDtype)memObjects[2],
                (int)uintObjects[2],
                (float)floatObjects[0],
                (float)floatObjects[1]);

            list.push_back(_kernel_normal);
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE SET ARGUMENT";

            err = LaunchKernel(cm, list);
            list.clear();

            if (!err) {
                LOG(ERROR) << "Fialed to set execution.";
                return SaberInvalidValue;
            }
        }

        if (isBias) {
            AMDKernel* kernel = _kernel_isBias.get();
            kernel->SetKernelArgs(
                (PtrDtype)memObjects[2],
                (PtrDtype)memObjects[2],
                (PtrDtype)memObjects[3],
                0.0f, // important
                (int)(inputs[0]->num()),
                (int)(param.weight()->num()),
                (int)(outputs[0]->height()),
                (int)(outputs[0]->width()),
                (int)isBias,
                (int)relu_flag);

            list.push_back(_kernel_isBias);
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE SET ARGUMENT";
        } else { // isBias==0
            if (relu_flag) {
                AMDKernel* kernel = _kernel_isBias.get();
                kernel->SetKernelArgs(
                    (PtrDtype)memObjects[2],
                    (PtrDtype)memObjects[2],
                    (PtrDtype)memObjects[3],
                    0.0f,
                    (int)(inputs[0]->num()),
                    (int)(param.weight()->num()),
                    (int)(outputs[0]->height()),
                    (int)(outputs[0]->width()),
                    (int)isBias,
                    (int)relu_flag);

                list.push_back(_kernel_isBias);
            }

            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE SET ARGUMENT";
        }

        err = LaunchKernel(cm, list);
        list.clear();

        if (!err) {
            LOG(ERROR) << "Fialed to set execution.";
            return SaberInvalidValue;
        }

    } else {
        cl_uint uintObjects[3]   = {0, 0, 0};
        cl_float floatObjects[2] = {1.0f, 0.0f};
        cl_mem memObjects[4]     = {(cl_mem)inputs[0]->data(),
                                    (cl_mem)param.weight()->data(),
                                    (cl_mem)outputs[0]->mutable_data(),
                                    (cl_mem)param.bias()->data()
                                   };

        AMDKernel* kernel;

        for (int i = 0; i < (inputs[0]->num()); i++) {

            uintObjects[2] =
                i * outputs[0]->channel() * outputs[0]->height() * outputs[0]->width();

            for (int k = 0; k < param.group; k++) {
                uintObjects[1] = k * (param.weight()->num()) * (param.weight()->channel() / param.group) *
                                 (param.weight()->height()) * (param.weight()->width());
                uintObjects[0] = (i * (inputs[0]->channel()) * (inputs[0]->height()) * (inputs[0]->width())) +
                                 (k * (inputs[0]->channel() / param.group) * inputs[0]->height() * inputs[0]->width());
                cl_uint gemmwksp_offset = k * (outputs[0]->channel()) * param.weight()->height() *
                                          param.weight()->width() * inputs[0]->height() * inputs[0]->width() / param.group;

                if (_multikernel) {
                    AMDKernel* kernel = _kernel_atomic.get();
                    kernel->SetKernelArgs(
                        (PtrDtype)_outGemmWorkspace->mutable_data(), gemmwksp_offset, (float)floatObjects[1]);
                    list.push_back(_kernel_atomic);
                }

                kernel = _kernel_normal.get();
                kernel->SetKernelArgs(
                    (PtrDtype)memObjects[1],
                    (int)uintObjects[1],
                    (PtrDtype)memObjects[0],
                    (int)uintObjects[0],
                    (PtrDtype)_outGemmWorkspace->mutable_data(),
                    gemmwksp_offset,
                    (float)floatObjects[0],
                    (float)floatObjects[1]);

                list.push_back(_kernel_normal);
                err = LaunchKernel(cm, list);
                list.clear();
            }

            kernel = _kernel_col2Im_bias_relu.get();
            kernel->SetKernelArgs(
                (PtrDtype)_outGemmWorkspace->mutable_data(),
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
                (int)outputs[0]->height(),
                (int)outputs[0]->width(),
                (PtrDtype)outputs[0]->mutable_data(),
                (int)uintObjects[2],

                (cl_mem) param.bias()->data(),
                0.0f, //slope==0.0f
                (int)(inputs[0]->num()),
                (int)(param.weight()->num()),
                (int) isBias,
                (int) relu_flag);
            list.push_back(_kernel_col2Im_bias_relu);
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE SET ARGUMENT";

            err = LaunchKernel(cm, list);
            list.clear();

            if (!err) {
                LOG(ERROR) << "Fialed to set execution.";
                return SaberInvalidValue;
            }
        }

    }
}

template <DataType OpDtype>
SaberStatus VenderDeconv2D<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch";

    AMD_API::stream_t cm    = this->_ctx->get_compute_stream();
    bool isBias             = (param.bias()->size() > 0) ? 1 : 0;
    bool isActive           = param.activation_param.has_active;
    float negative_slope    = 1.0f;

    amd_kernel_list list;
    list.clear();

    if (isActive) {

        if (param.activation_param.active == Active_relu) {
            negative_slope = 0.0f;
        } else {
            negative_slope = param.activation_param.negative_slope;
        }

        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << " param.activation_param.has_active="
                                             << param.activation_param.has_active
                                             << " param.activation_param.negative_slope=" << param.activation_param.negative_slope
                                             << " param.activation_param.active=" << param.activation_param.active
                                             << " param.activation_param.coef=" << param.activation_param.coef;
    }

    if (_use_gemm) {
        dispatch_gemm(inputs, outputs, param, list);
    } else {
        if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        bool err = false;
        for (int i = 0; i < _kernels_ptr.size(); i++) {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "kernel size:" << _kernels_ptr.size() << " name:" <<
                                                 _kernels_ptr[i].get()->GetName();

            if ((_kernels_ptr[i].get()->GetName() == "MIOpenConvUni")
                    || (_kernels_ptr[i].get()->GetName() == "MIOpenGroupConvUni")) {

                if (isBias) {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)inputs[0]->data(),
                                  (PtrDtype)param.weight()->data(),
                                  (PtrDtype)param.bias()->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  negative_slope,
                                  0.0f);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)inputs[0]->data(),
                                  (PtrDtype)param.weight()->data(),
                                  (PtrDtype)param.bias()->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  0.0f);
                    }
                } else {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)inputs[0]->data(),
                                  (PtrDtype)param.weight()->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  negative_slope,
                                  0.0f);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)inputs[0]->data(),
                                  (PtrDtype)param.weight()->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  0.0f);
                    }
                }

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i]);
            } else if (_kernels_ptr[i].get()->GetName() == "sp3AsmConvRxSU") {
                int d_n_groups  = 64;
                int d_flags     = 7;
                int reserved    = 0;
                int* return_addr = nullptr;

                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (unsigned int)inputs[0]->num(),
                          (unsigned int)inputs[0]->channel(),
                          (unsigned int)inputs[0]->height(),
                          (unsigned int)inputs[0]->width(),
                          (unsigned int)param.weight()->num(),
                          (unsigned int)d_n_groups,
                          (unsigned int)d_flags,
                          (unsigned int)reserved,
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)param.weight()->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)return_addr,
                          (unsigned int)param.weight()->height(),
                          (unsigned int)param.weight()->width(),
                          (unsigned int)(param.weight()->height() - param.pad_h - 1),
                          (unsigned int)(param.weight()->height() - param.pad_w - 1),
                          (unsigned int)outputs[0]->height(),
                          (unsigned int)outputs[0]->width());

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i]);
            } else if (_kernels_ptr[i].get()->GetName() == "sp3AsmConvRxSU_CBA") {
                int d_n_groups  = 64;
                int d_flags     = 7;
                int reserved    = 0;
                int* return_addr = nullptr;

                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (unsigned int)inputs[0]->num(),
                          (unsigned int)inputs[0]->channel(),
                          (unsigned int)inputs[0]->height(),
                          (unsigned int)inputs[0]->width(),
                          (unsigned int)param.weight()->num(),
                          (unsigned int)d_n_groups,
                          (unsigned int)d_flags,
                          (unsigned int)reserved,
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)param.weight()->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)return_addr,
                          (unsigned int)param.weight()->height(),
                          (unsigned int)param.weight()->width(),
                          (unsigned int)param.pad_h,
                          (unsigned int)param.pad_w,
                          (unsigned int)outputs[0]->height(),
                          (unsigned int)outputs[0]->width(),
                          (PtrDtype)param.bias()->data(),
                          negative_slope
                      );

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i]);
            } else if (_kernels_ptr[i].get()->GetName() == "BiasReluBoth") {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)param.bias()->data(),
                          negative_slope,
                          (inputs[0]->num()),
                          (outputs[0]->channel()),
                          (outputs[0]->height()),
                          (outputs[0]->width()),
                          1,
                          1);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i]);
            } else if (_kernels_ptr[i].get()->GetName() == "BiasOnly") {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)param.bias()->data(),
                          negative_slope,
                          (inputs[0]->num()),
                          (outputs[0]->channel()),
                          (outputs[0]->height()),
                          (outputs[0]->width()));

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i]);
            } else if (_kernels_ptr[i].get()->GetName() == "ReluOnly") {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          negative_slope);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i]);
            } else {
                LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "Not implement kernel name:" <<
                                                     _kernels_ptr[i].get()->GetName();
            }
        }

        if (list.size() > 0) {
            err = LaunchKernel(cm, list);
            list.clear();

            if (!err) {
                LOG(ERROR) << "Fail to set execution :" << err;
                return SaberInvalidValue;
            }
        }
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class VenderDeconv2D<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderDeconv2D, ConvParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(VenderDeconv2D, ConvParam, AMD, AK_HALF);
} // namespace saber

} // namespace anakin
