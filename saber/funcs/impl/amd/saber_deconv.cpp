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
#include "include/saber_deconv.h"
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
SaberStatus SaberDeconv2D<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberDeconv2D<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {

    KernelInfo kernelInfo;

    // bool isBias = (param.bias()->size() > 0) ? true : false;
    int isBias           = (param.bias()->size() > 0) ? 1 : 0;
    int relu_flag        = (param.activation_param.active == Active_relu) ? 1 : 0;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    // int K       = (param.weight()->num());
    int K       = (param.weight()->channel());
    int N       = (inputs[0]->height()) * (inputs[0]->width());
    int M       = (outputs[0]->channel()) * (param.weight()->height()) * (param.weight()->width());
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = true;
    bool tB     = false;
    bool tC     = false;
    int lda     = M;
    int ldb     = N;
    int ldc     = N;
    int workspace_req = param.weight()->channel() * param.weight()->height()
                        * param.weight()->width() * (inputs[0]->height()) * (inputs[0]->width());

    MIOpenGEMM::Geometry tgg {};
    tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');

    if (param.weight()->height() == 1 && param.weight()->width() == 1 && param.stride_h == 1
            && param.stride_w == 1 && param.pad_h == 0 && param.pad_w == 0) {
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
        int i = 0;

        if (soln.v_tgks.size() == 2) {
            _multikernel = true;

            // jn : the main kernel is at the back of the solution vector
            kernel_clstring = soln.v_tgks[i].kernstr;
            set_offsets_to_uint(kernel_clstring, 1);

            kernelInfo.kernel_type = SOURCE;
            kernelInfo.kernel_name = soln.v_tgks[i].fname;
            local_work_size        = soln.v_tgks[i].local_work_size;
            global_work_size       = soln.v_tgks[i].global_work_size;

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
            _kernels.push_back(kptr_atomic);

            i++;
        }

        // jn : the main kernel is at the back of the solution vector
        kernel_clstring = soln.v_tgks[i].kernstr;
        set_offsets_to_uint(kernel_clstring, 3);

        kernelInfo.kernel_name = soln.v_tgks[i].fname;
        local_work_size        = soln.v_tgks[i].local_work_size;
        global_work_size       = soln.v_tgks[i].global_work_size;

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
        _kernels.push_back(kptr_normal);

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
        _kernels.push_back(kptr_isBias);
    } else {
        _outGemmWorkspace = new Tensor<AMD>();
        Shape sh({param.weight()->channel(),
                  param.weight()->height(),
                  param.weight()->width(),
                  inputs[0]->width() * inputs[0]->width()
                 });
        _outGemmWorkspace->re_alloc(sh);

        _outCol2ImSpace = new Tensor<AMD>();

        Shape sh_col2Im({param.weight()->channel(),
                         param.weight()->height(),
                         param.weight()->width(),
                         inputs[0]->height() * inputs[0]->width()
                        });
        _outCol2ImSpace->re_alloc(sh_col2Im);
        bool miopengemm_verbose = false;

        bool miopengemm_warnings = false;

        MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                        0.003f,
                                        cm,
                                        (cl_mem)param.weight()->data(),
                                        (cl_mem)inputs[0]->data(),
                                        (cl_mem)_outCol2ImSpace->mutable_data(),
                                        false,
                                        tgg,
                                        miopengemm_verbose,
                                        miopengemm_warnings);
        std::string kernel_clstring;
        size_t local_work_size;
        size_t global_work_size;
        cl_int errCode;
        int i = 0;

        if (soln.v_tgks.size() == 2) {
            _multikernel = true;

            kernel_clstring = soln.v_tgks[i].kernstr;
            set_offsets_to_uint(kernel_clstring, 1);

            kernelInfo.kernel_name = soln.v_tgks[i].fname;
            local_work_size        = soln.v_tgks[i].local_work_size;
            global_work_size       = soln.v_tgks[i].global_work_size;

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
            _kernels.push_back(kptr_atomic);
            i++;
        }

        // jn : the main kernel is at the back of the solution vector
        kernel_clstring = soln.v_tgks[i].kernstr;
        set_offsets_to_uint(kernel_clstring, 3);

        kernelInfo.kernel_name = soln.v_tgks[i].fname;
        local_work_size        = soln.v_tgks[i].local_work_size;
        global_work_size       = soln.v_tgks[i].global_work_size;

        kernelInfo.kernel_file = kernel_clstring;

        kernelInfo.l_wk        = {local_work_size, 1, 1};
        kernelInfo.g_wk        = {global_work_size, 1, 1};
        kernelInfo.wk_dim      = 1;
        kernelInfo.kernel_type = SOURCE;

        AMDKernelPtr kptr_normal = CreateKernel(inputs[0]->device_id(), &kernelInfo);
        _kernel_normal           = kptr_normal;
        _kernels.push_back(kptr_normal);

        //kernelInfo.kernel_type = MIOPEN;
        kernelInfo.kernel_file = "MIOpenUtilKernels2.cl";
        kernelInfo.kernel_name = "Col2ImBiasRelu";
        kernelInfo.kernel_type = MIOPEN;
        //kernelInfo.kernel_type = SABER;
        kernelInfo.wk_dim      = 1;
        kernelInfo.l_wk        = {256, 1, 1};
        kernelInfo.g_wk        = {
            param.weight()->num()* outputs[0]->height()* outputs[0]->width(), 1, 1
        };

        AMDKernelPtr kptr_col2Im_bias_relu = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr_col2Im_bias_relu.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernel_col2Im_bias_relu = kptr_col2Im_bias_relu;
        _kernels.push_back(kptr_col2Im_bias_relu);

    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}
template <DataType OpDtype>

SaberStatus SaberDeconv2D<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param) {
    bool err;
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch";
    amd_kernel_list list;

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    int isBias    = (param.bias()->size() > 0) ? 1 : 0;
    int relu_flag = (param.activation_param.active == Active_relu) ? 1 : 0;

    if (param.weight()->height() == 1 && param.weight()->width() == 1 && param.stride_h == 1
            && param.stride_w == 1 && param.pad_h == 0 && param.pad_w == 0) {
        cl_uint uintObjects[3]   = {0, 0, 0};
        cl_float floatObjects[2] = {1.0f, 0.0f};
        cl_uint im_offset        = 0;
        cl_uint out_offset       = 0;
        cl_mem memObjects[4]     = {(cl_mem)inputs[0]->data(),
                                    (cl_mem)param.weight()->data(),
                                    (cl_mem)outputs[0]->mutable_data(),
                                    (cl_mem)param.bias()->data()
                                   };
        cl_mem gemmWrokspace     = (cl_mem)_outGemmWorkspace->mutable_data();
        int j                    = 0;

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

            err = kernel->SetKernelArgs(
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

            if (!err) {
                LOG(ERROR) << "Fialed to set execution.";
                return SaberInvalidValue;
            }

            list.clear();
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

        if (!err) {
            LOG(ERROR) << "Fialed to set execution.";
            return SaberInvalidValue;
        }

        list.clear();
    } else // param.weight()->height() != 1 && param.weight()->width() != 1 && param.stride_h != 1
        // && param.stride_w != 1
    {

        cl_uint uintObjects[3]   = {0, 0, 0};
        cl_float floatObjects[2] = {1.0f, 0.0f};
        cl_uint im_offset        = 0;
        cl_uint out_offset       = 0;
        cl_mem memObjects[4]     = {(cl_mem)inputs[0]->data(),
                                    (cl_mem)param.weight()->data(),
                                    (cl_mem)outputs[0]->mutable_data(),
                                    (cl_mem)param.bias()->data()
                                   };
        cl_mem gemmWrokspace     = (cl_mem)_outGemmWorkspace->mutable_data();

        int j = 0;
        AMDKernel* kernel;

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

            kernel = _kernel_normal.get();
            kernel->SetKernelArgs(
                (PtrDtype)memObjects[1],
                (int)uintObjects[1],
                (PtrDtype)memObjects[0],
                (int)uintObjects[0],
                (PtrDtype)_outCol2ImSpace->mutable_data(),
                (int)uintObjects[1],
                (float)floatObjects[0],
                (float)floatObjects[1]);

            list.push_back(_kernel_normal);

            kernel = _kernel_col2Im_bias_relu.get();
            kernel->SetKernelArgs(
                (PtrDtype)_outCol2ImSpace->mutable_data(),
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

            if (!err) {
                LOG(ERROR) << "Fialed to set execution.";
                return SaberInvalidValue;
            }

            list.clear();
        }

    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class SaberDeconv2D<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, AMD, AK_HALF);
} // namespace saber

} // namespace anakin
