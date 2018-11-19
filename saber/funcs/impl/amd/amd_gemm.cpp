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
#include "saber/funcs/impl/amd/include/amd_utils.h"
#include "saber/funcs/impl/amd/include/amd_gemm.h"
#include "saber/funcs/conv.h"

namespace anakin {
namespace saber {

bool findGenericGemm(bool fromSolver, std::vector<AMDKernelPtr>& vkptr,
                     const std::vector<Tensor<AMD>*>& inputs,
                     Tensor<AMD>*& output,
                     ConvParam<AMD>& param,
                     Tensor<AMD>*& workspace,
                     Context<AMD>& ctx,
                     bool& needBiasRelu) {

    KernelInfo kernelInfo;
    bool _multikernel = false;
    bool isBias             = (param.bias()->size() > 0) ? true : false;
    AMDKernelPtr kptr;
    needBiasRelu = false;

    if (fromSolver) {
        if ((inputs[0]->num() > 1
                && inputs[0]->width() <= 14 && param.stride_h == 1)
                || (param.stride_h == 2)) {
            ALOGD("GEMM 1x1, 14x14");
            int K       = (inputs[0]->channel());
            int M       = (param.weight()->num());
            int N       = (inputs[0]->num()) * (output->height()) * (output->width());
            float alpha = 1.0f;
            float beta  = 0.0f;
            bool tA     = false;
            bool tB     = false;
            bool tC     = false;
            int lda     = K;
            int ldb     = N;
            int ldc     = N;

            MIOpenGEMM::Geometry tgg {};
            tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');

            /////////////////////////////////////////////////////////////
            // transpose_NCHW2CNHW kernel
            transpose_NCHW2CNHW(
                kptr,
                inputs[0]->device_id(),
                (inputs[0]->num()),
                (inputs[0]->channel()),
                (inputs[0]->height()),
                (inputs[0]->width()),
                (output->height()),
                (output->width()),
                0,
                0,
                param.stride_h,
                param.stride_w);

            if (!kptr.get()->isInit()) {
                ALOGE("Failed to create kernel");
                return false;
            }

            vkptr.push_back(kptr);

            AMD_API::stream_t cm = ctx.get_compute_stream();

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
                                            (PtrDtype)param.weight()->data(),
                                            (PtrDtype)workspace->mutable_data(),
                                            false,
                                            tgg,
                                            miopengemm_verbose,
                                            miopengemm_warnings);

            if (soln.v_tgks.size() == 2) {
                _multikernel = true;
            }

            for (int i = 0; i < soln.v_tgks.size(); i++) {
                // jn : the main kernel is at the back of the solution vector
                std::string kernel_clstring = soln.v_tgks[i].kernstr;

                if (i == soln.v_tgks.size() - 1) {
                    tempfix::set_offsets_to_uint(kernel_clstring, 3);
                } else {
                    tempfix::set_offsets_to_uint(kernel_clstring, 1);
                }

                kernelInfo.kernel_name     = soln.v_tgks[i].fname;
                std::string network_config = tgg.get_networkconfig_string();
                size_t local_work_size     = soln.v_tgks[i].local_work_size;
                size_t global_work_size    = soln.v_tgks[i].global_work_size;

                kernelInfo.kernel_file  = kernel_clstring;
                kernelInfo.l_wk         = {local_work_size, 1, 1};
                kernelInfo.g_wk         = {global_work_size, 1, 1};
                kernelInfo.comp_options = "";
                kernelInfo.kernel_type  = SOURCE;

                kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    ALOGE("Failed to create kernel");
                    return false;
                }

                vkptr.push_back(kptr);
            }

            /////////////////////////////////////////////////////////////
            // transpose_CNHW2NCHW kernel
            size_t _x_t_size = (inputs[0]->num()) * (inputs[0]->channel())
                               * (output->height()) * (output->width());

            transpose_CNHW2NCHW(
                kptr,
                inputs[0]->device_id(),
                (inputs[0]->num()),
                (param.weight()->num()),
                (output->height()),
                (output->width()),
                (output->height()),
                (output->width()),
                _x_t_size,
                0,
                1,
                1,
                isBias);

            if (!kptr.get()->isInit()) {
                ALOGE("Failed to create kernel");
                return false;
            }

            vkptr.push_back(kptr);

        } else {
            ALOGD("GEMM 1x1");
            int K = (inputs[0]->channel()) * (param.weight()->height())
                    * (param.weight()->width());
            int M       = (param.weight()->num());
            int N       = (output->height()) * (output->width());
            float alpha = 1.0;
            float beta  = 0.0;
            bool tA     = false;
            bool tB     = false;
            bool tC     = false;
            int lda     = K;
            int ldb     = N;
            int ldc     = N;

            MIOpenGEMM::Geometry tgg {};
            tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
            AMD_API::stream_t cm = ctx.get_compute_stream();

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
                                            (PtrDtype)param.weight()->data(),
                                            (PtrDtype)workspace->mutable_data(),
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
                    ALOGE("Failed to create kernel");
                    return false;
                }

                vkptr.push_back(kptr);

                i++;
            }

            // jn : the main kernel is at the back of the solution vector
            kernel_clstring = soln.v_tgks[i].kernstr;
            tempfix::set_offsets_to_uint(kernel_clstring, 3);

            if (!_multikernel && inputs[0]->num() == 1) {
                if (isBias) {
                    tempfix::add_bias_relu(kernel_clstring);
                } else {
                    tempfix::add_relu(kernel_clstring);
                }
            } else {
                needBiasRelu = true;
            }

            kernelInfo.kernel_name = soln.v_tgks[i].fname;
            local_work_size        = soln.v_tgks[i].local_work_size;
            global_work_size       = soln.v_tgks[i].global_work_size;

            kernelInfo.kernel_file = kernel_clstring;
            kernelInfo.l_wk        = {local_work_size, 1, 1};
            kernelInfo.g_wk        = {global_work_size, 1, 1};
            kernelInfo.kernel_type = SOURCE;

            // To create the program
            kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                ALOGE("Failed to create kernel");
                return false;
            }

            vkptr.push_back(kptr);
        }
    } else {
        ALOGD("Not GEMM 1x1");
        needBiasRelu = true;
        int K = (inputs[0]->channel()) * (param.weight()->height())
                * (param.weight()->width());
        int M       = (param.weight()->num());
        int N       = (output->height()) * (output->width());
        float alpha = 1.0;
        float beta  = 0.0;
        bool tA     = false;
        bool tB     = false;
        bool tC     = false;
        int lda     = K;
        int ldb     = N;
        int ldc     = N;

        MIOpenGEMM::Geometry tgg {};
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');

        AMD_API::stream_t cm = ctx.get_compute_stream();

        /////////////////////////////////////////////////////////////
        // gemm kernel
        // jn : print search results to terminal
        bool miopengemm_verbose = false;

        // jn : print warning messages when the returned kernel(s) might be sub-optimal
        bool miopengemm_warnings = false;

        Im2ColGPU(
            kptr,
            inputs[0]->device_id(),
            inputs[0]->channel(),
            inputs[0]->height(),
            inputs[0]->width(),
            param.weight()->height(),
            param.weight()->width(),
            output->height(),
            output->width(),
            param.pad_h,
            param.pad_w,
            param.stride_h,
            param.stride_w,
            param.dilation_h,
            param.dilation_w);

        if (!kptr.get()->isInit()) {
            ALOGE("Failed to create kernel");
            return SaberInvalidValue;
        }

        vkptr.push_back(kptr);

        // jn : find with no workspace
        MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                        0.003f,
                                        cm,
                                        (PtrDtype)inputs[0]->data(),
                                        (PtrDtype)param.weight()->data(),
                                        (PtrDtype)workspace->mutable_data(),
                                        false,
                                        tgg,
                                        miopengemm_verbose,
                                        miopengemm_warnings);

        std::string kernel_clstring;
        size_t local_work_size;
        size_t global_work_size;
        int errCode;

        int i                   = 0;
        kernelInfo.comp_options = "";

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
                ALOGE("Failed to create kernel");
                return SaberInvalidValue;
            }

            vkptr.push_back(kptr);

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

        // To create the program
        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            ALOGE("Failed to create kernel");
            return SaberInvalidValue;
        }

        vkptr.push_back(kptr);
    }

    kptr = nullptr;
    return true;

}

} // namespace saber
} // namespace anakin
