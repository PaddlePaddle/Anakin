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

#ifndef ANAKIN_SABER_FUNCS_ALIGNED_MAT_MUL_H
#define ANAKIN_SABER_FUNCS_ALIGNED_MAT_MUL_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_aligned_mat_mul.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_aligned_mat_mul.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_aligned_mat_mul.h"
#endif

#ifdef AMD_GPU 
//#include "saber/funcs/impl/amd/saber_aligned_mat_mul.h"
#endif

#ifdef USE_ARM_PLACE
//#include "saber/funcs/impl/arm/saber_aligned_mat_mul.h"
#endif

#ifdef USE_BM_PLACE 
//#include "saber/funcs/impl/bm/vender_aligned_mat_mul.h"
#endif


namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class AlignedMatMul : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        AlignedMatMulParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            AlignedMatMulParam>::BaseFunc;

    AlignedMatMul() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef AlignedMatMulParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {

        auto seq_offset_0 = input[0]->get_seq_offset()[0];
        auto seq_offset_1 = input[1]->get_seq_offset()[0];
        int seq_num = seq_offset_0.size() - 1;
        int inner_size_A = input[0]->count_valid(1, input[0]->dims());
        int inner_size_B = input[1]->count_valid(1, input[1]->dims());
        int batch_A = seq_offset_0[1];
        int batch_B = seq_offset_1[1];
        int M = param.is_transpose_X ? inner_size_A : batch_A;
        int N = param.is_transpose_Y ? batch_B : inner_size_B;
        Shape output_shape({seq_num * M, N, 1, 1}, Layout_NCHW);
        output[0]->set_seq_offset(input[0]->get_seq_offset());
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderAlignedMatMul <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberAlignedMatMul <TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};



} // namespace saber
} // namespace anakin

#endif
