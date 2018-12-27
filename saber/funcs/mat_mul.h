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

#ifndef ANAKIN_SABER_FUNCS_MAT_MUL_H
#define ANAKIN_SABER_FUNCS_MAT_MUL_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_mat_mul.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_mat_mul.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/vender_mat_mul.h"
#endif

#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/vender_mat_mul.h"
#endif

namespace anakin{

namespace saber{

template<typename TargetType, DataType OpDtype>
class MatMul : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        MatMulParam> {
public:
    using BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        MatMulParam>::BaseFunc;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef MatMulParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    MatMul() = default;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override 
    {
        CHECK_EQ(input.size(), 2);
        CHECK_EQ(input[0]->num(), input[1]->num());
        CHECK_EQ(input[0]->channel(), input[1]->channel());
        Shape shape_output = input[0]->valid_shape();
        int M,N,K0,K1,B;
//        LOG(INFO)<<"input0 shape:";
//        LOG(INFO)<<"n:"<<input[0]->num()<<"c:"<<input[0]->channel()<<"h:"<<input[0]->height()<<"w:"<<input[0]->width();
//        LOG(INFO)<<"input1 shape:";
//        LOG(INFO)<<"n:"<<input[1]->num()<<"c:"<<input[1]->channel()<<"h:"<<input[1]->height()<<"w:"<<input[1]->width();
        if (param._is_transpose_X)
        {
            K0 = input[0]->height();
            M = input[0]->width();
        }else{
            M = input[0]->height();
            K0 = input[0]->width();
        }

        if (param._is_transpose_Y)
        {
            N = input[1]->height();
            K1 = input[1]->width();
        }else{
            K1 = input[1]->height();
            N = input[1]->width();
        }
        CHECK_EQ(K0, K1);

        param._b = input[0]->num() * input[0]->channel();
        param._m = M;
        param._n = N;
        param._k = K0;
        return output[0]->set_shape(Shape({input[0]->num(), input[0]->channel(), M, N}));
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderMatMul <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberMatMul <TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;            
        }
    }

private:

    virtual void pick_best_static() override {
        //! Fc only has saber implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_runtime(const Input_v input, Output_v output, Param_t& param, \
        Context<TargetType> &ctx) {
        this->_best_impl = this->_impl[0];
    }
    virtual void pick_best_specify(ImplEnum implenum) override {
        //! Fc only has saber implementation
        this->_best_impl = this->_impl[0];
    }

};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_MAT_MUL_H
