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

#ifndef ANAKIN_SABER_FUNCS_TRANSPOSE_H
#define ANAKIN_SABER_FUNCS_TRANSPOSE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_transpose.h"

#ifdef NVIDIA_GPU
//#include "saber/funcs/impl/cuda/saber_transpose.h"
#endif

namespace anakin{

namespace saber{

template <typename TargetType,
        DataType OpDtype
>
class Transpose : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        TransposeParam>
{
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            TransposeParam >::BaseFunc;
    Transpose() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef TransposeParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, \
        Output_v &output, Param_t& param) override {

        Shape output_shape = input[0]->valid_shape();

        output_shape.set_num(input[0]->num()); // N
        output_shape.set_channel(input[0]->channel()); // C
        output_shape.set_height(input[0]->width());
        output_shape.set_width(input[0]->height());

        return output[0]->set_shape(output_shape);

    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) { 
            case VENDER_IMPL: 
                this->_impl.push_back(new VenderTranspose <TargetType,
                        OpDtype>);
                return SaberSuccess; 
            case SABER_IMPL: 
                this->_impl.push_back(new SaberTranspose <TargetType,
                        OpDtype>);
                return SaberSuccess; 
            default: 
                return SaberUnImplError;
        }        
    }

private:

    virtual void pick_best_static() override {
        //! Transpose only has saber implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        //! Transpose only has saber implementation
        this->_best_impl = this->_impl[0];
    }

};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_TRANSPOSE_H