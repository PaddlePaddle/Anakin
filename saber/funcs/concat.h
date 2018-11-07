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
#ifndef ANAKIN_SABER_FUNCS_CONCAT_H
#define ANAKIN_SABER_FUNCS_CONCAT_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_concat.h"

#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_concat.h"
#endif

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_concat.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_concat.h"
#endif

#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/saber_concat.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>

class Concat : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        ConcatParam
> {
public:
    using BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        ConcatParam>::BaseFunc;

    Concat() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef ConcatParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        unsigned long input_size = input.size();

        Shape_v shapes_in;
        shapes_in.resize(input_size);
        //! get input size
        for (int i = 0; i < input_size; i++){
            shapes_in[i] = input[i]->valid_shape();
        }

        Shape shape_out = shapes_in[0];

        //! compute output shape
        for (int i = 1; i < input_size; ++i) {
            Shape sh = shapes_in[i];
            for (int j = 0; j < sh.dims(); ++j) {
                if (j == param.axis) { continue; }
                else if (sh[j] != -1) {
                            CHECK_EQ(shape_out[j], sh[j]) \
                        << "All inputs must have the same shape, except at concat_axis.";
                } else {
                    sh[j] = shape_out[j];
                    SABER_CHECK(input[i]->set_shape(sh));
                }
            }
            shape_out[param.axis] += sh[param.axis];
        }
        output[0]->set_seq_offset(input[0]->get_seq_offset());
        return output[0]->set_shape(shape_out);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderConcat <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberConcat <TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        if (true) // some condition?
            this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

} // namespace saber
} // namespace anakin


#endif
