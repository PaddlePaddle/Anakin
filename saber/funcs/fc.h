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

#ifndef ANAKIN_SABER_FUNCS_FC_H
#define ANAKIN_SABER_FUNCS_FC_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_fc.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_fc.h"
#include "saber/funcs/impl/cuda/vender_fc.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/vender_fc.h"
#endif

#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/saber_fc.h"
#endif

#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/vender_fc.h"
#endif

namespace anakin{

namespace saber {

template<typename TargetType, DataType OpDtype>
class Fc : public BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    FcParam > {
public:
    using BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    FcParam >::BaseFunc;

    Fc() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef FcParam<TargetType> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
            Param_t& param) override {

        int m = input[0]->count_valid(0, param.axis);
        int k = input[0]->count_valid(param.axis, input[0]->dims());
        int n = param.num_output;
        int weights_size = param.weights->valid_size();

        if (n <= 0) {
            n = weights_size / k;
        }

        CHECK_EQ(weights_size / n, k) << "weights size does not meet the input size";

        Shape shape_out({m, n, 1, 1}, Layout_NCHW);
        output[0]->set_seq_offset(input[0]->get_seq_offset());
        return output[0]->set_shape_without_layout(shape_out);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
        case VENDER_IMPL:
            this->_impl.push_back(new VenderFc<TargetType, OpDtype>);
            return SaberSuccess;

        case SABER_IMPL:
            this->_impl.push_back(new SaberFc<TargetType, OpDtype>);
            return SaberSuccess;

        default:
            return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        if (std::is_same<TargetType, NV>::value) {
            bool use_saber_fc = true;
            use_saber_fc &= this->_last_input_shape[0][0] > 1;
            use_saber_fc &= this->_last_input_shape[0][0] <= 32;

            if (use_saber_fc) {
                this->_best_impl = this->_impl[1];
            } else {
                this->_best_impl = this->_impl[0];
            }
        } else {
            this->_best_impl = this->_impl[0];
        }
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_FC_H
