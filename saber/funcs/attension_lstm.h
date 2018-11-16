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

#ifndef ANAKIN_SABER_FUNCS_ATTENSION_LSTM_H
#define ANAKIN_SABER_FUNCS_ATTENSION_LSTM_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_attension_lstm.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_attension_lstm.h"
//#include "saber/funcs/impl/cuda/vender_attension_lstm.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_attension_lstm.h"
//#include "saber/funcs/impl/x86/vender_attension_lstm.h"
#endif
#ifdef USE_ARM_PLACE
//todo
//#include "saber/funcs/impl/impl_attension_lstm.h"
#endif
namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype
         >
class AttensionLstm : public BaseFunc <
        TargetType,
        OpDtype,
        ImplBase,
    AttensionLstmParam
    > {
public:
    using BaseFunc <
            TargetType,
            OpDtype,
            ImplBase,
            AttensionLstmParam >::BaseFunc;

    AttensionLstm() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef AttensionLstmParam<TargetType> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;


    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
            Param_t& param) override {

        CHECK_GE(input.size(), 1) << "input must >= 1";

        int hidden_size = param.lstm_param.bias()->valid_size() / 4;

        int word_num = input[0]->num();

        Shape output_shape({word_num, hidden_size * param.lstm_param.num_direction, 1, 1});
        output[0]->set_shape(output_shape);
        output[0]->set_seq_offset(input[0]->get_seq_offset());
        return SaberSuccess;
    }


    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
        case VENDER_IMPL:
            return SaberUnImplError;

        case SABER_IMPL:
            this->_impl.push_back(new SaberAttensionLstm <TargetType, OpDtype>);
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

}
}
#endif //ANAKIN_SABER_FUNCS_ATTENSION_LSTM_H