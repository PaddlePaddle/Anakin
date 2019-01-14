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

#ifndef ANAKIN_SABER_FUNCS_SEQUENCE_CONCAT_H
#define ANAKIN_SABER_FUNCS_SEQUENCE_CONCAT_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_sequence_concat.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_sequence_concat.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_sequence_concat.h"
#endif

#ifdef AMD_GPU 
//#include "saber/funcs/impl/amd/saber_sequence_concat.h"
#endif

#ifdef USE_ARM_PLACE
//#include "saber/funcs/impl/arm/saber_sequence_concat.h"
#endif

#ifdef USE_BM_PLACE 
//#include "saber/funcs/impl/bm/vender_sequence_concat.h"
#endif


namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class SequenceConcat : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        SequenceConcatParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            SequenceConcatParam>::BaseFunc;

    SequenceConcat() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef SequenceConcatParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        Shape output_shape = (input[0]->valid_shape());
        CHECK_EQ(input[0]->num_index(), 0) << "num index must be zero";
        for (int i = 1; i < input.size(); i++) {
            output_shape[0] += input[i]->num();
        }
        std::vector<std::vector<int>> out_offset;
        out_offset.resize(1);
        int seq_len = input[0]->get_seq_offset()[0].size() - 1;
        out_offset[0].push_back(0);
        int cur_off = 0;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < input.size(); j++) {
                cur_off += input[j]->get_seq_offset()[0][i + 1];
            }
            out_offset[0].push_back(cur_off);
        }
        
        output[0]->set_seq_offset(out_offset);
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderSequenceConcat <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberSequenceConcat <TargetType, OpDtype>);
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
