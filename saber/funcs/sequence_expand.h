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

#ifndef ANAKIN_SABER_FUNCS_SEQUENCE_EXPAND_H
#define ANAKIN_SABER_FUNCS_SEQUENCE_EXPAND_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_activation.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_sequence_expand.h"
#endif

#ifdef USE_X86_PLACE
//#include "saber/funcs/impl/x86/saber_activation.h"
#endif

#ifdef USE_ARM_PLACE
//#include "saber/funcs/impl/arm/saber_activation.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_FLOAT,
        DataType outDtype = AK_FLOAT,
        typename LayOutType_op = NCHW,
        typename LayOutType_in = NCHW,
        typename LayOutType_out = NCHW
>
class SequenceExpand : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        SequenceExpandParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            SequenceExpandParam>::BaseFunc;

    SequenceExpand() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef SequenceExpandParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {

        Shape output_shape = input[0]->valid_shape();
        CHECK_EQ(input.size() == 2) << "sequence expand need two input but " << input.size << "is provided";
        Shape in_shape = input[0]->valid_shape();
        auto input_seq_offset = input[0]->get_seq_offset();
        auto ref_seq_offset = input[1]->get_seq_offset();
        if (input_seq_offset.size() == 0) {
            out_shape = in_shape;
            out_shape[0] = input_seq_offset[input_seq_offset.size() - 1];
            output[0]->set_seq_offset(ref_seq_offset);
            
        } else {
            CHECK_EQ(input_seq_offset.size(), ref_seq_offset.size()) <<"input and ref sequence offset must have the same size";
            int cum = 0;
            std::vector<int> off;
            off.push_back(cum);
            for (int i = 1; i <= ref_seq_offset.size(); i++) {
                cum += (ref_seq_offset[i] - ref_seq_offset[i - 1]) * (input_seq_offset[i] - input_seq_offset[i - 1]);
                off.push_back(cum);
            }
            out_shape[0] = cum;
            output[0]->set_seq_offset(off);
            
        }

        
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                //this->_impl.push_back(new VenderSequenceExpand <TargetType,
                //this->_impl.push_back(new VenderSequenceExpand <TargetType,
                //        OpDtype, inDtype, outDtype,
                //        LayOutType_op, LayOutType_in, LayOutType_out>);
                //return SaberSuccess;
                return SaberUnImplError;

            case SABER_IMPL:
                this->_impl.push_back(new SaberSequenceExpand <TargetType,
                        OpDtype, inDtype, outDtype,
                        LayOutType_op, LayOutType_in, LayOutType_out>);
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
