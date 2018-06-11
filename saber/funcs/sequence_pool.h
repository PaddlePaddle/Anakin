/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_SEQUENCE_POOL_H
#define ANAKIN_SABER_FUNCS_SEQUENCE_POOL_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/impl_sequence_pool.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_sequence_pool.h"
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
class SequencePool : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        SequencePoolParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            SequencePoolParam>::BaseFunc;

    SequencePool() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef SequencePoolParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, \
        Output_v &output, Param_t& param) override {
        Shape output_shape = (input[0]->valid_shape());
        int num_idx = input[0]->num_index();
        std::vector<int> offset = input[0]->get_seq_offset();
        //CHECK_GT(offset.size(), 1) << "seq num error! " << offset.size();
        if (offset.size() > 1) {
            output_shape[num_idx] = offset.size() - 1;
        } else {
            output_shape[num_idx] = input[0]->num();
        }
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderSequencePool <TargetType, OpDtype, inDtype, outDtype,
                        LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            case SABER_IMPL:
                this->_impl.push_back(new SaberSequencePool <TargetType, OpDtype, inDtype, outDtype,
                        LayOutType_op, LayOutType_in, LayOutType_out>);
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
}
}
#endif
