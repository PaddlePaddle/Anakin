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

#ifndef ANAKIN_SABER_FUNCS_PAD_H
#define ANAKIN_SABER_FUNCS_PAD_H

#include "saber/funcs/impl/impl_pad.h"

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_pad.h"
#endif
#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_pad.h"
#endif


namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype
>
class Pad : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        PadParam
> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            PadParam>::BaseFunc;

    Pad() = default;

    typedef PadParam<TargetType> Param_t;
    typedef std::vector<Tensor<TargetType> *> Input_v;
    typedef std::vector<Tensor<TargetType> *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {
        SaberStatus status;
        CHECK_EQ(2, param.pad_c.size());
        CHECK_EQ(2, param.pad_h.size());
        CHECK_EQ(2, param.pad_w.size());
        Shape output_shape = input[0]->valid_shape();
        int c_id = input[0]->channel_index();
        int h_id = input[0]->height_index();
        int w_id = input[0]->width_index();
        output_shape[c_id] += param.pad_c[0] + param.pad_c[1];
        output_shape[h_id] += param.pad_h[0] + param.pad_h[1];
        output_shape[w_id] += param.pad_w[0] + param.pad_w[1];
        output[0]->set_shape(output_shape);

        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderPad<TargetType,OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberPad<TargetType,OpDtype>);
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

#endif //ANAKIN_SABER_FUNCS_PAD_H
