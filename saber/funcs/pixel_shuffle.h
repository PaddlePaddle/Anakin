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

#ifndef ANAKIN_SABER_FUNCS_PIXEL_SHUFFLE_H
#define ANAKIN_SABER_FUNCS_PIXEL_SHUFFLE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_pixel_shuffle.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_pixel_shuffle.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_pixel_shuffle.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType, DataType OpDtype>
class PixelShuffle : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        PixelShuffleParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            PixelShuffleParam>::BaseFunc;

    PixelShuffle() = default;

    typedef PixelShuffleParam<TargetType> Param_t;
    typedef std::vector<Tensor<TargetType> *> Input_v;
    typedef std::vector<Tensor<TargetType> *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {

        int rh = param.rh;
        int rw = param.rw;

        Shape in_shape = input[0]->valid_shape();
        Shape out_shape = in_shape;
        int in_c = in_shape.channel();
        CHECK_EQ(in_c%(rw*rh), 0) << "input channel must mod rw*rh to 0";
   
        int oc = in_c/(rw*rh);
        int oh = in_shape.height() * rh;
        int ow = in_shape.width() * rw;
        

        if (param.channel_first){
            out_shape[1] = oc;
            out_shape[2] = oh;
            out_shape[3] = ow; 
        } else {
            out_shape[1] = oh;
            out_shape[2] = ow;
            out_shape[3] = oc; 
        }

        return output[0] -> set_shape(out_shape);

    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderPixelShuffle <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberPixelShuffle <TargetType, OpDtype>);
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

#endif //ANAKIN_SABER_FUNCS_PERMUTE_H
