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

#ifndef ANAKIN_SABER_FUNCS_POWER_H
#define ANAKIN_SABER_FUNCS_POWER_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_power.h"
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_power.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_power.h"
#endif
#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_power.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class Power : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        PowerParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            PowerParam>::BaseFunc;

    Power() = default;

    typedef PowerParam<TargetType> Param_t;
    typedef std::vector<Tensor<TargetType> *> Input_v;
    typedef std::vector<Tensor<TargetType> *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
        Param_t& param) override {
        SaberStatus status;
        for (int j = 0; j < input.size(); ++j) {
            Shape output_shape = input[j]->valid_shape();
            output[j]->set_shape(output_shape);
        }

        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderPower <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberPower <TargetType, OpDtype>);
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

#endif //ANAKIN_SABER_FUNCS_POWER_H
