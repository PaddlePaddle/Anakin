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

#ifndef ANAKIN_SABER_FUNCS_PS_ROI_POOLING_H
#define ANAKIN_SABER_FUNCS_PS_ROI_POOLING_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_ps_roi_pooling.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_ps_roi_pooling.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_ps_roi_pooling.h"
#endif
namespace anakin {
namespace saber {

template <typename TargetType,
        DataType OpDtype
>
class PsRoiPool : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        PsRoiPoolParam>
{
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            PsRoiPoolParam >::BaseFunc;

    PsRoiPool() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef PsRoiPoolParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, \
        Output_v &output, Param_t& param) override {

        CHECK_GE(input.size(), 2) << "psroipooling input must equal or greater than 2";

        Shape in_sh = input[0]->valid_shape();
        int rois_num = input[1]->num();
        Shape out_sh = in_sh;

        int size = param.pooled_width * param.pooled_height;
        CHECK_EQ(in_sh.channel()%size, 0);

        int new_c = in_sh.channel() / size;

        if (!param.global_pooling){
            out_sh.set_width(param.pooled_width);
            out_sh.set_height(param.pooled_height);
        } else {
            out_sh.set_width(1);
            out_sh.set_height(1);
        }
        out_sh.set_channel(new_c);
        out_sh.set_num(rois_num);

        return output[0]->set_shape(out_sh);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) { 
            case VENDER_IMPL: 
                this->_impl.push_back(new VenderPsRoiPool <TargetType,
                        OpDtype>);
                return SaberSuccess; 
            case SABER_IMPL: 
                this->_impl.push_back(new SaberPsRoiPool <TargetType,
                        OpDtype>);
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

#endif //ANAKIN_SABER_FUNCS_CROP_H
