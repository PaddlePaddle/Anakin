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
#ifndef ANAKIN_SABER_FUNCS_FUSION_H
#define ANAKIN_SABER_FUNCS_FUSION_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_fusion.h"

#ifdef USE_MLU
#include "saber/funcs/impl/mlu/saber_fusion.h"
#endif  // USE_MLU

#ifdef USE_BM_PLACE
#include "saber/funcs/impl/bm/saber_fusion.h"
#endif
namespace anakin {
namespace saber {

//fusion op: only used for dispatching.
template<typename TargetType,
        DataType OpDtype >
class Fusion : public BaseFunc<
        TargetType,
        OpDtype, 
        ImplBase, 
        FusionParam > {
public:
    Fusion() = default;
    
    typedef Tensor<TargetType> DataTensor;
    typedef FusionParam<TargetType> Param_t;
    typedef std::vector<DataTensor *> Input_v;
    typedef std::vector<DataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input,
                                             Output_v& output, Param_t& param) override {

        // fusion op do not need do anything in this function!
        return SaberSuccess; 
    }
    
    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderFusion <TargetType,
                        OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberFusion <TargetType,
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

} // namespace saber
} // namespace anakin


#endif
