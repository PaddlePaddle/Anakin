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
#ifndef ANAKIN_SABER_FUNCS_DETECTION_OUTPUT_H
#define ANAKIN_SABER_FUNCS_DETECTION_OUTPUT_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_detection_output.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_detection_output.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_detection_output.h"
#endif
#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_detection_output.h"
#endif
namespace anakin {
namespace saber {

template < typename TargetType,
         DataType OpDtype >
class DetectionOutput : public BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
        DetectionOutputParam > {
public:
    using BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    DetectionOutputParam >::BaseFunc;

    DetectionOutput() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef DetectionOutputParam<TargetType> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, \
            Output_v& output, Param_t& param) override {
        Shape shape_out = Shape({1, 1, param.keep_top_k* input[0]->num(), 7}, Layout_NCHW);
        return output[0]->set_shape(shape_out);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
        case VENDER_IMPL:
            this->_impl.push_back(new VenderDetectionOutput < TargetType,
                                  OpDtype >);
            return SaberSuccess;

        case SABER_IMPL:
            this->_impl.push_back(new SaberDetectionOutput < TargetType,
                                  OpDtype >);
            return SaberSuccess;

        default:
            return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        if (true) { // some condition?
            this->_best_impl = this->_impl[0];
        }
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

} // namespace saber
} // namespace anakin


#endif
