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
#ifndef ANAKIN_SABER_FUNCS_DECONV_H
#define ANAKIN_SABER_FUNCS_DECONV_H

#include "saber/funcs/base.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_deconv.h"

#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/vender_deconv.h"
#endif
#ifdef USE_CUDA
#include "saber/funcs/impl/cuda/saber_deconv.h"
#include "saber/funcs/impl/cuda/vender_deconv.h"
#endif
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_deconv.h"
//#include "saber/funcs/impl/x86/vender_deconv.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class Deconv : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        ConvParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            ConvParam>::BaseFunc;

    Deconv() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef ConvParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input, \
        Output_v &output, Param_t &param) override {

        Shape deconv_shape = deconv_compute_shape(input[0]->valid_shape(), param);
        deconv_shape.set_layout(Layout_NCHW);
        return output[0]->set_shape(deconv_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderDeconv2D <TargetType,
                        OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberDeconv2D <TargetType,
                        OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }
    SaberStatus trans_weights(Tensor<TargetType> &target_weights,
                              Tensor<TargetType> &target_bias,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dilation_h, int dilation_w,
                              int group,
                              ImplEnum implenum) {
        if (implenum == VENDER_IMPL) {
            return static_cast<VenderDeconv2D<TargetType, OpDtype> *>(this->_best_impl)->trans_weights(
                    target_weights, target_bias, stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, group);
        } else if (implenum == SABER_IMPL) {
            return static_cast<SaberDeconv2D<TargetType, OpDtype> *>(this->_best_impl)->trans_weights(
                    target_weights, target_bias, stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, group);
        } else {
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
