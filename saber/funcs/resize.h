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

#ifndef ANAKIN_SABER_FUNCS_RESIZE_H
#define ANAKIN_SABER_FUNCS_RESIZE_H

#include "saber/core/tensor.h"
#include "saber/funcs/base.h"
#include "saber/funcs/timer.h"   
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_define.h"

namespace anakin{

namespace saber{

template <typename TargetType,
    DataType OpDtype,
    DataType inDtype = AK_FLOAT,
    DataType outDtype = AK_FLOAT,
    typename LayOutType_op = NCHW,
    typename LayOutType_in = NCHW,
    typename LayOutType_out = NCHW
    >
class Resize : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        ResizeParam>
{
public:
    using BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        ResizeParam >::BaseFunc;
    Resize() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef ResizeParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, \
        Output_v &output, Param_t& param) override {

        Shape output_shape = input[0]->valid_shape();

        int num_idx = input[0]->num_index();
        int channel_idx = input[0]->channel_index();
        int height_idx = input[0]->height_index();
        int width_idx = input[0]->width_index();

        CHECK_GE(height_idx, 0) << "no height dim in tensor";
        CHECK_GE(width_idx, 0) << "no width dim in tensor";

        if (num_idx > -1) {
            output_shape[num_idx] = input[0]->num(); // N
        }
        if (channel_idx > -1) {
            output_shape[channel_idx] = input[0]->channel(); // C
        }
        if (height_idx > -1) {
            int height = floor(input[0]->height() * param.height_scale); // H
            output_shape[height_idx] = height;
        }
        if (width_idx > -1) {
            int width = floor(input[0]->width() * param.width_scale); //W
            output_shape[width_idx] = width;
        }

        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) { 
            case VENDER_IMPL: 
                return SaberUnImplError; 
            case SABER_IMPL: 
                this->_impl.push_back(new SaberResize<TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>); 
                return SaberSuccess;
            default: 
                return SaberUnImplError; 
        } 
        return SaberSuccess;
    };

private:

    virtual void pick_best_static() override {
        //! resize only has saber implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_runtime(Input_v input, Output_v output, \
        Param_t& param, Context<TargetType> &ctx) override {
        //! resize only has saber implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        //! resize only has saber implementation
        this->_best_impl = this->_impl[0];
    }

};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_RESIZE_H
