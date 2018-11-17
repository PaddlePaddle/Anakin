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

#ifndef ANAKIN_SABER_FUNCS_RESHAPE_H
#define ANAKIN_SABER_FUNCS_RESHAPE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"

namespace anakin {

namespace saber {

template <typename TargetType, DataType OpDtype>
class Reshape : public BaseFunc<
    TargetType,
    OpDtype,
    ImplBase,
    ReshapeParam>
{
public:
    using BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        ReshapeParam >::BaseFunc;
    Reshape() = default;

    typedef ReshapeParam<TargetType> Param_t;
    typedef std::vector<Tensor<TargetType> *> Input_v;
    typedef std::vector<Tensor<TargetType> *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, \
        Output_v &output, Param_t& param) override {

        Shape output_shape;
        output_shape.resize(param.shape_params.size());
        output_shape.set_layout(param.layout);

        CHECK_EQ(input[0] -> is_continue_mem(), true) << "input tensor must not have roi";

        Shape input_shape = input[0] -> valid_shape();
        int valid_size = input[0] -> valid_size();
        int infer_axis = -1;
        int count_axis = 1;

        for (int i = 0; i < param.shape_params.size(); ++i) {
            if (param.shape_params[i] == 0){
                CHECK_LT(i, input_shape.size()) << "wrong parameters, exceed input dims";
                output_shape[i] = input_shape[i];
                count_axis *= input_shape[i];
            } else if (param.shape_params[i] > 0){
                output_shape[i] = param.shape_params[i];
                count_axis *= param.shape_params[i];
            } else {
                output_shape[i] = -1;
                infer_axis = i;
            }
        }
        // The axis that needs to automatically infer the dimension
        if (infer_axis >= 0){
            output_shape[infer_axis] = valid_size / count_axis;
        }
        
        return output[0] -> set_shape(output_shape);
    }
    //Reshape ops do nothing
    virtual SaberStatus init_impl(ImplEnum implenum) override {
        return SaberSuccess;
    }

    //Reshape ops do nothing
    virtual SaberStatus init(const Input_v& input, Output_v& output, Param_t& param,
            SaberImplStrategy strategy, ImplEnum implenum, Context<TargetType > &ctx) {
        return SaberSuccess;
    }
    //Reshape ops do nothing
    virtual SaberStatus operator()(const Input_v& input, Output_v& output, Param_t& param, \
        Context<TargetType> &ctx) {
        return SaberSuccess;
    }
private:

    virtual void pick_best_static() override {
        //saber impl
        this -> _best_impl = this -> _impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        //saber impl
        this -> _best_impl = this -> _impl[0];
    }

};

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_RESHAPE_H
