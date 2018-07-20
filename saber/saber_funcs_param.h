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

#ifndef ANAKIN_SABER_FUNCS_PARAM_H
#define ANAKIN_SABER_FUNCS_PARAM_H
#include "anakin_config.h"
#include <vector>
#include <string>
#include "saber/core/shape.h"
#include "saber/core/tensor.h"
#include "saber/saber_types.h"

namespace anakin{

namespace saber {

template <typename TargetType>
struct PreluParam {
    PreluParam() = default;
    PreluParam(bool is_channel_shared, Tensor<TargetType>* input_slope) {
        channel_shared = is_channel_shared;
        slope = input_slope;
    }
    PreluParam(const PreluParam<TargetType>& right) {
        channel_shared = right.channel_shared;
        slope = right.slope;
    }
    PreluParam<TargetType>& operator=(const PreluParam<TargetType>& right) {
        this->channel_shared = right.channel_shared;
        this->slope = right.slope;
        return *this;
    }
    bool operator==(const PreluParam<TargetType>& right) {
        bool flag = this->channel_shared == right.channel_shared;
        return flag && (this->slope == right.slope);
    }
    bool channel_shared{false};
    Tensor<TargetType>* slope{nullptr};
};

template <typename TargetType>
struct ActivationParam {
    ActivationParam()
            : active(Active_unknow)
            , negative_slope(float(-1))
            , coef(float(-1))
            , prelu_param(PreluParam<TargetType>(false, nullptr)) {}
    ActivationParam(ActiveType act, float n_slope = float(0),
                    float co = float(1),
                    PreluParam<TargetType> prelu = PreluParam<TargetType>(false, nullptr))
            : active(act)
            , negative_slope(n_slope)
            , coef(co)
            , prelu_param(prelu)
    {}
    ActivationParam(const ActivationParam &right)
            : active(right.active)
            , negative_slope(right.negative_slope)
            , coef(right.coef)
            , prelu_param(right.prelu_param)
    {}
    ActivationParam &operator=(const ActivationParam &right) {
        active = right.active;
        negative_slope = right.negative_slope;
        coef = right.coef;
        prelu_param = right.prelu_param;
        return *this;
    }
    bool operator==(const ActivationParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (active == right.active);
        comp_eq = comp_eq && (negative_slope == right.negative_slope);
        comp_eq = comp_eq && (coef == right.coef);
        comp_eq = comp_eq && (prelu_param == right.prelu_param);
        return comp_eq;
    }
    bool has_negative_slope(){
        return (active == Active_relu) && (negative_slope != float (0));
    }
    ActiveType active;
    float negative_slope;
    float coef;
    PreluParam<TargetType> prelu_param;
};

template <typename TargetType>
struct LstmParam{
    typedef Tensor<TargetType> opTensor;
    LstmParam() :
            weight_tensor(nullptr)
            ,bias_tensor(nullptr)
            ,init_hidden_tensor(nullptr)
            ,dropout_param(1.0f)
            ,num_direction(1)
            ,num_layers(1)
            ,is_reverse(false)
            ,input_activity(Active_unknow)
            ,gate_activity(Active_sigmoid)
            ,cell_activity(Active_tanh)
            ,candidate_activity(Active_tanh)
            ,with_peephole(true)
            ,skip_input(false)

    {}

    LstmParam(opTensor* weight_in, opTensor* bias_in,
              opTensor* hidden_init_in = nullptr,
              ActiveType input_activity = Active_unknow,
              ActiveType gate_activity_in = Active_sigmoid,
              ActiveType cell_activity_in = Active_tanh,
              ActiveType candidate_activity_in = Active_tanh,
              bool with_peephole_in = true,
              bool skip_input_in = false,
              bool is_reverse_in = false,
              float dropout_param_in = 1.f,
              int num_direction_in = 1,
              int numLayers_in = 1)
            :
            weight_tensor(weight_in)
            ,bias_tensor(bias_in)
            ,dropout_param(dropout_param_in)
            ,num_direction(num_direction_in)
            ,num_layers(numLayers_in)
            ,is_reverse(is_reverse_in)
            ,input_activity(input_activity)
            ,gate_activity(gate_activity_in)
            ,candidate_activity(candidate_activity_in)
            ,cell_activity(cell_activity_in)
            ,init_hidden_tensor(hidden_init_in)
            ,with_peephole(with_peephole_in)
            ,skip_input(skip_input_in)
    {}


    LstmParam &operator=(const LstmParam &right) {
        weight_tensor = right.weight_tensor;
        dropout_param=right.dropout_param;
        num_direction=right.num_direction;
        num_layers=right.num_layers;
        bias_tensor = right.bias_tensor;
        input_activity=right.input_activity;
        gate_activity=right.gate_activity;
        cell_activity=right.cell_activity;
        candidate_activity=right.candidate_activity;
        with_peephole=right.with_peephole;
        skip_input=right.skip_input;
        is_reverse=right.is_reverse;
        init_hidden_tensor=right.init_hidden_tensor;
        return *this;
    }

    bool operator==(const LstmParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (weight_tensor == right.weight_tensor);
        comp_eq = comp_eq && (dropout_param == right.dropout_param);
        comp_eq = comp_eq && (num_direction == right.num_direction);
        comp_eq = comp_eq && (num_layers == right.num_layers);
        comp_eq = comp_eq && (bias_tensor == right.bias_tensor);
        comp_eq = comp_eq && (input_activity==right.input_activity);
        comp_eq = comp_eq && (gate_activity==right.gate_activity);
        comp_eq = comp_eq && (cell_activity==right.cell_activity);
        comp_eq = comp_eq && (with_peephole==right.with_peephole);
        comp_eq = comp_eq && (skip_input==right.skip_input);
        comp_eq = comp_eq && (candidate_activity==right.candidate_activity);
        comp_eq = comp_eq && (is_reverse=right.is_reverse);
        comp_eq = comp_eq && (init_hidden_tensor==right.init_hidden_tensor);
        return comp_eq;
    }

    inline const opTensor* weight() {
        return weight_tensor;
    }

    inline const opTensor* bias() {
        return bias_tensor;
    }

    inline const opTensor* init_hidden() {
        return init_hidden_tensor;
    }

    int num_direction;
    float dropout_param;
    int num_layers;
    ActiveType input_activity;
    ActiveType gate_activity;
    ActiveType cell_activity;
    ActiveType candidate_activity;
    bool is_reverse;
    bool with_peephole;
    // skip input (X * [Wix, Wfx, Wcx, Wox]) or not;
    // if true, the input's memory layout should be total_seq_len * (4 * hidden_size),
    // and you should calc this information in fc layer before;
    // otherwise the input's memory layout should be total_seq_len * input_size;
    bool skip_input;
private:
    opTensor* weight_tensor;
    opTensor* bias_tensor;
    opTensor* init_hidden_tensor;
};

}
}
#endif //SABER_FUNCS_PARAM_H
