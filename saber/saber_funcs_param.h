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
struct PreluParam;

template <typename TargetType>
struct ActivationParam {
    ActivationParam()
            : active(Active_unknow)
            , negative_slope(float(-1))
            , coef(float(-1))
            , prelu_param(PreluParam<TargetType>(false, nullptr))
            , has_active(false) {}
    ActivationParam(ActiveType act, float n_slope = float(0),
                    float co = float(1),
                    PreluParam<TargetType> prelu = PreluParam<TargetType>(false, nullptr))
            : active(act)
            , negative_slope(n_slope)
            , coef(co)
            , prelu_param(prelu)
            , has_active(true)
    {}
    ActivationParam(const ActivationParam &right)
            : active(right.active)
            , negative_slope(right.negative_slope)
            , coef(right.coef)
            , prelu_param(right.prelu_param)
            , has_active(right.has_active)
    {}
    ActivationParam &operator=(const ActivationParam &right) {
        active = right.active;
        negative_slope = right.negative_slope;
        coef = right.coef;
        prelu_param = right.prelu_param;
        has_active = right.has_active;
        return *this;
    }
    bool operator==(const ActivationParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (active == right.active);
        comp_eq = comp_eq && (negative_slope == right.negative_slope);
        comp_eq = comp_eq && (coef == right.coef);
        comp_eq = comp_eq && (prelu_param == right.prelu_param);
        comp_eq = comp_eq && (has_active == right.has_active);
        return comp_eq;
    }
    bool has_negative_slope(){
        return (active == Active_relu) && (negative_slope != float (0));
    }
    ActiveType active;
    float negative_slope;
    float coef;
    bool has_active;
    PreluParam<TargetType> prelu_param;
};

template <typename TargetType>
struct ConvParam {

    ConvParam()
            : group(-1), pad_h(-1), pad_w(-1)
            , stride_h(-1), stride_w(-1)
            , dilation_h(-1), dilation_w(-1)
            , weight_tensor(NULL), bias_tensor(NULL)
            , alpha(1.0), beta(0.0)
            , activation_param(ActivationParam<TargetType>()){}

    ConvParam(int group_in, int pad_h_in, int pad_w_in,
              int stride_h_in, int stride_w_in, int dilation_h_, int dilation_w_,
              Tensor<TargetType>* weight, Tensor<TargetType>* bias,
              ActivationParam<TargetType> activation_param_in = ActivationParam<TargetType>(),
              float alpha_in = 1.0, float beta_in = 0.0)
            : group(group_in), pad_h(pad_h_in), pad_w(pad_w_in)
            , stride_h(stride_h_in), stride_w(stride_w_in)
            , dilation_h(dilation_h_), dilation_w(dilation_w_)
            , weight_tensor(weight), bias_tensor(bias)
            , activation_param(activation_param_in)
            , alpha(alpha_in), beta(beta_in)
    {}

    ConvParam(const ConvParam &right)
            : group(right.group), pad_h(right.pad_h)
            , pad_w(right.pad_w), stride_h(right.stride_h)
            , stride_w(right.stride_w), dilation_h(right.dilation_h)
            , dilation_w(right.dilation_w)
            , weight_tensor(right.weight_tensor)
            , bias_tensor(right.bias_tensor)
            , alpha(right.alpha)
            , beta(right.beta)
            , activation_param(right.activation_param)
    {}

    ConvParam &operator=(const ConvParam &right) {
        group = right.group;
        pad_h = right.pad_h;
        pad_w = right.pad_w;
        stride_h = right.stride_h;
        stride_w = right.stride_w;
        dilation_h = right.dilation_h;
        dilation_w = right.dilation_w;
        weight_tensor = right.weight_tensor;
        bias_tensor = right.bias_tensor;
        alpha = right.alpha;
        beta = right.beta;
        activation_param = right.activation_param;
        return *this;
    }

    bool operator==(const ConvParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (group == right.group);
        comp_eq = comp_eq && (pad_h == right.pad_h);
        comp_eq = comp_eq && (pad_w == right.pad_w);
        comp_eq = comp_eq && (stride_h == right.stride_h);
        comp_eq = comp_eq && (stride_w == right.stride_w);
        comp_eq = comp_eq && (dilation_h == right.dilation_h);
        comp_eq = comp_eq && (dilation_w == right.dilation_w);
        comp_eq = comp_eq && (weight_tensor == right.weight_tensor);
        comp_eq = comp_eq && (bias_tensor == right.bias_tensor);
        comp_eq = comp_eq && (alpha == right.alpha);
        comp_eq = comp_eq && (beta == right.beta);
        comp_eq = comp_eq && (activation_param == right.activation_param);
        return comp_eq;
    }

    inline const Tensor<TargetType>* weight() {
        return weight_tensor;
    }

    inline const Tensor<TargetType>* bias() {
        return bias_tensor;
    }

    inline Tensor<TargetType>* mutable_weight() {
        return weight_tensor;
    }

    inline Tensor<TargetType>* mutable_bias() {
        return bias_tensor;
    }

    int group;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    float alpha;
    float beta;
    ActivationParam<TargetType> activation_param;
private:
    Tensor<TargetType>* weight_tensor;
    Tensor<TargetType>* bias_tensor;
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


template <typename TargetType>
struct NormalizeParam {
    NormalizeParam() = default;

    NormalizeParam(bool is_across_spatial, float eps_in = 1e-6f, int pin = 2) {
        across_spatial = is_across_spatial;
        p = pin;
        has_scale = false;
        scale = nullptr;
        eps = eps_in;
        CHECK_EQ(p == 2 || p == 1, true) << "only support L1 and L2 norm";
    }
    NormalizeParam(bool is_across_spatial, bool is_shared_channel, \
                   Tensor<TargetType>* input_scale, float eps_in = 1e-6f, int pin = 2) {

        across_spatial = is_across_spatial;
        channel_shared = is_shared_channel;
        p = pin;
        has_scale = true;
        scale = input_scale;
        eps = eps_in;
        CHECK_EQ(p == 2 || p == 1, true) << "only support L1 and L2 norm";
    }

    NormalizeParam(const NormalizeParam<TargetType>& right) {
        channel_shared = right.channel_shared;
        across_spatial = right.across_spatial;
        p = right.p;
        has_scale = right.has_scale;
        scale = right.scale;
        eps = right.eps;
    }

    NormalizeParam<TargetType>& operator=(const NormalizeParam<TargetType>& right) {
        this->channel_shared = right.channel_shared;
        this->across_spatial = right.across_spatial;
        this->scale = right.scale;
        this->p = right.p;
        this->has_scale = right.has_scale;
        this->eps = right.eps;
        return *this;
    }

    bool operator==(const NormalizeParam<TargetType>& right) {
        bool flag = this->across_spatial == right.across_spatial;
        flag = flag && (this->channel_shared == right.channel_shared);
        flag = flag && (this->has_scale == right.has_scale);
        flag = flag && (this->p == right.p);
        flag = flag && (fabsf(this->eps - right.eps) < 1e-7f);
        return flag && (this->scale == right.scale);
    }

    //! p = 1, L1 normalize, p = 2, L2 normalize
    int  p{2};
    //! whether normalize is across the spatial
    //! if not across spatial, do normalize across channel
    bool across_spatial{true};
    //! has_scale = true, result is multiplied by scale
    bool has_scale{false};
    //! if channel_shared = true, use one scale data
    bool channel_shared{false};
    //! scale tensor if has one
    Tensor<TargetType>* scale{nullptr};
    float eps{1e-6f};
};

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

/**
 * GRU_Formula,origin for paddle,Cudnn for cudnn,difference is w_h_r and weighted mean
 * weight for origin is [W_h_o][W_h_r,W_h_z]
 * weight for cudnn is [W_h_o,W_h_r,W_h_z]
 */

template <typename TargetType>
struct GruParam {

    typedef Tensor<TargetType> opTensor;

    GruParam() :
            weight_tensor(nullptr)
            ,bias_tensor(nullptr)
            ,init_hidden_tensor(nullptr)
            ,dropout_param(1.0f)
            ,num_direction(1)
            ,num_layers(1)
            ,is_reverse(false)
            ,gate_activity(Active_sigmoid)
            ,h_activity(Active_tanh)
            ,formula(GRU_ORIGIN)
    {}
    /**
     *
     * @param weight i2h,i2h_r,i2h_z,h2h,h2h_r,h2h_z (different from paddlepaddle h2h_z,h2h_r,h2h and i2h* is the fc weights before gru)
     * @param bias if bias is NULL bias will be zero
     * @param dropout_param_in default 1.0f
     * @param num_direction_in 1 or 2 ,output will be channged
     * @param numLayers_in
     * @param mode_in
     */
    GruParam(opTensor* weight_in, opTensor* bias_in,GruFormula formula_in,
             ActiveType gate_activity_in=Active_sigmoid, ActiveType h_activity_in=Active_tanh,
             bool is_reverse_in=false,opTensor* hidden_init_in=nullptr,
             float dropout_param_in=1.f
            ,int num_direction_in=1,int numLayers_in=1)
            :
            weight_tensor(weight_in)
            ,bias_tensor(bias_in)
            ,dropout_param(dropout_param_in)
            ,num_direction(num_direction_in)
            ,num_layers(numLayers_in)
            ,is_reverse(is_reverse_in)
            ,gate_activity(gate_activity_in)
            ,h_activity(h_activity_in)
            ,formula(formula_in)
            ,init_hidden_tensor(hidden_init_in)
    {}


    GruParam &operator=(const GruParam &right) {
        weight_tensor = right.weight_tensor;
        dropout_param=right.dropout_param;
        num_direction=right.num_direction;
        num_layers=right.num_layers;
        bias_tensor = right.bias_tensor;
        gate_activity=right.gate_activity;
        h_activity=right.h_activity;
        is_reverse=right.is_reverse;
        formula=right.formula;
        init_hidden_tensor=right.init_hidden_tensor;
        return *this;
    }

    bool operator==(const GruParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (weight_tensor == right.weight_tensor);
        comp_eq = comp_eq && (dropout_param == right.dropout_param);
        comp_eq = comp_eq && (num_direction == right.num_direction);
        comp_eq = comp_eq && (num_layers == right.num_layers);
        comp_eq = comp_eq && (bias_tensor == right.bias_tensor);
        comp_eq = comp_eq && (gate_activity=right.gate_activity);
        comp_eq = comp_eq && (h_activity=right.h_activity);
        comp_eq = comp_eq && (is_reverse=right.is_reverse);
        comp_eq = comp_eq && (formula=right.formula);
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
    ActiveType gate_activity;
    ActiveType h_activity;
    GruFormula formula;
    bool is_reverse;
private:
    opTensor* weight_tensor;
    opTensor* bias_tensor;
    opTensor* init_hidden_tensor;
};


}
}
#endif //SABER_FUNCS_PARAM_H
