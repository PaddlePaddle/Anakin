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

template <typename Type>
struct MatMulParam {
    MatMulParam():_is_transpose_X(false),_is_transpose_Y(false){}
    MatMulParam(bool x, bool y):_is_transpose_X(x),_is_transpose_Y(y){}
    MatMulParam &operator=(const MatMulParam &right)
    {
        _is_transpose_X = right._is_transpose_X;
        _is_transpose_Y = right._is_transpose_Y;
    }
    bool operator==(const MatMulParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (_is_transpose_X == right._is_transpose_X);        
        comp_eq = comp_eq && (_is_transpose_Y == right._is_transpose_Y);
        return comp_eq;
    }

    bool _is_transpose_X{false};
    bool _is_transpose_Y{false};
    int _m = 0;
    int _n = 0;
    int _k = 0;
    int _b = 0;//batch_size

};

//should design this one for pick_best_specify()
enum ImplEnum{
    VENDER_IMPL = 0,
    SABER_IMPL
};

enum SequencePoolType{
    Sequence_pool_unknow = 0,
    Sequence_pool_average,
    Sequence_pool_sum,
    Sequence_pool_sqrt,
    Sequence_pool_last,
    Sequence_pool_first,
    Sequence_pool_max
};

template <typename opTensor>
struct TransposeParam {
    TransposeParam() = default;
    TransposeParam(const TransposeParam& right){}
    TransposeParam& operator=(const TransposeParam& right){}
    bool operator==(const TransposeParam& right){
        return true;
    }
};
/**
 * GRU_Formula,origin for paddle,Cudnn for cudnn,difference is w_h_r and weighted mean
 * weight for origin is [W_h_o][W_h_r,W_h_z]
 * weight for cudnn is [W_h_o,W_h_r,W_h_z]
 */
enum GruFormula {
    GRU_ORIGIN = 0,
    GRU_CUDNN
};

template <typename opTensor>
struct GruParam {



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

template <typename opTensor>
struct LstmParam{

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

template <typename opTensor>
struct ConvParam {

    ConvParam() : group(-1), pad_h(-1), pad_w(-1),
                  stride_h(-1), stride_w(-1),
                  dilation_h(-1), dilation_w(-1),
                  weight_tensor(NULL), bias_tensor(NULL), alpha(1.0), beta(0.0) {}

    ConvParam(int group_in, int pad_h_in, int pad_w_in,
              int stride_h_in, int stride_w_in, int dilation_h_, int dilation_w_,
              opTensor* weight, opTensor* bias, float alpha_in = 1.0, float beta_in = 0.0)
            : group(group_in), pad_h(pad_h_in), pad_w(pad_w_in)
            , stride_h(stride_h_in), stride_w(stride_w_in)
            , dilation_h(dilation_h_), dilation_w(dilation_w_)
            , weight_tensor(weight), bias_tensor(bias)
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
        return comp_eq;
    }

    inline const opTensor* weight() {
        return weight_tensor;
    }

    inline const opTensor* bias() {
        return bias_tensor;
    }

    inline opTensor* mutable_weight() {
        return weight_tensor;
    }

    inline opTensor* mutable_bias() {
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

private:
    opTensor* weight_tensor;
    opTensor* bias_tensor;
};
// specify for int8
template <>
struct ConvParam<Tensor<NV, AK_INT8, NCHW> > {
    ConvParam() : group(-1), pad_h(-1), pad_w(-1),
                  stride_h(-1), stride_w(-1),
                  dilation_h(-1), dilation_w(-1),
                  weight_tensor(NULL), bias_tensor(NULL), alpha(1.0), beta(0.0){}
    ConvParam(int group_in, int pad_h_in, int pad_w_in,
              int stride_h_in, int stride_w_in, int dilation_h_, int dilation_w_,
              Tensor<NV, AK_INT8, NCHW>* weight, Tensor<NV, AK_FLOAT, NCHW>* bias, 
              float alpha_in = 1.0, float beta_in = 0.0)
            : group(group_in), pad_h(pad_h_in), pad_w(pad_w_in)
            , stride_h(stride_h_in), stride_w(stride_w_in)
            , dilation_h(dilation_h_), dilation_w(dilation_w_)
            , weight_tensor(weight), bias_tensor(bias)
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
            , beta(right.beta) {}
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
        return comp_eq;
    }
    inline const Tensor<NV, AK_INT8, NCHW>* weight() {
        return weight_tensor;
    }
    inline const Tensor<NV, AK_FLOAT, NCHW>* bias() {
        return bias_tensor;
    }
    inline Tensor<NV, AK_INT8, NCHW>* mutable_weight() {
        return weight_tensor;
    }
    inline Tensor<NV, AK_FLOAT, NCHW>* mutable_bias() {
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
private:
    Tensor<NV, AK_INT8, NCHW>* weight_tensor;
    Tensor<NV, AK_FLOAT, NCHW>* bias_tensor;
};

// specify for int8 NCHW_VECT_C
template <>
struct ConvParam<Tensor<NV, AK_INT8, NCHW_C4> > {

    ConvParam() : group(-1), pad_h(-1), pad_w(-1),
                  stride_h(-1), stride_w(-1),
                  dilation_h(-1), dilation_w(-1),
                  weight_tensor(NULL), bias_tensor(NULL) {}

    ConvParam(int group_in, int pad_h_in, int pad_w_in,
              int stride_h_in, int stride_w_in, int dilation_h_, int dilation_w_,
              Tensor<NV, AK_INT8, NCHW_C4>* weight, Tensor<NV, AK_FLOAT, NCHW>* bias)
            : group(group_in), pad_h(pad_h_in), pad_w(pad_w_in)
            , stride_h(stride_h_in), stride_w(stride_w_in)
            , dilation_h(dilation_h_), dilation_w(dilation_w_)
            , weight_tensor(weight), bias_tensor(bias)
    {}

    ConvParam(const ConvParam &right)
            : group(right.group), pad_h(right.pad_h)
            , pad_w(right.pad_w), stride_h(right.stride_h)
            , stride_w(right.stride_w), dilation_h(right.dilation_h)
            , dilation_w(right.dilation_w)
            , weight_tensor(right.weight_tensor)
            , bias_tensor(right.bias_tensor)
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
        return comp_eq;
    }
    inline const Tensor<NV, AK_INT8, NCHW_C4>* weight() {
        return weight_tensor;
    }
    inline const Tensor<NV, AK_FLOAT, NCHW>* bias() {
        return bias_tensor;
    }
    inline Tensor<NV, AK_INT8, NCHW_C4>* mutable_weight() {
        return weight_tensor;
    }
    inline Tensor<NV, AK_FLOAT, NCHW>* mutable_bias() {
        return bias_tensor;
    }
    int group;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
private:
    Tensor<NV, AK_INT8, NCHW_C4>* weight_tensor;
    Tensor<NV, AK_FLOAT, NCHW>* bias_tensor;
};
template <typename opTensor>
struct PermuteParam {
    PermuteParam() {}
    PermuteParam(std::vector<int> order):order(order) {}
    PermuteParam(const PermuteParam &right): order(right.order) {}
    PermuteParam &operator=(const PermuteParam &right) {
        order = right.order;
        return *this;
    }
    bool operator==(const PermuteParam &right) {
        bool comp_eq = true;
        comp_eq = order.size() == right.order.size();
        for (int i = 0; i < order.size(); ++i) {
            comp_eq = comp_eq && (order[i] == right.order[i]);
        }
        return comp_eq;
    }
    std::vector<int> order;
};
template<typename opTensor>
struct PowerParam {
    PowerParam() {}
    PowerParam(float power, float scale, float shift) 
            : power(power), scale(scale), shift(shift) {}
    PowerParam(const PowerParam &right): 
            power(right.power), scale(right.scale), shift(right.shift) {}
    bool operator==(const PowerParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (power == right.power);
        comp_eq = comp_eq && (scale == right.scale);
        comp_eq = comp_eq && (shift == right.shift);
        return comp_eq;
    }
    float power;
    float scale;
    float shift;
};
template<typename opTensor>
struct PermutePowerParam {
    PermutePowerParam() {}
    PermutePowerParam(PermuteParam<opTensor> permute_param):
            power_param(power_param), has_power_param(false) {}
    PermutePowerParam(PermuteParam<opTensor> permute_param, PowerParam<opTensor> power_param):
            power_param(power_param), permute_param(permute_param), has_power_param(true) {}
    PermutePowerParam(const PermutePowerParam & right):
        power_param(right.power_param), permute_param(right.permute_param) {}
    bool operator==(const PermutePowerParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (power_param == right.power_param);
        comp_eq = comp_eq && (permute_param == right.permute_param);
        return comp_eq;
    }
    PowerParam<opTensor> power_param;
    PermuteParam<opTensor> permute_param;
    bool has_power_param;
};
template <typename type>
struct ConcatParam {
    ConcatParam() = default;
    explicit ConcatParam(int axis_in){
        CHECK_GE(axis_in, 0) << "concat parameter should >= 0, current is " << axis_in;
        axis = axis_in;
    }
    ConcatParam(const ConcatParam<type> &right) {
        axis = right.axis;
    }
    ConcatParam<type> &operator=(const ConcatParam<type> &right) {
        axis = right.axis;
        return *this;
    }
    bool operator==(const ConcatParam<type> &right) {
        return axis == right.axis;
    }
    int axis;
};
template <typename type>
struct ReshapeParam {
    ReshapeParam() = default;
    explicit ReshapeParam(std::vector<int> shape_param_in){
        int count = 0;
        for (int i = 0; i < shape_param_in.size(); ++i) {
            if (shape_param_in[i] == -1){
                count ++;
            }
        }
        CHECK_LE(count, 1) << "shape parameter contains multiple -1 dims";
        shape_params = shape_param_in;
    }
    ReshapeParam(const ReshapeParam<type> &right) {
        shape_params = right.shape_params;
    }
    ReshapeParam<type> &operator=(const ReshapeParam<type> &right) {
        shape_params = right.shape_params;
        return *this;
    }
    bool operator==(const ReshapeParam &right) {
        bool comp_eq = shape_params.size() == right.shape_params.size();
        for (int i = 0; i < shape_params.size(); ++i) {
            if (!comp_eq){
                return false;
            }
            comp_eq = shape_params[i] == right.shape_params[i];
        }
        return true;
    }
    std::vector<int> shape_params;
};
template <typename type>
struct SliceParam {
    SliceParam() = default;
    explicit SliceParam(int axis_in, std::vector<int> slice_points_in){
        CHECK_GE(axis_in, 0) << "slice axis should >=0, current is " << axis_in;
        axis = axis_in;
        slice_points = slice_points_in;
    }
    SliceParam(const SliceParam<type> &right) {
        axis = right.axis;
        slice_points = right.slice_points;
    }
    SliceParam<type> &operator=(const SliceParam<type> &right) {
        axis = right.axis;
        slice_points = right.slice_points;
        return *this;
    }
    bool operator==(const SliceParam<type> &right) {
        bool comp_eq = slice_points.size() == right.slice_points.size();
        for (int i = 0; i < slice_points.size(); ++i) {
            if (!comp_eq){
                return false;
            }
            comp_eq = slice_points[i] == right.slice_points[i];
        }
        return axis == right.axis;
    }
    int axis;
    std::vector<int> slice_points;
};
template <typename type>
struct SoftmaxParam {
    SoftmaxParam() = default;
    explicit SoftmaxParam(int axis_in){
        CHECK_GE(axis_in, 0) << "input axis index should >= 0, current is " << axis_in;
        axis = axis_in;
    }
    SoftmaxParam(const SoftmaxParam<type>& right){
        axis = right.axis;
    }
    SoftmaxParam<type>& operator=(const SoftmaxParam<type>& right){
        this->axis = right.axis;
        return *this;
    }
    bool operator==(const SoftmaxParam<type>& right){
        return axis == right.axis;
    }
    int axis;
};
template <typename opTensor>
struct BatchnormParam {
    typedef typename opTensor::Dtype DataDtype;
    BatchnormParam()
        : scale(DataDtype(0))
        , use_global_stats(true)
        , moving_average_fraction(DataDtype(0.999))
        , eps(DataDtype(1e-5))
        , mean(), variance()
    {}
    //scale_factor = 1 / scale;
    BatchnormParam(std::vector<DataDtype> mean_in, std::vector<DataDtype> variance_in,
                DataDtype scale_in, DataDtype moving_average_fraction_in = DataDtype(0.999),
                DataDtype eps_in = DataDtype(1e-5), bool use_global_stats_in = true)
        : mean(mean_in), variance(variance_in), scale(scale_in)
        , moving_average_fraction(moving_average_fraction_in)
        , eps(eps_in), use_global_stats(use_global_stats_in)
    {}
    BatchnormParam &operator=(const BatchnormParam &right) {
        scale = right.scale;
        moving_average_fraction = right.moving_average_fraction;
        eps = right.eps;
        use_global_stats = right.use_global_stats;
        mean = right.mean;
        variance = right.variance;
        return *this;
    }
    bool operator==(const BatchnormParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (scale == right.scale);
        comp_eq = comp_eq && (moving_average_fraction == right.moving_average_fraction);
        comp_eq = comp_eq && (eps == right.eps);
        comp_eq = comp_eq && (use_global_stats == right.use_global_stats);
        comp_eq = comp_eq && (mean == right.mean);
        comp_eq = comp_eq && (variance == right.variance);
        return comp_eq;
    }
    DataDtype scale;
    DataDtype moving_average_fraction;
    DataDtype eps;
    bool use_global_stats;
    std::vector<DataDtype> mean;
    std::vector<DataDtype> variance;
};

template <typename opTensor>
struct PreluParam {
    PreluParam() = default;
    PreluParam(bool is_channel_shared, opTensor* input_slope) {
        channel_shared = is_channel_shared;
        slope = input_slope;
    }
    PreluParam(const PreluParam<opTensor>& right) {
        channel_shared = right.channel_shared;
        slope = right.slope;
    }
    PreluParam<opTensor>& operator=(const PreluParam<opTensor>& right) {
        this->channel_shared = right.channel_shared;
        this->slope = right.slope;
        return *this;
    }
    bool operator==(const PreluParam<opTensor>& right) {
        bool flag = this->channel_shared == right.channel_shared;
        return flag && (this->slope == right.slope);
    }
    bool channel_shared{false};
    opTensor* slope{nullptr};
};

template <typename opTensor>
struct ActivationParam {
    typedef typename opTensor::Dtype DataDtype;
    ActivationParam()
            : active(Active_unknow)
            , negative_slope(DataDtype(-1))
            , coef(DataDtype(-1))
            , prelu_param(PreluParam<opTensor>(false, nullptr)) {}
    ActivationParam(ActiveType act, DataDtype n_slope = DataDtype(0),
                    DataDtype co = DataDtype(1), 
                    PreluParam<opTensor> prelu = PreluParam<opTensor>(false, nullptr))
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
        return (active == Active_relu) && (negative_slope != DataDtype (0));
    }
    ActiveType active;
    DataDtype negative_slope;
    DataDtype coef;
    PreluParam<opTensor> prelu_param;
};
template <typename opTensor>
struct ScaleParam {
    typedef typename opTensor::Dtype DataDtype;
    ScaleParam()
            : axis(1), num_axes(1)
            , bias_term(false)
    {}
    ScaleParam(std::vector<DataDtype> scale_w_in, std::vector<DataDtype> scale_b_in,
               bool bias_term_in = true, int axis_in = 1, int num_axes_in = 1)
            : scale_w(scale_w_in), scale_b(scale_b_in)
            , bias_term(bias_term_in), axis(axis_in), num_axes(num_axes_in)
    {}
    ScaleParam(std::vector<DataDtype> scale_w_in,
               bool bias_term_in = false, int axis_in = 1, int num_axes_in = 1)
            : scale_w(scale_w_in)
            , bias_term(bias_term_in), axis(axis_in), num_axes(num_axes_in)
    {}
    ScaleParam(const ScaleParam &right)
            : scale_w(right.scale_w), scale_b(right.scale_b)
            , bias_term(right.bias_term), axis(right.axis), num_axes(right.num_axes)
    {}
    ScaleParam &operator=(const ScaleParam &right) {
        scale_w = right.scale_w;
        scale_b = right.scale_b;
        bias_term = right.bias_term;
        axis = right.axis;
        num_axes = right.num_axes;
        return *this;
    }
    bool operator==(const ScaleParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (scale_w == right.scale_w);
        comp_eq = comp_eq && (scale_b == right.scale_b);
        comp_eq = comp_eq && (bias_term == right.bias_term);
        comp_eq = comp_eq && (axis == right.axis);
        comp_eq = comp_eq && (num_axes == right.num_axes);
        return comp_eq;
    }
    int axis; // default is 1
    int num_axes; // default is 1
    bool bias_term; // default false
    std::vector<DataDtype> scale_w;
    std::vector<DataDtype> scale_b;
};
template <typename opTensor>
struct PoolingParam {
    PoolingParam() : window_h(-1), window_w(-1)
                   , pad_h(-1), pad_w(-1)
                   , stride_h(-1), stride_w(-1)
                   , pooling_type(Pooling_unknow)
                   , global_pooling(false)
                   , cmp_out_shape_floor_as_conv(false)
        {}
    PoolingParam(int window_h_in, int window_w_in, int pad_h_in
            , int pad_w_in, int stride_h_in, int stride_w_in, PoolingType type
            , bool global_pooling_in = false, bool cmp_out_shape_floor_as_conv_in = false)
            : window_h(window_h_in), window_w(window_w_in)
            , pad_h(pad_h_in), pad_w(pad_w_in)
            , stride_h(stride_h_in), stride_w(stride_w_in)
            , pooling_type(type)
            , global_pooling(global_pooling_in)
            , cmp_out_shape_floor_as_conv(cmp_out_shape_floor_as_conv_in)
    {}
    PoolingParam(const PoolingParam &right)
            : window_h(right.window_h)
            , window_w(right.window_w)
            , pad_h(right.pad_h)
            , pad_w(right.pad_w)
            , stride_h(right.stride_h)
            , stride_w(right.stride_w)
            , pooling_type(right.pooling_type)
            , global_pooling(right.global_pooling)
            , cmp_out_shape_floor_as_conv(right.cmp_out_shape_floor_as_conv)
    {}
    PoolingParam &operator=(const PoolingParam &right) {
        window_h = right.window_h;
        window_w = right.window_w;
        pad_h = right.pad_h;
        pad_w = right.pad_w;
        stride_h = right.stride_h;
        stride_w = right.stride_w;
        pooling_type = right.pooling_type;
        global_pooling = right.global_pooling;
        cmp_out_shape_floor_as_conv = right.cmp_out_shape_floor_as_conv;
        return *this;
    }
    bool operator==(const PoolingParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (window_h == right.window_h);
        comp_eq = comp_eq && (window_w == right.window_w);
        comp_eq = comp_eq && (pad_h == right.pad_h);
        comp_eq = comp_eq && (pad_w == right.pad_w);
        comp_eq = comp_eq && (stride_h == right.stride_h);
        comp_eq = comp_eq && (stride_w == right.stride_w);
        comp_eq = comp_eq && (pooling_type == right.pooling_type);
        comp_eq = comp_eq && (global_pooling == right.global_pooling);
        comp_eq = comp_eq && (cmp_out_shape_floor_as_conv == right.cmp_out_shape_floor_as_conv);
        return comp_eq;
    }
    inline bool pooling_padded() {
        return (pad_h || pad_w);
    }
    int window_h;
    int window_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    PoolingType pooling_type;
    bool global_pooling;
    bool cmp_out_shape_floor_as_conv;
};

template <typename opTensor>
struct SequenceConvParam {
    SequenceConvParam()
            : filter_tensor(nullptr),
              padding_tensor(nullptr),
              context_length(1),
              context_start(0),
              context_stride(1),
              padding_trainable(false)
    {}
    SequenceConvParam(opTensor* filter_tensor_in,int context_length_in,
                      int context_start_in=0,int context_stride_in=1,bool padding_trainable_in=false,
                              opTensor* padding_tensor_in= nullptr)
            : filter_tensor(filter_tensor_in),
              padding_tensor(padding_tensor_in),
              context_length(context_length_in),
              context_start(context_start_in),
              context_stride(context_stride_in),
              padding_trainable(padding_trainable_in)
    {}
    SequenceConvParam(const SequenceConvParam &right)
            : filter_tensor(right.filter_tensor),
              padding_tensor(right.padding_tensor),
              context_length(right.context_length),
              context_start(right.context_start),
              context_stride(right.context_stride),
              padding_trainable(right.padding_trainable)
    {}
    SequenceConvParam &operator=(const SequenceConvParam &right) {
        filter_tensor=right.filter_tensor;
        padding_tensor=right.padding_tensor;
        context_length=right.context_length;
        context_start=right.context_start;
        context_stride=right.context_stride;
        padding_trainable=right.padding_trainable;
        return *this;
    }
    bool operator==(const SequenceConvParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (filter_tensor=right.filter_tensor);
        comp_eq = comp_eq && (padding_tensor=right.padding_tensor);
        comp_eq = comp_eq && (context_length=right.context_length);
        comp_eq = comp_eq && (context_start=right.context_start);
        comp_eq = comp_eq && (context_stride=right.context_stride);
        comp_eq = comp_eq && (padding_trainable=right.padding_trainable);
        return comp_eq;
    }

    opTensor *filter_tensor;
    opTensor *padding_tensor;
    int context_length;
    int context_start;
    int context_stride;
    bool padding_trainable;
};

template <typename opTensor>
struct SequencePoolParam {
    SequencePoolParam()
            : sequence_pool_type(Sequence_pool_unknow)
    {}
    SequencePoolParam(SequencePoolType sequence_pool_type_in)
            : sequence_pool_type(sequence_pool_type_in)
    {}
    SequencePoolParam(const SequencePoolParam &right)
            : sequence_pool_type(right.sequence_pool_type)
    {}
    SequencePoolParam &operator=(const SequencePoolParam &right) {
        sequence_pool_type = right.sequence_pool_type;
        return *this;
    }
    bool operator==(const SequencePoolParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (sequence_pool_type == right.sequence_pool_type);
        return comp_eq;
    }
    SequencePoolType sequence_pool_type;
};
template <typename opTensor>
struct CrfDecodingParam {
    CrfDecodingParam()
            : weight_tensor(NULL)
            , tag_num(0)
    {}
    CrfDecodingParam(opTensor* weight_tensor_in, int tag_num_in = 0)
            : weight_tensor(weight_tensor_in) {
        if (tag_num_in == 0) {
            tag_num = weight_tensor->channel();
        } else {
            tag_num = tag_num_in;
        }
    }
    CrfDecodingParam(const CrfDecodingParam &right)
            : weight_tensor(right.weight_tensor)
            , tag_num(right.tag_num)
    {}
    CrfDecodingParam &operator=(const CrfDecodingParam &right) {
        weight_tensor = right.weight_tensor;
        tag_num = right.tag_num;
        return *this;
    }
    bool operator==(const CrfDecodingParam &right) {
        bool comp_eq = true;
        comp_eq &= (weight_tensor == right.weight_tensor);
        comp_eq &= (tag_num == right.tag_num);
        return comp_eq;
    }
    inline const opTensor* transition_weight() {
        return weight_tensor;
    }
    inline opTensor* mutable_transition_weight() {
        return weight_tensor;
    }
    int tag_num;
private:
    opTensor *weight_tensor;
};

template <typename opTensor>
struct EltwiseParam;
template <typename opTensor>
struct EltwiseActiveParam;
// Fusion conv with batchnorm, scale, activation, eltwise(sigmoid, relu, tanh, clipped_relu, elu)
template <typename opTensor>
struct ConvActiveParam {
    ConvActiveParam()
            : has_batchnorm(false)
            , has_scale(false)
            , has_active(false)
            , has_eltwise(false)
            , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in)
        : conv_param(conv_param_in), has_active(false)
        , has_batchnorm(false), has_scale(false), has_eltwise(false), has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in,
                    ActivationParam<opTensor> &activation_param_in)
        : conv_param(conv_param_in), activation_param(activation_param_in)
        , has_batchnorm(false)
        , has_scale(false)
        , has_active(true)
        , has_eltwise(false)
        , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
                    , ActivationParam<opTensor> &activation_param_in
                    , EltwiseParam<opTensor> &eltwise_param_in)
        : conv_param(conv_param_in)
        , activation_param(activation_param_in) 
        , eltwise_param(eltwise_param_in)
        , has_batchnorm(false)
        , has_scale(false)
        , has_active(true)
        , has_eltwise(true)
        , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
            , ActivationParam<opTensor> &activation_param_in
            , BatchnormParam<opTensor> &batchnorm_param_in)
            : conv_param(conv_param_in)
            , activation_param(activation_param_in)
            , batchnorm_param(batchnorm_param_in)
            , has_batchnorm(true)
            , has_scale(false)
            , has_active(true)
            , has_eltwise(false)
            , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
            , ActivationParam<opTensor> &activation_param_in
            , BatchnormParam<opTensor> &batchnorm_param_in
            , EltwiseParam<opTensor> &eltwise_param_in)
            : conv_param(conv_param_in)
            , activation_param(activation_param_in)
            , batchnorm_param(batchnorm_param_in)
            , eltwise_param(eltwise_param_in)
            , has_batchnorm(true)
            , has_scale(false)
            , has_active(true)
            , has_eltwise(true)
            , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
            , ActivationParam<opTensor> &activation_param_in
            , ScaleParam<opTensor> &scale_param_in)
            : conv_param(conv_param_in)
            , activation_param(activation_param_in)
            , scale_param(scale_param_in)
            , has_batchnorm(false)
            , has_scale(true)
            , has_active(true)
            , has_eltwise(false)
            , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
            , ActivationParam<opTensor> &activation_param_in
            , ScaleParam<opTensor> &scale_param_in
            , EltwiseParam<opTensor> &eltwise_param_in)
            : conv_param(conv_param_in)
            , activation_param(activation_param_in)
            , scale_param(scale_param_in)
            , eltwise_param(eltwise_param_in)
            , has_batchnorm(false)
            , has_scale(true)
            , has_active(true)
            , has_eltwise(true)
            , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
            , ActivationParam<opTensor> &activation_param_in
            , BatchnormParam<opTensor> &batchnorm_param_in
            , ScaleParam<opTensor> &scale_param_in)
            : conv_param(conv_param_in)
            , activation_param(activation_param_in)
            , batchnorm_param(batchnorm_param_in)
            , scale_param(scale_param_in)
            , has_batchnorm(true)
            , has_scale(true)
            , has_active(true)
            , has_eltwise(false)
            , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
            , BatchnormParam<opTensor> &batchnorm_param_in
            , ScaleParam<opTensor> &scale_param_in)
            : conv_param(conv_param_in)
            , batchnorm_param(batchnorm_param_in)
            , scale_param(scale_param_in)
            , has_batchnorm(true)
            , has_scale(true)
            , has_active(false)
            , has_eltwise(false)
            , has_eltwise_act(false)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
            , BatchnormParam<opTensor> &batchnorm_param_in
            , ScaleParam<opTensor> &scale_param_in
            , EltwiseActiveParam<opTensor> &elt_act_param_in)
            : conv_param(conv_param_in)
            , batchnorm_param(batchnorm_param_in)
            , scale_param(scale_param_in)
            , eltwise_act_param(elt_act_param_in)
            , has_batchnorm(true)
            , has_scale(true)
            , has_active(false)
            , has_eltwise(false)
            , has_eltwise_act(true)
    {}
    ConvActiveParam(ConvParam<opTensor> &conv_param_in
            , ActivationParam<opTensor> &activation_param_in
            , BatchnormParam<opTensor> &batchnorm_param_in
            , ScaleParam<opTensor> &scale_param_in
            , EltwiseParam<opTensor> &eltwise_param_in)
            : conv_param(conv_param_in)
            , activation_param(activation_param_in)
            , batchnorm_param(batchnorm_param_in)
            , scale_param(scale_param_in)
            , eltwise_param(eltwise_param_in)
            , has_batchnorm(true)
            , has_scale(true)
            , has_active(true)
            , has_eltwise(true)
            , has_eltwise_act(false)
    {}
    ConvActiveParam(const ConvActiveParam &right)
            : conv_param(right.conv_param)
            , activation_param(right.activation_param)
            , batchnorm_param(right.batchnorm_param)
            , scale_param(right.scale_param)
            , has_batchnorm(right.has_batchnorm)
            , has_scale(right.has_scale)
            , has_active(right.has_active)
            , has_eltwise_act(right.has_active)
    {}
    ConvActiveParam &operator=(const ConvActiveParam &right) {
        conv_param = right.conv_param;
        activation_param = right.activation_param;
        batchnorm_param = right.batchnorm_param;
        scale_param = right.scale_param;
        has_batchnorm = right.has_batchnorm;
        has_scale = right.has_scale;
        has_active = right.has_active;
        has_eltwise = right.has_eltwise;
        has_eltwise_act = right.has_eltwise_act;
        return *this;
    }
    bool operator==(const ConvActiveParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (conv_param == right.conv_param);
        comp_eq = comp_eq && (activation_param == right.activation_param);
        comp_eq = comp_eq && (batchnorm_param == right.batchnorm_param);
        comp_eq = comp_eq && (scale_param == right.scale_param);
        comp_eq = comp_eq && (has_batchnorm == right.has_batchnorm);
        comp_eq = comp_eq && (has_scale == right.has_scale);
        comp_eq = comp_eq && (has_active == right.has_active);
        comp_eq = comp_eq && (has_eltwise == right.has_eltwise);
        comp_eq = comp_eq && (has_eltwise_act == right.has_eltwise_act);
        return comp_eq;
    }
    ConvParam<opTensor> conv_param;
    ActivationParam<opTensor> activation_param;
    BatchnormParam<opTensor> batchnorm_param;
    ScaleParam<opTensor> scale_param;
    EltwiseParam<opTensor> eltwise_param;
    EltwiseActiveParam<opTensor> eltwise_act_param;

    bool has_batchnorm;
    bool has_scale;
    bool has_active;
    bool has_eltwise;
    bool has_eltwise_act;
};
// Fusion conv with batchnorm, scale, activation(sigmoid, relu, tanh, clipped_relu, elu)
template <typename opTensor>
struct ConvActivePoolingParam {
    ConvActivePoolingParam() : has_batchnorm(false), has_scale(false),
                               has_activation(false), has_pooling(false) {}
    ConvActivePoolingParam(ConvParam<opTensor> &conv_param_in)
            : conv_param(conv_param_in)
            , has_batchnorm(false), has_scale(false)
            , has_activation(false), has_pooling(false)
    {}
    ConvActivePoolingParam(ConvParam<opTensor> &conv_param_in,
    ActivationParam<opTensor> &activation_param_in)
        : conv_param(conv_param_in), activation_param(activation_param_in)
        , has_batchnorm(false), has_scale(false), has_activation(true), has_pooling(false)
    {}
    ConvActivePoolingParam(ConvParam<opTensor> &conv_param_in,
                           PoolingParam<opTensor> &pooling_param_in)
            : conv_param(conv_param_in), pooling_param(pooling_param_in)
            , has_batchnorm(false), has_scale(false), has_activation(false), has_pooling(false)
    {}
    ConvActivePoolingParam(ConvParam<opTensor> &conv_param_in
    , ActivationParam<opTensor> &activation_param_in
    , PoolingParam<opTensor> &pooling_param_in)
        : conv_param(conv_param_in)
        , activation_param(activation_param_in)
        , pooling_param(pooling_param_in)
        , has_batchnorm(false)
        , has_scale(false)
        , has_activation(true)
        , has_pooling(true)
    {}
    ConvActivePoolingParam(ConvParam<opTensor> &conv_param_in
            , BatchnormParam<opTensor> &batchnorm_param_in
            , ScaleParam<opTensor> &scale_param_in
            , ActivationParam<opTensor> &activation_param_in
            , PoolingParam<opTensor> &pooling_param_in)
            : conv_param(conv_param_in)
            , batchnorm_param(batchnorm_param_in)
            , scale_param(scale_param_in)
            , activation_param(activation_param_in)
            , pooling_param(pooling_param_in)
            , has_batchnorm(true)
            , has_scale(true)
            , has_activation(true)
            , has_pooling(true)
    {}
    ConvActivePoolingParam(const ConvActivePoolingParam &right)
            : conv_param(right.conv_param)
            , batchnorm_param(right.batchnorm_param)
            , scale_param(right.scale_param)
            , activation_param(right.activation_param)
            , pooling_param(right.pooling_param)
            , has_batchnorm(right.has_batchnorm)
            , has_scale(right.has_scale)
            , has_activation(right.has_activation)
            , has_pooling(right.has_pooling)
    {}
    ConvActivePoolingParam &operator=(const ConvActivePoolingParam &right) {
        conv_param = right.conv_param;
        batchnorm_param = right.batchnorm_param;
        scale_param = right.scale_param;
        activation_param = right.activation_param;
        pooling_param = right.pooling_param;
        has_batchnorm = right.has_batchnorm;
        has_scale = right.has_scale;
        has_activation = right.has_activation;
        has_pooling = right.has_pooling;
        return *this;
    }
    bool operator==(const ConvActivePoolingParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (conv_param == right.conv_param);
        comp_eq = comp_eq && (batchnorm_param == right.batchnorm_param);
        comp_eq = comp_eq && (scale_param == right.scale_param);
        comp_eq = comp_eq && (activation_param == right.activation_param);
        comp_eq = comp_eq && (pooling_param == right.pooling_param);
        comp_eq = comp_eq && (has_batchnorm == right.has_batchnorm);
        comp_eq = comp_eq && (has_scale == right.has_scale);
        comp_eq = comp_eq && (has_activation == right.has_activation);
        comp_eq = comp_eq && (has_pooling == right.has_pooling);
        return comp_eq;
    }
    ConvParam<opTensor> conv_param;
    BatchnormParam<opTensor> batchnorm_param;
    ScaleParam<opTensor> scale_param;
    ActivationParam<opTensor> activation_param;
    PoolingParam<opTensor> pooling_param;
    bool has_activation;
    bool has_pooling;
    bool has_batchnorm;
    bool has_scale;
};
template <typename opTensor>
struct ResizeParam {
    ResizeParam() = default;
    explicit ResizeParam(float scale_w, float scale_h){
        bool flag = scale_w > 0.f && scale_h > 0.f;
        CHECK_EQ(flag, true) << "wrong parameters";
        width_scale = scale_w;
        height_scale = scale_h;
    }
    ResizeParam(const ResizeParam<opTensor>& right){
        width_scale = right.width_scale;
        height_scale = right.height_scale;
    }
    ResizeParam<opTensor>& operator=(const ResizeParam<opTensor>& right){
        this->width_scale = right.width_scale;
        this->height_scale = right.height_scale;
        return *this;
    }
    bool operator==(const ResizeParam<opTensor> right){
        float eps = 1e-6f;
        bool flag = fabsf(width_scale - right.width_scale) < eps;
        flag &= fabsf(height_scale - right.height_scale) < eps;
        return flag;
    }
    float width_scale{0.f};
    float height_scale{0.f};
};

template <typename opTensor>
struct MvnParam {

    MvnParam() = default;

    MvnParam(bool normalize_variance_in, bool across_channels_in, float eps_in) {
        normalize_variance = normalize_variance_in;
        across_channels = across_channels_in;
        eps = eps_in;
    }

    MvnParam(const MvnParam<opTensor>& right) {
        normalize_variance = right.normalize_variance;
        across_channels = right.across_channels;
        eps = right.eps;
    }

    MvnParam<opTensor>& operator=(const MvnParam<opTensor>& right) {
        this->normalize_variance = right.normalize_variance;
        this->across_channels = right.across_channels;
        this->eps = right.eps;
        return *this;
    }

    bool operator==(const MvnParam<opTensor>& right) {
        bool flag = this->normalize_variance == right.normalize_variance;
        flag = flag && this->across_channels == right.across_channels;
        return flag && (this->eps == right.eps);
    }

    bool normalize_variance{true};
    bool across_channels{true};
    float eps{1e-9};
};


template <typename opTensor>
struct ArgmaxParam {

    ArgmaxParam() = default;

    ArgmaxParam(bool out_max_val_in,int top_k_in, int axis_in) {
        out_max_val = out_max_val_in;
        has_axis = true;
        top_k = top_k_in;
        axis = axis_in;
    }
    
    ArgmaxParam(bool out_max_val_in,int top_k_in) {
        out_max_val = out_max_val_in;
        top_k = top_k_in;
        has_axis = false;
        axis = 3;
    }
    

    ArgmaxParam(const ArgmaxParam<opTensor>& right) {
        out_max_val = right.out_max_val;
        top_k = right.top_k;
        has_axis = right.has_axis;
        axis = right.axis;
    }

    ArgmaxParam<opTensor>& operator=(const ArgmaxParam<opTensor>& right) {
        this->out_max_val = right.out_max_val;
        this->top_k = right.top_k;
        this->axis = right.axis;
        this->has_axis = right.has_axis;
        return *this;
    }

    bool operator==(const ArgmaxParam<opTensor>& right) {
        bool flag = this->out_max_val == right.out_max_val;
        flag = flag && this->top_k == right.top_k;
        flag = flag && this->has_axis == right.has_axis;
        return flag && (this->axis == right.axis);
    }
    bool out_max_val{false};
    bool has_axis{true};
    int top_k{1};
    int axis{3};
};



template <typename opTensor>
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
        opTensor* input_scale, float eps_in = 1e-6f, int pin = 2) {

        across_spatial = is_across_spatial;
        channel_shared = is_shared_channel;
        p = pin;
        has_scale = true;
        scale = input_scale;
        eps = eps_in;
        CHECK_EQ(p == 2 || p == 1, true) << "only support L1 and L2 norm";
    }

    NormalizeParam(const NormalizeParam<opTensor>& right) {
        channel_shared = right.channel_shared;
        across_spatial = right.across_spatial;
        p = right.p;
        has_scale = right.has_scale;
        scale = right.scale;
        eps = right.eps;
    }

    NormalizeParam<opTensor>& operator=(const NormalizeParam<opTensor>& right) {
        this->channel_shared = right.channel_shared;
        this->across_spatial = right.across_spatial;
        this->scale = right.scale;
        this->p = right.p;
        this->has_scale = right.has_scale;
        this->eps = right.eps;
        return *this;
    }

    bool operator==(const NormalizeParam<opTensor>& right) {
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
    opTensor* scale{nullptr};
    float eps{1e-6f};
};

template <typename opTensor>
struct FcParam {
    FcParam() = default;

    FcParam(opTensor* input_weight, int output_num, int in_axis = 1,
            bool trans = false) {

        num_output = output_num;
        weights = input_weight;
        bias = nullptr;
        axis = in_axis;
        is_transpose_weights = trans;
    }
    FcParam(opTensor* input_weight, opTensor* input_bias, int output_num,
            int in_axis = 1, bool trans = false) {

        num_output = output_num;
        weights = input_weight;
        bias = input_bias;
        axis = in_axis;
        is_transpose_weights = trans;
    }

    FcParam(const FcParam<opTensor>& right) {
        weights = right.weights;
        bias = right.bias;
        num_output = right.num_output;
        axis = right.axis;
        is_transpose_weights = right.is_transpose_weights;
    }

    FcParam<opTensor>& operator=(const FcParam<opTensor>& right) {
        this->weights = right.weights;
        this->bias = right.bias;
        this->num_output = right.num_output;
        this->axis = right.axis;
        this->is_transpose_weights = right.is_transpose_weights;
        return *this;
    }

    bool operator==(const FcParam<opTensor>& right) {
        bool flag = this->is_transpose_weights == right.is_transpose_weights;
        flag = flag && (this->num_output == right.num_output) && (this->axis == right.axis);
        return flag && (this->weights == right.weights) && (this->bias == right.bias);
    }

    bool is_transpose_weights{false};
    int num_output;
    int axis{1};
    opTensor* weights{nullptr};
    opTensor* bias{nullptr};
};

template <typename opTensor>
struct EltwiseParam {
    typedef typename opTensor::Dtype DataDtype;
    EltwiseParam()
        : operation(Eltwise_unknow)
        , coeff()
    {}
    EltwiseParam(EltwiseType operation_in
            , std::vector<DataDtype> coeff_in = std::vector<DataDtype>({1,1}))
        : operation(operation_in)
        , coeff(coeff_in)
    {
        if ((operation == Eltwise_sum) && (coeff.size() == 0)) {
            coeff.push_back(1);
            coeff.push_back(1);
        }
    }

    EltwiseParam(const EltwiseParam<opTensor>& right)
        : operation(right.operation)
        , coeff(right.coeff)
    {}

    EltwiseParam<opTensor>& operator=(const EltwiseParam<opTensor>& right) {
        operation = right.operation;
        coeff.resize(right.coeff.size());
        for (int i = 0; i < coeff.size(); ++i) {
            coeff[i] = right.coeff[i];
        }
        return *this;
    }

    bool operator==(const EltwiseParam<opTensor>& right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (operation == right.operation);
        comp_eq = comp_eq && (coeff.size() == right.coeff.size());
        if (!comp_eq) {
            return comp_eq;
        }
        for (int i = 0; i < coeff.size(); ++i) {
            comp_eq = comp_eq && (coeff[i] == right.coeff[i]);
        }
        return comp_eq;
    }
    EltwiseType operation;
    std::vector<DataDtype> coeff;
};

template <typename opTensor>
struct EltwiseActiveParam {
    EltwiseActiveParam()
        : eltwise_param()
        , activation_param()
        , has_activation(false)
    {}
    EltwiseActiveParam(EltwiseParam<opTensor> &eltwise_param_in,
                ActivationParam<opTensor> &activation_param_in)
            : eltwise_param(eltwise_param_in)
            , activation_param(activation_param_in)
            , has_activation(true)
    {}
    EltwiseActiveParam(const EltwiseActiveParam &right)
            : eltwise_param(right.eltwise_param)
            , activation_param(right.activation_param)
            , has_activation(right.has_activation)
    {}
    EltwiseActiveParam &operator=(const EltwiseActiveParam &right) {
        eltwise_param = right.eltwise_param;
        activation_param = right.activation_param;
        has_activation = right.has_activation;
        return *this;
    }
    bool operator==(const EltwiseActiveParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (eltwise_param == right.eltwise_param);
        comp_eq = comp_eq && (activation_param == right.activation_param);
        comp_eq = comp_eq && (has_activation == right.has_activation);
        return comp_eq;
    }

    EltwiseParam<opTensor> eltwise_param;
    ActivationParam<opTensor> activation_param;
    bool has_activation;
};

template <typename opTensor>
struct PriorBoxParam {

    PriorBoxParam(){}
    PriorBoxParam(std::vector<float> min_in, std::vector<float> max_in, \
        std::vector<float> aspect_in, std::vector<float> variance_in,
        bool flip, bool clip, int image_width, int image_height, \
        float step_width, float step_height, float offset_in, std::vector<PriorType> order_in) {
        is_flip = flip;
        is_clip = clip;
        min_size = min_in;
        img_w = image_width;
        img_h = image_height;
        step_w = step_width;
        step_h = step_height;
        offset = offset_in;
        order = order_in;
        aspect_ratio.clear();
        aspect_ratio.push_back(1.f);

        variance.clear();
        if (variance_in.size() == 1) {
            variance.push_back(variance_in[0]);
            variance.push_back(variance_in[0]);
            variance.push_back(variance_in[0]);
            variance.push_back(variance_in[0]);
        } else {
            CHECK_EQ(variance_in.size(), 4) << "variance size must = 1 or = 4";
            variance.push_back(variance_in[0]);
            variance.push_back(variance_in[1]);
            variance.push_back(variance_in[2]);
            variance.push_back(variance_in[3]);
        }

        for (int i = 0; i < aspect_in.size(); ++i) {
            float ar = aspect_in[i];
            bool already_exist = false;
            for (int j = 0; j < aspect_ratio.size(); ++j) {
                if (fabs(ar - aspect_ratio[j]) < 1e-6) {
                    already_exist = true;
                    break;
                }
            }
            if (!already_exist) {
                aspect_ratio.push_back(ar);
                if (is_flip) {
                    aspect_ratio.push_back(1.f/ar);
                }
            }
        }
        prior_num = min_size.size() * aspect_ratio.size();
        max_size.clear();
        if (max_in.size() > 0) {
            CHECK_EQ(max_in.size(), min_size.size()) << "max_size num must = min_size num";
            for (int i = 0; i < max_in.size(); ++i) {
                CHECK_GT(max_in[i], min_size[i]) << "max_size val must > min_size val";
                max_size.push_back(max_in[i]);
                prior_num++;
            }
        }
    }
    PriorBoxParam(const PriorBoxParam<opTensor>& right) {
        is_flip = right.is_flip;
        is_clip = right.is_clip;
        min_size = right.min_size;
        max_size = right.max_size;
        aspect_ratio = right.aspect_ratio;
        variance = right.variance;
        img_w = right.img_w;
        img_h = right.img_h;
        step_w = right.step_w;
        step_h = right.step_h;
        offset = right.offset;
        order = right.order;
        prior_num = right.prior_num;
    }
    PriorBoxParam<opTensor>& operator=(const PriorBoxParam<opTensor>& right) {
        this->is_flip = right.is_flip;
        this->is_clip = right.is_clip;
        this->min_size = right.min_size;
        this->max_size = right.max_size;
        this->aspect_ratio = right.aspect_ratio;
        this->variance = right.variance;
        this->img_w = right.img_w;
        this->img_h = right.img_h;
        this->step_w = right.step_w;
        this->step_h = right.step_h;
        this->offset = right.offset;
        this->order = right.order;
        this->prior_num = right.prior_num;
        return *this;
    }
    bool operator==(const PriorBoxParam<opTensor>& right) {
        bool flag = is_flip == right.is_flip;
        flag = flag && (is_clip == right.is_clip);
        if (min_size.size() != right.min_size.size()) {
            return false;
        }
        for (int i = 0; i < min_size.size(); ++i) {
            if (min_size[i] != right.min_size[i]) {
                return false;
            }
        }
        if (max_size.size() != right.max_size.size()) {
            return false;
        }
        for (int i = 0; i < max_size.size(); ++i) {
            if (max_size[i] != right.max_size[i]) {
                return false;
            }
        }
        if (aspect_ratio.size() != right.aspect_ratio.size()) {
            return false;
        }
        for (int i = 0; i < aspect_ratio.size(); ++i) {
            if (aspect_ratio[i] != right.aspect_ratio[i]) {
                return false;
            }
        }
        if (variance.size() != right.variance.size()) {
            return false;
        }
        for (int i = 0; i < variance.size(); ++i) {
            if (variance[i] != right.variance[i]) {
                return false;
            }
        }
        flag = flag && (img_w == right.img_w);
        flag = flag && (img_h == right.img_h);
        flag = flag && (step_w == right.step_w);
        flag = flag && (step_h == right.step_h);
        flag = flag && (offset == right.offset);
        flag = flag && (order == right.order);
        flag = flag && (prior_num == right.prior_num);
        return flag;
    }

    bool is_flip;
    bool is_clip;
    std::vector<float> min_size;
    std::vector<float> max_size;
    std::vector<float> aspect_ratio;
    std::vector<float> variance;
    int img_w{0};
    int img_h{0};
    float step_w{0};
    float step_h{0};
    float offset{0.5};
    int prior_num{0};
    std::vector<PriorType> order;
};
template <typename opTensor>
struct DeformableConvParam {

    DeformableConvParam() : group(-1), pad_h(-1), pad_w(-1),
                  stride_h(-1), stride_w(-1),
                  dilation_h(-1), dilation_w(-1), axis(-1),
                  weight_tensor(NULL), bias_tensor(NULL), alpha(1.0), beta(0.0) {}

    DeformableConvParam(int group_in, int pad_h_in, int pad_w_in, int stride_h_in,
              int stride_w_in, int dilation_h_, int dilation_w_, opTensor* weight,
              opTensor* bias,  int axis_in = 1, float alpha_in = 1.0, float beta_in = 0.0)
            : group(group_in), pad_h(pad_h_in), pad_w(pad_w_in)
            , stride_h(stride_h_in), stride_w(stride_w_in)
            , dilation_h(dilation_h_), dilation_w(dilation_w_)
            , axis(axis_in)
            , weight_tensor(weight), bias_tensor(bias)
            , alpha(alpha_in), beta(beta_in)
    {}

    DeformableConvParam(const DeformableConvParam &right)
            : group(right.group), pad_h(right.pad_h)
            , pad_w(right.pad_w), stride_h(right.stride_h)
            , stride_w(right.stride_w), dilation_h(right.dilation_h)
            , dilation_w(right.dilation_w)
            , axis(right.axis)
            , weight_tensor(right.weight_tensor)
            , bias_tensor(right.bias_tensor)
            , alpha(right.alpha)
            , beta(right.beta)
    {}

    DeformableConvParam &operator=(const DeformableConvParam &right) {
        group = right.group;
        pad_h = right.pad_h;
        pad_w = right.pad_w;
        stride_h = right.stride_h;
        stride_w = right.stride_w;
        dilation_h = right.dilation_h;
        dilation_w = right.dilation_w;
        axis = right.axis;
        weight_tensor = right.weight_tensor;
        bias_tensor = right.bias_tensor;
        alpha = right.alpha;
        beta = right.beta;
        return *this;
    }

    bool operator==(const DeformableConvParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (group == right.group);
        comp_eq = comp_eq && (pad_h == right.pad_h);
        comp_eq = comp_eq && (pad_w == right.pad_w);
        comp_eq = comp_eq && (stride_h == right.stride_h);
        comp_eq = comp_eq && (stride_w == right.stride_w);
        comp_eq = comp_eq && (dilation_h == right.dilation_h);
        comp_eq = comp_eq && (dilation_w == right.dilation_w);
        comp_eq = comp_eq && (axis == right.axis);
        comp_eq = comp_eq && (weight_tensor == right.weight_tensor);
        comp_eq = comp_eq && (bias_tensor == right.bias_tensor);
        comp_eq = comp_eq && (alpha == right.alpha);
        comp_eq = comp_eq && (beta == right.beta);
        return comp_eq;
    }

    inline const opTensor* weight() {
        return weight_tensor;
    }

    inline const opTensor* bias() {
        return bias_tensor;
    }

    inline opTensor* mutable_weight() {
        return weight_tensor;
    }

    inline opTensor* mutable_bias() {
        return bias_tensor;
    }

    int group;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int axis;
    float alpha;
    float beta;

private:
    opTensor* weight_tensor;
    opTensor* bias_tensor;
};

template <typename opTensor>
struct SPPParam {
    SPPParam() = default;
    SPPParam(int pyramid_height_in, PoolingType pool_type_in)
            : pyramid_height(pyramid_height_in)
            , pool_type(pool_type_in)
    {}
    SPPParam(const SPPParam &right)
            : pyramid_height(right.pyramid_height)
            , pool_type(right.pool_type)
    {}
    SPPParam &operator=(const SPPParam &right) {
        pyramid_height = right.pyramid_height;
        pool_type = right.pool_type;
        return *this;
    }
    bool operator==(const SPPParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (pyramid_height == right.pyramid_height);
        comp_eq = comp_eq && (pool_type == right.pool_type);
        return comp_eq;
    }

    int pyramid_height;
    PoolingType pool_type;
};

template <typename opTensor>
struct CropParam {
    CropParam() = default;
    CropParam(int axis_in, std::vector<int> offset_in, std::vector<int> shape_in)
            : axis(axis_in)
            , offset(offset_in)
            , shape(shape_in)
    {}
    CropParam(const CropParam &right)
            : axis(right.axis)
            , offset(right.offset)
            , shape(right.shape)
    {}
    CropParam &operator=(const CropParam &right) {
        axis = right.axis;
        offset = right.offset;
        shape = right.shape;
        return *this;
    }
    bool operator==(const CropParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (axis == right.axis);
        comp_eq = comp_eq && (offset == right.offset);
        comp_eq = comp_eq && (shape == right.shape);
        return comp_eq;
    }
    int axis = 1;
    std::vector<int>  offset;
    std::vector<int>  shape;
};


template <typename opTensor>
struct PadParam {
    PadParam() = default;
    PadParam(std::vector<int> pad_c_in, std::vector<int> pad_h_in, std::vector<int> pad_w_in)
            : pad_c(pad_c_in)
            , pad_h(pad_h_in)
            , pad_w(pad_w_in)
    {}
    PadParam(const PadParam &right)
            : pad_c(right.pad_c)
            , pad_h(right.pad_h)
            , pad_w(right.pad_w)
    {}
    PadParam &operator=(const PadParam &right) {
        pad_c = right.pad_c;
        pad_h = right.pad_h;
        pad_w = right.pad_w;
        return *this;
    }
    bool operator==(const PadParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (pad_c == right.pad_c);
        comp_eq = comp_eq && (pad_h == right.pad_h);
        comp_eq = comp_eq && (pad_w == right.pad_w);
        return comp_eq;
    }
    std::vector<int>  pad_c;
    std::vector<int>  pad_h;
    std::vector<int>  pad_w;
};

template <typename opTensor>
struct LrnParam {
    LrnParam() = default;
    LrnParam(int local_size_in, float alpha_in,  float beta_in, float k_in, NormRegion norm_region_in)
            : local_size(local_size_in)
            , alpha(alpha_in)
            , beta(beta_in)
            , k(k_in)
            , norm_region(norm_region_in)
    {}
    LrnParam(const LrnParam &right)
            : local_size(right.local_size)
            , alpha(right.alpha)
            , beta(right.beta)
            , k(right.k)
            , norm_region(right.norm_region)
    {}
    LrnParam &operator=(const LrnParam &right) {
        local_size = right.local_size;
        alpha = right.alpha;
        beta = right.beta;
        k = right.k;
        norm_region = right.norm_region;
        return *this;
    }
    bool operator==(const LrnParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (local_size == right.local_size);
        comp_eq = comp_eq && (alpha == right.alpha);
        comp_eq = comp_eq && (beta == right.beta);
        comp_eq = comp_eq && (k == right.k);
        comp_eq = comp_eq && (norm_region == right.norm_region);
        return comp_eq;
    }
    int local_size{5};
    float alpha{1.};
    float beta{0.75};
    float k{1.};
    NormRegion norm_region{ACROSS_CHANNELS};
};

template <class opTensor>
struct MultiClassNMSParam {
    MultiClassNMSParam() = default;

    MultiClassNMSParam(int bg_id, int keep_topk, int nms_topk, float nms_threshold, \
        float confidence_threshold, float eta = 1.f) {
        background_id = bg_id;
        keep_top_k = keep_topk;
        nms_top_k = nms_topk;
        nms_thresh = nms_threshold;
        conf_thresh = confidence_threshold;
        nms_eta = eta;
    }

    void init(int bg_id, int keep_topk, int nms_topk, float nms_threshold, \
        float confidence_threshold, float eta = 1.f) {
        background_id = bg_id;
        keep_top_k = keep_topk;
        nms_top_k = nms_topk;
        nms_thresh = nms_threshold;
        conf_thresh = confidence_threshold;
        nms_eta = eta;
    }

    MultiClassNMSParam(const MultiClassNMSParam<opTensor> &right) {
        background_id = right.background_id;
        keep_top_k = right.keep_top_k;
        nms_top_k = right.nms_top_k;
        nms_thresh = right.nms_thresh;
        conf_thresh = right.conf_thresh;
        nms_eta = right.nms_eta;
    }

    MultiClassNMSParam<opTensor> &operator=(const MultiClassNMSParam<opTensor> &right) {
        this->background_id = right.background_id;
        this->keep_top_k = right.keep_top_k;
        this->nms_top_k = right.nms_top_k;
        this->nms_thresh = right.nms_thresh;
        this->conf_thresh = right.conf_thresh;
        this->nms_eta = right.nms_eta;
        return *this;
    }

    bool operator==(const MultiClassNMSParam<opTensor> &right) {
        bool flag = (background_id == right.background_id);
        flag = flag && (keep_top_k == right.keep_top_k);
        flag = flag && (nms_top_k == right.nms_top_k);
        flag = flag && (nms_thresh == right.nms_thresh);
        flag = flag && (conf_thresh == right.conf_thresh);
        flag = flag && (nms_eta == right.nms_eta);
        return flag;
    }

    int background_id{0};
    int keep_top_k{-1};
    float conf_thresh;
    int nms_top_k;
    float nms_thresh{0.3f};
    float nms_eta{1.f};
};

template<typename opTensor>
struct DetectionOutputParam {

    DetectionOutputParam() = default;

    DetectionOutputParam(int classes, int bg_id, int keep_topk, int nms_topk, float nms_threshold, \
        float confidence_threshold, bool share_loc = true, bool variance_in_target = false, \
        int codetype = 1, float eta = 1.f) {
        class_num = classes;
        background_id = bg_id;
        keep_top_k = keep_topk;
        nms_top_k = nms_topk;
        nms_thresh = nms_threshold;
        conf_thresh = confidence_threshold;
        share_location = share_loc;
        variance_encode_in_target = variance_in_target;
        type = (CodeType) codetype;
        nms_eta = eta;
    }

    void init(int classes, int bg_id, int keep_topk, int nms_topk, float nms_threshold, \
        float confidence_threshold, bool share_loc = true, bool variance_in_target = false, \
        int codetype = 1, float eta = 1.f) {
        class_num = classes;
        background_id = bg_id;
        keep_top_k = keep_topk;
        nms_top_k = nms_topk;
        nms_thresh = nms_threshold;
        conf_thresh = confidence_threshold;
        share_location = share_loc;
        variance_encode_in_target = variance_in_target;
        type = (CodeType) codetype;
        nms_eta = eta;
    }

    DetectionOutputParam(const DetectionOutputParam<opTensor> &right) {
        class_num = right.class_num;
        background_id = right.background_id;
        keep_top_k = right.keep_top_k;
        nms_top_k = right.nms_top_k;
        nms_thresh = right.nms_thresh;
        conf_thresh = right.conf_thresh;
        share_location = right.share_location;
        variance_encode_in_target = right.variance_encode_in_target;
        type = right.type;
        nms_eta = right.nms_eta;
    }

    DetectionOutputParam<opTensor> &operator=(const DetectionOutputParam<opTensor> &right) {
        this->class_num = right.class_num;
        this->background_id = right.background_id;
        this->keep_top_k = right.keep_top_k;
        this->nms_top_k = right.nms_top_k;
        this->nms_thresh = right.nms_thresh;
        this->conf_thresh = right.conf_thresh;
        this->share_location = right.share_location;
        this->variance_encode_in_target = right.variance_encode_in_target;
        this->type = right.type;
        this->nms_eta = right.nms_eta;
        return *this;
    }

    bool operator==(const DetectionOutputParam<opTensor> &right) {
        bool flag = class_num == right.class_num;
        flag = flag && (background_id == right.background_id);
        flag = flag && (keep_top_k == right.keep_top_k);
        flag = flag && (nms_top_k == right.nms_top_k);
        flag = flag && (nms_thresh == right.nms_thresh);
        flag = flag && (conf_thresh == right.conf_thresh);
        flag = flag && (share_location == right.share_location);
        flag = flag && (variance_encode_in_target == right.variance_encode_in_target);
        flag = flag && (type == right.type);
        flag = flag && (nms_eta == right.nms_eta);
        return flag;
    }

    bool share_location{true};
    bool variance_encode_in_target{false};
    int class_num;
    int background_id{0};
    int keep_top_k{-1};
    CodeType type{CORNER};
    float conf_thresh;
    int nms_top_k;
    float nms_thresh{0.3f};
    float nms_eta{1.f};
};

template <typename opTensor>
struct BoxCoderParam {

    BoxCoderParam() = default;

    BoxCoderParam(int coder_type) {
        type = BoxCoderType(coder_type);
    }

    BoxCoderParam(const BoxCoderParam& right) {
        type = right.type;
    }

    BoxCoderParam& operator=(const BoxCoderParam& right) {
        this->type = right.type;
        return *this;
    }

    bool operator==(const BoxCoderParam& right) {
        return this->type == right.type;
    }

    BoxCoderType type{ENCODE_CENTER};
};

template <typename opTensor>
struct DfmbpsRoiAlignParam {
    DfmbpsRoiAlignParam()
        : heat_map_a(0)
        , output_dim(0)
    {}
    DfmbpsRoiAlignParam(float heat_map_a_in, int output_dim_in,
        float heat_map_b_in = 0, float pad_ratio_in = 0,
        float trans_std_in = 0.1, int sample_per_part_in = 4,
        int group_height_in = 7,  int group_width_in = 7,
        int pooled_height_in = 7, int pooled_width_in = 7,
        int part_height_in = 7, int part_width_in = 7)
        : heat_map_a(heat_map_a_in), output_dim(output_dim_in)
        , heat_map_b(heat_map_b_in), pad_ratio(pad_ratio_in)
        , trans_std(trans_std_in), sample_per_part(sample_per_part_in)
        , group_height(group_height_in), group_width(group_width_in)
        , pooled_height(pooled_height_in), pooled_width(pooled_width_in)

    {}
    ~DfmbpsRoiAlignParam(){}

    DfmbpsRoiAlignParam(const DfmbpsRoiAlignParam &right)
        : heat_map_a(right.heat_map_a)
        , output_dim(right.output_dim)
        , heat_map_b(right.heat_map_b)
        , pad_ratio(right.pad_ratio)
        , trans_std(right.trans_std)
        , sample_per_part(right.sample_per_part)
        , group_height(right.group_height)
        , group_width(right.group_width)
        , pooled_height(right.pooled_height)
        , pooled_width(right.pooled_width)
    {}

    DfmbpsRoiAlignParam &operator=(const DfmbpsRoiAlignParam &right) {
        heat_map_a = right.heat_map_a;
        output_dim = right.output_dim;
        heat_map_b = right.heat_map_b;
        pad_ratio = right.pad_ratio;
        trans_std = right.trans_std;
        sample_per_part = right.sample_per_part;
        group_height = right.group_height;
        group_width = right.group_width;
        pooled_height = right.pooled_height;
        pooled_width = right.pooled_width;
        return *this;
    }

    bool operator==(const DfmbpsRoiAlignParam &right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (heat_map_a == right.heat_map_a);
        comp_eq &= comp_eq && (output_dim == right.output_dim);
        comp_eq &= comp_eq && (heat_map_b == right.heat_map_b);
        comp_eq &= comp_eq && (pad_ratio == right.pad_ratio);
        comp_eq &= comp_eq && (trans_std == right.trans_std);
        comp_eq &= comp_eq && (sample_per_part == right.sample_per_part);
        comp_eq &= comp_eq && (group_height == right.group_height);
        comp_eq &= comp_eq && (group_width == right.group_width);
        comp_eq &= comp_eq && (pooled_height == right.pooled_height);
        comp_eq &= comp_eq && (pooled_width == right.pooled_width);
        return comp_eq;
    }

    float heat_map_a;
    int output_dim;

    float heat_map_b;
    float pad_ratio;
    float trans_std;
    int sample_per_part;
    int group_height;
    int group_width;
    int pooled_height;
    int pooled_width;
    int part_height;
    int part_width;
};
template <typename vectors>
inline
bool compare_vectors(vectors &a, const vectors &b) {
    if (a.size()!= b.size()){
        return false;
    }
    bool comp = true;
    for (int i = 0; i < a.size(); ++i) {
        comp &= comp && (a[i] == b[i]);
    }
    return comp;
}
template <typename vectors>
inline
void copy_vectors(vectors &out, const vectors &in) {
    out.resize(in.size());
    for (int i = 0; i < out.size(); ++i) {
        out[i] = in[i];
    }
}

template <typename opTensor>
struct RoiPoolParam {
    RoiPoolParam() = default;
    RoiPoolParam(int pooled_height_in, int pooled_width_in, float spatial_scale_in,
            int height_in, int width_in)
            : pooled_height(pooled_height_in)
            , pooled_width(pooled_width_in)
            , spatial_scale(spatial_scale_in)
            , height(height_in)
            , width(width_in)
    {}
    RoiPoolParam(int pooled_height_in, int pooled_width_in, float spatial_scale_in)
            : pooled_height(pooled_height_in)
            , pooled_width(pooled_width_in)
            , spatial_scale(spatial_scale_in)
    {}
    RoiPoolParam(const RoiPoolParam &right)
            : pooled_height(right.pooled_height)
            , pooled_width(right.pooled_width)
            , spatial_scale(right.spatial_scale)
            , height(right.height)
            , width(right.width)
    {}
    RoiPoolParam &operator=(const RoiPoolParam &right) {
        pooled_height = right.pooled_height;
        pooled_width = right.pooled_width;
        spatial_scale = right.spatial_scale;
        height = right.height;
        width = right.width;
        return *this;
    }
    bool operator==(const RoiPoolParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (pooled_height == right.pooled_height);
        comp_eq = comp_eq && (pooled_width == right.pooled_width);
        comp_eq = comp_eq && (spatial_scale == right.spatial_scale);
        comp_eq = comp_eq && (height == right.height);
        comp_eq = comp_eq && (width == right.width);
        return comp_eq;
    }
    int pooled_height;
    int pooled_width;
    float spatial_scale;
    int height{1};
    int width{1};
};

template <class opTensor>
struct FlattenParam {
    FlattenParam() = default;
    FlattenParam(const FlattenParam& right) {}
    FlattenParam& operator=(const FlattenParam& right){ return *this;}
    bool operator==(const FlattenParam& right){
        return true;
    }
};
template <class opTensor>
struct AxpyParam {
    AxpyParam() = default;
    AxpyParam(const AxpyParam& right) { }
    AxpyParam& operator=(const AxpyParam& right){ return *this;}
    bool operator==(const AxpyParam& right){
        return true;
    }
};
template <typename opTensor>
struct CtcAlignParam {
    CtcAlignParam() = default;
    CtcAlignParam(int blank_in, bool merge_repeated_in)
            : blank(blank_in)
            , merge_repeated(merge_repeated_in)
    {}
    CtcAlignParam(const CtcAlignParam &right)
            : blank(right.blank)
            , merge_repeated(right.merge_repeated)
    {}
    CtcAlignParam &operator=(const CtcAlignParam &right) {
        blank = right.blank;
        merge_repeated = right.merge_repeated;
        return *this;
    }
    bool operator==(const CtcAlignParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (blank == right.blank);
        comp_eq = comp_eq && (merge_repeated == right.merge_repeated);
        return comp_eq;
    }
    int blank;
    bool merge_repeated;
};

template <typename opTensor>
struct Im2SequenceParam {
    Im2SequenceParam() = default;
    Im2SequenceParam(int window_h_in,
                     int window_w_in,
                     int pad_up_in,
                     int pad_down_in,
                     int pad_left_in,
                     int pad_right_in,
                     int stride_h_in,
                     int stride_w_in,
                     int dilation_h_in,
                     int dilation_w_in)
               : window_h(window_h_in)
                , window_w(window_w_in)
                , pad_up(pad_up_in)
                , pad_down(pad_down_in)
                , pad_left(pad_left_in)
                , pad_right(pad_right_in)
                , stride_h(stride_h_in)
                , stride_w(stride_w_in)
                , dilation_h(dilation_h_in)
                , dilation_w(dilation_w_in)
    {}
    Im2SequenceParam(const Im2SequenceParam &right)
                : window_h(right.window_h)
                , window_w(right.window_w)
                , pad_up(right.pad_up)
                , pad_down(right.pad_down)
                , pad_left(right.pad_left)
                , pad_right(right.pad_right)
                , stride_h(right.stride_h)
                , stride_w(right.stride_w)
                , dilation_h(right.dilation_h)
                , dilation_w(right.dilation_w)
    {}
    Im2SequenceParam &operator=(const Im2SequenceParam &right) {
        window_h = right.window_h;
        window_w = right.window_w;
        pad_up = right.pad_up;
        pad_down = right.pad_down;
        pad_left = right.pad_left;
        pad_right = right.pad_right;
        stride_h = right.stride_h;
        stride_w = right.stride_w;
        dilation_h = right.dilation_h;
        dilation_w = right.dilation_w;
        return *this;
    }
    bool operator==(const Im2SequenceParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (window_h == right.window_h);
        comp_eq = comp_eq && (window_w == right.window_w);
        comp_eq = comp_eq && (pad_up == right.pad_up);
        comp_eq = comp_eq && (pad_down == right.pad_down);
        comp_eq = comp_eq && (pad_left == right.pad_left);
        comp_eq = comp_eq && (pad_right == right.pad_right);
        comp_eq = comp_eq && (stride_h == right.stride_h);
        comp_eq = comp_eq && (stride_w == right.stride_w);
        comp_eq = comp_eq && (dilation_h == right.dilation_h);
        comp_eq = comp_eq && (dilation_w == right.dilation_w);
        return comp_eq;
    }
    int window_h;
    int window_w;
    int pad_up;
    int pad_down;
    int pad_left;
    int pad_right;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
};

template <typename opTensor>
struct CastParam {
    CastParam() = default;
    CastParam(int in_type_in, int out_type_in)
            : in_type(in_type_in)
            , out_type(out_type_in)
    {}
    CastParam(const CastParam &right)
            : in_type(right.in_type)
            , out_type(right.out_type)
    {}
    CastParam &operator=(const CastParam &right) {
        in_type = right.in_type;
        out_type = right.out_type;
        return *this;
    }
    bool operator==(const CastParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (in_type == right.in_type);
        comp_eq = comp_eq && (out_type == right.out_type);
        return comp_eq;
    }
    int in_type;
    int out_type;
};
template <typename opTensor>
struct EmbeddingParam {
    EmbeddingParam() = default;
    EmbeddingParam(int word_num_in, int emb_dim_in, int padding_idx_in,
             opTensor* weight_tensor_in)
            : word_num(word_num_in)
            , emb_dim(emb_dim_in)
            , padding_idx(padding_idx_in)
            , weight_tensor(weight_tensor_in)
    {}
    EmbeddingParam(const EmbeddingParam &right)
            : word_num(right.word_num)
            , emb_dim(right.emb_dim)
            , padding_idx(right.padding_idx)
            , weight_tensor(right.weight_tensor)
    {}
    EmbeddingParam &operator=(const EmbeddingParam &right) {
        word_num = right.word_num;
        emb_dim = right.emb_dim;
        padding_idx = right.padding_idx;
        weight_tensor = right.weight_tensor;
        return *this;
    }
    bool operator==(const EmbeddingParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (word_num == right.word_num);
        comp_eq = comp_eq && (emb_dim == right.emb_dim);
        comp_eq = comp_eq && (padding_idx == right.padding_idx);
        comp_eq = comp_eq && (weight_tensor == right.weight_tensor);
        return comp_eq;
    }
    inline const opTensor* weight() {
        return weight_tensor;
    }

    inline opTensor* mutable_weight() {
        return weight_tensor;
    }
    int emb_dim;
    int word_num;
    int padding_idx;
private:
    opTensor* weight_tensor;
};

template <typename opTensor>
struct LayerNormParam {
    LayerNormParam() = default;
    LayerNormParam(int axis_in, float eps_in, opTensor* weights_scale, opTensor* weights_bias) {
        axis = axis_in;
        eps = eps_in;
        scale = weights_scale;
        bias = weights_bias;
    }
    LayerNormParam(const LayerNormParam &right) {
        axis = right.axis;
        eps = right.eps;
        scale = right.scale;
        bias = right.bias;
    }
    LayerNormParam &operator=(const LayerNormParam &right) {
        this->axis = right.axis;
        this->eps = right.eps;
        this->scale = right.scale;
        this->bias = right.bias;
        return *this;
    }
    bool operator==(const LayerNormParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (axis == right.axis);
        comp_eq = comp_eq && (fabsf(eps - right.eps) < 1e-7f);
        comp_eq = comp_eq && (scale == scale);
        comp_eq = comp_eq && (bias == bias);
        return comp_eq;
    }
    inline const opTensor* scale_weights() {
        return scale;
    }

    inline opTensor* mutable_scale_weights() {
        return scale;
    }

    inline const opTensor* bias_weights() {
        return bias;
    }

    inline opTensor* mutable_bias_weights() {
        return bias;
    }

    int axis;
    float eps{1e-5f};

private:
    opTensor* scale;
    opTensor* bias;
};

}
}
#endif //SABER_FUNCS_PARAM_H
