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
struct ConvParam {

    ConvParam()
            : group(-1), pad_h(-1), pad_w(-1),
              stride_h(-1), stride_w(-1),
              dilation_h(-1), dilation_w(-1),
              weight_tensor(NULL), bias_tensor(NULL), alpha(1.0), beta(0.0) {}

    ConvParam(int group_in, int pad_h_in, int pad_w_in,
              int stride_h_in, int stride_w_in, int dilation_h_, int dilation_w_,
              Tensor<TargetType>* weight, Tensor<TargetType>* bias,
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
        comp_eq &= (group == right.group);
        comp_eq &= (pad_h == right.pad_h);
        comp_eq &= (pad_w == right.pad_w);
        comp_eq &= (stride_h == right.stride_h);
        comp_eq &= (stride_w == right.stride_w);
        comp_eq &= (dilation_h == right.dilation_h);
        comp_eq &= (dilation_w == right.dilation_w);
        comp_eq &= (weight_tensor == right.weight_tensor);
        comp_eq &= (bias_tensor == right.bias_tensor);
        comp_eq &= (alpha == right.alpha);
        comp_eq &= (beta == right.beta);
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

private:
    Tensor<TargetType>* weight_tensor;
    Tensor<TargetType>* bias_tensor;

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

template <typename TargetType>
struct EmbeddingParam {
    EmbeddingParam() = default;
    EmbeddingParam(int word_num_in, int emb_dim_in, int padding_idx_in,
            Tensor<TargetType>* weight_tensor_in)
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
    inline const Tensor<TargetType>* weight() {
        return weight_tensor;
    }

    inline Tensor<TargetType>* mutable_weight() {
        return weight_tensor;
    }
    int emb_dim;
    int word_num;
    int padding_idx;
private:
    Tensor<TargetType>* weight_tensor;
};

template <typename TargetType>
struct FcParam {
    FcParam() = default;

    FcParam(Tensor<TargetType>* input_weight, int output_num, int in_axis = 1,
            bool trans = false) {

        num_output = output_num;
        weights = input_weight;
        bias = nullptr;
        axis = in_axis;
        is_transpose_weights = trans;
    }
    FcParam(Tensor<TargetType>* input_weight, Tensor<TargetType>* input_bias, int output_num,
            int in_axis = 1, bool trans = false) {

        num_output = output_num;
        weights = input_weight;
        bias = input_bias;
        axis = in_axis;
        is_transpose_weights = trans;
    }

    FcParam(const FcParam &right) {
        weights = right.weights;
        bias = right.bias;
        num_output = right.num_output;
        axis = right.axis;
        is_transpose_weights = right.is_transpose_weights;
    }

    FcParam& operator=(const FcParam &right) {
        this->weights = right.weights;
        this->bias = right.bias;
        this->num_output = right.num_output;
        this->axis = right.axis;
        this->is_transpose_weights = right.is_transpose_weights;
        return *this;
    }

    bool operator==(const FcParam &right) {
        bool flag = this->is_transpose_weights == right.is_transpose_weights;
        flag = flag && (this->num_output == right.num_output) && (this->axis == right.axis);
        return flag && (this->weights == right.weights) && (this->bias == right.bias);
    }

    bool is_transpose_weights{false};
    int num_output;
    int axis{1};
    Tensor<TargetType>* weights{nullptr};
    Tensor<TargetType>* bias{nullptr};
};

}
}
#endif //SABER_FUNCS_PARAM_H
