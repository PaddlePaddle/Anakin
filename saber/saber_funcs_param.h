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

template<typename TargetType>
struct ResizeParam{
    ResizeParam() = default;
    explicit ResizeParam(float scale_w, float scale_h){
        bool flag = scale_w > 0.f && scale_h > 0.f;
        CHECK_EQ(flag, true) << "wrong parameters";
        width_scale = scale_w;
        height_scale = scale_h;
    }
    ResizeParam(const ResizeParam<TargetType>& right){
        width_scale = right.width_scale;
        height_scale = right.height_scale;
    }
    ResizeParam<TargetType>& operator=(const ResizeParam<TargetType>& right){
        this->width_scale = right.width_scale;
        this->height_scale = right.height_scale;
        return *this;
    }
    bool operator==(const ResizeParam<TargetType>& right){
        float eps = 1e-6;
        bool flag = fabsf(width_scale - right.width_scale) < eps;
        flag &= fabsf(height_scale - right.height_scale) < eps;
        return flag;
    }
    float width_scale{0.0f};
    float height_scale{0.0f};
};
template <typename TargetType>
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

template <typename TargetType>
struct ScaleParam {
    typedef float DataDtype;
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
    std::vector<float> scale_w;
    std::vector<float> scale_b;
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

template <typename TargetType>
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

template <typename TargetType>
struct FlattenParam {
    FlattenParam() = default;
    FlattenParam(const FlattenParam& right) {}
    FlattenParam& operator=(const FlattenParam& right){ return *this;}
    bool operator==(const FlattenParam& right){
        return true;
    }
};

template <typename TargetType>
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

template <typename TargetType>
struct LayerNormParam {
    LayerNormParam() = default;
    LayerNormParam(int axis_in, float eps_in, Tensor<TargetType>* weights_scale, Tensor<TargetType>* weights_bias) {
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
    inline const Tensor<TargetType>* scale_weights() {
        return scale;
    }

    inline Tensor<TargetType>* mutable_scale_weights() {
        return scale;
    }

    inline const Tensor<TargetType>* bias_weights() {
        return bias;
    }

    inline Tensor<TargetType>* mutable_bias_weights() {
        return bias;
    }

    int axis;
    float eps{1e-5f};

private:
    Tensor<TargetType>* scale;
    Tensor<TargetType>* bias;
};

template <typename TargetType>
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

template <typename TargetType>
struct MvnParam {

    MvnParam() = default;

    MvnParam(bool normalize_variance_in, bool across_channels_in, float eps_in) {
        normalize_variance = normalize_variance_in;
        across_channels = across_channels_in;
        eps = eps_in;
    }

    MvnParam(const MvnParam<TargetType>& right) {
        normalize_variance = right.normalize_variance;
        across_channels = right.across_channels;
        eps = right.eps;
    }

    MvnParam<TargetType>& operator=(const MvnParam<TargetType>& right) {
        this->normalize_variance = right.normalize_variance;
        this->across_channels = right.across_channels;
        this->eps = right.eps;
        return *this;
    }

    bool operator==(const MvnParam<TargetType>& right) {
        bool flag = this->normalize_variance == right.normalize_variance;
        flag = flag && this->across_channels == right.across_channels;
        return flag && (this->eps == right.eps);
    }

    bool normalize_variance{true};
    bool across_channels{true};
    float eps{1e-9};
};

}
}
#endif //SABER_FUNCS_PARAM_H
