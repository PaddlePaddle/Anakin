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
#ifndef ANAKIN_SABER_LITE_FUNCS_OP_PARAM_H
#define ANAKIN_SABER_LITE_FUNCS_OP_PARAM_H

#include "saber/lite/core/common_lite.h"
namespace anakin{

namespace saber{

namespace lite{

struct ParamBase {
    ParamBase(){}
    virtual ~ParamBase(){}
    ParamBase(const ParamBase& param){}
    ParamBase&operator=(const ParamBase& param){ return *this;}
};

struct ActivationParam : public ParamBase {
    ActivationParam(){}
    ActivationParam(ActiveType act_type, float neg_slope = 0.f, float coef = 1.f, \
        bool channel_shared = false, const float* weights = nullptr, int size = 0) {
        _act_type = act_type;
        _neg_slope = neg_slope;
        _coef = coef;
        _prelu_channel_shared = channel_shared;
        if (_act_type == Active_prelu) {
            int size_r = _prelu_channel_shared? 1 : size;
            _prelu_weights.clear();
            for (int i = 0; i < size_r; ++i) {
                _prelu_weights.push_back(weights[i]);
            }

        }
    }
    ActivationParam(const ActivationParam& param) : ParamBase(param) {
        _act_type = param._act_type;
        _neg_slope = param._neg_slope;
        _coef = param._coef;
        _prelu_channel_shared = param._prelu_channel_shared;
        _prelu_weights = param._prelu_weights;
    }
    ActivationParam&operator=(const ActivationParam& param) {
        _act_type = param._act_type;
        _neg_slope = param._neg_slope;
        _coef = param._coef;
        _prelu_channel_shared = param._prelu_channel_shared;
        _prelu_weights = param._prelu_weights;
        return *this;
    }
    ActiveType _act_type{Active_relu};
    float _neg_slope{0.f};
    float _coef{1.f};
    bool _prelu_channel_shared{false};
    std::vector<float> _prelu_weights;
};
struct PowerParam : public ParamBase {
        PowerParam(){}
        PowerParam(float scale,float shift,float power ) {
            _scale = scale;
            _shift=shift;
            _power=power;
        }
        PowerParam(const PowerParam& param) : ParamBase(param) {
            _scale =param._scale;
            _shift=param._shift;
            _power=param._power;
        }
        PowerParam&operator=(const PowerParam& param) {
            _scale =param._scale;
            _shift=param._shift;
            _power=param._power;
            return *this;
        }
        float _scale;
        float _shift;
        float _power;
    };
struct Conv2DParam : public ParamBase{
    Conv2DParam(){}
    Conv2DParam(int weighs_size, int num_out, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        const void* weights, const void* bias = nullptr, bool flag_eltwise = false, \
        bool flag_act = false, ActiveType act_type = Active_relu, float neg_slope = 0.f, float coef = 1.f, \
        bool channel_shared = false, const float* act_weights = nullptr) {
        _weights_size = weighs_size;
        _num_output = num_out;
        _group = group;
        _kw = kw;
        _kh = kh;
        _stride_w = stride_w;
        _stride_h = stride_h;
        _pad_w = pad_w;
        _pad_h = pad_h;
        _dila_w = dila_w;
        _dila_h = dila_h;
        _bias_term = flag_bias;
        int chin = weighs_size / (num_out * kw * kh);
        _weights.re_alloc(Shape(num_out, chin, kh, kw), AK_FLOAT);
        memcpy(_weights.mutable_data(), weights, sizeof(float) * weighs_size);
        if (_bias_term) {
            _bias.re_alloc(Shape(num_out), AK_FLOAT);
            memcpy(_bias.mutable_data(), bias, sizeof(float) * num_out);
        }
        _flag_eltwise = flag_eltwise;
        _flag_act = flag_act;
        _act_type = act_type;
        _neg_slope = neg_slope;
        _act_coef = coef;
        _act_channel_shared = channel_shared;
        if (_flag_act && (_act_type == Active_prelu)) {
            int size = _act_channel_shared? 1 : num_out;
            _act_weights.clear();
            for (int i = 0; i < size; ++i) {
                _act_weights.push_back(act_weights[i]);
            }
        }
    }

    Conv2DParam(const Conv2DParam& param) : ParamBase(param) {
        _weights_size = param._weights_size;
        _num_output = param._num_output;
        _group = param._group;
        _kw = param._kw;
        _kh = param._kh;
        _stride_w = param._stride_w;
        _stride_h = param._stride_h;
        _pad_w = param._pad_w;
        _pad_h = param._pad_h;
        _dila_w = param._dila_w;
        _dila_h = param._dila_h;
        _bias_term = param._bias_term;
        _weights = param._weights;
        _bias = param._bias;
        _flag_eltwise = param._flag_eltwise;
        _flag_act = param._flag_act;
        _act_type = param._act_type;
        _neg_slope = param._neg_slope;
        _act_coef = param._act_coef;
        _act_channel_shared = param._act_channel_shared;
        _act_weights = param._act_weights;
    }

    Conv2DParam&operator=(const Conv2DParam& param) {
        _weights_size = param._weights_size;
        _num_output = param._num_output;
        _group = param._group;
        _kw = param._kw;
        _kh = param._kh;
        _stride_w = param._stride_w;
        _stride_h = param._stride_h;
        _pad_w = param._pad_w;
        _pad_h = param._pad_h;
        _dila_w = param._dila_w;
        _dila_h = param._dila_h;
        _bias_term = param._bias_term;
        _weights = param._weights;
        _bias = param._bias;
        _flag_eltwise = param._flag_eltwise;
        _flag_act = param._flag_act;
        _act_type = param._act_type;
        _neg_slope = param._neg_slope;
        _act_coef = param._act_coef;
        _act_channel_shared = param._act_channel_shared;
        _act_weights = param._act_weights;
        return *this;
    }

    int _weights_size;
    int _num_output;
    int _group;
    int _kw;
    int _kh;
    int _stride_w;
    int _stride_h;
    int _pad_w;
    int _pad_h;
    int _dila_w;
    int _dila_h;
    bool _bias_term{true};
    Tensor<CPU> _weights;
    Tensor<CPU> _bias;
    bool _flag_eltwise{false};
    bool _flag_act{false};
    ActiveType _act_type;
    float _neg_slope{0.f};
    float _act_coef{1.f};
    bool _act_channel_shared{false};
    std::vector<float> _act_weights;
};

struct PoolParam : public ParamBase {
    PoolParam(){}
    PoolParam(PoolingType pool_type, bool is_global_pool, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h) {
        _pool_type = pool_type;
        _flag_global = is_global_pool;
        _pool_kw = kw;
        _pool_kh = kh;
        _pool_stride_w = stride_w;
        _pool_stride_h = stride_h;
        _pool_pad_w = pad_w;
        _pool_pad_h = pad_h;
    }
    PoolParam(const PoolParam& param) : ParamBase(param) {
        _pool_type = param._pool_type;
        _flag_global = param._flag_global;
        _pool_kw = param._pool_kw;
        _pool_kh = param._pool_kh;
        _pool_stride_w = param._pool_stride_w;
        _pool_stride_h = param._pool_stride_h;
        _pool_pad_w = param._pool_pad_w;
        _pool_pad_h = param._pool_pad_h;
    }
    PoolParam&operator=(const PoolParam& param) {
        _pool_type = param._pool_type;
        _flag_global = param._flag_global;
        _pool_kw = param._pool_kw;
        _pool_kh = param._pool_kh;
        _pool_stride_w = param._pool_stride_w;
        _pool_stride_h = param._pool_stride_h;
        _pool_pad_w = param._pool_pad_w;
        _pool_pad_h = param._pool_pad_h;
        return *this;
    }

    PoolingType _pool_type;
    bool _flag_global;
    int _pool_kw;
    int _pool_kh;
    int _pool_stride_w;
    int _pool_stride_h;
    int _pool_pad_w;
    int _pool_pad_h;
};

struct ConvPool2DParam : public ParamBase {
    ConvPool2DParam(){}
    ConvPool2DParam(int weighs_size, int num_out, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        const float* weights, const float* bias, \
        bool flag_eltwise, bool flag_act, ActiveType act_type, float neg_slop, float coef, \
        bool act_channel_shared, const float* act_weights_ptr, \
        PoolingType pool_type, bool flag_global_pool, \
        int pool_kw, int pool_kh, int pool_stride_w, int pool_stride_h, int pool_pad_w, int pool_pad_h) : \
        _conv_param(weighs_size, num_out, group, kw, kh, \
            stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias, flag_eltwise, \
            flag_act, act_type, neg_slop, coef, act_channel_shared, act_weights_ptr), \
             _pool_param(pool_type, flag_global_pool, pool_kw, pool_kh, \
             pool_stride_w, pool_stride_h, pool_pad_w, pool_pad_h) {}

    ConvPool2DParam(const Conv2DParam& conv_param, const PoolParam& pool_param) : _conv_param(conv_param), \
             _pool_param(pool_param) {}

    ConvPool2DParam(const ConvPool2DParam& param) : ParamBase(param),
                                                    _conv_param(param._conv_param),
                                                    _pool_param(param._pool_param) {}

    ConvPool2DParam&operator=(const ConvPool2DParam& param) {
        _conv_param = param._conv_param;
        _pool_param = param._pool_param;
        return *this;
    }
    Conv2DParam _conv_param;
    PoolParam _pool_param;
};

struct ConcatParam : public ParamBase {

    ConcatParam(){}
    ConcatParam(int axis) {
        _axis = axis;
    }
    ConcatParam(const ConcatParam& param) : ParamBase(param) {
        _axis = param._axis;
    }
    ConcatParam&operator=(const ConcatParam& param) {
        _axis = param._axis;
        return *this;
    }
    int _axis{0};
};

struct DetectionOutputParam : public ParamBase {
    DetectionOutputParam(){}
    DetectionOutputParam(int class_num, float conf_thresh, int nms_topk, int background_id = 0, \
        int keep_topk = -1, CodeType code_type = CORNER, float nms_thresh = 0.3f, float nms_eta = 1.f, \
        bool share_location = true, bool encode_in_target = false) {
        _class_num = class_num;
        _conf_thresh = conf_thresh;
        _nms_top_k = nms_topk;
        _background_id = background_id;
        _keep_top_k = keep_topk;
        _code_type = code_type;
        _nms_thresh = nms_thresh;
        _nms_eta = nms_eta;
        _share_location = share_location;
        _variance_encode_in_target = encode_in_target;
    }
    DetectionOutputParam(const DetectionOutputParam& param) : ParamBase(param) {
        _class_num = param._class_num;
        _conf_thresh = param._conf_thresh;
        _nms_top_k = param._nms_top_k;
        _background_id = param._background_id;
        _keep_top_k = param._keep_top_k;
        _code_type = param._code_type;
        _nms_thresh = param._nms_thresh;
        _nms_eta = param._nms_eta;
        _share_location = param._share_location;
        _variance_encode_in_target = param._variance_encode_in_target;
    }
    DetectionOutputParam&operator=(const DetectionOutputParam& param) {
        _class_num = param._class_num;
        _conf_thresh = param._conf_thresh;
        _nms_top_k = param._nms_top_k;
        _background_id = param._background_id;
        _keep_top_k = param._keep_top_k;
        _code_type = param._code_type;
        _nms_thresh = param._nms_thresh;
        _nms_eta = param._nms_eta;
        _share_location = param._share_location;
        _variance_encode_in_target = param._variance_encode_in_target;
        return *this;
    }
    int _class_num;
    float _conf_thresh;
    int _nms_top_k;
    int _background_id{0};
    int _keep_top_k{-1};
    CodeType _code_type{CORNER};
    float _nms_thresh{0.3f};
    float _nms_eta{1.f};
    bool _share_location{true};
    bool _variance_encode_in_target{false};
};

struct EltwiseParam : public ParamBase {

    EltwiseParam(){}
    EltwiseParam(EltwiseType elt_type, std::vector<float> coef) {
        _elt_type = elt_type;
        _coef = coef;
    }
    EltwiseParam(const EltwiseParam& param) : ParamBase(param) {
        _elt_type = param._elt_type;
        _coef= param._coef;
    }
    EltwiseParam&operator=(const EltwiseParam& param) {
        _elt_type = param._elt_type;
        _coef= param._coef;
        return *this;
    }

    EltwiseType _elt_type;
    std::vector<float> _coef;
};

struct EltwiseActParam : public ParamBase {
    EltwiseActParam()
        : _eltwise_param()
        , _activation_param()
        , _has_activation(false)
    {}
    EltwiseActParam(EltwiseType elt_type, std::vector<float> coef, \
         ActiveType act_type, float neg_slope, float act_coef, bool channel_shared, const float* weights)
        : _eltwise_param(elt_type, coef)
        , _activation_param(act_type, neg_slope, act_coef, channel_shared, weights)
        , _has_activation(true)
    {}
    EltwiseActParam(EltwiseParam &eltwise_param_in,
                ActivationParam &activation_param_in)
            : _eltwise_param(eltwise_param_in)
            , _activation_param(activation_param_in)
            , _has_activation(true)
    {}
    EltwiseActParam(const EltwiseActParam &right)
            : ParamBase(right), _eltwise_param(right._eltwise_param)
            , _activation_param(right._activation_param)
            , _has_activation(right._has_activation)
    {}
    EltwiseActParam&operator=(const EltwiseActParam &right) {
        _eltwise_param = right._eltwise_param;
        _activation_param = right._activation_param;
        _has_activation = right._has_activation;
        return *this;
    }
    EltwiseParam _eltwise_param;
    ActivationParam _activation_param;
    bool _has_activation;
};


struct FcParam : public ParamBase {
    FcParam(){}
    FcParam(int axis, int num_output, bool flag_bias, const float* weights, int size, const float* bias = nullptr, bool flag_trans = false) {
        _axis = axis;
        _num_output = num_output;
        _flag_bias = flag_bias;
        int k = size / num_output;
        _weights.re_alloc(Shape(num_output, k), AK_FLOAT);
        memcpy(_weights.mutable_data(), weights, sizeof(float) * size);
        if (flag_bias) {
            _bias.re_alloc(Shape(num_output), AK_FLOAT);
            memcpy(_bias.mutable_data(), bias, sizeof(float) * num_output);
        }
        _flag_trans = flag_trans;
    }
    FcParam(const FcParam& param) : ParamBase(param) {
        _axis = param._axis;
        _num_output = param._num_output;
        _flag_bias = param._flag_bias;
        _weights = param._weights;
        _bias = param._bias;
        _flag_trans = param._flag_trans;
    }
    FcParam&operator=(const FcParam& param) {
        _axis = param._axis;
        _num_output = param._num_output;
        _flag_bias = param._flag_bias;
        _weights = param._weights;
        _bias = param._bias;
        _flag_trans = param._flag_trans;
        return *this;
    }
    int _axis;
    int _num_output;
    bool _flag_bias;
    bool _flag_trans{false};
    Tensor<CPU> _weights;
    Tensor<CPU> _bias;
};

struct PermuteParam : public ParamBase {
    PermuteParam(){}
    PermuteParam(std::vector<int> order) {
        _order = order;
    }
    PermuteParam(const PermuteParam& param) : ParamBase(param) {
        _order = param._order;
    }
    PermuteParam&operator=(const PermuteParam& param) {
        _order = param._order;
        return *this;
    }
    std::vector<int> _order;
};

struct PriorBoxParam : public ParamBase {
    PriorBoxParam(){}
   PriorBoxParam(std::vector<float> variance_in, \
        bool flip, bool clip, int image_width, int image_height, \
        float step_width, float step_height, float offset_in, std::vector<PriorType> order_in, \
        std::vector<float> min_in= std::vector<float>(), std::vector<float> max_in= std::vector<float>(), \
        std::vector<float> aspect_in= std::vector<float>(),
        std::vector<float> fixed_in = std::vector<float>(), std::vector<float> fixed_ratio_in = std::vector<float>(), \
        std::vector<float> density_in = std::vector<float>()) {

        _is_flip = flip;
        _is_clip = clip;
        _min_size = min_in;
        //add
        _fixed_size = fixed_in;
        _density_size = density_in;

        _img_w = image_width;
        _img_h = image_height;
        _step_w = step_width;
        _step_h = step_height;
        _offset = offset_in;
        _order = order_in;
        _aspect_ratio.clear();
        _aspect_ratio.push_back(1.f);

        _variance.clear();
        if (variance_in.size() == 1) {
            _variance.push_back(variance_in[0]);
            _variance.push_back(variance_in[0]);
            _variance.push_back(variance_in[0]);
            _variance.push_back(variance_in[0]);
        } else {
            LCHECK_EQ(variance_in.size(), 4, "ERROR: variance size must = 1 or = 4\n");
            _variance.push_back(variance_in[0]);
            _variance.push_back(variance_in[1]);
            _variance.push_back(variance_in[2]);
            _variance.push_back(variance_in[3]);
        }

        //add
        if (_fixed_size.size() > 0){
            LCHECK_GT(_density_size.size(), 0, "ERROR: if use fixed_size then you must provide density\n");
        }

        //add
         _fixed_ratio.clear();

         for (int i = 0; i < fixed_ratio_in.size(); i++){
            _fixed_ratio.push_back(fixed_ratio_in[i]);
         }
         //end

        for (int i = 0; i < aspect_in.size(); ++i) {
            float ar = aspect_in[i];
            bool already_exist = false;
            for (int j = 0; j < _aspect_ratio.size(); ++j) {
                if (fabsf(ar - _aspect_ratio[j]) < 1e-6f) {
                    already_exist = true;
                    break;
                }
            }
            if (!already_exist) {
                _aspect_ratio.push_back(ar);
                if (_is_flip) {
                    _aspect_ratio.push_back(1.f/ar);
                }
            }
        }

        //add
        if (_min_size.size() > 0)
            _prior_num = _min_size.size() * _aspect_ratio.size();
        if (_fixed_size.size() > 0){
            _prior_num = _fixed_size.size() * _fixed_ratio.size();
        }

        if (_density_size.size() > 0){
            for (int i = 0; i < _density_size.size(); i++){
                if (_fixed_ratio.size() > 0){
                    _prior_num += (_fixed_ratio.size() * ((pow(_density_size[i], 2))-1));
                }else{
                    _prior_num += ((_fixed_ratio.size() + 1) * ((pow(_density_size[i], 2))-1));
                }
            }
        }
        //end
        _max_size.clear();
        if (max_in.size() > 0) {
            LCHECK_EQ(max_in.size(), _min_size.size(), "ERROR: max_size num must = min_size num\n");
            for (int i = 0; i < max_in.size(); ++i) {
                LCHECK_GT(max_in[i], _min_size[i], "ERROR: max_size val must > min_size val\n");
                _max_size.push_back(max_in[i]);
                _prior_num++;
            }
        }

    }
    PriorBoxParam(const PriorBoxParam& param) : ParamBase(param) {
        _is_flip = param._is_flip;
        _is_clip = param._is_clip;
        _min_size = param._min_size;
        _max_size = param._max_size;
        _aspect_ratio = param._aspect_ratio;
        _fixed_size = param._fixed_size;
        _fixed_ratio = param._fixed_ratio;
        _density_size = param._density_size;
        _variance = param._variance;
        _img_w = param._img_w;
        _img_h = param._img_h;
        _step_w = param._step_w;
        _step_h = param._step_h;
        _offset = param._offset;
        _order = param._order;
        _prior_num = param._prior_num;
    }
    PriorBoxParam& operator=(const PriorBoxParam& param) {
        _is_flip = param._is_flip;
        _is_clip = param._is_clip;
        _min_size = param._min_size;
        _max_size = param._max_size;
        _aspect_ratio = param._aspect_ratio;
        _fixed_size = param._fixed_size;
        _fixed_ratio = param._fixed_ratio;
        _density_size = param._density_size;
        _variance = param._variance;
        _img_w = param._img_w;
        _img_h = param._img_h;
        _step_w = param._step_w;
        _step_h = param._step_h;
        _offset = param._offset;
        _order = param._order;
        _prior_num = param._prior_num;
        return *this;
    }
    bool _is_flip;
    bool _is_clip;
    std::vector<float> _min_size;
    std::vector<float> _max_size;
    std::vector<float> _aspect_ratio;
    std::vector<float> _fixed_size;
    std::vector<float> _fixed_ratio;
    std::vector<float> _density_size;
    std::vector<float> _variance;
    int _img_w{0};
    int _img_h{0};
    float _step_w{0};
    float _step_h{0};
    float _offset{0.5};
    int _prior_num{0};
    std::vector<PriorType> _order;
};
struct ResizeParam: public ParamBase{

    ResizeParam(){}
    ResizeParam(float width_scale_in, float height_scale_in){
        _width_scale = width_scale_in;
        _height_scale = height_scale_in;
    }
    ResizeParam(const ResizeParam &right):ParamBase(right){
        _width_scale = right._width_scale;
        _height_scale = right._height_scale;
    }
    ResizeParam &operator=(const ResizeParam &right){
        _width_scale = right._width_scale;
        _height_scale = right._height_scale;
        return *this;
    }
    bool operator==(const ResizeParam &right){
        bool comp_eq = (_width_scale == right._width_scale) &&\
                        (_height_scale == right._height_scale);
        return comp_eq;
    }
    float _width_scale{0.0f};
    float _height_scale{0.0f};
};
struct SliceParam : public ParamBase {

    SliceParam(){}
    SliceParam(int axis, std::vector<int> points) {
        _axis = axis;
        _points = points;
    }
    SliceParam(const SliceParam& param) : ParamBase(param) {
        _axis = param._axis;
        _points = param._points;
    }
    SliceParam&operator=(const SliceParam& param) {
        _axis = param._axis;
        _points = param._points;
        return *this;
    }
    int _axis;
    std::vector<int> _points;
};

struct SoftmaxParam : public ParamBase {
    SoftmaxParam(){}
    SoftmaxParam(int axis) {
        _axis = axis;
    }
    SoftmaxParam(const SoftmaxParam& param) : ParamBase(param) {
        _axis = param._axis;
    }
    SoftmaxParam&operator=(const SoftmaxParam& param) {
        _axis = param._axis;
        return *this;
    }

    int _axis;
};

struct SplitParam : public ParamBase {
    SplitParam(){}
    SplitParam(const SplitParam& param) : ParamBase(param) {}
    SplitParam&operator=(const SplitParam& param) { return *this; }
};

struct FlattenParam : public ParamBase {
    FlattenParam() = default;
    FlattenParam(const FlattenParam& right) : ParamBase(right) {}
    FlattenParam& operator=(const FlattenParam& right){ return *this;}
    bool operator==(const FlattenParam& right){
        return true;
    }
};

struct ReshapeParam : public ParamBase {
    ReshapeParam() = default;
    explicit ReshapeParam(std::vector<int> shape_param_in) {
        int count = 0;
        for (int i = 0; i < shape_param_in.size(); ++i) {
            if (shape_param_in[i] == -1){
                count ++;
            }
        }
        LCHECK_LE(count, 1, "ERROR: shape parameter contains multiple -1 dims\n");
        _shape_params = shape_param_in;
    }
    ReshapeParam(const ReshapeParam& right) : ParamBase(right) {
        _shape_params = right._shape_params;
    }
    ReshapeParam& operator=(const ReshapeParam& right) {
        _shape_params = right._shape_params;
        return *this;
    }
    bool operator==(const ReshapeParam& right) {
        bool comp_eq = _shape_params.size() == right._shape_params.size();
        for (int i = 0; i < _shape_params.size(); ++i) {
            if (!comp_eq){
                return false;
            }
            comp_eq = _shape_params[i] == right._shape_params[i];
        }
        return true;
    }
    std::vector<int> _shape_params;
};

struct ScaleParam : public ParamBase {
    ScaleParam() {
        _axis = 1;
        _num_axes = 1;
        _bias_term = false;
    }

    ScaleParam(const float* scale_w_in, const float* scale_b_in, int w_size, int b_size, \
               bool bias_term_in = true, int axis_in = 1, int num_axes_in = 1) {
        _scale_w.clear();
        _scale_b.clear();
        for (int i = 0; i < w_size; ++i) {
            _scale_w.push_back(scale_w_in[i]);
        }
        for (int i = 0; i < b_size; ++i) {
            _scale_b.push_back(scale_b_in[i]);
        }
        _bias_term = bias_term_in;
        _axis = axis_in;
        _num_axes = num_axes_in;
    }

    ScaleParam(const float*  scale_w_in, int size_w, \
               bool bias_term_in = false, int axis_in = 1, int num_axes_in = 1) {
        _scale_w.clear();
        for (int i = 0; i < size_w; ++i) {
            _scale_w.push_back(scale_w_in[i]);
        }
        _bias_term = bias_term_in;
        _axis = axis_in;
        _num_axes = num_axes_in;
    }

    ScaleParam(const ScaleParam &right) : ParamBase(right) {
        _scale_w = right._scale_w;
        _scale_b = right._scale_b;
        _bias_term = right._bias_term;
        _axis = right._axis;
        _num_axes = right._num_axes;
    }

    ScaleParam &operator=(const ScaleParam &right) {
        _scale_w = right._scale_w;
        _scale_b = right._scale_b;
        _bias_term = right._bias_term;
        _axis = right._axis;
        _num_axes = right._num_axes;
        return *this;
    }
    bool operator==(const ScaleParam &right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (_scale_w == right._scale_w);
        comp_eq = comp_eq && (_scale_b == right._scale_b);
        comp_eq = comp_eq && (_bias_term == right._bias_term);
        comp_eq = comp_eq && (_axis == right._axis);
        comp_eq = comp_eq && (_num_axes == right._num_axes);
        return comp_eq;
    }
    int _axis; // default is 1
    int _num_axes; // default is 1
    bool _bias_term; // default false
    std::vector<float>  _scale_w;
    std::vector<float>  _scale_b;
};

struct ShuffleChannelParam : public ParamBase {
    ShuffleChannelParam() : _group(1) {}
    ShuffleChannelParam(int group) : _group(group) {}
    ShuffleChannelParam(const ShuffleChannelParam& right) : ParamBase(right) {
        _group = right._group;
    }
    ShuffleChannelParam& operator=(const ShuffleChannelParam& right){
        _group = right._group;
        return *this;
    }
    bool operator==(const ShuffleChannelParam& right){
        return _group == right._group;
    }
    int _group;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_OP_PARAM_H
