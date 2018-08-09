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
        bool channel_shared = false, float* weights = nullptr) {
        _act_type = act_type;
        _neg_slope = neg_slope;
        _coef = coef;
        _prelu_channel_shared = channel_shared;
        _prelu_weights = weights;
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
    const float* _prelu_weights{nullptr};
};

struct Conv2DParam : public ParamBase{
    Conv2DParam(){}
    Conv2DParam(int weighs_size, int num_out, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        const float* weights, const float* bias) {
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
        _weights = weights;
        _bias = bias;
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
        return *this;
    }

    bool _bias_term{true};
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
    const float* _weights{nullptr};
    const float* _bias{nullptr};
    int _weights_size;
};

struct DeConv2DParam : public ParamBase{
    DeConv2DParam(){}
    DeConv2DParam(int weighs_size, int num_out, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        const float* weights, const float* bias) {
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
        _weights = weights;
        _bias = bias;
    }

    DeConv2DParam(const DeConv2DParam& param) : ParamBase(param) {
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
    }

    DeConv2DParam&operator=(const DeConv2DParam& param) {
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
        return *this;
    }

    bool _bias_term{true};
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
    const float* _weights{nullptr};
    const float* _bias{nullptr};
    int _weights_size;
};

struct ConvAct2DParam : public ParamBase {
    ConvAct2DParam(){}
    ConvAct2DParam(int weighs_size, int num_out, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        ActiveType act_type, bool flag_act, \
        const float* weights, const float* bias) : _conv_param(weighs_size, num_out, group, kw, kh, \
            stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias) {
        _act_type = act_type;
        _flag_act = flag_act;
    }

    ConvAct2DParam(const ConvAct2DParam& param) : ParamBase(param),
                                                  _conv_param(param._conv_param) {
        _act_type = param._act_type;
        _flag_act = param._act_type;
    }

    ConvAct2DParam&operator=(const ConvAct2DParam& param) {
        _conv_param = param._conv_param;
        _act_type = param._act_type;
        _flag_act = param._flag_act;
        return *this;
    }
    Conv2DParam _conv_param;
    ActiveType _act_type;
    bool _flag_act;
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

struct ConvActPool2DParam : public ParamBase {
    ConvActPool2DParam(){}
    ConvActPool2DParam(int weighs_size, int num_out, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        ActiveType act_type, bool flag_act, PoolingType pool_type, bool flag_global_pool, \
        int pool_kw, int pool_kh, int pool_stride_w, int pool_stride_h, int pool_pad_w, int pool_pad_h, \
        const float* weights, const float* bias) : _conv_act_param(weighs_size, num_out, group, kw, kh, \
            stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias, act_type, flag_act, weights, bias), \
             _pool_param(pool_type, flag_global_pool, pool_kw, pool_kh, pool_stride_w, pool_stride_h, pool_pad_w, pool_pad_h) {}

    ConvActPool2DParam(const ConvAct2DParam& conv_act_param, const PoolParam& pool_param) : _conv_act_param(conv_act_param), \
             _pool_param(pool_param) {}

    ConvActPool2DParam(const ConvActPool2DParam& param) : ParamBase(param),
                                                          _conv_act_param(param._conv_act_param),
                                                          _pool_param(param._pool_param) {}

    ConvActPool2DParam&operator=(const ConvActPool2DParam& param) {
        _conv_act_param = param._conv_act_param;
        _pool_param = param._pool_param;
        return *this;
    }
    ConvAct2DParam _conv_act_param;
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

struct FcParam : public ParamBase {
    FcParam(){}
    FcParam(int axis, int num_output, bool flag_bias, const float* weights, const float* bias = nullptr, bool flag_trans = false) {
        _axis = axis;
        _num_output = num_output;
        _flag_bias = flag_bias;
        _weights = weights;
        _bias = bias;
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
    const float* _weights{nullptr};
    const float* _bias{nullptr};
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
    PriorBoxParam(std::vector<float> min_in, std::vector<float> max_in, \
        std::vector<float> aspect_in, std::vector<float> variance_in,
                  bool flip, bool clip, int image_width, int image_height, \
        float step_width, float step_height, float offset_in, std::vector<PriorType> order_in) {

        _is_flip = flip;
        _is_clip = clip;
        _min_size = min_in;
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
            LCHECK_EQ(variance_in.size(), 4, "variance size must = 1 or = 4");
            _variance.push_back(variance_in[0]);
            _variance.push_back(variance_in[1]);
            _variance.push_back(variance_in[2]);
            _variance.push_back(variance_in[3]);
        }

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
        _prior_num = _min_size.size() * _aspect_ratio.size();
        _max_size.clear();
        if (max_in.size() > 0) {
            LCHECK_EQ(max_in.size(), _min_size.size(), "max_size num must = min_size num");
            for (int i = 0; i < max_in.size(); ++i) {
                LCHECK_GT(max_in[i], _min_size[i], "max_size val must > min_size val");
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
    std::vector<float> _variance;
    int _img_w{0};
    int _img_h{0};
    float _step_w{0};
    float _step_h{0};
    float _offset{0.5};
    int _prior_num{0};
    std::vector<PriorType> _order;
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

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_OP_PARAM_H
