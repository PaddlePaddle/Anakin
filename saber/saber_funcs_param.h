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
struct ArgmaxParam {

    ArgmaxParam() = default;

    ArgmaxParam(bool out_max_val_in,int top_k_in, bool has_axis_in, int axis_in) {
        out_max_val = out_max_val_in;
        top_k = top_k_in;
        has_axis = has_axis_in;
        axis = axis_in;
    }

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
    

    ArgmaxParam(const ArgmaxParam<TargetType>& right) {
        out_max_val = right.out_max_val;
        top_k = right.top_k;
        has_axis = right.has_axis;
        axis = right.axis;
    }

    ArgmaxParam<TargetType>& operator=(const ArgmaxParam<TargetType>& right) {
        this->out_max_val = right.out_max_val;
        this->top_k = right.top_k;
        this->axis = right.axis;
        this->has_axis = right.has_axis;
        return *this;
    }

    bool operator==(const ArgmaxParam<TargetType>& right) {
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

template <typename opTensor>
struct ActivationParam {
    ActivationParam()
            : active(Active_unknow)
            , negative_slope(float(-1))
            , coef(float(-1))
            , prelu_param(PreluParam<opTensor>(false, nullptr)) {}
    ActivationParam(ActiveType act, float n_slope = float(0),
                    float co = float(1),
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
        return (active == Active_relu) && (negative_slope != float (0));
    }
    ActiveType active;
    float negative_slope;
    float coef;
    PreluParam<opTensor> prelu_param;
};

}
}
#endif //SABER_FUNCS_PARAM_H
