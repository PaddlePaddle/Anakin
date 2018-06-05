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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_POOLING_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_POOLING_H

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

typedef void (*pool_func)(Tensor<float>& tensor_out, Tensor<float>& tensor_in, \
    PoolingType type, bool global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h);

template <typename Dtype>
class SaberPooling {

public:
    SaberPooling() {}
    ~SaberPooling() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     PoolingParam<Tensor<Dtype>> &param) {
        Shape output_shape = inputs[0]->valid_shape();

        int in_height = inputs[0]->height();
        int in_width = inputs[0]->width();

        int window_h = param.window_h;
        int window_w = param.window_w;
        int pad_h = param.pad_h;
        int pad_w = param.pad_w;
        int stride_h = param.stride_h;
        int stride_w = param.stride_w;
        int out_height;
        int out_width;
        if (param.global_pooling) {
            out_height = 1;
            out_width = 1;
        } else {
            if (param.cmp_out_shape_floor_as_conv) {
                out_height = static_cast<int>((static_cast<float>(\
                    in_height + 2 * pad_h - window_h) / stride_h)) + 1;
                out_width = static_cast<int>((static_cast<float>(\
                    in_width + 2 * pad_w - window_w) / stride_w)) + 1;
            } else {
                out_height = static_cast<int>(ceilf(static_cast<float>(\
                    in_height + 2 * pad_h - window_h) / stride_h)) + 1;
                out_width = static_cast<int>(ceilf(static_cast<float>(\
                    in_width + 2 * pad_w - window_w) / stride_w)) + 1;
            }
        }

        if (param.pooling_padded()) {
            if ((out_height - 1) * stride_h >= in_height + pad_h) {
                -- out_height;
            }
            if ((out_width - 1) * stride_w >= in_width + pad_w) {
                -- out_width;
            }
        }

        int height_idx = inputs[0]->height_index();
        int width_idx = inputs[0]->width_index();

        output_shape[height_idx] = out_height;
        output_shape[width_idx] = out_width;

        return outputs[0]->set_shape(output_shape);
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs, \
        std::vector<Tensor<Dtype>*>& outputs, \
        PoolingParam<Tensor<Dtype>> &param, Context &ctx) {
        return create(inputs, outputs, param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,
                               std::vector<Tensor<Dtype>*>& outputs,
                               PoolingParam<Tensor<Dtype>> &param, Context &ctx);

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs, \
        std::vector<Tensor<Dtype>*>& outputs, PoolingParam<Tensor<Dtype>> &param);

private:
    pool_func _impl{nullptr};
    Context _ctx;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_POOLING_H
