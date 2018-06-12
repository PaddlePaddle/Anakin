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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_PRIORBOX_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_PRIORBOX_H

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"
#include "saber/saber_funcs_param.h"

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
class SaberPriorBox {
public:

    SaberPriorBox() = default;
    ~SaberPriorBox() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     PriorBoxParam<Tensor<Dtype>> &param) {

        //! output tensor's dims = 3 (1, 2, 4 * num_priors)
        Shape shape_out = outputs[0]->valid_shape();
        shape_out[0] = 1;
        shape_out[1] = 2;

        int win1 = inputs[0]->width();
        int hin1 = inputs[0]->height();

        int wout = win1 * hin1 * param.prior_num * 4;
        shape_out[2] = wout;

        return outputs[0]->set_shape(shape_out);
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                      std::vector<Tensor<Dtype>*>& outputs,
                      PriorBoxParam<Tensor<Dtype>> &param, Context &ctx) {
        // get context
        _ctx = ctx;
        return create(inputs, outputs, param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,
                        std::vector<Tensor<Dtype>*>& outputs,
                        PriorBoxParam<Tensor<Dtype>> &param, Context &ctx) {

        SABER_CHECK(_output_arm.reshape(outputs[0]->valid_shape()));
        float* output_host = _output_arm.mutable_data();

        const int width = inputs[0]->width();
        const int height = inputs[0]->height();
        int img_width = param.img_w;
        int img_height = param.img_h;
        if (img_width == 0 || img_height == 0) {
            img_width = inputs[1]->width();
            img_height = inputs[1]->height();
        }

        float step_w = param.step_w;
        float step_h = param.step_h;
        if (step_w == 0 || step_h == 0) {
            step_w = static_cast<float>(img_width) / width;
            step_h = static_cast<float>(img_height) / height;
        }
        float offset = param.offset;

        int channel_size = height * width * param.prior_num * 4;
        int idx = 0;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float center_x = (w + offset) * step_w;
                float center_y = (h + offset) * step_h;
                float box_width;
                float box_height;
                for (int s = 0; s < param.min_size.size(); ++s) {
                    float min_size = param.min_size[s];
                    //! first prior: aspect_ratio = 1, size = min_size
                    box_width = box_height = min_size;
                    //! xmin
                    output_host[idx++] = (center_x - box_width / 2.f) / img_width;
                    //! ymin
                    output_host[idx++] = (center_y - box_height / 2.f) / img_height;
                    //! xmax
                    output_host[idx++] = (center_x + box_width / 2.f) / img_width;
                    //! ymax
                    output_host[idx++] = (center_y + box_height / 2.f) / img_height;

                    if (param.max_size.size() > 0) {

                        int max_size = param.max_size[s];
                        //! second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        box_width = box_height = sqrtf(min_size * max_size);
                        //! xmin
                        output_host[idx++] = (center_x - box_width / 2.f) / img_width;
                        //! ymin
                        output_host[idx++] = (center_y - box_height / 2.f) / img_height;
                        //! xmax
                        output_host[idx++] = (center_x + box_width / 2.f) / img_width;
                        //! ymax
                        output_host[idx++] = (center_y + box_height / 2.f) / img_height;
                    }

                    //! rest of priors
                    for (int r = 0; r < param.aspect_ratio.size(); ++r) {
                        float ar = param.aspect_ratio[r];
                        if (fabs(ar - 1.f) < 1e-6f) {
                            continue;
                        }
                        box_width = min_size * sqrtf(ar);
                        box_height = min_size / sqrtf(ar);
                        //! xmin
                        output_host[idx++] = (center_x - box_width / 2.f) / img_width;
                        //! ymin
                        output_host[idx++] = (center_y - box_height / 2.f) / img_height;
                        //! xmax
                        output_host[idx++] = (center_x + box_width / 2.f) / img_width;
                        //! ymax
                        output_host[idx++] = (center_y + box_height / 2.f) / img_height;
                    }
                }
            }
        }
        //! clip the prior's coordidate such that it is within [0, 1]
        if (param.is_clip) {
            for (int d = 0; d < channel_size; ++d) {
                output_host[d] = std::min(std::max(output_host[d], 0.f), 1.f);
            }
        }
        //! set the variance.

        float* ptr = output_host + channel_size;
        int count = 0;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int i = 0; i < param.prior_num; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        ptr[count] = param.variance[j];
                        ++count;
                    }
                }
            }
        }
        return SaberSuccess;
    }

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs,
                          std::vector<Tensor<Dtype>*>& outputs,
                          PriorBoxParam<Tensor<Dtype>> &param) {
        memcpy(outputs[0]->mutable_data(), _output_arm.data(), \
            outputs[0]->valid_size() * sizeof(float));
        return SaberSuccess;
    }

private:
    Context _ctx;
    Tensor<Dtype> _output_arm;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_PRIORBOX_H
