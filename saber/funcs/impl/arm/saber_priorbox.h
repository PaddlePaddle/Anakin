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

#ifndef ANAKIN_SABER_FUNCS_ARM_SABER_PRIORBOX_H
#define ANAKIN_SABER_FUNCS_ARM_SABER_PRIORBOX_H

#include "saber/funcs/impl/impl_priorbox.h"
#include "saber/core/tensor.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberPriorBox<ARM, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        PriorBoxParam<Tensor<ARM, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberPriorBox() = default;
    ~SaberPriorBox() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                      std::vector<DataTensor_out*>& outputs,
                      PriorBoxParam<OpTensor> &param, Context<ARM> &ctx) override {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                        std::vector<DataTensor_out*>& outputs,
                        PriorBoxParam<OpTensor> &param, Context<ARM> &ctx) override {

        SABER_CHECK(_output_arm.reshape(outputs[0]->valid_shape()));
        float* output_host = _output_arm.mutable_data();

        float* min_buf = (float*)fast_malloc(sizeof(float) * 4);
        float* max_buf = (float*)fast_malloc(sizeof(float) * 4);
        float* com_buf = (float*)fast_malloc(sizeof(float) * param.aspect_ratio.size() * 4);

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
                    int min_idx = 0;
                    int max_idx = 0;
                    int com_idx = 0;
                    int min_size = param.min_size[s];
                    //! first prior: aspect_ratio = 1, size = min_size
                    box_width = box_height = min_size;
                    //! xmin
                    min_buf[min_idx++] = (center_x - box_width / 2.f) / img_width;
                    //! ymin
                    min_buf[min_idx++] = (center_y - box_height / 2.f) / img_height;
                    //! xmax
                    min_buf[min_idx++] = (center_x + box_width / 2.f) / img_width;
                    //! ymax
                    min_buf[min_idx++] = (center_y + box_height / 2.f) / img_height;

                    if (param.max_size.size() > 0) {

                        int max_size = param.max_size[s];
                        //! second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        box_width = box_height = sqrtf(min_size * max_size);
                        //! xmin
                        max_buf[max_idx++] = (center_x - box_width / 2.f) / img_width;
                        //! ymin
                        max_buf[max_idx++] = (center_y - box_height / 2.f) / img_height;
                        //! xmax
                        max_buf[max_idx++] = (center_x + box_width / 2.f) / img_width;
                        //! ymax
                        max_buf[max_idx++] = (center_y + box_height / 2.f) / img_height;
                    }

                    //! rest of priors
                    for (int r = 0; r < param.aspect_ratio.size(); ++r) {
                        float ar = param.aspect_ratio[r];
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }
                        box_width = min_size * sqrt(ar);
                        box_height = min_size / sqrt(ar);
                        //! xmin
                        com_buf[com_idx++] = (center_x - box_width / 2.f) / img_width;
                        //! ymin
                        com_buf[com_idx++] = (center_y - box_height / 2.f) / img_height;
                        //! xmax
                        com_buf[com_idx++] = (center_x + box_width / 2.f) / img_width;
                        //! ymax
                        com_buf[com_idx++] = (center_y + box_height / 2.f) / img_height;
                    }

                    for (const auto &type : param.order) {
                        if (type == PRIOR_MIN) {
                            memcpy(output_host + idx, min_buf, sizeof(float) * min_idx);
                            idx += min_idx;
                        } else if (type == PRIOR_MAX) {
                            memcpy(output_host + idx, max_buf, sizeof(float) * max_idx);
                            idx += max_idx;
                        } else if (type == PRIOR_COM) {
                            memcpy(output_host + idx, com_buf, sizeof(float) * com_idx);
                            idx += com_idx;
                        }
                    }
                }
            }
        }

        fast_free(min_buf);
        fast_free(max_buf);
        fast_free(com_buf);

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

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          PriorBoxParam<OpTensor> &param) override {
        memcpy(outputs[0]->mutable_data(), _output_arm.data(), \
            outputs[0]->valid_size() * sizeof(float));
        return SaberSuccess;
    }

private:
    Tensor<ARM, AK_FLOAT, NCHW> _output_arm;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_ARM_SABER_PRIORBOX_H
