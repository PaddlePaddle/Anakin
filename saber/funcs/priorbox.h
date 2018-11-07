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
#ifndef ANAKIN_SABER_FUNCS_PRIORBOX_H
#define ANAKIN_SABER_FUNCS_PRIORBOX_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"

namespace anakin {

namespace saber {

template<typename TargetType,
        DataType OpDtype>
class PriorBox : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        PriorBoxParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            PriorBoxParam>::BaseFunc;
    PriorBox() = default;
    ~PriorBox() {
        if (_cpu_data) {
            fast_free(_cpu_data);
            _cpu_data = nullptr;
        }
    }
    typedef TargetWrapper<TargetType> API;
    typedef std::vector<Tensor<TargetType> *> Input_v;
    typedef std::vector<Tensor<TargetType> *> Output_v;
    typedef PriorBoxParam<TargetType> Param_t;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        //! priorbox layout NHW
        //! N = 1, H = 2
        //! W = 4 * feature_map_width * feature_map_height * num_of_priors
        int win1 = input[0]->width();
        int hin1 = input[0]->height();
        int wout = win1 * hin1 * param.prior_num * 4;
        Shape shape_out({1, 2, wout}, Layout_NHW);
        return output[0]->set_shape(shape_out);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        return SaberSuccess;
    }

    SaberStatus compute_priorbox_kernel(const Input_v& input, Output_v& output, Param_t& param) {

        LOG(INFO) << "input tensor size: " << input.size();

        unsigned long long out_size = output[0]->valid_size();
        if (_cpu_data == nullptr) {
            _size = out_size;
            _cpu_data = static_cast<float*>(fast_malloc(sizeof(float) * _size));
        } else {
            if (out_size > _size) {
                _size = out_size;
                fast_free(_cpu_data);
                _cpu_data = static_cast<float*>(fast_malloc(sizeof(float) * _size));
            }
        }
        _tensor_tmp.reshape(output[0]->valid_shape());

        float* min_buf = (float*)fast_malloc(sizeof(float) * 4);
        float* max_buf = (float*)fast_malloc(sizeof(float) * 4);
        float* com_buf = (float*)fast_malloc(sizeof(float) * param.aspect_ratio.size() * 4);

        const int width = input[0]->width();
        const int height = input[0]->height();
        int img_width = param.img_w;
        int img_height = param.img_h;
        if (img_width == 0 || img_height == 0) {
            img_width = input[1]->width();
            img_height = input[1]->height();
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
                        if (fabsf(ar - 1.f) < 1e-6f) {
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
                            memcpy(_cpu_data + idx, min_buf, sizeof(float) * min_idx);
                            idx += min_idx;
                        } else if (type == PRIOR_MAX) {
                            memcpy(_cpu_data + idx, max_buf, sizeof(float) * max_idx);
                            idx += max_idx;
                        } else if (type == PRIOR_COM) {
                            memcpy(_cpu_data + idx, com_buf, sizeof(float) * com_idx);
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
                _cpu_data[d] = std::min(std::max(_cpu_data[d], 0.f), 1.f);
            }
        }
        //! set the variance.

        float* ptr = _cpu_data + channel_size;
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
        //! copy data to tensor
        typedef typename TargetTypeTraits<TargetType>::target_category target_category;
        typedef typename IF<std::is_same<target_category, __host_target>::value, __HtoH, __HtoD>::Type copy_type;
        API::sync_memcpy(_tensor_tmp.mutable_data(), 0, API::get_device_id(), \
            _cpu_data, 0, 0, sizeof(float) * out_size, copy_type());

        return SaberSuccess;
    }

    //PriorBox do computation in init
    virtual SaberStatus init(const Input_v& input, Output_v& output, Param_t& param,
                             SaberImplStrategy strategy, ImplEnum implenum, Context<TargetType > &ctx) {

        this->_last_input_shape.clear();
        for (int i = 0; i < input.size(); ++i) {
            this->_last_input_shape.push_back(input[i]->valid_shape());
        }

        if (output[0]->get_dtype() != AK_FLOAT) {
            return SaberInvalidValue;
        } else {
            compute_priorbox_kernel(input, output, param);
        }
        return SaberSuccess;
    }
    //copy data to output
    virtual SaberStatus operator() (const Input_v& input, Output_v& output, Param_t& param, \
        Context<TargetType> &ctx) {
        typename Tensor<TargetType>::API::stream_t stream = ctx.get_compute_stream();
        bool flag = (this->_param == param);
        for (int i = 0; i < input.size(); ++i) {
            flag = flag && input[i]->valid_shape() == this->_last_input_shape[i];
        }
        if (!flag) {
            this->_param = param;
            this->_last_input_shape.clear();
            for (int i = 0; i < input.size(); ++i) {
                this->_last_input_shape.push_back(input[i]->valid_shape());
            }
            compute_output_shape(input, output, param);
            compute_priorbox_kernel(input, output, param);
        }
        return output[0]->async_copy_from(_tensor_tmp, stream);
    }

private:
    float* _cpu_data{nullptr};
    Tensor<TargetType> _tensor_tmp;
    unsigned long long _size{0};

    virtual void pick_best_static() override {
        // do nothing
        return;
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        //do nothing
        return;
    }

};

} // namespace saber

} // namespace anakin

#endif //ANAKIN_SABER_FUNCS_PRIORBOX_H
