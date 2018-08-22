#include "saber/lite/funcs/saber_priorbox.h"
#include <cmath>
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

SaberPriorBox::SaberPriorBox(const ParamBase *param) {
    _param = (const PriorBoxParam*)param;
    this->_flag_param = true;
}

SaberStatus SaberPriorBox::load_param(const ParamBase *param) {
    _param = (const PriorBoxParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberPriorBox::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_param) {
        printf("load priorbox param first\n");
        return SaberNotInitialized;
    }

    //! output tensor's dims = 4 (1, 1, 2, 4 * num_priors)

    Shape shape_out = outputs[0]->valid_shape();
    shape_out[0] = 1;
    shape_out[1] = 1;
    shape_out[2] = 2;

    int win1 = inputs[0]->width();
    int hin1 = inputs[0]->height();

    int wout = win1 * hin1 * this->_param->_prior_num * 4;
    shape_out[3] = wout;

    return outputs[0]->set_shape(shape_out);
}

SaberStatus SaberPriorBox::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {

    if (!this->_flag_param) {
        printf("load priorbox param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;

    LITE_CHECK(_output_arm.reshape(outputs[0]->valid_shape()));
    float* output_host = _output_arm.mutable_data();

    const int width = inputs[0]->width();
    const int height = inputs[0]->height();
    int img_width = _param->_img_w;
    int img_height = _param->_img_h;
    if (img_width == 0 || img_height == 0) {
        img_width = inputs[1]->width();
        img_height = inputs[1]->height();
    }

    float step_w = _param->_step_w;
    float step_h = _param->_step_h;
    if (step_w == 0 || step_h == 0) {
        step_w = static_cast<float>(img_width) / width;
        step_h = static_cast<float>(img_height) / height;
    }
    float offset = _param->_offset;

    int channel_size = height * width * this->_param->_prior_num * 4;
    int idx = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width;
            float box_height;
            for (int s = 0; s < _param->_min_size.size(); ++s) {
                float min_size = _param->_min_size[s];
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

                if (_param->_max_size.size() > 0) {

                    int max_size = _param->_max_size[s];
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
                for (int r = 0; r < _param->_aspect_ratio.size(); ++r) {
                    float ar = _param->_aspect_ratio[r];
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
    if (_param->_is_clip) {
        for (int d = 0; d < channel_size; ++d) {
            output_host[d] = std::min(std::max(output_host[d], 0.f), 1.f);
        }
    }
    //! set the variance.
    float* ptr = output_host + channel_size;
    int count = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int i = 0; i < this->_param->_prior_num; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ptr[count] = _param->_variance[j];
                    ++count;
                }
            }
        }
    }
    this->_flag_init = true;
    return SaberSuccess;
}

SaberStatus SaberPriorBox::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                    std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init priorbox first\n");
        return SaberNotInitialized;
    }

    memcpy(outputs[0]->mutable_data(), _output_arm.data(), \
            outputs[0]->valid_size() * sizeof(float));
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif

