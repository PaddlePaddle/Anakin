#include "saber/lite/funcs/saber_priorbox.h"
#include <cmath>
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

SaberPriorBox::SaberPriorBox(bool is_flip, bool is_clip, std::vector<float> min_size, std::vector<float> max_size,
                             std::vector<float> aspect_ratio, std::vector<float> variance, int img_width,
                             int img_height, float step_w, float step_h, float offset) {
    _is_flip = is_flip;
    _is_clip = is_clip;
    _min_size = min_size;
    _max_size = max_size;
    _aspect_ratio = aspect_ratio;
    _variance = variance;
    _img_width = img_width;
    _img_height = img_height;
    _step_w = step_w;
    _step_h = step_h;
    _offset = offset;
}

SaberStatus SaberPriorBox::load_param(bool is_flip, bool is_clip, std::vector<float> min_size,
                                      std::vector<float> max_size, std::vector<float> aspect_ratio,
                                      std::vector<float> variance, int img_width, int img_height, float step_w,
                                      float step_h, float offset) {
    _is_flip = is_flip;
    _is_clip = is_clip;
    _min_size = min_size;
    _max_size = max_size;
    _img_width = img_width;
    _img_height = img_height;
    _step_w = step_w;
    _step_h = step_h;
    _offset = offset;

    _aspect_ratio.clear();
    _aspect_ratio.push_back(1.f);

    _variance.clear();
    if (variance.size() == 1) {
        _variance.push_back(variance[0]);
        _variance.push_back(variance[0]);
        _variance.push_back(variance[0]);
        _variance.push_back(variance[0]);
    } else {
        LCHECK_EQ(variance.size(), 4, "variance size must = 1 or = 4");
        _variance.push_back(variance[0]);
        _variance.push_back(variance[1]);
        _variance.push_back(variance[2]);
        _variance.push_back(variance[3]);
    }

    for (int i = 0; i < aspect_ratio.size(); ++i) {
        float ar = aspect_ratio[i];
        bool already_exist = false;
        for (int j = 0; j < aspect_ratio.size(); ++j) {
            if (fabsf(ar - aspect_ratio[j]) < 1e-6f) {
                already_exist = true;
                break;
            }
        }
        if (!already_exist) {
            _aspect_ratio.push_back(ar);
            if (_is_flip) {
                _aspect_ratio.push_back(1.f / ar);
            }
        }
    }
    _num_priors = min_size.size() * aspect_ratio.size();
    _max_size.clear();
    if (max_size.size() > 0) {
        LCHECK_EQ(max_size.size(), min_size.size(), "max_size num must = min_size num");
        for (int i = 0; i < max_size.size(); ++i) {
            LCHECK_GT(max_size[i], min_size[i], "max_size val must > min_size val");
            _max_size.push_back(max_size[i]);
            _num_priors++;
        }
    }

    return SaberSuccess;
}

SaberStatus SaberPriorBox::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    //! output tensor's dims = 3 (1, 2, 4 * num_priors)
    Shape shape_out = outputs[0]->valid_shape();
    shape_out[0] = 1;
    shape_out[1] = 2;

    int win1 = inputs[0]->width();
    int hin1 = inputs[0]->height();

    int wout = win1 * hin1 * _num_priors * 4;
    shape_out[2] = wout;

    return outputs[0]->set_shape(shape_out);
}

SaberStatus SaberPriorBox::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    _ctx = ctx;

    LITE_CHECK(_output_arm.reshape(outputs[0]->valid_shape()));
    float* output_host = _output_arm.mutable_data();

    const int width = inputs[0]->width();
    const int height = inputs[0]->height();
    int img_width = _img_width;
    int img_height = _img_height;
    if (img_width == 0 || img_height == 0) {
        img_width = inputs[1]->width();
        img_height = inputs[1]->height();
    }

    float step_w = _step_w;
    float step_h = _step_h;
    if (step_w == 0 || step_h == 0) {
        step_w = static_cast<float>(img_width) / width;
        step_h = static_cast<float>(img_height) / height;
    }
    float offset = _offset;

    int channel_size = height * width * _num_priors * 4;
    int idx = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width;
            float box_height;
            for (int s = 0; s < _min_size.size(); ++s) {
                float min_size = _min_size[s];
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

                if (_max_size.size() > 0) {

                    int max_size = _max_size[s];
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
                for (int r = 0; r < _aspect_ratio.size(); ++r) {
                    float ar = _aspect_ratio[r];
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
    if (_is_clip) {
        for (int d = 0; d < channel_size; ++d) {
            output_host[d] = std::min(std::max(output_host[d], 0.f), 1.f);
        }
    }
    //! set the variance.

    float* ptr = output_host + channel_size;
    int count = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int i = 0; i < _num_priors; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ptr[count] = _variance[j];
                    ++count;
                }
            }
        }
    }
    return SaberSuccess;
}

SaberStatus SaberPriorBox::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                    std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    memcpy(outputs[0]->mutable_data(), _output_arm.data(), \
            outputs[0]->valid_size() * sizeof(float));
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif

