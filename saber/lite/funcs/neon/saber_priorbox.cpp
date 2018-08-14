#include "saber/lite/funcs/saber_priorbox.h"
#include "saber/lite/net/saber_factory_lite.h"
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
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const PriorBoxParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberPriorBox::load_param(FILE *fp, const float *weights) {
    int size_min;
    int size_max;
    int size_as;
    //add
    int size_fixed;
    int size_ratio;
    int size_density;
    int size_var;
    std::vector<float> min_size;
    std::vector<float> max_size;
    std::vector<float> as;
    //add
    std::vector<float> fixed_size;
    std::vector<float> fixed_ratio;
    std::vector<float> density;

    std::vector<float> var;
    std::vector<int> order;
    

    //! others
    int flip_flag;
    int clip_flag;
    int img_w;
    int img_h;
    float step_w;
    float step_h;
    float offset;
     //! var
    fscanf(fp, "%d ", &size_var);
    var.resize(size_var);
    for (int i = 0; i < size_var; ++i) {
        fscanf(fp, "%f ", &var[i]);
    }

    std::vector<int> type(3);
    std::vector<PriorType> ptype(3);
    fscanf(fp, ",%d,%d,%d,%d,%f,%f,%f,%d,%d,%d", &flip_flag, &clip_flag,
           &img_w, &img_h, &step_w, &step_h, &offset, &type[0], &type[1], &type[2]);
    ptype[0] = (PriorType)type[0];
    ptype[1] = (PriorType)type[1];
    ptype[2] = (PriorType)type[2];

    //! min
    fscanf(fp, ", %d ", &size_min);
    min_size.resize(size_min);
    for (int i = 0; i < size_min; ++i) {
        fscanf(fp, "%f ", &min_size[i]);
    }

    //! max
    fscanf(fp, ", %d ", &size_max);
    max_size.resize(size_max);
    for (int i = 0; i < size_max; ++i) {
        fscanf(fp, "%f ", &max_size[i]);
    }

    //! as
    fscanf(fp, ", %d ", &size_as);
    as.resize(size_as);
    for (int i = 0; i < size_as; ++i) {
        fscanf(fp, "%f ", &as[i]);
    }

    //! fixed
    fscanf(fp, ", %d ", &size_fixed);
    fixed_size.resize(size_fixed);
    for (int i = 0; i < size_fixed; ++i) {
        fscanf(fp, "%f ", &fixed_size[i]);
    }

    //! fixed_ratio
    fscanf(fp, ", %d ", &size_ratio);
    fixed_ratio.resize(size_ratio);
    for (int i = 0; i < size_ratio; ++i) {
        fscanf(fp, "%f ", &fixed_ratio[i]);
    }

    //! density
    fscanf(fp, ", %d ", &size_density);
    density.resize(size_density);
    for (int i = 0; i < size_density; ++i) {
        fscanf(fp, "%f ", &density[i]);
    }
    fscanf(fp, "\n");
    _param = new PriorBoxParam(var, \
        flip_flag>0, clip_flag>0, img_w, img_h, step_w, step_h, offset, ptype, \
        min_size, max_size, as, \
        fixed_size, fixed_ratio, density);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberPriorBox::~SaberPriorBox() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberPriorBox::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_param) {
        printf("load priorbox param first\n");
        return SaberNotInitialized;
    }

    //! output tensor's dims = 4 (1, 1, 2, 4 * num_priors)
    Shape shape_out = inputs[0]->valid_shape();
    shape_out[0] = 1;
    shape_out[1] = 1;
    shape_out[2] = 2;

    int win1 = inputs[0]->width();
    int hin1 = inputs[0]->height();

    int wout = win1 * hin1 * _param->_prior_num * 4;
    shape_out[3] = wout;

   // printf("shape: %d, %d, %d, %d \n", shape_out[0], shape_out[1], shape_out[2], shape_out[3]);
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
    int step_average = static_cast<int>((step_w + step_h) * 0.5); //add
    int channel_size = height * width * this->_param->_prior_num * 4;
    int idx = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width;
            float box_height;
             //LOG(INFO) << " ****** center_x = " << center_x << ", center_y = " << center_y << " ******";
            if(_param->_fixed_size.size() > 0){
                //add
                for (int s = 0; s < _param->_fixed_size.size(); ++s) {
                    int fixed_size_ = _param->_fixed_size[s];
                    int com_idx = 0;
                    box_width = fixed_size_;
                    box_height = fixed_size_;

                    if (_param->_fixed_ratio.size() > 0){
                        for (int r = 0; r < _param->_fixed_ratio.size(); ++r) {
                            float ar = _param->_fixed_ratio[r];
                            int density_ = _param->_density_size[s];
                            //int shift = fixed_sizes_[s] / density_; 
                            int shift = step_average / density_;
                            float box_width_ratio =  _param->_fixed_size[s] * sqrt(ar);
                            float box_height_ratio =  _param->_fixed_size[s] / sqrt(ar);

                            for (int p = 0 ; p < density_ ; ++p) {
                                for (int c = 0 ; c < density_ ; ++c) {
                                    // liu@20171207 changed to fix density bugs at anchor = 64
                                    float center_x_temp = center_x - step_average / 2 + 
                                    shift / 2. + c * shift;
                                    float center_y_temp = center_y - step_average / 2 + 
                                    shift / 2. + p * shift;
                                    //float center_x_temp = center_x - fixed_size_ / 2 + shift/2. + c*shift;
                                    //float center_y_temp = center_y - fixed_size_ / 2 + shift/2. + r*shift;
                                    //LOG(INFO) << " dense_center_x = " << center_x_temp << ", dense_center_y = " << center_y_temp;
                                    // xmin
                                    output_host[idx++] = (center_x_temp - box_width_ratio / 2.) 
                                                        / img_width >= 0 ?
                                                  (center_x_temp - box_width_ratio / 2.) 
                                                        / img_width : 0 ;
                                    // ymin
                                    output_host[idx++] = (center_y_temp - box_height_ratio / 2.) 
                                                        / img_height >= 0 ?
                                                  (center_y_temp - box_height_ratio / 2.) 
                                                        / img_height : 0;
                                    // xmax
                                    output_host[idx++] = (center_x_temp + box_width_ratio / 2.) 
                                                        / img_width <= 1 ?
                                                  (center_x_temp + box_width_ratio / 2.) 
                                                        / img_width : 1;
                                    // ymax
                                    output_host[idx++] = (center_y_temp + box_height_ratio / 2.) 
                                                        / img_height <= 1 ?
                                                  (center_y_temp + box_height_ratio / 2.) 
                                                        / img_height : 1;
                                }
                            }
                        }
                    } else {
                    //this code for density anchor box
                        if (_param->_density_size.size() > 0) {
                            LCHECK_EQ(_param->_fixed_size.size(), _param->_density_size.size(), "fixed_size should be same with denstiy_size");
                            int density_ = _param->_density_size[s];
                            int shift = _param->_fixed_size[s] / density_;

                            for (int r = 0 ; r < density_ ; ++r) {
                                for (int c = 0 ; c < density_ ; ++c) {
                                    float center_x_temp = center_x - fixed_size_ / 2 
                                    + shift / 2. + c * shift;
                                    float center_y_temp = center_y - fixed_size_ / 2 
                                    + shift / 2. + r * shift;
                                    // xmin
                                    output_host[idx++] = (center_x_temp - box_width / 2.) 
                                                        / img_width >= 0 ?
                                                  (center_x_temp - box_width / 2.) 
                                                        / img_width : 0 ;
                                    // ymin
                                    output_host[idx++] = (center_y_temp - box_height / 2.) 
                                                        / img_height >= 0 ?
                                                  (center_y_temp - box_height / 2.) 
                                                        / img_height : 0;
                                    // xmax
                                    output_host[idx++] = (center_x_temp + box_width / 2.) 
                                                        / img_width <= 1 ?
                                                  (center_x_temp + box_width / 2.) 
                                                        / img_width : 1;
                                    // ymax
                                    output_host[idx++] = (center_y_temp + box_height / 2.) 
                                                        / img_height <= 1 ?
                                                  (center_y_temp + box_height / 2.) 
                                                        / img_height : 1;
                                }
                            }
                        }

                        //rest of priors :will never come here!!!
                        for (int r = 0; r < _param->_aspect_ratio.size(); ++r) {
                            float ar = _param->_aspect_ratio[r];

                            if (fabs(ar - 1.) < 1e-6) {
                                //LOG(INFO) << "returning for aspect == 1";
                                continue;
                            }

                            int density_ = _param->_density_size[s];
                            int shift = _param->_fixed_size[s] / density_;
                            float box_width_ratio = _param->_fixed_size[s] * sqrt(ar);
                            float box_height_ratio = _param->_fixed_size[s] / sqrt(ar);

                            for (int p = 0 ; p < density_ ; ++p) {
                                for (int c = 0 ; c < density_ ; ++c) {
                                    float center_x_temp = center_x - fixed_size_ / 2 
                                    + shift / 2. + c * shift;
                                    float center_y_temp = center_y - fixed_size_ / 2 
                                    + shift / 2. + p * shift;
                                    // xmin
                                    output_host[idx++] = (center_x_temp - box_width_ratio / 2.) 
                                                        / img_width >= 0 ?
                                                  (center_x_temp - box_width_ratio / 2.) 
                                                        / img_width : 0 ;
                                    // ymin
                                    output_host[idx++] = (center_y_temp - box_height_ratio / 2.) 
                                                        / img_height >= 0 ?
                                                  (center_y_temp - box_height_ratio / 2.) 
                                                        / img_height : 0;
                                    // xmax
                                    output_host[idx++] = (center_x_temp + box_width_ratio / 2.) 
                                                        / img_width <= 1 ?
                                                  (center_x_temp + box_width_ratio / 2.) 
                                                        / img_width : 1;
                                    // ymax
                                    output_host[idx++] = (center_y_temp + box_height_ratio / 2.)
                                                        / img_height <= 1 ?
                                                  (center_y_temp + box_height_ratio / 2.) 
                                                        / img_height : 1;
                                }
                            }
                        }
                    }
                }
            }else{
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
REGISTER_LAYER_CLASS(SaberPriorBox);
} //namespace lite

} //namespace saber

} //namespace anakin

#endif

