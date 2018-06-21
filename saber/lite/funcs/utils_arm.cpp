#include "saber/lite/funcs/utils_arm.h"
#include <cmath>
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void update_weights(Tensor<CPU, AK_FLOAT>& new_weight, Tensor<CPU, AK_FLOAT>& new_bias, \
    const float* weights, const float* bias, int num, int ch, int kh, int kw, bool conv_bias_term, \
    float batchnorm_scale, float batchnorm_eps, \
    std::vector<float> batchnorm_mean, std::vector<float> batchnorm_variance, \
    std::vector<float> scale_w, std::vector<float> scale_b, bool scale_bias_term){

    Shape weight_shape = {num, ch, kh, kw};
    int weight_size = num * ch * kh * kw;
    new_weight.reshape(weight_shape);
    memcpy(new_weight.mutable_data(), weights, sizeof(float) * weight_size);

    Shape bias_shape = {num};
    new_bias.reshape(bias_shape);

    if (conv_bias_term) {
        memcpy(new_bias.mutable_data(), bias, sizeof(float) * num);
    } else {
        memset(new_bias.mutable_data(), 0, sizeof(float) * num);
    }

    int filter_num = new_weight.num();
    int chw = new_weight.channel();

    float* weight_data = new_weight.mutable_data();
    float* bias_data = new_bias.mutable_data();

    chw *= new_weight.height();
    chw *= new_weight.width();

    for (int i = 0; i < filter_num; ++i) {
        float alpha = 1.f;
        float beta = 0.f;

        //! process batchnorm
        float scale_factor = 1.f;
        scale_factor = (batchnorm_scale == 0) ? 1 : 1.f / batchnorm_scale;
        float eps = batchnorm_eps;
        float variance;
        float mean;
        alpha = batchnorm_variance[i] * scale_factor + eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * scale_factor);
        beta *= alpha;

        //! process scale
        alpha *= scale_w[i];

        if (scale_bias_term) {
            beta = beta * scale_w[i] + scale_b[i];
        } else {
            beta *= scale_w[i];
        }

        for (int j = 0; j < chw; ++j) {
            weight_data[i * chw + j] *= alpha;
        }

        bias_data[i] *= alpha;
        bias_data[i] += beta;
    }
}


} //namespace lite

} //namespace saber

} //namespace anakin


#endif

