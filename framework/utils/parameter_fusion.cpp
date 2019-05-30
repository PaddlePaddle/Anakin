/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#include "framework/utils/parameter_fusion.h"
namespace anakin {

static void basic_x86_gemm(const int m, const int n, const int k,
                const float* a, const float* b, float* c,
                const float alpha, const float beta,
                const bool trans_a, const bool trans_b) {
    if (!trans_a && !trans_b) {
        int lda = k;
        int ldb = n;
        int ldc = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;

                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[m_i * lda + k_i] * b[k_i * ldb + n_i];
                }
            }
        }
    } else if (!trans_a && trans_b) {
        int lda = k;
        int ldb = k;
        int ldc = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;

                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[m_i * lda + k_i] * b[n_i * ldb + k_i];
                }
            }
        }
    } else if (trans_a && !trans_b) {
        int lda = m;
        int ldb = n;
        int ldc = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;

                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[k_i * lda + m_i] * b[k_i * ldb + n_i];
                }
            }
        }
    } else {
        int lda = m;
        int ldb = k;
        int ldc = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;

                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[k_i * lda + m_i] * b[n_i * ldb + k_i];
                }
            }
        }
    }

}
/**
 * \brief  update fp32 conv weights with batchnorm and scale parameters.
 */
template<typename T>
void WeightsFusion<float, T>::update_weights(
                    PBlock<T> weights, PBlock<T> bias,
                    int n, int c, int h, int w, bool conv_bias_term,
                    float batchnorm_scale, float batchnorm_eps,
                    std::vector<float> batchnorm_mean,
                    std::vector<float> batchnorm_variance,
                    std::vector<float> scale_w,
                    std::vector<float> scale_b,
                    bool scale_bias_term) {
    float* weights_p = (float*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();
    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;

        // insert scale parameters
        alpha = scale_w[i] * alpha;
        if (scale_bias_term) {
            beta = beta * scale_w[i] + scale_b[i];
        } else {
            beta = beta * scale_w[i];
        }
        int start_index = i * chw;
        for (int j = 0; j < chw; j++) {
            weights_p[start_index + j] *= alpha;
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
 * \brief  update fp32 conv weights with affine channel parameters.
 */
template<typename T>
void WeightsFusion<float, T>::update_conv_affine_channel_weights(
                    PBlock<T> weights, PBlock<T> bias,
                    int n, int c, int h, int w,
                    std::vector<float> affine_channel_w,
                    std::vector<float> affine_channel_b) {
    float* weights_p = (float*)(weights.h_tensor().mutable_data());
    float* bias_p = (float* )(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();
    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < chw; j++) {
            weights_p[i * chw + j] *= affine_channel_w[i];
        }
        bias_p[i] = bias_p[i] * affine_channel_w[i] + affine_channel_b[i];
    }
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
 * \brief  update fp32 conv weights with batchnorm.
 */
template<typename T>
void WeightsFusion<float, T>::update_weights_without_scale(
                    PBlock<T> weights, PBlock<T> bias,
                    int n, int c, int h, int w, bool conv_bias_term,
                    float batchnorm_scale, float batchnorm_eps,
                    std::vector<float> batchnorm_mean,
                    std::vector<float> batchnorm_variance) {
    float* weights_p = (float* )(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();

    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;
        int start_index = i * chw;
        for (int j = 0; j < chw; j++) {
            weights_p[start_index + j] *= alpha;
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

template<typename T>
void WeightsFusion<float, T>::update_weights_conv_scale(PBlock<T> weights, PBlock<T> bias,
                               int n, int c, int h, int w, bool conv_bias_term,
                               std::vector<float> scale_w,
                               std::vector<float> scale_b,
                               bool scale_bias_term){
    float* weights_p = (float*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, scale_w.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();

    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        float alpha = scale_w[i];
        float beta = 0.f;
        if (scale_bias_term) {
            beta = scale_b[i];
        }
        int start_index = i * chw;
        for (int j = 0; j < chw; j++) {
            weights_p[start_index + j] *= alpha;
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
 * \brief  update fp32 deconv weights with batchnorm and scale parameters.
 */
template<typename T>
void WeightsFusion<float, T>::update_deconv_weights(
                    PBlock<T> weights, PBlock<T> bias,
                    int n, int c, int h, int w, bool conv_bias_term,
                    float batchnorm_scale, float batchnorm_eps,
                    std::vector<float> batchnorm_mean,
                    std::vector<float> batchnorm_variance,
                    std::vector<float> scale_w,
                    std::vector<float> scale_b,
                    bool scale_bias_term) {
    float* weights_p = (float*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();

    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    //swap n and c
    int tn = c;
    c = n;
    n = tn;

    int chw = c * h * w;
    int hw = h * w;
    for (int i = 0; i < c; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;

        // insert scale parameters
        alpha = scale_w[i] * alpha;
        if (scale_bias_term) {
            beta = beta * scale_w[i] + scale_b[i];
        } else {
            beta = beta * scale_w[i];
        }
        for (int ni = 0; ni < n; ++ni){
            for (int j=0; j < hw; j++) {
                weights_p[ni * chw + i * hw + j] *= alpha;
            }
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
 * \brief  update fp32 deconv weights with batchnorm.
 */
template<typename T>
void WeightsFusion<float, T>::update_deconv_weights_without_scale(
                    PBlock<T> weights, PBlock<T> bias,
                    int n, int c, int h, int w, bool conv_bias_term,
                    float batchnorm_scale, float batchnorm_eps,
                    std::vector<float> batchnorm_mean,
                    std::vector<float> batchnorm_variance) {
    float* weights_p = (float*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();

    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    //swap n and c
    int tn = c;
    c = n;
    n = tn;

    int chw = c * h * w;
    int hw = h * w;
    for (int i = 0; i < c; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;
        for (int ni = 0; ni < n; ++ni){
            for (int j=0; j < hw; j++){
                weights_p[ni * chw + i * hw + j] *= alpha;
            }
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
* \brief  update dense weights.
*/
template<typename T>
void WeightsFusion<float, T>::update_dense_weights(PBlock<T> weights_0, PBlock<T> bias_0, 
    bool bias_term_0, int out_dim_0, bool is_trans_0,
    PBlock<T> weights_1, PBlock<T> bias_1, 
    bool bias_term_1, int out_dim_1, bool is_trans_1){
        typedef typename target_host<T>::type T_HOST;
        auto& w_0_tensor = weights_0.h_tensor();
        auto& w_1_tensor = weights_1.h_tensor();
        auto& b_0_tensor = bias_0.h_tensor();
        auto& b_1_tensor = bias_1.h_tensor();

        const float* w_0_data = static_cast<const float*>(w_0_tensor.data());
        const float* w_1_data = static_cast<const float*>(w_1_tensor.data());
        CHECK_GT(out_dim_1, 0) << "dense out dim must > 0";
        CHECK_GT(out_dim_0, 0) << "dense out dim must > 0";
        CHECK_GE(w_0_tensor.valid_size(), out_dim_0);
        CHECK_GE(w_1_tensor.valid_size(), out_dim_0 * out_dim_1);
        int m = w_0_tensor.valid_size() / out_dim_0;
        int k = out_dim_0;
        int n = out_dim_1;

        Tensor<T_HOST> temp_tensor;
        temp_tensor.re_alloc(Shape({1, 1, m, n}));
        float* w_fusion_data = static_cast<float*>(temp_tensor.mutable_data());

        
        basic_x86_gemm(n, m, k, w_1_data, w_0_data, w_fusion_data, 
                       1, 0, is_trans_1, is_trans_0);
        
        
        weights_0.re_alloc(temp_tensor.valid_shape());

        w_0_tensor.copy_from(temp_tensor);
        weights_0.map_to_device();

        if (bias_term_0){
            int n = out_dim_1;
            int k = w_1_tensor.valid_size() / n;
            CHECK_GE(b_0_tensor.valid_size(), k);
            int m = 1;

            Tensor<T_HOST> temp_bias_tensor;
            temp_bias_tensor.re_alloc(Shape({1, n, 1, 1}));
            float* b_fusion_data = static_cast<float*>(temp_bias_tensor.mutable_data());            
            const float* b_0_data = static_cast<float*>(b_0_tensor.data());
            int beta = 0;
            if (bias_term_1){
                CHECK_GE(b_1_tensor.valid_size(), n);
                temp_bias_tensor.copy_from(b_1_tensor);
                beta = 1;
            }

            basic_x86_gemm(m, n, k, b_0_data, w_1_data, b_fusion_data, 1, beta, false, !is_trans_1);

            b_1_tensor.copy_from(temp_bias_tensor);
            bias_1.map_to_device();
            
        }
}

/**
 * \brief  update int8 conv weights with batchnorm and scale parameters.
 */
template<typename T>
void WeightsFusion<char, T>::update_weights(
        PBlock<T> weights, PBlock<T> bias,
        int n, int c, int h, int w, bool conv_bias_term,
        float batchnorm_scale, float batchnorm_eps,
        std::vector<float> batchnorm_mean,
        std::vector<float> batchnorm_variance,
        std::vector<float> scale_w,
        std::vector<float> scale_b,
        bool scale_bias_term) {
    char* weights_p = (char*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();
    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;

        // insert scale parameters
        alpha = scale_w[i] * alpha;
        if (scale_bias_term) {
            beta = beta * scale_w[i] + scale_b[i];
        } else {
            beta = beta * scale_w[i];
        }
        // change weights scale
        w_scale[i] *= alpha;
        if (w_scale[i] < 0){
            w_scale[i] = fabs(w_scale[i]);
            for (int j = 0; j < chw; ++j){
                weights_p[i * chw + j] *= -1;
            }
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.h_tensor().set_scale(w_scale);
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

template<typename T>
void WeightsFusion<char, T>::update_weights_conv_scale(PBlock<T> weights, PBlock<T> bias,
                               int n, int c, int h, int w, bool conv_bias_term,
                               std::vector<float> scale_w,
                               std::vector<float> scale_b,
                               bool scale_bias_term){
    char* weights_p = (char*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, scale_w.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();

    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        float alpha = scale_w[i];
        float beta = 0.f;
        // insert scale parameters
        if (scale_bias_term) {
            beta = scale_b[i];
        }
        int start_index = i * chw;
        for (int j = 0; j < chw; j++) {
            weights_p[start_index + j] *= alpha;
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.h_tensor().set_scale(w_scale);
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
 * \brief  update int8 conv weights with affine channel parameters.
 */
template<typename T>
void WeightsFusion<char, T>::update_conv_affine_channel_weights(
        PBlock<T> weights, PBlock<T> bias,
        int n, int c, int h, int w,
        std::vector<float> affine_channel_w,
        std::vector<float> affine_channel_b) {
    char* weights_p = (char*)(weights.h_tensor().mutable_data());
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();
    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        // change weights scale
        w_scale[i] *= affine_channel_w[i];
        if (w_scale[i] < 0){
            w_scale[i] = fabs(w_scale[i]);
            for (int j = 0; j < chw; ++j){
                weights_p[i * chw + j] *= -1;
            }
        }
        bias_p[i] = bias_p[i] * affine_channel_w[i] + affine_channel_b[i];
    }
    weights.h_tensor().set_scale(w_scale);
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
 * \brief  update int8 conv weights with batchnorm.
 */
template<typename T>
void WeightsFusion<char, T>::update_weights_without_scale(
        PBlock<T> weights, PBlock<T> bias,
        int n, int c, int h, int w, bool conv_bias_term,
        float batchnorm_scale, float batchnorm_eps,
        std::vector<float> batchnorm_mean,
        std::vector<float> batchnorm_variance) {
    char* weights_p = (char*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());
    std::vector<float> w_scale = weights.h_tensor().get_scale();
    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;

        // change weights scale
        w_scale[i] *= alpha;
        if (w_scale[i] < 0){
            w_scale[i] = fabs(w_scale[i]);
            for (int j = 0; j < chw; ++j){
                int start_index = i * chw;
                weights_p[start_index + j] *= -1;
            }
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.h_tensor().set_scale(w_scale);
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}
/**
 * \brief  update int8 deconv weights with batchnorm and scale parameters.
 */
template<typename T>
void WeightsFusion<char, T>::update_deconv_weights(
        PBlock<T> weights, PBlock<T> bias,
        int n, int c, int h, int w, bool conv_bias_term,
        float batchnorm_scale, float batchnorm_eps,
        std::vector<float> batchnorm_mean,
        std::vector<float> batchnorm_variance,
        std::vector<float> scale_w,
        std::vector<float> scale_b,
        bool scale_bias_term) {
    char* weights_p = (char*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());

    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    std::vector<float> w_scale = weights.h_tensor().get_scale();
    //swap n and c
    int tn = c;
    c = n;
    n = tn;

    int chw = c * h * w;
    int hw = h * w;
    for (int i = 0; i < c; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;

        // insert scale parameters
        alpha = scale_w[i] * alpha;
        if (scale_bias_term) {
            beta = beta * scale_w[i] + scale_b[i];
        } else {
            beta = beta * scale_w[i];
        }
        // change weights scale
        w_scale[i] *= alpha;
        if (w_scale[i] < 0){
            w_scale[i] = fabs(w_scale[i]);
            for (int ni = 0; ni < n; ++ni){
                for (int j = 0; j < hw; j++) {
                    weights_p[ni * chw + i * hw + j] *= -1;
                }
            }
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.h_tensor().set_scale(w_scale);
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
* \brief  update int8 deconv weights with batchnorm.
*/
template<typename T>
void WeightsFusion<char, T>::update_deconv_weights_without_scale(
        PBlock<T> weights, PBlock<T> bias,
        int n, int c, int h, int w, bool conv_bias_term,
        float batchnorm_scale, float batchnorm_eps,
        std::vector<float> batchnorm_mean,
        std::vector<float> batchnorm_variance) {
    char* weights_p = (char*)(weights.h_tensor().mutable_data());
    if (!conv_bias_term) {
        bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, sizeof(float) * bias.h_tensor().size());
    }
    float* bias_p = (float*)(bias.h_tensor().mutable_data());

    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    std::vector<float> w_scale = weights.h_tensor().get_scale();
    //swap n and c
    int tn = c;
    c = n;
    n = tn;

    int chw = c * h * w;
    int hw = h * w;
    for (int i = 0; i < c; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;
        w_scale[i] *= alpha;
        if (w_scale[i] < 0){
            w_scale[i] = fabs(w_scale[i]);
            for (int ni = 0; ni < n; ++ni){
                for (int j = 0; j < hw; j++) {
                    weights_p[ni * chw + i * hw + j] *= -1;
                }
            }
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
    weights.h_tensor().set_scale(w_scale);
    weights.d_tensor().copy_from(weights.h_tensor());
    weights.d_tensor().set_scale(w_scale);
    bias.d_tensor().copy_from(bias.h_tensor());
}
#if defined USE_CUDA
template class WeightsFusion<float, NV>;
template class WeightsFusion<char, NV>;
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
template class WeightsFusion<float, X86>;
template class WeightsFusion<char, X86>;
#endif
#if defined USE_ARM_PLACE
template class WeightsFusion<float, ARM>;
template class WeightsFusion<char, ARM>;
#endif

#if defined AMD_GPU
template class WeightsFusion<float, AMD>;
template class WeightsFusion<char, AMD>;
#endif

}
