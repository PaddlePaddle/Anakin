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

#ifndef SABER_FUNCS_UTILS_H
#define SABER_FUNCS_UTILS_H

#include <iostream>
#include <map>
#include "saber/core/tensor.h"
#include "saber/core/tensor_op.h"
namespace anakin{
namespace saber{

template <typename Dtype>

void transpose_inplace(float* output, const float* input, const int num,
                       const int channel,
                       const int height, const int width) {
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channel; ++c) {
            int offset = n * channel * height * width + c * height * width;

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    output[(x * height) + y + offset] = input[(y * width) + x + offset];
                }
            }
        }
    }
}

template <typename Dtype>
void extract_matrix_from_matrix_in_leddim(const Dtype* input,
                Dtype* output,int start_index,int end_index,int stride,int dimsize){
    for(int i=start_index;i<end_index;i+=stride){
        int output_height=(i-start_index)/stride;
        for(int j=0;j<dimsize;j++){
            output[output_height*dimsize+j]=input[i+j];
        }
    }
}



template <typename Dtype>
void merge_matrix_to_matrix_in_leddim(const Dtype* input,
                                          Dtype* output,int start_index,int end_index,int stride,int dimsize){
    for(int i=start_index;i<end_index;i+=stride){
        int input_height=(i-start_index)/stride;
        for(int j=0;j<dimsize;j++){
            output[i+j]=input[input_height*dimsize+j];
        }
    }
}

template <typename Dtype>
void transform_3x3_weight_2_4x4(const Dtype* input, 
    Dtype* output,
    int K,
    int k_align_up,
    int C,
    int c_alignup)
{
    float g[3][3];
    float G[4][4];
    //for(int k = 0; k < k_align_up; k++)
    for (int k = 0; k < k_align_up; k++)
    {
        for (int c = 0; c < c_alignup; c++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (k < K && c < C)
                    {
                        g[i][j] = input[k*C*9 + c*9 + i*3 + j];
                    }else{
                        g[i][j] = 0.f;
                    }
             
                }
            }
            G[0][0] = g[0][0];
            G[0][1] = 0.5*(g[0][0] + g[0][1] + g[0][2]);
            G[0][2] = 0.5*(g[0][0] - g[0][1] + g[0][2]);
            G[0][3] = g[0][2];

            G[1][0] = 0.50*(g[0][0] + g[1][0] + g[2][0]);
            G[1][1] = 0.25*(g[0][0] + g[0][1] + g[0][2]  
                + g[1][0] + g[1][1] + g[1][2]  
                + g[2][0] + g[2][1] + g[2][2]);
            G[1][2] = 0.25*(g[0][0] - g[0][1] + g[0][2]  
                + g[1][0] - g[1][1] + g[1][2]  
                + g[2][0] - g[2][1] + g[2][2]);
            G[1][3] = 0.50*(g[0][2] + g[1][2] + g[2][2]);

            G[2][0] = 0.50*(g[0][0] - g[1][0] + g[2][0]);
            G[2][1] = 0.25*(g[0][0] + g[0][1] + g[0][2]  
                - g[1][0] - g[1][1] - g[1][2]  
                + g[2][0] + g[2][1] + g[2][2]);
            G[2][2] = 0.25*(g[0][0] - g[0][1] + g[0][2]  
                - g[1][0] + g[1][1] - g[1][2]  
                + g[2][0] - g[2][1] + g[2][2]);
            G[2][3] = 0.50*(g[0][2] - g[1][2] + g[2][2]);

            G[3][0] = g[2][0];
            G[3][1] = 0.50*(g[2][0] + g[2][1] + g[2][2]);
            G[3][2] = 0.50*(g[2][0] - g[2][1] + g[2][2]);
            G[3][3] = g[2][2];

            int kidx_0 = k % 32;
            int kidx_1 = k / 32;

            int kidx_16 = kidx_0 / 16;
            int kidx_16_0 = kidx_0 % 16;
            int kidx_height = kidx_16_0 % 4;
            int kidx_width = kidx_16_0 / 4;
            int cidx_0 = c % 8;
            int cidx_1 = c / 8;

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    int idx_0 = (i * 4 + j) % 2;
                    int idx_1 = (i * 4 + j) / 2;

                    int offset = 
                        kidx_1 * 32 * 2 * 8
                        + cidx_1 * (k_align_up * 2 * 8 * 8) 
                        + cidx_0 * 2 * 32 + idx_1 * (k_align_up * 2 * 8) 
                        + idx_0 * 32 + kidx_16 * 16 + kidx_height * 4 + kidx_width;
                    output[offset] = G[i][j];
                }
            }
        }
    }
}

// transform 
    // PAY ATTENTION!!!![zs]
    // The shape of weights is suppose to be {in_channel, out_channel, kernel_size, kernel_size};
    // but caffe is reshaped their shape as {out, in, kernel_size, kernel_size}
    // so this need to reshaped like caffe as {out_channel, in_channel, kernel_size, kernel_size}
    // The param of transform_weights_deconv:
    // int in_channel  : the in_channel of the img(where loop running on!)
    //                   this param must be the seam with img in_channel,
    //
    // int out_channel : the real output filter num(as much as you can, this is the proto param)
    //
    // const float *
    //     weights_src : the real data is orgnized as 
    //                   (in_channel, out_channel, kernel_size, kernel_size)
    // const float *
    //     XX_out      : the output data is orgnized as
    //                   (out_channel, in_channel, kernel_size, kernel_size)
    //                   just like normal convolution weights

template<int kernel_size, bool cross>
void scale_weight_deconv_w4x4(float *w1,
                                float *w2,
                                float *w3,
                                float *w4,
                                const float* src_weights,
                                int in_channel, int out_channel) {

    int flag = cross ? 1: (-1);
    int pass = cross ? 15 : 0;
    for (int o = 0; o < out_channel; ++o) {
        for (int c = 0; c < in_channel; ++c) {
            int i_off = c * out_channel * 16 + o * 16;
            int o_off = o * in_channel * 4 + c * 4;

            w1[o_off + 0] = src_weights[i_off + pass - (flag * 0)];
            w2[o_off + 0] = src_weights[i_off + pass - (flag * 1)];
            w1[o_off + 1] = src_weights[i_off + pass - (flag * 2)];
            w2[o_off + 1] = src_weights[i_off + pass - (flag * 3)];
            w3[o_off + 0] = src_weights[i_off + pass - (flag * 4)];
            w4[o_off + 0] = src_weights[i_off + pass - (flag * 5)];
            w3[o_off + 1] = src_weights[i_off + pass - (flag * 6)];
            w4[o_off + 1] = src_weights[i_off + pass - (flag * 7)];
            w1[o_off + 2] = src_weights[i_off + pass - (flag * 8)];
            w2[o_off + 2] = src_weights[i_off + pass - (flag * 9)];
            w1[o_off + 3] = src_weights[i_off + pass - (flag * 10)];
            w2[o_off + 3] = src_weights[i_off + pass - (flag * 11)];
            w3[o_off + 2] = src_weights[i_off + pass - (flag * 12)];
            w4[o_off + 2] = src_weights[i_off + pass - (flag * 13)];
            w3[o_off + 3] = src_weights[i_off + pass - (flag * 14)];
            w4[o_off + 3] = src_weights[i_off + pass - (flag * 15)];
        }
    }
}

template <typename Dtype>
void transpose_filter_KCRS_2_CRSK(const Dtype *input, Dtype *output, \
    int K, int C, int R, int S) {
    const int CRS = C * R * S;
    for (int var_k = 0; var_k < K; var_k++) {
        for (int var_crs = 0; var_crs < CRS; var_crs++) {
            output[var_crs * K + var_k] = input[var_k * CRS + var_crs];
        }
    }
}

template < typename Tensor_t, template <typename T> class Param >
void update_conv_weights(Param<Tensor_t>& param)
{
#ifdef USE_ARM_PLACE
    Tensor<ARM, AK_FLOAT, NCHW> new_weight;
    Tensor<ARM, AK_FLOAT, NCHW> new_bias;
#else
    Tensor<X86, AK_FLOAT, NCHW> new_weight;
    Tensor<X86, AK_FLOAT, NCHW> new_bias;
#endif //USE_ARM_PLACE
    typedef typename Tensor_t::Dtype Dtype;

    Shape weight_shape = param.conv_param.weight()->shape();
    new_weight.re_alloc(weight_shape);
    new_weight.copy_from(*(param.conv_param.weight()));
    Shape bias_shape;

    if (param.conv_param.bias()->size() > 0) {
        bias_shape = param.conv_param.bias()->shape();
        new_bias.re_alloc(bias_shape);
        new_bias.copy_from(*(param.conv_param.bias()));

    } else if (param.has_batchnorm) {
        bias_shape = {1, param.batchnorm_param.mean.size(), 1, 1};
        new_bias.re_alloc(bias_shape);
        void* new_bias_data = new_bias.mutable_data();
        memset(new_bias_data, 0, sizeof(Dtype) * new_bias.size());

    } else if (param.has_scale) {
        bias_shape = {1, param.scale_param.scale_w.size(), 1, 1};
        new_bias.re_alloc(bias_shape);
        void* new_bias_data = new_bias.mutable_data();
        memset(new_bias_data, 0, sizeof(Dtype) * new_bias.size());
    } else {
        return;
    }

    int filter_num = new_weight.num();
    int chw = new_weight.channel();

    Dtype* weight_data = new_weight.mutable_data();
    Dtype* bias_data = new_bias.mutable_data();

    chw *= new_weight.height();
    chw *= new_weight.width();

    for (int i = 0; i < filter_num; ++i) {
        Dtype alpha = 1.f;
        Dtype beta = 0.f;

        if (param.has_batchnorm) {
            float scale_factor = 1.f;
            scale_factor = (param.batchnorm_param.scale == 0) ?
                           1 : 1.f / param.batchnorm_param.scale;
            float eps = param.batchnorm_param.eps;
            float variance;
            float mean;
            alpha = param.batchnorm_param.variance[i] * scale_factor + eps;
            alpha = 1.f / sqrtf(alpha);
            beta = -1.f * (param.batchnorm_param.mean[i] * scale_factor);
            beta *= alpha;
        }

        if (param.has_scale) {
            alpha *= param.scale_param.scale_w[i];

            if (param.scale_param.bias_term) {
                beta = beta * param.scale_param.scale_w[i]
                       + param.scale_param.scale_b[i];
            } else {
                beta *= param.scale_param.scale_w[i];
            }
        }

        for (int j = 0; j < chw; ++j) {
            weight_data[i * chw + j] *= alpha;
        }

        bias_data[i] *= alpha;
        bias_data[i] += beta;
    }

    param.conv_param.mutable_weight()->copy_from(new_weight);
    Shape new_bias_shape = new_bias.shape();
    param.conv_param.mutable_bias()->re_alloc(new_bias_shape);
    param.conv_param.mutable_bias()->copy_from(new_bias);
}

template < typename Tensor_t, template <typename T> class Param >
void update_deconv_weights(Param<Tensor_t>& param)
{
#ifdef USE_ARM_PLACE
    Tensor<ARM, AK_FLOAT, NCHW> new_weight;
    Tensor<ARM, AK_FLOAT, NCHW> new_bias;
#else
    Tensor<X86, AK_FLOAT, NCHW> new_weight;
    Tensor<X86, AK_FLOAT, NCHW> new_bias;
#endif //USE_ARM_PLACE
    typedef typename Tensor_t::Dtype dtype;

    Shape weight_shape = param.conv_param.weight()->shape();
    new_weight.re_alloc(weight_shape);
    new_weight.copy_from(*(param.conv_param.weight()));
    Shape bias_shape;

    if (param.conv_param.bias()->size() > 0) {
        bias_shape = param.conv_param.bias()->shape();
        new_bias.re_alloc(bias_shape);
        new_bias.copy_from(*(param.conv_param.bias()));

    } else if (param.has_batchnorm) {
        bias_shape = {1, param.batchnorm_param.mean.size(), 1, 1};
        new_bias.re_alloc(bias_shape);
        void* new_bias_data = new_bias.mutable_data();
        memset(new_bias_data, 0, sizeof(dtype) * new_bias.size());

    } else if (param.has_scale) {
        bias_shape = {1, param.scale_param.scale_w.size(), 1, 1};
        new_bias.re_alloc(bias_shape);
        void* new_bias_data = new_bias.mutable_data();
        memset(new_bias_data, 0, sizeof(dtype) * new_bias.size());
    } else {
        return;
    }
    int filter_num = new_weight.num();
    int channel_num_per_group = new_weight.channel();
    std::vector<dtype> scale(new_weight.num(), 0);
    std::vector<dtype> shift(new_weight.num(), 0);

    for (int i = 0; i < filter_num; ++i) {
        dtype alpha = 1.f;
        dtype beta = 0.f;

        if (param.has_batchnorm) {
            float scale_factor = 1.f;
            scale_factor = (param.batchnorm_param.scale == 0) ?
                           1 : 1.f / param.batchnorm_param.scale;
            float eps = param.batchnorm_param.eps;
            float variance;
            float mean;
            alpha = param.batchnorm_param.variance[i] * scale_factor + eps;
            alpha = 1.f / sqrtf(alpha);
            beta = -1.f * (param.batchnorm_param.mean[i] * scale_factor);
            beta *= alpha;
        }

        if (param.has_scale) {
            alpha *= param.scale_param.scale_w[i];

            if (param.scale_param.bias_term) {
                beta = beta * param.scale_param.scale_w[i]
                       + param.scale_param.scale_b[i];
            } else {
                beta *= param.scale_param.scale_w[i];
            }
        }
        scale[i] = alpha;
        shift[i] = beta;
    }


    dtype* weight_data = new_weight.mutable_data();
    dtype* bias_data = new_bias.mutable_data();
    // {Ic, Oc/group, K_h, K_w} real shape
    // {Oc, Ic/group, K_h, K_w} parser return back shape
    // filter_num = Oc;
    // channel_num_per_group = Ic/group;
    // [group, Ic/group, Oc/group, K_h, k_w]

    int hw = new_weight.height() * new_weight.width();
    int group = param.conv_param.group;
    int filter_num_per_group = filter_num / group;
    int id = 0;
    for (int i = 0; i < group; i++) {
        for (int j = 0; j < channel_num_per_group; j++) {
            for (int k = 0; k < filter_num_per_group; k++) {
                int out_channel_id = i * filter_num_per_group + k;
                for (int m = 0; m < hw; m++) {
                    weight_data[id] = weight_data[id]* scale[out_channel_id];
                    id++;
                }
            }
        }
    }

    for (int i = 0; i < filter_num; i++) {
        bias_data[i] *= scale[i];
        bias_data[i] += shift[i];
    }

    param.conv_param.mutable_weight()->copy_from(new_weight);
    Shape new_bias_shape = new_bias.shape();
    param.conv_param.mutable_bias()->re_alloc(new_bias_shape);
    param.conv_param.mutable_bias()->copy_from(new_bias);
}

} // namespace saber

} // namespace anakin
#endif //SABER_FUNCS_UTILS_H



