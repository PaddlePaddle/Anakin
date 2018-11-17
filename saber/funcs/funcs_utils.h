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
#include "saber/saber_funcs_param.h"

namespace anakin{
namespace saber{

template <class Param >
Shape conv_compute_shape(const Shape input_shape, Param &param) {
    Shape output_shape = (input_shape);
    CHECK_GE(input_shape.size(), 4) << "using reshape2d to reshape a 1d conv?";

    output_shape.set_num(input_shape.num()); // N
    output_shape.set_channel(param.weight()->num()); // K

    int input_dim = input_shape.height(); // P
    int kernel_exten = param.dilation_h * (param.weight()->height() - 1) + 1;
    int output_height = (input_dim + 2 * param.pad_h - kernel_exten)
                     / param.stride_h + 1;
    output_shape.set_height(output_height);

    input_dim = input_shape.width(); // Q
    kernel_exten = param.dilation_w * (param.weight()->width() - 1) + 1;
    int output_width = (input_dim + 2 * param.pad_w - kernel_exten)
                 / param.stride_w + 1;
    output_shape.set_width(output_width);
    return output_shape;
}

template <typename TargetType>
Shape deconv_compute_shape(const Shape input_shape, ConvParam<TargetType> &param) {
    Shape output_shape = input_shape;
    CHECK_GE(input_shape.size(), 4) << "using reshape2d to reshape a 1d deconv?";

    // append the $n and $c/$k, output: N * K * P * Q

    output_shape.set_num(input_shape.num()); // N
    output_shape.set_channel(param.weight()->num() * param.group); // K

    int kernel_extent_h = param.dilation_h *
                          (param.weight()->height() - 1) + 1;
    int output_dim_h = (input_shape.height() - 1) *
                       param.stride_h + kernel_extent_h - 2 * param.pad_h;
    int kernel_extent_w = param.dilation_w *
                          (param.weight()->width() - 1) + 1;
    int output_dim_w = (input_shape.width() - 1) *
                       param.stride_w + kernel_extent_w - 2 * param.pad_w;

    output_shape.set_height(output_dim_h);
    output_shape.set_width(output_dim_w);
    return output_shape;
}

template <class Param >
Shape pool_compute_shape(const Shape input_shape, Param &param) {

    Shape output_shape = input_shape;

    int in_height = input_shape.height();
    int in_width = input_shape.width();

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
        param.stride_h = in_height;
        param.stride_w = in_width;
        window_h = in_height;
        window_w = in_width;
        param.window_h = in_height;
        param.window_w = in_width;
    } else {
        if (param.cmp_out_shape_floor_as_conv) {
            out_height = static_cast<int>((static_cast<float>(
                                                   in_height + 2 * pad_h - window_h) / stride_h)) + 1;

            out_width = static_cast<int>((static_cast<float>(
                                                  in_width + 2 * pad_w - window_w) / stride_w)) + 1;
        } else {
            out_height = static_cast<int>(ceilf(static_cast<float>(
                                                        in_height + 2 * pad_h - window_h) / stride_h)) + 1;

            out_width = static_cast<int>(ceilf(static_cast<float>(
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
    output_shape.set_height(out_height);
    output_shape.set_width(out_width);
    return output_shape;
}

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
void update_conv_weights(Param<Tensor_t>& param) {
#ifdef USE_ARM_PLACE
    Tensor<ARM> new_weight;
    Tensor<ARM> new_bias;
#elif defined(USE_CUDA)
    Tensor<NVHX86> new_weight;
    Tensor<NVHX86> new_bias;
#else
    Tensor<X86> new_weight;
    Tensor<X86> new_bias;
#endif //USE_ARM_PLACE
    typedef typename Tensor_t::FDtype Dtype;
    DataType dtype = param.conv_param.weight()->get_dtype();
    CHECK_EQ(dtype, AK_FLOAT) << "only support float type weights";

    Shape weight_shape = param.conv_param.weight()->shape();
    new_weight.re_alloc(weight_shape, AK_FLOAT);
    new_weight.copy_from(*(param.conv_param.weight()));
    Shape bias_shape;

    if (param.conv_param.bias()->size() > 0) {
        bias_shape = param.conv_param.bias()->shape();
        new_bias.re_alloc(bias_shape, AK_FLOAT);
        new_bias.copy_from(*(param.conv_param.bias()));

    } else if (param.has_batchnorm) {
        bias_shape = {1, param.batchnorm_param.mean.size(), 1, 1};
        new_bias.re_alloc(bias_shape, AK_FLOAT);
        void* new_bias_data = new_bias.mutable_data();
        memset(new_bias_data, 0, sizeof(Dtype) * new_bias.size());

    } else if (param.has_scale) {
        bias_shape = {1, param.scale_param.scale_w.size(), 1, 1};
        new_bias.re_alloc(bias_shape, AK_FLOAT);
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
    Tensor<ARM> new_weight;
    Tensor<ARM> new_bias;
#elif defined(USE_CUDA)
    Tensor<NVHX86> new_weight;
    Tensor<NVHX86> new_bias;
#else
    Tensor<X86> new_weight;
    Tensor<X86> new_bias;
#endif //USE_ARM_PLACE
    //typedef typename Tensor_t::FDtype dtype;
    CHECK_EQ(AK_FLOAT, param.conv_param.weight()->get_dtype()) << "only support float weights";

    Shape weight_shape = param.conv_param.weight()->shape();
    new_weight.re_alloc(weight_shape, AK_FLOAT);
    new_weight.copy_from(*(param.conv_param.weight()));
    Shape bias_shape;

    if (param.conv_param.bias()->size() > 0) {
        bias_shape = param.conv_param.bias()->shape();
        new_bias.re_alloc(bias_shape, AK_FLOAT);
        new_bias.copy_from(*(param.conv_param.bias()));

    } else if (param.has_batchnorm) {
        bias_shape = {1, param.batchnorm_param.mean.size(), 1, 1};
        new_bias.re_alloc(bias_shape, AK_FLOAT);
        void* new_bias_data = new_bias.mutable_data();
        memset(new_bias_data, 0, sizeof(float) * new_bias.size());

    } else if (param.has_scale) {
        bias_shape = {1, param.scale_param.scale_w.size(), 1, 1};
        new_bias.re_alloc(bias_shape, AK_FLOAT);
        void* new_bias_data = new_bias.mutable_data();
        memset(new_bias_data, 0, sizeof(float) * new_bias.size());
    } else {
        return;
    }
    int filter_num = new_weight.num();
    int channel_num_per_group = new_weight.channel();
    std::vector<float> scale(new_weight.num(), 0);
    std::vector<float> shift(new_weight.num(), 0);

    for (int i = 0; i < filter_num; ++i) {
        float alpha = 1.f;
        float beta = 0.f;

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


    float* weight_data = (float*)new_weight.mutable_data();
    float* bias_data = (float*)new_bias.mutable_data();
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

template<typename Tensor_d, typename Tensor_h, int k>
void scale_to_new_tensor_k4_s2_p1_deconv (Tensor_d &weight, int in_channel, int out_channel,
                                          bool in_place = false, Tensor_d *weight_dev = nullptr) {
    Tensor_h new_weights_h;
    Tensor_h temp_weights;
//    new_weights_dev.reshape(weight->valid_shape());
    new_weights_h.reshape(weight.valid_shape());
    temp_weights.reshape(weight.valid_shape());
    temp_weights.copy_from(weight);
    int offset = in_channel * out_channel * k;
    float* trans_w = (float*)new_weights_h.mutable_data();
    scale_weight_deconv_w4x4<k, true>(trans_w + 0 * offset,
                                      trans_w + 1 * offset,
                                      trans_w + 2 * offset,
                                      trans_w + 3 * offset,
                                      static_cast<float*>(temp_weights.data()),
                                      in_channel, out_channel);
    if (in_place) {
        weight.copy_from(new_weights_h);
    } else {
        weight_dev->re_alloc(weight.valid_shape(), AK_FLOAT);
        weight_dev->copy_from(new_weights_h);
    }
}

inline int align_up(int a, int b) {
    return (a % b != 0) ? (a - a % b + b) : a;
}

template <typename TargetType, typename TargetType_H>
void conv_trans_weights(Tensor<TargetType> &target_weights,
        int stride_h, int stride_w, int group,
        bool in_place = false, Tensor<TargetType>* weight_dev = nullptr,
        int dilation_h = 1, int dilation_w = 1) {

    Tensor<TargetType_H> trans_weights_host;
    if (stride_h == 1 &&
    stride_w == 1 &&
    target_weights.height() == 3 &&
    target_weights.width() == 3 && group == 1
    && dilation_h == 1 && dilation_w == 1) {
        //Update weights if need
        Shape weight_shape = target_weights.valid_shape();
        Tensor<TargetType_H> new_weight;
        new_weight.re_alloc(weight_shape, target_weights.get_dtype());
        new_weight.copy_from(target_weights);
        float *weight_data = (float *)new_weight.mutable_data();
        int round_in_channel = align_up(target_weights.channel(), 8);
        int round_out_channel = align_up(target_weights.num(), 32);
        int weight4x4_size = round_in_channel * round_out_channel * 4 * 4;
        Shape old_shape = target_weights.valid_shape();
        Shape new_trans_weights_shape({{weight4x4_size, 1, 1 ,1}}, target_weights.get_layout());
        trans_weights_host.re_alloc(new_trans_weights_shape, target_weights.get_dtype());
        float* _host_work_space = (float*)trans_weights_host.mutable_data();
        transform_3x3_weight_2_4x4(weight_data, _host_work_space, target_weights.num(),
        round_out_channel, target_weights.channel(), round_in_channel);
        Shape new_weights_shape({weight4x4_size, 1, 1, 1}, target_weights.get_layout());
        if (in_place) {
            target_weights.re_alloc(new_weights_shape, target_weights.get_dtype());
            target_weights.copy_from(trans_weights_host);
            target_weights.set_shape(old_shape);
        } else {
            weight_dev->re_alloc(new_weights_shape, target_weights.get_dtype());
            weight_dev->copy_from(trans_weights_host);
            weight_dev->set_shape(old_shape);
        }
    } else if (group == 1) {

        int weight_size = (target_weights.valid_shape()).count();
        Tensor<TargetType_H> weight_host;
        weight_host.re_alloc(target_weights.valid_shape(), target_weights.get_dtype());
        weight_host.copy_from(target_weights);
        const float *weight_data = (const float *)weight_host.data();
        trans_weights_host.re_alloc(target_weights.valid_shape(), target_weights.get_dtype());
        float* _host_work_space = (float*)trans_weights_host.mutable_data();

        transpose_filter_KCRS_2_CRSK(weight_data, _host_work_space, \
                                                 target_weights.num(), \
                                                 target_weights.channel(), \
                                                 target_weights.height(), \
                                                 target_weights.width());
        if (in_place) {
            target_weights.re_alloc(target_weights.valid_shape(), target_weights.get_dtype());
            target_weights.copy_from(trans_weights_host);
        } else {
            weight_dev->re_alloc(target_weights.valid_shape(), target_weights.get_dtype());
            weight_dev->copy_from(trans_weights_host);
        }

    }
//    cudaDeviceSynchronize();
}

} // namespace saber

} // namespace anakin
#endif //SABER_FUNCS_UTILS_H



