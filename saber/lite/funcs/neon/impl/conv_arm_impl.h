/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#ifndef ANAKIN_SABER_LITE_FUNCS_NEON_IMPL_CONV_ARM_IMPL_H
#define ANAKIN_SABER_LITE_FUNCS_NEON_IMPL_CONV_ARM_IMPL_H

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"
#ifdef USE_ARM_PLACE

//#include "saber/lite/funcs/neon/impl/sgemm_arm.h"
namespace anakin{

namespace saber{

namespace lite{

void conv_3x3s1_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_3x3s1_direct_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_3x3s2_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_3x3s2_direct_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_3x3s2_2_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_7x7s1_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_1x5s1_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_5x1s1_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv1x1s1_gemm(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv1x1s1_gemm_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_im2col_gemm(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void conv_im2col_gemm_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias
 */
void conv_depthwise_3x3(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);


void conv_depthwise_3x3_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, \
                          Context* ctx, void* work_space, const void* idx_ptr);

void conv_arm_winograd3x3(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);

void winograd_transform_weights(void* dout, const void* din, int ch_out, \
    int ch_in, void* work_space);

void compute_offset(int* idx_out, int h, int w, int kernel_h, int kernel_w, int height, int width, int pad_h, int pad_w, int dilation_h, int dilation_w);

void fill_bias(float* tensor, const float* bias, int channel, int channel_size);

void fill_bias_int8(int* tensor, const int* bias, int channel, int channel_size);

void conv3x3s2_trans_weights4c(void* dout, const void* din, int chout, int chin);

void conv3x3s2_trans_weights4c_int8(void* dout, const void* din, int chout, int chin);
#if 0
class ConvWinogradF63 {
public:
    ConvWinogradF63();
    ~ConvWinogradF63();
    bool init(const size_t l1_cache, const size_t l2_cache, \
        const int chout, const int chin, const int hin, const int win, const int threads = 4);
    bool operator()(const float* trans_weights, const float* din, float* dout, \
        void* workspace = nullptr);

private:

    unsigned int _k_block{0};
    unsigned int _x_block{0};
    unsigned int _Mround{0};

    unsigned int _loop_count{0};
    unsigned int _cblock_size{0};
    int _thread_num{1};

    void* _work_space_ptr{nullptr};

    size_t _work_size{0};
    size_t _a_worksize{0};
    size_t _b_worksize{0};
    load_data _load_a;
    load_data _load_b;

    bool _init_flag{false};
};
#endif

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_NEON_IMPL_CONV_ARM_IMPL_H
