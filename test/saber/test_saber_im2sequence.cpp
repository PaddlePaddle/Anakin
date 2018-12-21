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
#include "saber/core/context.h"
#include "saber/funcs/im2sequence.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>

using namespace anakin::saber;

/**
 * @brief Extract image patches from input tensor to a tensor with the shape 
 *                  [batch_size * output_h * ouput_w, window_h * window_w * channels]
 * output_h = (padding_up + padding_down + input_h - window_h)/strid_h + 1;
 * output_w = (padding_left + padding_right + input_w - windwo_w)/strid_w + 1;
 * 
 * @tparam dtype 
 * @tparam TargetType_D 
 * @tparam TargetType_H 
 * @param input
 * @param output : with shape [batch_size * output_h * ouput_w, window_h * window_w * channels]
 * @param param 
 */
template <typename dtype,typename TargetType_D,typename TargetType_H>
void im2sequence_cpu_base(const std::vector<Tensor<TargetType_H>* > &input, std::vector<Tensor<TargetType_H>* > &output, Im2SequenceParam<TargetType_D> &param) {
    
    int N = input[0]->num();
    int C = input[0]->channel();
    int H = input[0]->height();
    int W = input[0]->width();
    Shape output_shape = input[0]->valid_shape();

    //compute output shape.
    int kernel_extern_h = param.dilation_h * (param.window_h - 1) + 1;
    int output_height = (H + param.pad_up + param.pad_down - kernel_extern_h)
                        / param.stride_h + 1;

    int input_width = input[0]->width(); // Q
    int kernel_extern_w = param.dilation_w * (param.window_w - 1) + 1;
    int output_width = (W + param.pad_left + param.pad_right - kernel_extern_w)
                    / param.stride_w + 1;

    output_shape.set_num(input[0]->num() * output_height * output_width); // N
    output_shape.set_channel(input[0]->channel() * param.window_h * param.window_w); // K
    output_shape.set_height(1);
    output_shape.set_width(1);

    output[0]->set_shape(output_shape);

    /**
     * @brief for each channel:
     *     get patches[kernel_extern_w * kernel_extern_h] to dst tensor util the channel has been finished.
     * 
     */
    int out_rows_id = 0;
    int old_row;
    int out_cols = output_shape.channel();
    const dtype* input_ptr = (const dtype*)input[0]->data();
    dtype* output_ptr = (dtype*)output[0]->mutable_data();
    int H_pad = H + param.pad_up + param.pad_down;
    int W_pad = W + param.pad_left + param.pad_right;
    int wd_id = 0;
    int wd_num_each_channel = output_height * output_width;
    int wd_size = param.window_h * param.window_w;
    int m = 0; //the id which is mapped to the j th element of i th window
    int input_id;
    int st_id;
    int get_stride_h = param.dilation_h ? param.dilation_h : 1;
    int get_stride_w = param.dilation_w ? param.dilation_w : 1;
    for (int i = 0; i < N; i++) {
        wd_id = 0;
        out_rows_id = i * wd_num_each_channel + wd_id % wd_num_each_channel;
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < H_pad - kernel_extern_h + 1; k += param.stride_h) {
                for (int l = 0; l < W_pad - kernel_extern_w + 1; l += param.stride_w) {
                    m = 0;
                    //consider dilation.
                    for (int wd_h = k; wd_h < k + kernel_extern_h; wd_h += get_stride_h) {
                        for (int wd_w = l; wd_w < l + kernel_extern_w; wd_w += get_stride_w) {
                            input_id = i * C * H_pad * W_pad + j * H_pad * W_pad + wd_h * W_pad + wd_w;
                            st_id = out_rows_id * out_cols + j * wd_size + m;
                            output_ptr[st_id] = input_ptr[input_id];    
                            m++;
                        }
                    }
                    wd_id++;
                    out_rows_id = i * wd_num_each_channel + wd_id % wd_num_each_channel;
                }
            }
        }
    } 
}

TEST(TestSaberFunc, test_op_im2sequence) {

    int N = 2;
    int C = 2;
    int H = 3;
    int W = 3;

    int window_h = 2;
    int window_w = 2;
    int pad_up = 0;
    int pad_down = 0;
    int pad_left = 0;
    int pad_right = 0;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, Im2Sequence, Im2SequenceParam> testbase;
    Im2SequenceParam<NV> param_nv(window_h, window_w, pad_up, pad_down, pad_left, pad_right, \
                                    stride_h, stride_w, dilation_h, dilation_w);

    for(int w_in : {8, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    Shape shape({num_in, ch_in, h_in, w_in});
                    testbase.set_param(param_nv);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(im2sequence_cpu_base<float, NV, NVHX86>);
                }
            }
        }
    }

#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    TestSaberBase<AMD, AMDHX86, AK_FLOAT, Im2Sequence, Im2SequenceParam> testbase;
    Im2SequenceParam<AMD> param_amd(window_h, window_w, pad_up, pad_down, pad_left, pad_right, \
                                    stride_h, stride_w, dilation_h, dilation_w);

    for(int w_in : {8, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    Shape shape({num_in, ch_in, h_in, w_in});
                    testbase.set_param(param_amd);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(im2sequence_cpu_base<float, AMD, AMDHX86>);
                }
            }
        }
    }

#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, Im2Sequence, Im2SequenceParam> testbase_x86;
    Im2SequenceParam<X86> param_x86(window_h, window_w, pad_up, pad_down, pad_left, pad_right, \
                                    stride_h, stride_w, dilation_h, dilation_w);

    for(int w_in : {8, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    Shape shape({num_in, ch_in, h_in, w_in});
                    testbase_x86.set_param(param_x86);
                    testbase_x86.set_rand_limit(-5.0, 5.0);
                    testbase_x86.set_input_shape(shape);
                    testbase_x86.run_test(im2sequence_cpu_base<float, X86, X86>);
                }
            }
        }
    }
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
