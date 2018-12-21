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
#include "saber/funcs/lrn.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cmath>

using namespace anakin::saber;

/**
 * @brief get sum of x^2 between channels [size elements]
 *
 * @tparam dtype
 * @tparam TargetType_H
 * @param input
 * @param num_id:  the i-th graph.
 * @param channel_id: the j-th channel within i-th graph.
 * @param offset_within_channel: the pixel's offset within a channel.
 * @param offset_num: the first address of i-th graph.
 * @param c
 * @param h
 * @param w
 * @param size
 * @return dtype
 */
template <typename dtype, typename TargetType_H>
dtype lrn_square(const Tensor<TargetType_H>& input, int channel_id, int offset_within_channel,
                 int offset_num,
                 int c, int h, int w, int size) {

    int pre_pad = (size - 1) / 2;
    dtype res = 0;
    const dtype* src = (const dtype*)input.data() + offset_num;

    //handle left channels with padding situation.
    if (channel_id - pre_pad < 0) {
        for (int i = 0; i <= channel_id; i++) {
            res += src[i * h * w + offset_within_channel] * src[i * h * w + offset_within_channel];
        }
    }

    //handle left channels.
    if (channel_id - pre_pad >= 0) {
        for (int i = channel_id - pre_pad; i <= channel_id; i++) {
            res += src[i * h * w + offset_within_channel] * src[i * h * w + offset_within_channel];
        }
    }

    //handle right channels.
    if (channel_id + pre_pad < c) {
        for (int i = channel_id + 1; i <= channel_id + pre_pad; i++) {
            res += src[i * h * w + offset_within_channel] * src[i * h * w + offset_within_channel];
        }
    }

    //handle right channels with padding situation.
    if (channel_id + pre_pad >= c && channel_id + 1 < c) {
        for (int i = channel_id + 1; i < c; i++) {
            res += src[i * h * w + offset_within_channel] * src[i * h * w + offset_within_channel];
        }
    }

    return res;
}

/**
 * @brief   formula: (k + alpha * sigma((x(i))^2)) ^ beta.
 *              where,
 *                      local_size = 5(default), means 5 channels in succession.
 *                      sigma((x(i))^2): sum of x^2 of k channels in succession.
 *
 *
 * @tparam dtype
 * @tparam TargetType_D
 * @tparam TargetType_H
 * @param input
 * @param output
 * @param param
 */
template <typename dtype, typename TargetType_D, typename TargetType_H>
void lrn_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                  std::vector<Tensor<TargetType_H>* >& output, LrnParam<TargetType_D>& param) {

    int N = input[0]->num();
    int C = input[0]->channel();
    int H = input[0]->height();
    int W = input[0]->width();

    const dtype* src = (const dtype*)input[0]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();
    dtype square;
    int offset_within_channel = 0;
    int offset_num = 0;
    int dst_id;
    int size = param.local_size;
    int pre_pad = (size - 1) / 2;

    for (int i = 0; i < N; i++) {
        offset_num = i * C * H * W;

        for (int j = 0; j < C; j++) {
            for (int l = 0; l < H; l++) {
                for (int m = 0; m < W; m++) {
                    offset_within_channel = l * W + m;
                    dst_id = offset_num + j * H * W + offset_within_channel;
                    square = lrn_square<dtype, TargetType_H>(*input[0], j, offset_within_channel, offset_num, C, H, W,
                             size);
                    dst[dst_id] = src[dst_id] * pow(param.k + param.alpha * square, -param.beta);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_lrn) {

    int local_size = 5;
    float alpha = 1.0f;
    float beta = 0.75;
    float k = 1.0f;
    NormRegion norm_region = ACROSS_CHANNELS;

#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, Lrn, LrnParam> testbase;

    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {2, 3, 8, 64}) {
                for (int num_in : {1, 21, 32}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    LrnParam<NV> param(local_size, alpha, beta, k, norm_region);
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(lrn_cpu_base<float, NV, NVHX86>, 2.1e-5f);
                }
            }
        }
    }
#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, Lrn, LrnParam> testbase_x86;

    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {2, 3, 8, 64}) {
                for (int num_in : {1, 21, 32}) {
                    Shape shape_x86({num_in, ch_in, h_in, w_in});
                    LrnParam<X86> param_x86(local_size, alpha, beta, k, norm_region);
                    testbase_x86.set_param(param_x86);
                    testbase_x86.set_rand_limit(-5.0, 5.0);
                    testbase_x86.set_input_shape(shape_x86);
                    testbase_x86.run_test(lrn_cpu_base<float, X86, X86>);
                }
            }
        }
    }
#endif

#ifdef AMD_GPU
    Env<AMD>::env_init();
    TestSaberBase<AMD, AMDHX86, AK_FLOAT, Lrn, LrnParam> testbase;
    
    for(int w_in : {8, 8, 16}){
        for(int h_in : {2, 8, 32}){
	    for(int ch_in : {2, 3, 8, 64}){
	        for(int num_in : {1, 21, 32}){
		    Shape shape({num_in, ch_in, h_in, w_in});
		    LrnParam<AMD> param(local_size, alpha, beta, k, norm_region);
		    testbase.set_param(param);
		    testbase.set_rand_limit(-0.5, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(lrn_cpu_base<float, AMD, AMDHX86>, 2.1e-5f);
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
