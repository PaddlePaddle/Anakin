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
#include "saber/funcs/layer_norm.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>

using namespace anakin::saber;

/**
 * @brief normalize each layer.
 *         Given tensor with shape NCHW, set axis = 1, for example.
 *              inner_size = C * H * W (i=0,1,2,..., CHW-1); outer_size = N (j=0,1,2...N-1);
 *        compute: 
 *              x' = inner_size elements' mean, y' = inner_size elements' standard deviation.
 *              for each element x[i] in inner_size:
 *                   (x[i]-x') / y' .
 *              
 * 
 * @tparam dtype 
 * @tparam TargetType_D 
 * @tparam TargetType_H 
 * @param input 
 * @param output 
 * @param param 
 */
template <typename dtype,typename TargetType_D,typename TargetType_H>
void layerNorm_cpu_base(const std::vector<Tensor<TargetType_H>* > &input, std::vector<Tensor<TargetType_H>* > &output, LayerNormParam<TargetType_D> &param) {

    int inner_size = input[0]->count_valid(param.axis, input[0]->dims());
    int outer_size = input[0]->count_valid(0, param.axis);
    const dtype* src = (const dtype*)input[0]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();

    Tensor<TargetType_H> bias_h(param.bias_weights()->valid_shape());
    Tensor<TargetType_H> scale_h(param.scale_weights()->valid_shape());
    bias_h.copy_from(*param.bias_weights());
    scale_h.copy_from(*param.scale_weights());
    const dtype* bias = (const dtype*)bias_h.data();
    const dtype* scale = (const dtype*)scale_h.data();
    
    bool flag_bias = true;
    bool flag_scale = true;

    for (int i = 0; i < outer_size; ++i) {
        dtype mean = 0;
        dtype std = 0;
        const dtype* src_ptr = src + i * inner_size;
        dtype* dst_ptr = dst + i * inner_size;
        for (int j = 0; j < inner_size; ++j) {
            mean += src_ptr[j];
        }
        mean /= inner_size;
        for (int j = 0; j < inner_size; ++j) {
            std += (src_ptr[j] - mean) * (src_ptr[j] - mean);
        }
        std = std / inner_size;
        //printf("std pre: %.6f\n", std);
        std = 1.f / (sqrtf(std) + param.eps);
        //printf("mean: %.6f, std: %.6f\n", mean, std);
        for (int j = 0; j < inner_size; ++j) {
           dst_ptr[j] = (flag_scale? scale[j] : 1) * (src_ptr[j] - mean) * std + (flag_bias? bias[j] : 0);
        }
    }

}

TEST(TestSaberFunc, test_op_layer_norm) {

float eps = 1e-6f;
int axis = 1;

#ifdef AMD_GPU
    TestSaberBase<AMD, AMDHX86, AK_FLOAT, LayerNorm, LayerNormParam> testbase;
    for(int w_in : {8, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    Shape shape({num_in, ch_in, h_in, w_in});
                    int inner_size = shape.count(axis);
                    Shape bias_scale_shape({1, 1, 1, inner_size});
                    Tensor<AMD> bias(bias_scale_shape);
                    Tensor<AMD> scale(bias_scale_shape);
                    fill_tensor_rand(bias, -1.0f, 1.0f);
                    fill_tensor_rand(scale, -1.0f, 1.0f);
                    LayerNormParam<AMD> param(axis, eps, &bias, &scale);
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(layerNorm_cpu_base<float, AMD, AMDHX86>);
                }
            }
        }
    }
#endif

#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, LayerNorm, LayerNormParam> testbase;
    for(int w_in : {8, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    Shape shape({num_in, ch_in, h_in, w_in});
                    int inner_size = shape.count(axis);
                    Shape bias_scale_shape({1, 1, 1, inner_size});
                    Tensor<NV> bias(bias_scale_shape);
                    Tensor<NV> scale(bias_scale_shape);
                    fill_tensor_rand(bias, -1.0f, 1.0f);
                    fill_tensor_rand(scale, -1.0f, 1.0f);
                    LayerNormParam<NV> param(axis, eps, &bias, &scale);
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(layerNorm_cpu_base<float, NV, NVHX86>);
                }
            }
        }
    }
#endif 

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, LayerNorm, LayerNormParam> testbase_x86;
      for(int w_in : {8, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    Shape shape({num_in, ch_in, h_in, w_in});
                    int inner_size = shape.count(axis);
                    Shape bias_scale_shape({1, 1, 1, inner_size});
                    Tensor<X86> bias(bias_scale_shape);
                    Tensor<X86> scale(bias_scale_shape);
                    fill_tensor_rand(bias, -1.0f, 1.0f);
                    fill_tensor_rand(scale, -1.0f, 1.0f);
                    LayerNormParam<X86> param(axis, eps, &bias, &scale);
                    testbase_x86.set_param(param);
                    testbase_x86.set_rand_limit(-5.0, 5.0);
                    testbase_x86.set_input_shape(shape);
                    testbase_x86.run_test(layerNorm_cpu_base<float, X86, X86>);
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

#ifdef AMD_GPU
    Env<AMD>::env_init();
#endif
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
