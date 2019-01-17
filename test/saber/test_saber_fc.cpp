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
#include "saber/funcs/fc.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <ctime>

using namespace anakin::saber;


//fc compute (native cpu version)
template <typename dtype,typename TargetType_D,typename TargetType_H>
void fc_cpu_base(const std::vector<Tensor<TargetType_H>* > &input, std::vector<Tensor<TargetType_H>* > &output, FcParam<TargetType_D> &param) {

    const dtype *data_in = (const dtype*)input[0]->data();
    const dtype *bias = param.bias ? (const dtype*)param.bias->data() : nullptr;

    Tensor<TargetType_H> weights_h(param.weights->valid_shape());
    weights_h.copy_from(*param.weights);

    const dtype *weights = (const dtype*)weights_h.data();
    dtype *data_out = (dtype*)output[0]->mutable_data();

    //is_trans: flase.
    //output: data_out; inputs: data_in ; weights: weights.
    //data_out = data_in * weights. Get weights' elements continuosly.
    int out_rows = input[0]->num();
    int in_cols = input[0]->valid_size() / out_rows;
    int out_cols = param.weights->valid_size() / in_cols;
    int index_out;
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            index_out = i * out_cols + j;
            data_out[index_out] = bias ? bias[j] : 0;
            for (int k = 0; k < in_cols; k++) {
                //data_out[index_out] += data_in[i * in_cols + k] * weights[k * out_cols + j];
                data_out[index_out] += data_in[i * in_cols + k] * weights[j * in_cols + k];
            }
        }
    }
}


TEST(TestSaberFunc, test_op_fc) {

#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, Fc, FcParam> testbase;

    Tensor<NVHX86> weights_h;
    Tensor<NV> weights_d;

    //Shape shape_weight({})
    for(int w_in : {2, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    int out_num = w_in * 2;
                    Shape shape({num_in, ch_in, h_in, w_in});
                    Shape shape_w({ch_in, h_in, w_in, out_num});
                    weights_h.re_alloc(shape_w, AK_FLOAT);
                    weights_d.re_alloc(shape_w, AK_FLOAT);
                    fill_tensor_rand(weights_h, 0.1, 1.5);
                    weights_d.copy_from(weights_h);
                    FcParam<NV> param(&weights_d, out_num);
                    testbase.set_param(param);
                    testbase.set_rand_limit(1, 12);
                    testbase.set_input_shape(shape);
                    testbase.run_test(fc_cpu_base<float, NV, NVHX86>, 2.1e-5f);
                }
            }
        }
    }

#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, Fc, FcParam> testbase0;

    Tensor<X86> weights_h0;

    //Shape shape_weight({})
    for(int w_in : {2, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    int out_num = w_in * 2;
                    Shape shape({num_in, ch_in, h_in, w_in});
                    Shape shape_w({ch_in, h_in, w_in, out_num});
                    weights_h0.re_alloc(shape_w, AK_FLOAT);
                    fill_tensor_rand(weights_h0, 0.1, 1.5);
                    FcParam<X86> param(&weights_h0, out_num);
                    testbase0.set_param(param);
                    testbase0.set_rand_limit(1, 12);
                    testbase0.set_input_shape(shape);
                    testbase0.run_test(fc_cpu_base<float, X86, X86>, 2.1e-5f);
                }
            }
        }
    }
#endif

#ifdef AMD_GPU
    TestSaberBase<AMD, AMDHX86, AK_FLOAT, Fc, FcParam> testbase;

    Tensor<AMDHX86> weights_h;
    Tensor<AMD> weights_d;
    Env<AMD>::env_init();
    //Shape shape_weight({})
    for(int w_in : {2, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    int out_num = w_in * 2;
                    Shape shape({num_in, ch_in, h_in, w_in});
                    Shape shape_w({ch_in, h_in, w_in, out_num});
                    weights_h.re_alloc(shape_w, AK_FLOAT);
                    weights_d.re_alloc(shape_w, AK_FLOAT);
                    fill_tensor_rand(weights_h, 0.1, 1.5);
                    weights_d.copy_from(weights_h);
                    FcParam<AMD> param(&weights_d, out_num);
                    testbase.set_param(param);
                    testbase.set_rand_limit(1, 12);
                    testbase.set_input_shape(shape);
                    testbase.run_test(fc_cpu_base<float, AMD, AMDHX86>, 2.1e-5f);
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
