/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

#include "saber/core/context.h"
#include "saber/funcs/mat_mul.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>

using namespace anakin::saber;

/**
 * @brief matrix transpose.
 *
 * @tparam dtype : data type.
 * @tparam TargetType : device type.
 * @param data: input/output
 */
template<typename dtype, typename TargetType>
void batch_transpos(const Tensor<TargetType>& data, Tensor<TargetType>& trans_data) {
    int M = data.height();
    int N = data.width();
    int B = data.num() * data.channel();
    const dtype* data_ptr = (const dtype*)data.data();
    dtype* trans_data_ptr = (dtype*)trans_data.mutable_data();

    for (int b = 0; b < B; b++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                trans_data_ptr[b * M * N + n * M + m] = data_ptr[b * M * N + m * N + n];
            }
        }
    }

    trans_data.set_height(N);
    trans_data.set_width(M);
}


/**
 * @brief mat_mul compute (native cpu version)
 *
 * @tparam dtype
 * @tparam TargetType_D
 * @tparam TargetType_H
 * @param input: input[0] with size M*K, input[1] with N*K
 * @param output: output = input[0] * input[1].
 * @param param
 */
template <typename dtype, typename TargetType_D, typename TargetType_H>
void mat_mul_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                      std::vector<Tensor<TargetType_H>* >& output, MatMulParam<TargetType_D>& param) {

    int M, N, K, B;
    //bath size
    B = input[0]->num() * input[0]->channel();

    Tensor<TargetType_H> trans_input0(input[0]->valid_shape());
    trans_input0.copy_from(*input[0]);
    Tensor<TargetType_H> trans_input1(input[1]->valid_shape());
    trans_input1.copy_from(*input[1]);

    //whether input[0] trans.
    if (param._is_transpose_X) {
        batch_transpos<dtype, TargetType_H>(*input[0], trans_input0);
    }

    //whether input[1] trans
    if (param._is_transpose_Y) {
        batch_transpos<dtype, TargetType_H>(*input[1], trans_input1);
    }

    CHECK_EQ(trans_input0.width(), trans_input1.height()) << "can't do matrix multiplay";

    M = trans_input0.height();
    N = trans_input1.width();
    K = trans_input1.height();

    dtype* out_ptr = (dtype*)output[0]->mutable_data();
    const dtype* in_ptr0 = (const dtype*)trans_input0.data();
    const dtype* in_ptr1 = (const dtype*)trans_input1.data();

    for (int b = 0; b < B; b++) {
        float* optr = out_ptr + b * M * N;
        const float* iptr0 = in_ptr0 + b * M * K;
        const float* iptr1 = in_ptr1 + b * K * N;

        for (int i = 0; i < M; ++i) {
            float* pdout = optr + i * N;

            for (int j = 0; j < N; ++j) {
                pdout[j] = 0;

                for (int l = 0; l < K; ++l) {
                    pdout[j] += iptr0[i * K + l] * iptr1[l * N + j];
                }
            }
        }
    }

}
TEST(TestSaberFunc, test_op_mat_mul) {

    int input_num = 2;
    bool trans_A = true;
    bool trans_B = false;

#ifdef USE_CUDA
    //2 inputs
    TestSaberBase<NV, NVHX86, AK_FLOAT, MatMul, MatMulParam> testbase(input_num);

    for (int w_in : {
                2, 8, 16
            }) {
        for (int h_in : {
                    2, 8, 32
                }) {
            for (int ch_in : {
                        2, 3, 8, 64
                    }) {
                for (int num_in : {
                            1, 21, 32
                        }) {
                    Shape shape0({num_in, ch_in, w_in, h_in});
                    Shape shape1({num_in, ch_in, w_in, h_in});
                    std::vector<Shape> shapes;
                    shapes.push_back(shape0);
                    shapes.push_back(shape1);
                    MatMulParam<NV> param(trans_A, trans_B);
                    testbase.set_param(param);
                    testbase.set_rand_limit(1, 12);
                    testbase.set_input_shape(shapes);
                    testbase.run_test(mat_mul_cpu_base<float, NV, NVHX86>);
                }
            }
        }
    }

#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, MatMul, MatMulParam> testbase_x86(input_num);

    for (int w_in : {
                2, 8, 16
            }) {
        for (int h_in : {
                    2, 8, 32
                }) {
            for (int ch_in : {
                        2, 3, 8, 64
                    }) {
                for (int num_in : {
                            1, 21, 32
                        }) {
                    Shape shape0({num_in, ch_in, w_in, h_in});
                    Shape shape1({num_in, ch_in, w_in, h_in});
                    std::vector<Shape> shapes;
                    shapes.push_back(shape0);
                    shapes.push_back(shape1);
                    MatMulParam<X86> param(trans_A, trans_B);
                    testbase_x86.set_param(param);
                    testbase_x86.set_rand_limit(1, 12);
                    testbase_x86.set_input_shape(shapes);
                    testbase_x86.run_test(mat_mul_cpu_base<float, X86, X86>);
                }
            }
        }
    }

#endif

#ifdef AMD_GPU
    Env<AMD>::env_init();
    TestSaberBase<AMD, AMDHX86, AK_FLOAT, MatMul, MatMulParam> testbase(input_num);

        for(int w_in : {2, 8, 16}) {
            for(int h_in : {2, 8, 32}){
                for(int ch_in : {2, 3, 8, 64}){
                    for(int num_in:{1, 21, 32}){
                        Shape shape0({num_in, ch_in, w_in, h_in});
                        Shape shape1({num_in, ch_in, w_in, h_in});
                        std::vector<Shape> shapes;
                        shapes.push_back(shape0);
                        shapes.push_back(shape1);
                        MatMulParam<AMD> param(trans_A, trans_B);
                        testbase.set_param(param);
                        testbase.set_rand_limit(1, 12);
                        testbase.set_input_shape(shapes);
                        testbase.run_test(mat_mul_cpu_base<float, AMD, AMDHX86>);
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
