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
 *     
 */

#include "saber/core/context.h"
#include "saber/funcs/embedding.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_base.h"
#include "test_saber_func.h"
#include <vector>

using namespace anakin::saber;



//native cpu version
template <typename dtype, typename TargetType_D, typename TargetType_H>
void embedding_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                        std::vector<Tensor<TargetType_H>* >& output, EmbeddingParam<TargetType_D>& param) {

    const float* in_data = (const float*)input[0]->data();
    int num = input[0]->valid_size();
    float* out_data = (float*)output[0]->mutable_data();
    //host weight
    Tensor<TargetType_H> weight_h(param.weight()->valid_shape());
    weight_h.copy_from(*param.weight());

    for (int i = 0; i < num; i++) {
        if (in_data[i] == param.padding_idx) {
            memset(out_data + i * param.emb_dim, 0, sizeof(float) * param.emb_dim);
        } else {
            CHECK_GE(int(in_data[i]), 0);
            CHECK_LT(int(in_data[i]), param.word_num);
            memcpy(out_data + i * param.emb_dim, \
                   (const float*)weight_h.data() + int(in_data[i]) * param.emb_dim, \
                   sizeof(float) * param.emb_dim);
        }
    }
}


TEST(TestSaberFunc, test_op_embedding) {


    //set param.
    int word_num = 128;
    int emb_dim = 10;
    int padding_idx = -1;

    Shape weights_s({1, 1, word_num, emb_dim});

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, Embedding, EmbeddingParam> testbase;
    Tensor<X86> weight_h(weights_s);
    fill_tensor_rand(weight_h, -0.5, 0.5);

    EmbeddingParam<X86> param(word_num, emb_dim, padding_idx, &weight_h);
    testbase.set_param(param);

    //test for nchw
    for (int w_in : {
                32, 64
            }) {
        for (int h_in : {
                    32, 64
                }) {
            for (int ch_in : {
                        3, 8
                    }) {
                for (int num_in : {
                            1, 2
                        }) {
                    testbase.set_rand_limit(1, 128);
                    testbase.set_input_shape(Shape({num_in, ch_in, h_in, w_in}));
                    testbase.run_test(embedding_cpu_base<float, X86, X86>);//run test
                }
            }
        }
    }

    //test for nc
    for (int ch_in : {
                3, 8, 16, 64
            }) {
        for (int num_in : {
                    1, 2, 32, 64
                }) {
            testbase.set_rand_limit(1, 128);
            testbase.set_input_shape(Shape({num_in, ch_in}, Layout_HW));
            testbase.run_test(embedding_cpu_base<float, X86, X86>);//run test
        }
    }

#endif
#ifdef AMD_GPU
    TestSaberBase<AMD, AMDHX86, AK_FLOAT, Embedding, EmbeddingParam> testbase0;
    Tensor<AMDHX86> weight_h0(weights_s);
    Tensor<AMD> weight_d0(weights_s);
    fill_tensor_rand(weight_h0, -0.5, 0.5);
    weight_d0.copy_from(weight_h0);

    EmbeddingParam<AMD> param0(word_num, emb_dim, padding_idx, &weight_d0);
    testbase0.set_param(param0);


    //test for nchw
    for (int w_in : {
                32, 64
            }) {
        for (int h_in : {
                    32, 64
                }) {
            for (int ch_in : {
                        3, 8
                    }) {
                for (int num_in : {
                            1, 2
                        }) {
                    testbase0.set_rand_limit(1, 128); //random interval [1, 128].
                    testbase0.set_input_shape(Shape({num_in, ch_in, h_in, w_in}));
                    testbase0.run_test(embedding_cpu_base<float, AMD, AMDHX86>);//run test
                }
            }
        }
    }

    //test for nc
    for (int ch_in : {
                3, 8, 16, 64
            }) {
        for (int num_in : {
                    1, 2, 32, 64
                }) {
            testbase0.set_rand_limit(1, 128);
            testbase0.set_input_shape(Shape({num_in, ch_in}, Layout_HW));
            testbase0.run_test(embedding_cpu_base<float, AMD, AMDHX86>);//run test
        }
    }

#endif


#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, Embedding, EmbeddingParam> testbase0;
    Tensor<NVHX86> weight_h0(weights_s);
    Tensor<NV> weight_d0(weights_s);
    fill_tensor_rand(weight_h0, -0.5, 0.5);
    weight_d0.copy_from(weight_h0);

    EmbeddingParam<NV> param0(word_num, emb_dim, padding_idx, &weight_d0);
    testbase0.set_param(param0);


    //test for nchw
    for (int w_in : {
                32, 64
            }) {
        for (int h_in : {
                    32, 64
                }) {
            for (int ch_in : {
                        3, 8
                    }) {
                for (int num_in : {
                            1, 2
                        }) {
                    testbase0.set_rand_limit(1, 128); //random interval [1, 128].
                    testbase0.set_input_shape(Shape({num_in, ch_in, h_in, w_in}));
                    testbase0.run_test(embedding_cpu_base<float, NV, NVHX86>);//run test
                }
            }
        }
    }

    //test for nc
    for (int ch_in : {
                3, 8, 16, 64
            }) {
        for (int num_in : {
                    1, 2, 32, 64
                }) {
            testbase0.set_rand_limit(1, 128);
            testbase0.set_input_shape(Shape({num_in, ch_in}, Layout_HW));
            testbase0.run_test(embedding_cpu_base<float, NV, NVHX86>);//run test
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

