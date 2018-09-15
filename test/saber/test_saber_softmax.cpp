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
#include "saber/funcs/softmax.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>
#include <cmath>
using namespace anakin::saber;

inline int Count(Shape sh, int start, int end) {
    int result = 1;

    for (int i = start; i < end; ++i) {
        result *= sh[i];
    }

    return result;
}

template <typename dtype, typename TargetType_D, typename TargetType_H>
void softmax_cpu(const std::vector<Tensor<TargetType_H>*>& input,
                 std::vector<Tensor<TargetType_H>*>& output, \
                 SoftmaxParam<TargetType_D>& param) {
    const dtype* in_data = (const dtype*)input[0]->data();
    dtype* out_data = (dtype*)output[0]->mutable_data();
    Shape sh_in = input[0]->valid_shape();
    Shape sh_out = output[0]->valid_shape();
    CHECK_EQ(sh_in == sh_out, true) << "input and output valid size must be same";
    int dims = input[0]->dims();
    int axis = param.axis;
    int axis_size = sh_in[axis];
    Shape in_stride = input[0]->get_stride();
    Shape out_stride = output[0]->get_stride();
    int inner_num = Count(sh_in, 0, axis);
    int outer_num = Count(sh_in, axis + 1, dims);
    int total_num = inner_num * outer_num;
    dtype* data = (dtype*)malloc(axis_size * sizeof(dtype));

    for (int num = 0; num < total_num; ++num) {
        int num_tmp = num;
        int in_index = 0, out_index = 0;

        for (int i = dims - 1; i >= 0; --i) {
            if (i == axis) {
                continue;
            }

            int pos = num_tmp % sh_in[i];
            in_index += pos * in_stride[i];
            out_index += pos * out_stride[i];
            num_tmp /= sh_in[i];
        }

        dtype max = std::numeric_limits<dtype>::lowest();

        for (int i = 0; i < axis_size; ++i) {
            max = in_data[in_index] > max ? in_data[in_index] : max;
            in_index += in_stride[axis];
        }

        dtype sum = (dtype)0;

        for (int i = 0; i < axis_size; ++i) {
            in_index -= in_stride[axis];
            data[axis_size - i - 1] = exp(in_data[in_index] - max);
            sum += data[axis_size - i - 1];
        }

        for (int i = 0; i < axis_size; ++i) {
            out_data[out_index] = data[i] / sum;
            out_index += out_stride[axis];
        }
    }

    free(data);

}

TEST(TestSaberFunc, test_func_softmax) {
#ifdef USE_CUDA
    LOG(INFO) << "NV test......";
    TestSaberBase<NV, NVHX86, AK_FLOAT, Softmax, SoftmaxParam> testbase;

    for (auto num : {
                1, 3, 4, 11
            }) {
        for (auto c : {
                    1, 3, 11, 4
                }) {
            for (auto h : {
                        3, 1, 11, 4
                    }) {
                for (auto w : {
                            1, 3, 4, 12
                        }) {
                    for (auto axis : {
                                0, 1, 2, 3
                            }) {
                        SoftmaxParam<NV> param(axis);
                        testbase.set_param(param);
                        testbase.set_input_shape(Shape({num, c, h, w}));
                        testbase.run_test(softmax_cpu<float, NV, NVHX86>);
                    }
                }
            }
        }
    }
    // softmax roi test will add later
    /*
        TestSaberBase<NV,NVHX86,AK_FLOAT,Softmax, SoftmaxParam> testbase1;
        for(auto num:{10,20,32}){
            for(auto c:{5,22,32}){
                for(auto h:{11,22,32}){
                    for(auto w:{11,22,32}){
                        for(auto axis:{0,1,2,3}){
                            Tensor<NV> bigtensor;
                            Tensor<NV> subtensor;
                            Shape sh({num, c, h, w});
                            Shape sh_roi({num/2, c/2, h/2, w/2});
                            Shape sh_offset({num/4, c/4, h/4, w/4});
                            bigtensor.re_alloc(sh, AK_FLOAT);
                            fill_tensor_rand(bigtensor);
                            subtensor.share_sub_buffer(bigtensor, sh_roi, sh_offset);
                            std::vector<Tensor<NV> *> input;
                            input.push_back(&subtensor);
                            testbase1.add_custom_input(input);
                            SoftmaxParam<NV> param(axis);
                            testbase1.set_param(param);
                            testbase1.run_test(softmax_cpu<float, NV, NVHX86>);
                        }
                    }
                }
            }
        }
    */
    LOG(INFO) << "NV test end.";
#endif

#ifdef USE_X86_PLACE
    LOG(INFO) << "x86 test......";
    TestSaberBase<X86, X86, AK_FLOAT, Softmax, SoftmaxParam> testbase2;

    for (auto num : {
                1, 3, 4, 12
            }) {
        for (auto c : {
                    1, 3, 11, 3
                }) {
            for (auto h : {
                        3, 1, 11, 2
                    }) {
                for (auto w : {
                            1, 3, 4, 11
                        }) {
                    for (auto axis : {
                                0, 1, 2, 3
                            }) {
                        SoftmaxParam<X86> param(axis);
                        testbase2.set_param(param);
                        testbase2.set_input_shape(Shape({num, c, h, w}));
                        testbase2.run_test(softmax_cpu<float, X86, X86>);
                    }
                }
            }
        }
    }
    LOG(INFO) << "x86 test end.";
#endif


#if 0
    Env<AMD>::env_init();
    TestSaberBase<AMD, AMDHX86, AK_FLOAT, Softmax, SoftmaxParam> testbase3;

    for (auto num : {
                1, 3, 4, 12
            }) {
        for (auto c : {
                    1, 3, 11, 3
                }) {
            for (auto h : {
                        3, 1, 11, 2
                    }) {
                for (auto w : {
                            1, 3, 4, 11
                        }) {
                    for (auto axis : {
                                0, 1, 2, 3
                            }) {
                        SoftmaxParam<AMD> param(axis);
                        testbase3.set_param(param);
                        testbase3.set_input_shape(Shape({num, c, h, w}));
                        testbase3.run_test(softmax_cpu<float, AMD, AMDHX86>, 0.0001, true);
                    }
                }
            }
        }
    }
#endif


#ifdef AMD_GPU
    LOG(INFO) << "AMD test......";
    TestSaberBase<AMD, AMDHX86, AK_FLOAT, Softmax, SoftmaxParam> testbase3;

    for (auto num : {
                1, 3, 4, 12
            }) {
        for (auto c : {
                    1, 3, 11, 3
                }) {
            for (auto h : {
                        3, 1, 11, 2
                    }) {
                for (auto w : {
                            1, 3, 4, 11
                        }) {
                    for (auto axis : {
                                0, 1, 2, 3
                            }) {
                        SoftmaxParam<AMD> param(axis);
                        testbase3.set_param(param);
                        testbase3.set_input_shape(Shape({num, c, h, w}));
                        testbase3.run_test(softmax_cpu<float, AMD, AMDHX86>);
                    }
                }
            }
        }
    }
    LOG(INFO) << "AMD test end.";
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
#ifdef AMD_GPU
    Env<AMD>::env_init();
#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
