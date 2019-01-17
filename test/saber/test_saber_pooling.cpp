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

#include <vector>
#include <limits>

#include "saber/core/context.h"
#include "test/saber/test_saber_base.h"
#include "test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/pooling.h"
#include "saber/core/data_traits.h"

using namespace anakin::saber;

template<typename dtype, typename TargetType_D, typename TargetType_H>
void pooling_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
                      std::vector<Tensor<TargetType_H>*>& output, PoolingParam<TargetType_D>& param) {
    const dtype* src_ptr = static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());

    int in_n = input[0]->num();
    int in_c = input[0]->channel();
    int in_h = input[0]->height();
    int in_w = input[0]->width();
    int size_in_n = in_c * in_h * in_w;
    int size_in_c = in_h * in_w;

    int out_h = output[0]->height();
    int out_w = output[0]->width();
    LOG(INFO) << "out = " << out_h << "," << out_w;
    int size_out_n = in_c * out_h * out_w;
    int size_out_c = out_h * out_w;

    for (int ind_n = 0; ind_n < in_n; ++ind_n) {
        for (int ind_c = 0; ind_c < in_c; ++ind_c) {
            for (int ind_h = 0; ind_h < out_h; ++ind_h) {
                int sh = ind_h * param.stride_h;
                int eh = sh + param.window_h;
                sh = (sh - param.pad_h) < 0 ? 0 : sh - param.pad_h;
                eh = (eh - param.pad_h) > in_h ? in_h : eh - param.pad_h;

                for (int ind_w = 0; ind_w < out_w; ++ind_w) {
                    int sw = ind_w * param.stride_w;
                    int ew = sw + param.window_w;
                    sw = (sw - param.pad_w) < 0 ? 0 : sw - param.pad_w;
                    ew = (ew - param.pad_w) > in_w ? in_w : ew - param.pad_w;


                    dtype result= static_cast<dtype>(0);

                    int dst_ind = ind_n * size_out_n + ind_c * size_out_c + ind_h * out_w + ind_w;

                    for (int kh = sh; kh < eh; ++kh) {
                        for (int kw = sw; kw < ew; ++kw) {
                            int src_ind = ind_n * size_in_n + ind_c * size_in_c + kh * in_w + kw;

                            if (kh == sh && kw == sw) {
                                result = src_ptr[src_ind];
                            } else {
                                if (param.pooling_type == Pooling_max) {
                                    result = result >= src_ptr[src_ind] ? result : src_ptr[src_ind];
                                }

                                if (param.pooling_type == Pooling_average_include_padding) {
                                    result += src_ptr[src_ind];
                                }

                                if (param.pooling_type == Pooling_average_exclude_padding) {
                                    result += src_ptr[src_ind];
                                }
                            }

                        }
                    }

                    if (param.pooling_type == Pooling_average_include_padding) {
                        //result /= param.window_h * param.window_w;
                        //LOG(ERROR)<<"cpu"<<param.window_h * param.window_w;
                            int bh = param.window_h;
                            int bw = param.window_w;
                            if (ew == in_w)
                            {
                                bw = sw + param.window_w >= in_w + param.pad_w ? in_w + param.pad_w : sw + param.window_w;
                                bw -= sw;
                            }
                            if (eh == in_h)
                            {
                                bh = sh + param.window_h >= in_h + param.pad_h ? in_h + param.pad_h: sh + param.window_h;
                                bh -=sh;
                            }
                            result /= bh * bw;
                    }

                    if (param.pooling_type == Pooling_average_exclude_padding) {
                        result /= (ew - sw) * (eh - sh);
                    }

                    dst_ptr[dst_ind] = result;

                }
            }
        }

    }
}

template<typename TargetType, typename TargetType_H>
int test_pooling_results(int window_h,int window_w,int pad_h,int pad_w,PoolingType pooling_type,int stride_h,int stride_w,
                         int in_n,int in_c,int in_h,int in_w) {

    Env<TargetType_H>::env_init();
    Shape input_s({in_n, in_c, in_h, in_w}, Layout_NCHW);
    Shape input_nchwc8({in_n, in_c,in_h,in_w}, Layout_NCHW_C8R);
    int out_h = static_cast<int>((static_cast<float>(
                                           in_h + 2 * pad_h - window_h) / stride_h)) + 1;

    int out_w = static_cast<int>((static_cast<float>(
                                          in_w + 2 * pad_w - window_w) / stride_w)) + 1;
    Shape output_s({in_n, in_c, out_h, out_w}, Layout_NCHW);
    Shape output_nchwc8({in_n, in_c, out_h, out_w}, Layout_NCHW_C8R);
    // init input Tensor
    Tensor<TargetType> input_dev(input_nchwc8);
    Tensor<TargetType_H> input_host(input_nchwc8);
    fill_tensor_rand(input_dev, -10.0f, 10.0f);
    input_host.copy_from(input_dev);

    Tensor<TargetType> output_dev(output_nchwc8);
    Tensor<TargetType_H> output_host(output_nchwc8);
    Tensor<TargetType_H> check_host;

    Context<TargetType> ctx1(0, 1, 1);
//    ActivationParam<TargetType> act_param(Active_relu);
    PoolingParam<TargetType> param(window_h,window_w,pad_h,pad_w,stride_h,stride_w,pooling_type);

    Pooling<TargetType, AK_FLOAT> pooling;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
    pooling.compute_output_shape(input_v, output_v, param);
//    output_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);

    pooling.init(input_v, output_v, param, SPECIFY, SABER_IMPL, ctx1);
    pooling(input_v, output_v, param, ctx1);

    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    output_host.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    output_host.copy_from(output_dev);

    Tensor<TargetType_H> input_check(input_s);
    Tensor<TargetType_H> output_check(output_s);
    Tensor<TargetType_H> output_check_from_dev(output_s);
    reorder_nchwc8_nchw(input_host,input_check);
    reorder_nchwc8_nchw(output_dev,output_check_from_dev);
    std::vector<Tensor<TargetType_H>* > input_v_h;
    std::vector<Tensor<TargetType_H>* > output_v_h;
    input_v_h.push_back(&input_check);
    output_v_h.push_back(&output_check);
    pooling_cpu_func<float>(input_v_h,output_v_h,param);

//    print_tensor_valid(check_host);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)output_check.data(), (const float*)output_check_from_dev.data(),
                    check_host.valid_size(), max_ratio, max_diff);
//    print_tensor(input_check);
//    print_tensor(output_check);
//    print_tensor(output_dev);
    if (max_ratio > 1e-3) {
        print_tensor(output_check);
        print_tensor_valid(output_check_from_dev);
        LOG(FATAL) << " max_ratio = " << max_ratio << " max_diff = " << max_diff;
    }else{
        LOG(INFO)<<"passed";
    }
    return 0;
}


//test template for different device and dtype
template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_pooling() {
    typedef typename DataTrait<TargetType_D, OpDtype> :: Dtype dtype;
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, Pooling, PoolingParam> testbase;

    for (int window_h : {2, 3, 5, 7}) {
        for (int window_w : {2, 3, 5, 7}) {
            for (int pad_h : {1, 2}) {
                for (int pad_w : {1, 2}) {
                    if (pad_h >= window_h || pad_w >= window_w){
                        continue;
                    }
                    for (PoolingType pooling_type : {Pooling_max, Pooling_average_include_padding, Pooling_average_exclude_padding}) {
                        for (int stride_h : {1, 2 }) {
                            for (int stride_w : {1, 2}) {
                                PoolingParam<TargetType_D> param(window_h, window_w, pad_h, pad_w, stride_h, stride_w,
                                                                 pooling_type);
                                LOG(INFO) << "win_h:" << window_h << "win_w:" << window_w \
                                          << "pad_h:" << pad_h << "pad_w:" << pad_w \
                                          << "stride_h:" << stride_h << "stride_w:" << stride_w \
                                          << "pooling_type:" << pooling_type;

                                for (int in_n : {1, 2}) {
                                    for (int in_c : {1, 3}) {
                                        for (int in_h : {7, 8, 13, 28, 32, 64}) {
                                            for (int in_w : {7, 8, 13, 28, 32, 64}) {
                                                LOG(INFO) << "n:" << in_n << ",in_c:" << in_c << ",in_h:" << in_h << ",in_w:" << in_w;
                                                testbase.set_param(param);//set param
                                                testbase.set_input_shape(Shape({in_n, in_c, in_h, in_w})); //add some input shape
                                                testbase.run_test(pooling_cpu_func<dtype, TargetType_D, TargetType_H>, 0.0001);//run test

                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

TEST(TestSaberFunc, test_func_pool) {
#ifdef USE_CUDA
    test_pooling<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
//    test_pooling<X86, X86, AK_FLOAT>();
    int window_h=2;
    int window_w=2;
    int pad_h=1;
    int pad_w=1;
    PoolingType pooling_type=Pooling_max;
    int stride_h=2;
    int stride_w=2;
    int in_n=1;
    int in_c=64;
    int in_h=96;
    int in_w=64;
    test_pooling_results<X86,X86>( window_h, window_w, pad_h, pad_w, pooling_type, stride_h, stride_w,
             in_n, in_c, in_h, in_w);
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_pooling<AMD, AMDHX86, AK_FLOAT>();
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
