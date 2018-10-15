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
#include "saber/funcs/roi_pooling.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>
#include <float.h>
#include <cmath>
using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void roi_pool_cpu(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,\
                RoiPoolParam<TargetType_D>& param){
    int roi_num = output[0]->num();
    int num_in = input[0]->num();
    int c_in = input[0]->channel();
    int h_in = input[0]->height();
    int w_in = input[0]->width();
    int pool_h = param.pooled_height;
    int pool_w = param.pooled_width;
    float spatial_scale = param.spatial_scale;
    const float* input_data = (const float*)input[0]->data();
    const float* roi = (const float*)input[1]->data();
    float* output_data = (float*)output[0]->mutable_data();
    int num_out = roi_num;
    int c_out = c_in;
    int h_out = pool_h;
    int w_out = pool_w;
    int in_stride_num = c_in * h_in * w_in;
    int in_stride_c = h_in * w_in;
    int in_stride_h = w_in;
    int in_stride_w = 1;
    int out_stride_num = c_out * h_out * w_out;
    int out_stride_c = h_out * w_out;
    int out_stride_h = w_out;
    int out_stride_w = 1;
    for(int n = 0; n < num_out; ++n){
        int in_index_n = roi[n * 5] * in_stride_num;
        int in_w_start = round(roi[n * 5 + 1] * spatial_scale);
        int in_h_start = round(roi[n * 5 + 2] * spatial_scale);
        int in_w_end = round(roi[n * 5 + 3] * spatial_scale);
        int in_h_end = round(roi[n * 5 + 4] * spatial_scale);
        float roi_rate_w = (float)(in_w_end - in_w_start + 1) / w_out;
        float roi_rate_h = (float)(in_h_end - in_h_start + 1) / h_out;
        for(int c = 0; c < c_out; ++c){
            int in_index = in_index_n + c * in_stride_c;
            for(int h = 0; h < h_out; ++h){
                for(int w = 0; w < w_out; ++w){
                    int w_start = floor(w * roi_rate_w) + in_w_start;
                    int h_start = floor(h * roi_rate_h) + in_h_start;
                    int w_end = ceil((w+1) * roi_rate_w) + in_w_start;
                    int h_end = ceil((h+1) * roi_rate_h) + in_h_start;
                    w_end = w_end > w_in ? w_in : w_end;
                    h_end = h_end > h_in ? h_in : h_end;
                    int out_index = n * out_stride_num + c * out_stride_c + h * out_stride_h + w * out_stride_w;
                    bool is_empty = (h_start >= h_end) || (w_start >= w_end);

                    float max = is_empty ? 0.0f : -FLT_MAX;
                    for(int j = h_start; j < h_end; ++j){
                        for(int i = w_start; i < w_end; ++i){
                            float data_in = input_data[in_index + i * in_stride_w + j * in_stride_h];
                            if(data_in > max)
                                max = data_in;
                        }
                    }
                    output_data[out_index] = max;
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_roi_pooling){

#ifdef AMD_GPU
    LOG(INFO)<<"AMD test......";
    TestSaberBase<AMD,AMDHX86,AK_FLOAT,RoiPool,RoiPoolParam> testbase(2);
    float spatial_scale = 2.0f;
    for(auto num_in :{1,3,10}){
        for(auto c_in:{1,3,32}){
            for(auto h_in:{9,16}){
                for(auto w_in:{9,16}){
                    for(auto roi_num:{1,3,6}){
                        for(auto pool_h:{1,2,4}){
                            for(auto pool_w:{1,2,4}){
                                Shape in_shape({num_in, c_in, h_in, w_in}, Layout_NCHW);
                                Shape roi_shape({roi_num, 5, 1, 1}, Layout_NCHW);
                                Tensor<X86> th_in, th_roi;
                                Tensor<AMD> td_in, td_roi;
                                th_in.re_alloc(in_shape, AK_FLOAT);
                                th_roi.re_alloc(roi_shape, AK_FLOAT);
                                td_in.re_alloc(in_shape, AK_FLOAT);
                                td_roi.re_alloc(roi_shape, AK_FLOAT);
                                // prepare host data
                                fill_tensor_rand(td_in, 0.0, 1.0);
                                // prepare roi data
                                float* roi_data = (float*)th_roi.mutable_data();
                                srand(time(0));
                                for(int i = 0; i < roi_num; ++i){
                                    roi_data[i * 5] = rand() % num_in;
                                    roi_data[i * 5 + 1] = floor(rand() % (w_in/2) / spatial_scale);
                                    roi_data[i * 5 + 2] = floor(rand() % (h_in/2) / spatial_scale);
                                    roi_data[i * 5 + 3] = floor((rand() % (w_in/2) + w_in/2) / spatial_scale);
                                    roi_data[i * 5 + 4] = floor((rand() % (h_in/2) + h_in/2) / spatial_scale);
                                }
                                th_in.copy_from(td_in);
                                td_roi.copy_from(th_roi);
                                std::vector<Tensor<AMD>*> input;
                                input.push_back(&td_in);
                                input.push_back(&td_roi);
                                testbase.add_custom_input(input);
                                RoiPoolParam<AMD> param(pool_h, pool_w, spatial_scale);
                                testbase.set_param(param);
                                testbase.run_test(roi_pool_cpu<float, AMD, AMDHX86>);
                            }
                        }
                    }
                }
            }
        }
    }
    LOG(INFO)<<"AMD test end.";
#endif


#ifdef USE_CUDA
    LOG(INFO)<<"NV test......";
    TestSaberBase<NV,NVHX86,AK_FLOAT,RoiPool,RoiPoolParam> testbase(2);
    float spatial_scale = 2.0f;
    for (auto num_in :{1,3,10}){
        for (auto c_in:{1,3,32}){
            for (auto h_in:{9,16}){
                for (auto w_in:{9,16}){
                    for (auto roi_num:{1,3,6}){
                        for (auto pool_h:{1,2,4}){
                            for (auto pool_w:{1,2,4}){
                                Shape in_shape({num_in, c_in, h_in, w_in}, Layout_NCHW);
                                Shape roi_shape({roi_num, 5, 1, 1}, Layout_NCHW);
                                Tensor<X86> th_in, th_roi;
                                Tensor<NV> td_in, td_roi;
                                th_in.re_alloc(in_shape, AK_FLOAT);
                                th_roi.re_alloc(roi_shape, AK_FLOAT);
                                td_in.re_alloc(in_shape, AK_FLOAT);
                                td_roi.re_alloc(roi_shape, AK_FLOAT);
                                // prepare host data
                                fill_tensor_rand(th_in, 0.0, 1.0);
                                // prepare roi data
                                float* roi_data = (float*)th_roi.mutable_data();
                                srand(time(0));
                                for(int i = 0; i < roi_num; ++i){
                                    roi_data[i * 5] = rand() % num_in;
                                    roi_data[i * 5 + 1] = floor(rand() % (w_in/2) / spatial_scale);
                                    roi_data[i * 5 + 2] = floor(rand() % (h_in/2) / spatial_scale);
                                    roi_data[i * 5 + 3] = floor((rand() % (w_in/2) + w_in/2) / spatial_scale);
                                    roi_data[i * 5 + 4] = floor((rand() % (h_in/2) + h_in/2) / spatial_scale);
                                }
                                td_in.copy_from(th_in);
                                td_roi.copy_from(th_roi);
                                std::vector<Tensor<NV>*> input;
                                input.push_back(&td_in);
                                input.push_back(&td_roi);
                                testbase.add_custom_input(input);
                                RoiPoolParam<NV> param(pool_h, pool_w, spatial_scale);
                                testbase.set_param(param);
                                testbase.run_test(roi_pool_cpu<float, NV, NVHX86>);
                            }
                        }
                    }
                }
            }
        }
    }
    LOG(INFO)<<"NV test end.";
#endif

#ifdef USE_X86_PLACE
    LOG(INFO)<<"x86 test......";
    do
    {
        TestSaberBase<X86,X86,AK_FLOAT,RoiPool,RoiPoolParam> testbase(2);
        float spatial_scale = 2.0f;
        for (auto num_in :{1,3,10}){
            for (auto c_in:{1,3,32}){
                for (auto h_in:{9,16}){
                    for (auto w_in:{9,16}){
                        for (auto roi_num:{1,3,6}){
                            for (auto pool_h:{1,2,4}){
                                for (auto pool_w:{1,2,4}){
                                    Shape in_shape({num_in, c_in, h_in, w_in}, Layout_NCHW);
                                    Shape roi_shape({roi_num, 5, 1, 1}, Layout_NCHW);
                                    Tensor<X86> th_in, th_roi;
                                    Tensor<X86> td_in, td_roi;
                                    th_in.re_alloc(in_shape, AK_FLOAT);
                                    th_roi.re_alloc(roi_shape, AK_FLOAT);
                                    td_in.re_alloc(in_shape, AK_FLOAT);
                                    td_roi.re_alloc(roi_shape, AK_FLOAT);
                                    // prepare host data
                                    fill_tensor_rand(th_in, 0.0, 1.0);
                                    // prepare roi data
                                    float* roi_data = (float*)th_roi.mutable_data();
                                    srand(time(0));
                                    for(int i = 0; i < roi_num; ++i){
                                        roi_data[i * 5] = rand() % num_in;
                                        roi_data[i * 5 + 1] = floor(rand() % (w_in/2) / spatial_scale);
                                        roi_data[i * 5 + 2] = floor(rand() % (h_in/2) / spatial_scale);
                                        roi_data[i * 5 + 3] = floor((rand() % (w_in/2) + w_in/2) / spatial_scale);
                                        roi_data[i * 5 + 4] = floor((rand() % (h_in/2) + h_in/2) / spatial_scale);
                                    }
                                    td_in.copy_from(th_in);
                                    td_roi.copy_from(th_roi);
                                    std::vector<Tensor<X86>*> input;
                                    input.push_back(&td_in);
                                    input.push_back(&td_roi);
                                    testbase.add_custom_input(input);
                                    RoiPoolParam<X86> param(pool_h, pool_w, spatial_scale);
                                    testbase.set_param(param);
                                    testbase.run_test(roi_pool_cpu<float, X86, X86>);
                                }
                            }
                        }
                    }
                }
            }
        }
    }while(0);
    LOG(INFO)<<"x86 test end.";
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
