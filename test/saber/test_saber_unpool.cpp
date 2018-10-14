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
#include "saber/funcs/unpool.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>

using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void unpool_cpu(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,\
                PoolingParam<TargetType_D>& param){
    const dtype* in_data = (const dtype*)input[0]->data();
    const dtype* max_data = (const dtype*)input[1]->data();
    dtype* out_data = (dtype*)output[0]->mutable_data();
    memset(out_data, 0, output[0]->valid_size() * sizeof(dtype));
    int num_in = input[0]->num();
    int c_in = input[0]->channel();
    int num_stride_in = input[0]->get_stride()[input[0]->num_index()];
    int c_stride_in = input[0]->get_stride()[input[0]->channel_index()];
    int num_stride_out = output[0]->get_stride()[output[0]->num_index()];
    int c_stride_out = output[0]->get_stride()[output[0]->channel_index()];
    int total_count = input[0]->valid_size();
    for(int i = 0; i < total_count; ++i){
        int num_out = i / num_stride_in;
        int c_out = (i / c_stride_in) % c_in;
        int out_index = num_out * num_stride_out + c_out * c_stride_out;
        int max_index = max_data[i];
        out_data[out_index + max_index] = in_data[i];
    }
   
}
template<typename TargetType>
void create_pooling_index(float* pool_index, int n, int c, int h, int w, PoolingParam<TargetType> param){
    int out_w = (w - 1) * param.stride_w + param.window_w - 2 * param.pad_w;
    int count  = 0;
    for (int n_id = 0; n_id < n; n_id++) {
        for (int c_id = 0; c_id < c; c_id++) {
            for (int h_id = 0; h_id < h; h_id++) {
                for (int w_id = 0; w_id < w; w_id++) {
                    pool_index[count] = (h_id * param.stride_h + rand() % param.window_h) * out_w +
                                   (w_id * param.stride_w + rand() % param.window_w);
                    count++;
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_resize){

#ifdef AMD_GPU 
    LOG(INFO)<<"AMD test......";
    //Init the test_base
    TestSaberBase<AMD,AMDHX86,AK_FLOAT,Unpool, PoolingParam> testbase(2);
    for(int num_in:{1,3,32}){
        for(int c_in:{1,3,12}){
            for(int h_in:{2,3,25}){
                for(int w_in:{2,3,32}){

                    PoolingParam<AMD> param(3, 3, 0, 0, 3, 3,Pooling_max, false);
                    Tensor<AMD> td, td_pool_index;
                    Tensor<X86> th_pool_index;
                    Shape sh_in({num_in, c_in, h_in, w_in});
                    td.re_alloc(sh_in, AK_FLOAT);
                    th_pool_index.re_alloc(sh_in, AK_FLOAT);
                    td_pool_index.re_alloc(sh_in, AK_FLOAT);
                    fill_tensor_rand(td, -1.0, 1.0);
                    float* pool_index_data = (float*)th_pool_index.mutable_data();
                    create_pooling_index(pool_index_data, num_in, c_in, h_in, w_in, param);
                    td_pool_index.copy_from(th_pool_index);
                    std::vector<Tensor<AMD>*> input;
                    input.push_back(&td);
                    input.push_back(&td_pool_index);
                    testbase.add_custom_input(input);
                    testbase.set_param(param);
                    testbase.run_test(unpool_cpu<float, AMD, AMDHX86>);
                }
            }
        }
    }  
#endif


#ifdef USE_CUDA
    LOG(INFO)<<"NV test......";
    //Init the test_base
    TestSaberBase<NV,NVHX86,AK_FLOAT,Unpool, PoolingParam> testbase(2);
    for(int num_in:{1,3,32}){
        for(int c_in:{1,3,12}){
            for(int h_in:{2,3,25}){
                for(int w_in:{2,3,32}){

                    PoolingParam<NV> param(3, 3, 0, 0, 3, 3,Pooling_max, false);
                    Tensor<NV> td, td_pool_index;
                    Tensor<X86> th_pool_index;
                    Shape sh_in({num_in, c_in, h_in, w_in});
                    td.re_alloc(sh_in, AK_FLOAT);
                    th_pool_index.re_alloc(sh_in, AK_FLOAT);
                    td_pool_index.re_alloc(sh_in, AK_FLOAT);
                    fill_tensor_rand(td, -1.0, 1.0);
                    float* pool_index_data = (float*)th_pool_index.mutable_data();
                    create_pooling_index(pool_index_data, num_in, c_in, h_in, w_in, param);
                    td_pool_index.copy_from(th_pool_index);
                    std::vector<Tensor<NV>*> input;
                    input.push_back(&td);
                    input.push_back(&td_pool_index);
                    testbase.add_custom_input(input);
                    testbase.set_param(param);
                    testbase.run_test(unpool_cpu<float, NV, NVHX86>);
                }
            }
        }
    }
#endif

#ifdef USE_X86_PLACE
    LOG(INFO)<<"x86 test......";
    //Init the test_base
    do
    {
        TestSaberBase<X86,X86,AK_FLOAT,Unpool, PoolingParam> testbase(2);
        for(int num_in:{1,3,32}){
            for(int c_in:{1,3,12}){
                for(int h_in:{2,3,25}){
                    for(int w_in:{2,3,32}){

                        PoolingParam<X86> param(3, 3, 0, 0, 3, 3,Pooling_max, false);
                        Tensor<X86> td, td_pool_index;
                        Tensor<X86> th_pool_index;
                        Shape sh_in({num_in, c_in, h_in, w_in});
                        td.re_alloc(sh_in, AK_FLOAT);
                        th_pool_index.re_alloc(sh_in, AK_FLOAT);
                        td_pool_index.re_alloc(sh_in, AK_FLOAT);
                        fill_tensor_rand(td, -1.0, 1.0);
                        float* pool_index_data = (float*)th_pool_index.mutable_data();
                        create_pooling_index(pool_index_data, num_in, c_in, h_in, w_in, param);
                        td_pool_index.copy_from(th_pool_index);
                        std::vector<Tensor<X86>*> input;
                        input.push_back(&td);
                        input.push_back(&td_pool_index);
                        testbase.add_custom_input(input);
                        testbase.set_param(param);
                        testbase.run_test(unpool_cpu<float, X86, X86>);
                    }
                }
            }
        }
    }while(0);
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
