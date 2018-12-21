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
#include "saber/funcs/permute_power.h"

#include "saber/core/context.h"
#include "test/saber/test_saber_func.h"
#include "test/saber/test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/core/data_traits.h"

#include <vector>

using namespace anakin::saber;
template <typename dtype, typename TargetType_D, typename TargetType_H>
void permute_power_cpu_func(const std::vector<Tensor<TargetType_H>*>& input, std::vector<Tensor<TargetType_H>*>& output, PermutePowerParam<TargetType_D>& param) {
    const dtype* src_ptr = static_cast<const dtype*>(input[0] -> data());
    dtype* dst_ptr = static_cast<dtype*>(output[0] -> mutable_data());
    //permute
    std::vector<int> orders = param.permute_param.order;
    int out_size = output[0] -> valid_size();
    int num_axes = input[0] -> valid_shape().size();
    std::vector<int> new_steps = output[0] -> get_stride();
    std::vector<int> old_steps = input[0] -> get_stride();
    std::vector<int> new_valid_shape = output[0] -> valid_shape();
    for (int j=0; j<out_size; ++j){
        int in_idx = 0;
        int out_idx  = 0;
        int new_valid_stride = 1;
        for (int i = num_axes - 1; i >= 0; --i) {
            int order = orders[i];
            int new_step = new_steps[i];
            int old_step = old_steps[order];
            int id = (j / new_valid_stride) % new_valid_shape[i];
            in_idx += id * old_step;
            out_idx += id * new_step;
            new_valid_stride *= new_valid_shape[i];
        }
        dst_ptr[out_idx] = src_ptr[in_idx];
    }
    
    //power
    float p = param.power_param.power;
    float scale = param.power_param.scale;
    float shift = param.power_param.shift;
    
    for (int i=0; i < out_size; ++i){
        dst_ptr[i] = pow(dst_ptr[i] * scale + shift, p);
    }
    
}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_permute_power(){
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, PermutePower, PermutePowerParam> testbase;
    typedef typename DataTrait<TargetType_H, OpDtype> :: Dtype dtype;
    
    for (int s0 : {0, 1, 2, 3}){
        for (int s1 : {0, 1, 2, 3}){
            for (int s2 : {0, 1, 2, 3}){
                for (int s3 : {0, 1, 2, 3}){
                    if (s0 != s1 && s0 != s2 && s0 != s3 && s1 != s2 && s1 != s3 && s2 != s3){
                        PermuteParam<TargetType_D> permute_param({s0, s1,s2, s3});
                        for (float p : {0, 1, 2}){
                            for (float scale : {0.5, 1.0, 2.0}){
                                for (float shift : {0, 1, 2}){
                                    PowerParam<TargetType_D> power_param(p, scale, shift);
                                    LOG(INFO)<<"permute:("<<s0<<","<<s1<<","<<s2<<","<<s3<<")";
                                    LOG(INFO)<<"power_param: p:"<<p<<" scale: "<<scale<<" shift:"<<shift;
                                    PermutePowerParam<TargetType_D> param(permute_param, power_param);
                                    for (int n : {1, 2}){
                                        for (int c : {1, 3}){
                                            for (int h: {32, 64}){
                                                for (int w : {32, 64}){
                                                    testbase.set_param(param);
                                                    testbase.set_input_shape(Shape({n, c, h, w}));
                                                    testbase.run_test(permute_power_cpu_func<dtype, TargetType_D, TargetType_H>);
                                                }
                                            }
                                        }
                                    }//for after permute_power_param
                                }
                            }
                        } //for after power_param
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_normalize) {
#ifdef USE_CUDA
    test_permute_power<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_permute_power<AMD, AMDHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    test_permute_power<X86, X86, AK_FLOAT>();
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}
