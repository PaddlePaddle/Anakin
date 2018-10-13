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
#include <vector>
#include <cmath>

#include "saber/core/context.h"
#include "saber/funcs/power.h"
#include "test/saber/test_saber_func.h"
#include "test/saber/test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"

using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void power_cpu_func(const std::vector<Tensor<TargetType_H>*>& input, std::vector<Tensor<TargetType_H>*>& output, PowerParam<TargetType_D>& param) {
    float p = param.power;
    float scale = param.scale;
    float shift = param.shift;
    
    const dtype* src_ptr = static_cast<const dtype*>(input[0] -> data());
    dtype* dst_ptr = static_cast<dtype*>(output[0] -> mutable_data());
    
    for (int i=0; i < input[0] -> valid_size(); ++i){
        dst_ptr[i] = pow(src_ptr[i]* scale +shift, p);
    }
}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_power(){
    
    typedef typename DataTrait<TargetType_D, OpDtype> :: Dtype dtype;
    //Init the test_base
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, Power, PowerParam> testbase;
    for (float p : {0, 1, 2}){
        for (float scale : {0.5, 1.0, 2.0}){
            for (float shift : {0, 1, 2}){
                PowerParam<TargetType_D> param(p, scale, shift);
                
                for (int n : {1, 2}){
                    for (int c : {1, 3}){
                        for (int h: {32, 64}){
                            for (int w : {32, 64}){
                                testbase.set_param(param);
                                testbase.set_input_shape(Shape({n, c, h, w}));
                                testbase.run_test(power_cpu_func<dtype, TargetType_D, TargetType_H>);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_power) {
#ifdef USE_CUDA
    test_power<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_power<AMD, AMDHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    test_power<X86, X86, AK_FLOAT>();
#endif
}



int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    
    return 0;
}
