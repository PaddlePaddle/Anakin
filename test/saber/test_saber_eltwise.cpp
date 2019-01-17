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
#include "funcs/eltwise.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "saber/core/tensor_op.h"
#include "saber_types.h"
#include <vector>

#include <chrono>


using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void eltwise_cpu(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,\
                EltwiseParam<TargetType_D>& param){
    int out_size = output[0]->size();
    int in_size = input[0]->size();
    const int num_arrs = input.size();
    const dtype* src = (const dtype*)input[0]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();
    switch (param.operation) {
        case Eltwise_sum:
            for (int e = 0; e < in_size; e++) {
                dst[e] = param.coeff[0] * src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] += param.coeff[a] * src[e];
                }
            }
            break;
        case Eltwise_prod:
            for (int e = 0; e < in_size; e++) {
                dst[e] =  src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] *=  src[e];
                }
            }
            break;
        case Eltwise_max:
            for (int e = 0; e < in_size; e++) {
                dst[e] =  src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] =  (dst[e]>src[e])?dst[e]:src[e];
                }
            }
            break;
           
        default:
           break;
    }
    if(param.activation_param.has_active){
        switch (param.activation_param.active) {
            case Active_relu:
                for (int a = 0; a < num_arrs; a++) {
                    for (int e = 0; e < in_size; e++) {
                       dst[e] =  (dst[e]>0.0f)?dst[e]:0.0f;
                    }
                }
                break;
            default:
                break;
        }
    }
}


TEST(TestSaberFunc, test_func_eltwise){

#ifdef AMD_GPU
    //Init the test_base
    Env<AMD>::env_init();
    TestSaberBase<AMD,AMDHX86,AK_FLOAT,Eltwise, EltwiseParam> testbase_amd(2,1);
#endif
#ifdef USE_CUDA
    //Init the test_base
    TestSaberBase<NV,NVHX86,AK_FLOAT,Eltwise, EltwiseParam> testbase_nv(2,1);
#endif
#ifdef USE_X86_PLACE
    //Init the test_base
    TestSaberBase<X86,X86,AK_FLOAT,Eltwise, EltwiseParam> testbase_x86(2,1);
#endif
    //Eltwise<NV,AK_FLOAT> test;
    for(int num_in:{2,3,32}){
        for(int c_in:{1,3,32}){
            for(int h_in:{2,3,32}){
                for(int w_in:{2,3,32}){
                	for(EltwiseType type:{Eltwise_prod,Eltwise_sum,Eltwise_max}){
                	    LOG(INFO)<<"input = "<<num_in<<", type = "<<type;
                	    std::vector<float> coeff({-1.5f,-2.f,3.f});
                    #ifdef AMD_GPU
                        ActivationParam<AMD> activationparam_amd(Active_relu);
                        EltwiseParam<AMD> param_amd(type,coeff,activationparam_amd);
                        testbase_amd.set_param(param_amd);
                        testbase_amd.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                        testbase_amd.run_test(eltwise_cpu<float, AMD, AMDHX86>);

                        ActivationParam<AMD> activationparam_amd_no;
                        EltwiseParam<AMD> param_amd_noactivate(type,coeff,activationparam_amd_no);
                        testbase_amd.set_param(param_amd_noactivate);
                        testbase_amd.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                        testbase_amd.run_test(eltwise_cpu<float, AMD, AMDHX86>);
                    #endif
                    #ifdef USE_CUDA
                        ActivationParam<NV> activationparam_nv(Active_relu);
                        EltwiseParam<NV> param_nv(type,coeff,activationparam_nv);
                        testbase_nv.set_param(param_nv);
                        testbase_nv.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                        testbase_nv.run_test(eltwise_cpu<float, NV, NVHX86>);

                        ActivationParam<NV> activationparam_nv_no;
                        EltwiseParam<NV> param_nv_noactivate(type,coeff,activationparam_nv_no);
                        testbase_nv.set_param(param_nv_noactivate);
                        testbase_nv.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                        testbase_nv.run_test(eltwise_cpu<float, NV, NVHX86>);
                    #endif
                    #ifdef USE_X86_PLACE
                        ActivationParam<X86> activationparam_x86(Active_relu);
                        EltwiseParam<X86> param_x86(type,coeff,activationparam_x86);
                        testbase_x86.set_param(param_x86);
                        testbase_x86.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                        testbase_x86.run_test(eltwise_cpu<float, X86, X86>);

                        ActivationParam<X86> activationparam_x86_no;
                        EltwiseParam<X86> param_x86_noactivate(type,coeff,activationparam_x86_no);
                        testbase_x86.set_param(param_x86_noactivate);
                        testbase_x86.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                        testbase_x86.run_test(eltwise_cpu<float, X86, X86>);
                    #endif
                	}
                }
            }
        }
    }
}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
