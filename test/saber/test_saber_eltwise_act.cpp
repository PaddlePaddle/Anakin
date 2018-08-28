
#include "saber/core/context.h"
#include "funcs/eltwise_act.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "saber/core/tensor_op.h"
#include "saber_types.h"
#include <vector>
#include <chrono>


using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void eltwise_act_cpu(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,\
                EltwiseActiveParam<TargetType_D>& param){
    int out_size = output[0]->size();
    int in_size = input[0]->size();
    const int num_arrs = input.size();
    const dtype* src = (const dtype*)input[0]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();
    switch (param.eltwise_param.operation) {
        case Eltwise_sum:
            for (int e = 0; e < in_size; e++) {
                dst[e] = param.eltwise_param.coeff[0] * src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] += param.eltwise_param.coeff[a] * src[e];
                }
            }
            for (int e = 0; e < in_size; e++) {
                dst[e] = dst[e]>0 ? dst[e]:0;
            }
            return;
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
            for (int e = 0; e < in_size; e++) {
                dst[e] = dst[e]>0 ? dst[e]:0;
            }
            return;
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
            for (int e = 0; e < in_size; e++) {
                dst[e] = dst[e]>0 ? dst[e]:0;
            }
            return;
           
        default:
           return;
    }
}


TEST(TestSaberFunc, test_func_eltwise){

#ifdef USE_CUDA
    //Init the test_base
    TestSaberBase<NV,NVHX86,AK_FLOAT,EltwiseActive, EltwiseActiveParam> testbase_nv(2,1);
#endif
#ifdef USE_X86_PLACE
    //Init the test_base
    TestSaberBase<X86,X86,AK_FLOAT,EltwiseActive, EltwiseActiveParam> testbase_x86(2,1);
#endif
    //Eltwise<NV,AK_FLOAT> test;
    for(int num_in:{1,3,32}){
        for(int c_in:{1,3,32}){
            for(int h_in:{2,3,32}){
                for(int w_in:{2,3,32}){
                	for(int type:{2}){
                	std::vector<float> coeff({1.0f,1.0f,1.0f,.0f,.0f});
                    #ifdef USE_CUDA
                 	EltwiseParam<NV> eltwiseparam_nv((EltwiseType)type,coeff);
                    ActivationParam<NV> activationparam_nv(Active_relu);
                    EltwiseActiveParam<NV> param_nv(eltwiseparam_nv,activationparam_nv);
              	    testbase_nv.set_param(param_nv);
                 	testbase_nv.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                    testbase_nv.run_test(eltwise_act_cpu<float, NV, NVHX86>);
                    #endif
                    #ifdef USE_X86_PLACE
                    EltwiseParam<X86> eltwiseparam_x86((EltwiseType)type,coeff);
                    ActivationParam<X86> activationparam_x86(Active_relu);
                    EltwiseActiveParam<X86> param_x86(eltwiseparam_x86,activationparam_x86);
                    testbase_x86.set_param(param_x86);
                    testbase_x86.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                    testbase_x86.run_test(eltwise_act_cpu<float, X86, X86>);
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
