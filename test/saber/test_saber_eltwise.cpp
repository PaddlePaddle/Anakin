
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
        case Eltwise_div:
            for (int e = 0; e < in_size; e++) {
                dst[e] =  src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] =  dst[e] / src[e];
                }
            }
            break;
           
        case Eltwise_mul:
            for (int e = 0; e < in_size; e++) {
                dst[e] =  src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] =  dst[e] * src[e];
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
template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_eltwise() {
    //Eltwise<NV,AK_FLOAT> test;
    for (int inputs_num: {2, 3}) {
        TestSaberBase<TargetType_D, TargetType_H, Dtype, Eltwise, EltwiseParam> testbase(inputs_num, 1);
        for (int num_in:{2, 3, 32}) {
            for (int c_in:{1, 3, 32}) {
                for (int h_in:{2, 3, 32}) {
                    for (int w_in:{2, 3, 32}) {
#ifdef USE_MLU
						for (EltwiseType type:{Eltwise_prod, Eltwise_sum, Eltwise_max}) {
#else
						for (EltwiseType type:{Eltwise_prod, Eltwise_sum, Eltwise_max, Eltwise_div, Eltwise_mul}) {
#endif
                    	    LOG(INFO)<<"input = "<<num_in<<", type = "<<type;
                    	    std::vector<float> coeff(inputs_num, 1);
                            ActivationParam<TargetType_D> activationparam(Active_relu);
                            EltwiseParam<TargetType_D> param(type, coeff, activationparam);
                            testbase.set_param(param);
                            //testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                            Shape input_shape({num_in, c_in, h_in, w_in}, Layout_NCHW);
                            std::vector<Tensor<TargetType_D>*> inputs;
                            for (int i = 0; i < inputs_num; i++) {
                                Tensor<TargetType_D>* input = new Tensor<TargetType_D>(input_shape);
                                fill_tensor_rand(*input, 0.5, 1.0);
                                inputs.push_back(input);
                            }
                            testbase.add_custom_input(inputs);
							if (std::is_same<TargetType_D, MLU>::value) {
                                testbase.run_test(eltwise_cpu<float, TargetType_D, TargetType_H>, 0.02);
							}else {
                                testbase.run_test(eltwise_cpu<float, TargetType_D, TargetType_H>);
							}

                            ActivationParam<TargetType_D> activationparam_no;
                            EltwiseParam<TargetType_D> param_noactivate(type, coeff, activationparam_no);
                            testbase.set_param(param_noactivate);
                            testbase.set_rand_limit(-5.0, 5.0);
                            testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
							if (std::is_same<TargetType_D, MLU>::value) {
                                testbase.run_test(eltwise_cpu<float, TargetType_D, TargetType_H>, 0.02);
							}else {
                                testbase.run_test(eltwise_cpu<float, TargetType_D, TargetType_H>);
							}
                            for (int i = 0; i < inputs_num; i++) {
                                delete inputs[i];
                            }
                        }
                    }
                }
            }
        }
    }
}


TEST(TestSaberFunc, test_func_eltwise){

#ifdef USE_CUDA
    //Init the test_base
    test_eltwise<AK_FLOAT, NV, NVHX86>();
#endif  
#ifdef USE_X86_PLACE
    test_eltwise<AK_FLOAT, X86, X86>();
#endif        
#ifdef USE_MLU
    //Init the test_base
    Env<MLUHX86>::env_init();
    Env<MLU>::env_init();
    test_eltwise<AK_FLOAT, MLU, MLUHX86>();
#endif  // USE_MLU
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
