#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/cumsum.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
int g_num = 1;
int g_channel = 2;
int g_height = 4;
int g_width = 4;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void cumsum_basic(const std::vector<Tensor<TargetType_H>*>& inputs, std::vector<Tensor<TargetType_H>*>& outputs, CumsumParam<TargetType_D>& param){

    CHECK_EQ(inputs.size(), 1) << "cumsum input tensor number must be one";
    CHECK_EQ(outputs.size(), 1) << "cumsum output tensor number must be one";
    const dtype* input_data = (const dtype*)inputs[0]->data();
    dtype* output_data = (dtype*) outputs[0]->data();
    auto valid_shape = inputs[0]->valid_shape();
    outputs[0]->reshape(valid_shape);
    int axis = param.axis < 0 ? param.axis + inputs[0]->dims() : param.axis;
    int dims = valid_shape.size();
    int pre = inputs[0]->count_valid(0, axis);
    int post = inputs[0]->count_valid(axis+1, dims);
    print_tensor<TargetType_H>(*inputs[0]);
    int idx = 0;
    if (param.reverse == false) {
        if (param.exclusive == true) {
            for (size_t i = 0; i < pre; i++) {
                idx = i * valid_shape[axis] * post;
                memset(output_data + idx, 0, sizeof(dtype) * post);
                idx += post;
                for (size_t j = 1; j < valid_shape[axis]; j++) {
                    for (size_t k = 0; k < post; k++) {
                        output_data[idx] = output_data[idx - post] + input_data[idx - post];
                        idx++;
                    }
                }
            }
        } else {
            for (size_t i = 0; i < pre; i++) {
                idx = i * valid_shape[axis] * post;
                memcpy(output_data + idx, input_data + idx, sizeof(dtype) * post);
                idx += post;
                for (size_t j = 1; j < valid_shape[axis]; j++) {
                    for (size_t k = 0; k < post; k++) {
                        output_data[idx] = output_data[idx - post] + input_data[idx];
                        idx++;
                    }
                }
            }
        }
    } else {
        if (param.exclusive == true) {
            for (size_t i = 0; i < pre; i++) {
                idx = i * valid_shape[axis] * post;
                memset(output_data + idx, 0, sizeof(dtype) * post);
                auto out_tmp = output_data + idx;
                for (size_t j = 1; j < valid_shape[axis]; j++) {
                    for (size_t k = 0; k < post; k++) {
                        out_tmp[j * post + k] = out_tmp[(j - 1) * post + k] 
                            + input_data[idx + (valid_shape[param.axis]  - j) * post + k];
                    }
                }
            } 
        } else {
            for (size_t i = 0; i < pre; i++) {
                idx = i * valid_shape[axis] * post;
                memcpy(output_data + idx, input_data + idx + (valid_shape[axis] - 1) * post, 
                        sizeof(dtype) * post);
                auto out_tmp = output_data + idx;
                for (size_t j = 1; j < valid_shape[axis]; j++) {
                    for (size_t k = 0; k < post; k++) {
                        out_tmp[j * post + k] = out_tmp[(j - 1) * post + k] 
                            + input_data[idx + (valid_shape[axis] - 1 - j) * post + k];
                    }
                }
            } 
        }      
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
}

template <DataType Dtype,typename TargetType_D,typename TargetType_H>
void test_model(){

    int num = g_num;
    int channel = g_channel;
    int height = g_height;
    int width = g_width;

    TestSaberBase<TargetType_D, TargetType_H, Dtype, Cumsum, CumsumParam> testbase(1,1);
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    
    Shape input_shape2({2, 3, 32, 24}, Layout_NCHW);
    //test example
    for (auto shape: {input_shape}) {
        for (auto exclusive: {false, true}) {
            for (auto reverse: {false, true}) {
                LOG(INFO)<< "exclusive "<< exclusive << " reverse " << reverse; 
                CumsumParam<TargetType_D> param(1, exclusive, reverse);
                testbase.set_param(param);//set param
                testbase.set_input_shape(shape);
                testbase.run_test(cumsum_basic<float, TargetType_D, TargetType_H>, 0.00001, true);//run test
            }
        }
    }
}
TEST(TestSaberFunc, test_func_activation) {
   
#ifdef USE_CUDA
   //Init the test_base
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_model<AK_FLOAT, ARM, ARM>();
#endif
#ifdef USE_BM
   // Env<BM>::env_init();
    //test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    //if(argc >= 3) {
    //    num_in = atoi(argv[2]);
    //    ch_in = atoi(argv[3]);
    //    h_in = atoi(argv[4]);
    //    w_in = atoi(argv[5]);
    //}
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

