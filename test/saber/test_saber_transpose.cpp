
#include "saber/core/context.h"
#include "saber/funcs/transpose.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>

#include <chrono>


using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void transpose_cpu(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,\
                TransposeParam<TargetType_D>& param){
    const dtype* in_data = (const dtype*)input[0]->data();
    dtype* out_data = (dtype*)output[0]->mutable_data();
    int w_out = output[0]->width();
    int h_out = output[0]->height();
    int c_out = output[0]->channel();
    int n_out = output[0]->num();

    int w_in = input[0]->width();
    int h_in = input[0]->height();
    int c_in = input[0]->channel();
    int n_in = input[0]->num();
   
    CHECK_EQ(c_in, c_out) << "input channel should = output channel";
    CHECK_EQ(n_in, n_out) << "input batch size should = output batch size";
    CHECK_EQ(h_in, w_out) << "input width size should = output height size";
    CHECK_EQ(w_in, h_out) << "input height size should = output width size";
    for(int k = 0; k < n_in * c_in; ++k){  
        for(int j = 0; j < h_in; ++j){
            for(int i = 0; i < w_in; ++i){
                out_data[i * w_out + j] = in_data[j * w_in + i];
            }
        }
        in_data += h_in * w_in;
        out_data += h_out * w_out;
    }
}


TEST(TestSaberFunc, test_func_transpose){

#ifdef USE_CUDA
    //Init the test_base
    TestSaberBase<NV,NVHX86,AK_FLOAT, Transpose, TransposeParam> testbase;
    for(int num_in:{1,3,32}){
        for(int c_in:{1,3,128}){
            for(int h_in:{2,3,256}){
                for(int w_in:{2,3,64}){
                    TransposeParam<NV> param;
                    testbase.set_param(param);
                    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                    testbase.run_test(transpose_cpu<float, NV, NVHX86>);

                }
            }
        }
    }

    
#endif
}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
