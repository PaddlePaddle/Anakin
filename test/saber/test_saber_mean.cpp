#include "saber/core/context.h"
#include "saber/funcs/mean.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>

using namespace anakin::saber;
/**
 * @brief compute a mean of input tensor's all elements. 
 *         
 * 
 * @tparam dtype 
 * @tparam TargetType_D 
 * @tparam TargetType_H 
 * @param input  
 * @param output 
 * @param param 
 */
template <typename dtype, typename TargetType_D, typename TargetType_H>
void mean_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                      std::vector<Tensor<TargetType_H>* >& output, MeanParam<TargetType_D>& param) {
    
    int n = input[0]->valid_size();
    const dtype* input_ptr = (const dtype*)input[0]->data();
    dtype* output_ptr = (dtype*)output[0]->mutable_data();
    dtype s = (dtype)0.0;
    for (int i = 0; i < n; i++) {
        s += input_ptr[i];
    }
    s /= n;
    output_ptr[0] = s;
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_mean(){
    TestSaberBase<TargetType_D, TargetType_H, Dtype, Mean, MeanParam> testbase;
    MeanParam<TargetType_D> param;

    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {3, 4, 8, 64}) {
                for (int num_in:{1, 21, 32}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    testbase.set_param(param);
                    //testbase.set_rand_limit();
                    testbase.set_input_shape(shape);
                    testbase.run_test(mean_cpu_base<float, TargetType_D, TargetType_H>);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_Mean) {

#ifdef USE_CUDA
   //Init the test_base
    test_mean<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_mean<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_Mean<AK_FLOAT, ARM, ARM>();
#endif
#ifdef USE_BM
   // Env<BM>::env_init();
    //test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
