#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/cos_sim.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;

template <typename dtype, typename TargetType_D, typename TargetType_H>
void cossim_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                    std::vector<Tensor<TargetType_H>*>& outputs,
                    CosSimParam<TargetType_D>& param) {
    CHECK_EQ(inputs.size(), 2) << "CosSim input num need be  2, but is" << inputs.size();
    CHECK_EQ(outputs.size(), 1) << "CosSim input num need be  1, but is" << outputs.size();
    size_t count_0 = inputs[0]->valid_size();
    size_t count_1 = inputs[1]->valid_size();
    CHECK_EQ(count_0, count_1) << "input0 and input1 valid size is not equal";

    size_t num = inputs[0]->num();
    size_t inner_size = count_0 / inputs[0]->num();
    const dtype *input0_data = (const dtype*)inputs[0]->data();
    const dtype *input1_data = (const dtype*)inputs[1]->data();
    dtype *output_data = (dtype*)outputs[0]->mutable_data();

    //z = x'y/ (|x|*|y|)
    for (size_t n = 0; n < num; n++) {
        auto input0_square_sum = (dtype)0;
        auto input1_square_sum = (dtype)0;
        auto input01_prod_sum = (dtype)0;
        for (size_t i = 0; i < inner_size; i++) {
            input0_square_sum += input0_data[i] * input0_data[i];
            input1_square_sum += input1_data[i] * input1_data[i];
            input01_prod_sum += input0_data[i] * input1_data[i];
        }
        float bc = input0_square_sum * input1_square_sum;
        if (bc < param.epsilon) {
            output_data[n] = 0;
        } else {
            output_data[n] = input01_prod_sum / sqrt(bc);
        }
        input0_data += inner_size;
        input1_data += inner_size;
    }

}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {

    TestSaberBase<TargetType_D, TargetType_H, Dtype, CosSim, CosSimParam> testbase(2, 1);
    //test example
    for (auto num : {1, 2, 16}) {
        for (auto channel : {1, 16, 32}) {
            for (auto height : {8, 15, 32}) {
                for (auto width: {8, 13, 45}) {
                    Shape shape({num, channel, height, width}, Layout_NCHW);
                    CosSimParam<TargetType_D> param(0.f);
                    testbase.set_param(param);//set param
                    testbase.set_input_shape(shape);
                    testbase.run_test(cossim_basic<float, TargetType_D, TargetType_H>, true);//run test
                }
            }
        }
    }
}
TEST(TestSaberFunc, test_func_cos_sim) {

#ifdef USE_CUDA
    //Init the test_base
    //Env<NV>::env_init();
    //test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    test_model<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_model<AK_FLOAT, ARM, ARM>();
#endif
#ifdef AMD_GPU
    //    Env<AMD>::env_init();
    //    test_model<AK_FLOAT, AMD, AMDHX86>();
#endif
#ifdef USE_BM_PLACE
    //    Env<BM>::env_init();
    //    test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

