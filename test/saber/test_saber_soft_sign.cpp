#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/soft_sign.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;

template <typename dtype, typename TargetType_D, typename TargetType_H>
void softsign_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                    std::vector<Tensor<TargetType_H>*>& outputs,
                    SoftSignParam<TargetType_D>& param) {

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();

    dtype* dout = (dtype*)outputs[0]->mutable_data();
    const dtype* din = (const dtype*)inputs[0]->data();
    size_t count = inputs[0]->valid_size();

    //y = x / (1 + |x|)
    for (size_t i = 0; i < count; i++) {
        dtype tmp = din[i] > 0 ? din[i] : -din[i];
        dout[i] = din[i] / (1 + tmp);
    }

}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {

    TestSaberBase<TargetType_D, TargetType_H, Dtype, SoftSign, SoftSignParam> testbase(1, 1);
    //test example
    for (auto num : {1, 2, 16}) {
        for (auto channel : {1, 16, 32}) {
            for (auto height : {8, 15, 32}) {
                for (auto width: {8, 13, 45}) {
                    Shape shape({num, channel, height, width}, Layout_NCHW);
                    SoftSignParam<TargetType_D> param;
                    testbase.set_param(param);//set param
                    testbase.set_input_shape(shape);
                    testbase.run_test(softsign_basic<float, TargetType_D, TargetType_H>);//run test
                }
            }
        }
    }
}
TEST(TestSaberFunc, test_func_soft_sign) {

#ifdef USE_CUDA
    //Init the test_base
    Env<NV>::env_init();
    test_model<AK_FLOAT, NV, NVHX86>();
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

