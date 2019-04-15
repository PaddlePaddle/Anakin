#include "saber/core/context.h"
#include "test_saber_base.h"
#include "test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/one_hot.h"
#include "saber/core/data_traits.h"

using namespace anakin::saber;

template<typename TargetType_D, typename TargetType_H>
void one_hot_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
                      std::vector<Tensor<TargetType_H>*>& output,
                      OneHotParam<TargetType_D>& param) {

    memset(output[0]->mutable_data(), 0, output[0]->valid_size() * output[0]->get_dtype_size());

    int depth = param.depth;
    const float* in_ptr = (const float*)input[0]->data();
    float* out_ptr = (float*)output[0]->mutable_data();
    int dims = input[0]->valid_size();
    for (int i = 0; i < dims; ++i) {
        out_ptr[i * depth + (int)in_ptr[i]] = 1.0;
    }
}

//test template for different device and dtype
template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_one_hot() {

    std::vector<int> in_n_v{2, 3, 4, 5, 6};
    std::vector<int> in_c_v{2, 3, 4, 5, 6};
    std::vector<int> in_h_v{2, 3, 4, 5, 6};
    std::vector<int> in_w_v{1};

    std::vector<int> depth_v{4, 5, 6, 7, 8, 9};
    Env<TargetType_D>::env_init();
    Env<TargetType_H>::env_init();
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, OneHot, OneHotParam> testbase;

    for (int in_n : in_n_v)
    for (int in_c : in_c_v)
    for (int in_h : in_h_v)
    for (int in_w : in_w_v)
    for (int depth : depth_v) {
        OneHotParam<TargetType_D> param(depth);
        testbase.set_param(param);//set param
        testbase.set_rand_limit(0, depth);
        testbase.set_input_shape(Shape({in_n, in_c, in_h, in_w})); //add some input shape
        testbase.run_test(one_hot_cpu_func<TargetType_D, TargetType_H>, 0.0001);//run test

    }
}

TEST(TestSaberFunc, test_func_pool) {
#ifdef USE_CUDA
    test_one_hot<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    test_one_hot<X86, X86, AK_FLOAT>();
#endif
}

int main(int argc, const char** argv) {
    // initial logger
//    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
