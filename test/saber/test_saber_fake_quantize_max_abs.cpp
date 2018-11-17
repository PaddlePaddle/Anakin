#include "saber/core/context.h"
#include "saber/funcs/fake_quantize_abs_max.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cmath>

using namespace anakin::saber;

/**
 * @brief   formula: x * scale /  max(max(abs(x)) .
 *              where,
 *                      local_size = 5(default), means 5 channels in succession.
 *                      sigma((x(i))^2): sum of x^2 of k channels in succession.
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
void fake_quantize_abs_max_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                  std::vector<Tensor<TargetType_H>* >& output, FakeQuantizeAbsMaxParam<TargetType_D>& param) {
    const dtype* src = (const dtype*)input[0]->data();
    auto dst = output[0]->mutable_data();
    int valid_size = input[0]->valid_size();
    auto max_data = 0.f;
    for (int i = 0; i < valid_size; i++) {
        auto abs_data = src[i] > 0.f ? src[i] : -src[i];
        max_data = abs_data > max_data ? abs_data : max_data;
    }
    auto range = (1<< (param.bit_length - 1)) - 1;
    auto scale = 1.f / max_data * range;
   LOG(INFO) <<"max_data" << max_data ;
   LOG(INFO) << "range" << range;
    if (param.bit_length == 8) {
        char* dst_tmp = (char*)dst;
        for (int i = 0; i < valid_size; i++) {
            dst_tmp[i] = round(src[i] * scale);
            //LOG(INFO) << i << " " << int(dst_tmp[i]);
        }
    } else if (param.bit_length == 16) {
        int16_t* dst_tmp = (int16_t*)dst;
        for (int i = 0; i < valid_size; i++) {
            dst_tmp[i] = round(src[i] * scale);
            LOG(INFO) << i << " " << dst_tmp[i];
        }
    } else {
        //LOG(FATAL) <<"other bit length has not been supported";
    }
}

TEST(TestSaberFunc, test_op_fake_quantize_abs_max) {

#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, FakeQuantizeAbsMax, FakeQuantizeAbsMaxParam> testbase;

    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {2, 3, 8, 64}) {
                for (int num_in : {1, 21, 32}) {
    //for (int w_in : {8,}) {
    //    for (int h_in : {2,}) {
    //        for (int ch_in : {2,}) {
    //            for (int num_in : {3}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    for(int bit_length: {8}) {
                        FakeQuantizeAbsMaxParam<NV> param(bit_length);
                        testbase.set_param(param);
                        testbase.set_rand_limit(-5.0, 5.0);
                        testbase.set_input_shape(shape);
                        testbase.run_test(fake_quantize_abs_max_cpu_base<float, NV, NVHX86>, 2.1e-5f);
                    }
                }
            }
        }
    }
#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, FakeQuantizeAbsMax, FakeQuantizeAbsMaxParam> testbase_x86;

    //for (int w_in : {8,}) {
    //    for (int h_in : {2,}) {
    //        for (int ch_in : {2,}) {
    //            for (int num_in : {3}) {
    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {2, 3, 8, 64}) {
                for (int num_in : {1, 21, 32}) {
                    Shape shape_x86({num_in, ch_in, h_in, w_in});
                    for (int bit_length : {8}) {
                        FakeQuantizeAbsMaxParam<X86> param_x86(bit_length);
                        testbase_x86.set_param(param_x86);
                        testbase_x86.set_rand_limit(-5.0, 5.0);
                        testbase_x86.set_input_shape(shape_x86);
                        testbase_x86.run_test(fake_quantize_abs_max_cpu_base<float, X86, X86>);
                    }
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
