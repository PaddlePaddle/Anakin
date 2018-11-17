#include "saber/core/context.h"
#include "saber/funcs/conv_shift.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cmath>

using namespace anakin::saber;

/**
 * @brief   formula: input0:[x0, x1, x2, x3, x4]. input1:[y0, y1, y2]
 *          out:z0 = y0 * x4 + y1 * x0 + y2 * x1
 *              z1 = y0 * x0 + y1 * x1 + y2 * x2
 *              z2 = y0 * x1 + y1 * x2 + y2 * x3
 *              z3 = y0 * x2 + y1 * x3 + y2 * x4
 *              z4 = y0 * x3 + y1 * x4 + y2 * x0. 
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
void conv_shift_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                  std::vector<Tensor<TargetType_H>* >& output, ConvShiftParam<TargetType_D>& param) {

    int num = input[0]->num();
    int width_x = input[0]->count_valid(1, input[0]->dims());
    int width_y = input[1]->count_valid(1, input[1]->dims());
    print_tensor(*input[0]);
    print_tensor(*input[1]);

    const dtype* src_x = (const dtype*)input[0]->data();
    const dtype* src_y = (const dtype*)input[1]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();

    for (int i = 0; i < num; i++) {
        auto x_tmp = src_x + i * width_x;
        auto y_tmp = src_y + i * width_y;
        auto dst_tmp = dst + i * width_x;
        for (int j = 0; j < width_x; j++) {
            dtype res = 0.f;
            for (int k = 0; k < width_y; k++) {
                int index = (j - (width_y - 1) / 2 + k + width_x) % width_x;
                res += y_tmp[k] * x_tmp[index];
            }
            dst_tmp[j] = res;
        }
    }
}

TEST(TestSaberFunc, test_op_conv_shift) {

#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, ConvShift, ConvShiftParam> testbase(2, 1);

    for (int num : {8}) {
        for (int x_width : {10}) {
            for (int y_width : {3}) {
                std::vector<Shape> input_shapes;
                Shape shape_x({num, x_width, 1, 1});
                Shape shape_y({num, y_width, 1, 1});
                input_shapes.push_back(shape_x);
                input_shapes.push_back(shape_y);
                ConvShiftParam<NV> param;
                testbase.set_param(param);
                testbase.set_rand_limit(-5.0, 5.0);
                testbase.set_input_shape(input_shapes);
                testbase.run_test(conv_shift_cpu_base<float, NV, NVHX86>, 2.1e-5f, true);
            }
        }
    }
#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, ConvShift, ConvShiftParam> testbase_cpu(2, 1);

    for (int num : {2}) {
        for (int x_width : {10}) {
            for (int y_width : {3}) {
                std::vector<Shape> input_shapes;
                Shape shape_x({num, x_width, 1, 1});
                Shape shape_y({num, y_width, 1, 1});
                input_shapes.push_back(shape_x);
                input_shapes.push_back(shape_y);
                ConvShiftParam<X86> param;
                testbase_cpu.set_param(param);
                testbase_cpu.set_rand_limit(-5.0, 5.0);
                testbase_cpu.set_input_shape(input_shapes);
                testbase_cpu.run_test(conv_shift_cpu_base<float, X86, X86>, 2.1e-5f, true);
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
