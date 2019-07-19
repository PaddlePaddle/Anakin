#include "saber/core/context.h"
#include "saber/funcs/scale.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>

using namespace anakin::saber;

template <typename Dtype>
void fill_vector_rand(std::vector<Dtype>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = rand() * 1.0f / RAND_MAX - 0.5;
    }
}
template <typename Dtype>
void fill_vector_const(std::vector<Dtype>& vec, float val) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = val;
    }
}

static int count(const int start, const int end, int n, int c, int w, int h) {
    int _layout[4] = {n, c, w, h};
    int result = 1;

    for (int i = start; i < end; ++i) {
        result *= _layout[i];
    }

    return result;
}

template <typename dtype, typename TargetType_D, typename TargetType_H>
void scale_cpu(const std::vector<Tensor<TargetType_H>*>& input,
               std::vector<Tensor<TargetType_H>*>& output, \
               ScaleParam<TargetType_D>& param) {
    LOG(INFO) << "input layout " << input[0]->get_layout();
    int axis = param.axis;
    int num_axes = param.num_axes;
    bool bias_term = param.bias_term;
    const dtype* scale_data = (input.size() > 1) ? (const dtype*)input[1]->data() : &(param.scale_w[0]);
    //std::vector<dtype>& scale_data = param.scale_w;
    std::vector<dtype>& bias_data = param.scale_b;
    int num_in = input[0]->num();
    int c_in = input[0]->channel();
    int h_in = input[0]->height();
    int w_in = input[0]->width();
    const dtype* src = (const dtype*)input[0]->mutable_data();
    Tensor<TargetType_H> inner_src;
    Tensor<TargetType_H> inner_dst;
    Tensor<X86> tensor_x;
    tensor_x.get_layout();
    dtype* dst = (dtype*)output[0]->data();

    axis = num_axes == 0 ? 0 : axis;
    num_axes  = num_axes >= 0 ? num_axes : 4 - axis;
    int outer_dim = count(0, axis, num_in, c_in, w_in, h_in);
    int inner_dim = count(axis + num_axes, 4, num_in, c_in, w_in, h_in);
    int scale_dim = count(axis, axis + num_axes, num_in, c_in, w_in, h_in);
    if(input.size() > 1) {
        scale_dim = input[1]->valid_size();
    }
    if (scale_dim > 1) {
        for (int i = 0; i < num_in * c_in * w_in * h_in; ++i) {
            int scale_id = (i / inner_dim) % scale_dim;
            dtype scale = scale_data[scale_id];

            if (bias_term) {
                dst[i] = scale * src[i] + bias_data[scale_id];
            } else {
                dst[i] = scale * src[i];
            }
        }
    } else {
        dtype scale = scale_data[0];

        for (int i = 0; i < num_in * c_in * w_in * h_in; ++i) {
            if (bias_term) {
                dtype bias = bias_data[0];
                dst[i] = scale * src[i] + bias;
            } else {
                dst[i] = scale * src[i];
            }
        }
    }
}

TEST(TestSaberFunc, test_func_scale) {
#ifdef USE_CUDA
    LOG(INFO) << "NV test......";
    TestSaberBase<NV, NVHX86, AK_FLOAT, Scale, ScaleParam> testbase;
    //test1
    int num_in = 2;
    int c_in = 2;
    int h_in = 4;
    int w_in = 4;
    int axis = 1;
    int num_axes = 1;
    bool bias_term = true;
    int scale_dim = 2;
    std::vector<float> scale_data(scale_dim);
    std::vector<float> bias_data(scale_dim);
    fill_vector_rand(scale_data);
    fill_vector_rand(bias_data);
    ScaleParam<NV> param1(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param1);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, NV, NVHX86>);
    //test2
    bias_term = false;
    ScaleParam<NV> param2(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param2);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, NV, NVHX86>);
    //test3
    axis = 0;
    num_axes = -1;
    bias_term = true;
    scale_dim = 64;
    scale_data.resize(scale_dim);
    bias_data.resize(scale_dim);
    fill_vector_rand(scale_data);
    fill_vector_rand(bias_data);
    ScaleParam<NV> param3(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param3);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, NV, NVHX86>);
    //test4
    bias_term = false;
    ScaleParam<NV> param4(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param4);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, NV, NVHX86>);
    //test5
    axis = 0;
    num_axes = 0;
    bias_term = true;
    scale_dim = 1;
    scale_data.resize(scale_dim);
    bias_data.resize(scale_dim);
    fill_vector_rand(scale_data);
    fill_vector_rand(bias_data);
    ScaleParam<NV> param5(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param5);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, NV, NVHX86>);
    //test6
    bias_term = false;
    ScaleParam<NV> param6(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param6);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, NV, NVHX86>);

#endif

#ifdef USE_X86_PLACE
    LOG(INFO) << "x86 test......";

    do {
        TestSaberBase<X86, X86, AK_FLOAT, Scale, ScaleParam> testbase;
        //test1
        int num_in = 3;
        int c_in = 256;
        int h_in = 23;
        int w_in = 32;
        int axis = 1;
        int num_axes = 1;
        bool bias_term = true;
        int scale_dim = c_in;
        std::vector<float> scale_data(scale_dim);
        std::vector<float> bias_data(scale_dim);
        fill_vector_rand(scale_data);
        fill_vector_rand(bias_data);
        ScaleParam<X86> param1(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param1);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}, Layout_NCHW_C8R), RANDOM, 1,
                                 Layout_NCHW_C8R);
        testbase.run_test(scale_cpu<float, X86, X86>);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, X86, X86>);
        //test2
        bias_term = false;
        ScaleParam<X86> param2(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param2);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}, Layout_NCHW_C8R), RANDOM, 1,
                                 Layout_NCHW_C8R);
        testbase.run_test(scale_cpu<float, X86, X86>);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, X86, X86>);
        //test3
        axis = 0;
        num_axes = -1;
        bias_term = true;
        scale_dim = num_in * c_in * h_in * w_in;
        scale_data.resize(scale_dim);
        bias_data.resize(scale_dim);
        fill_vector_rand(scale_data);
        fill_vector_rand(bias_data);
        ScaleParam<X86> param3(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param3);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, X86, X86>);
        //test4
        bias_term = false;
        ScaleParam<X86> param4(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param4);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, X86, X86>);
        //test5
        axis = 0;
        num_axes = 0;
        bias_term = true;
        scale_dim = 1;
        scale_data.resize(scale_dim);
        bias_data.resize(scale_dim);
        fill_vector_rand(scale_data);
        fill_vector_rand(bias_data);
        ScaleParam<X86> param5(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param5);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}, Layout_NCHW_C8R), RANDOM, 1,
                                 Layout_NCHW_C8R);
        testbase.run_test(scale_cpu<float, X86, X86>);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, X86, X86>);
        //test6
        bias_term = false;
        ScaleParam<X86> param6(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param6);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}, Layout_NCHW_C8R), RANDOM, 1,
                                 Layout_NCHW_C8R);
        testbase.run_test(scale_cpu<float, X86, X86>);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, X86, X86>);

        //test7
        TestSaberBase<X86, X86, AK_FLOAT, Scale, ScaleParam> testbase2(2,1);
        num_in = 1;
        c_in = 64;
        h_in = 32;
         w_in = 32;
        axis = 1;
        num_axes = 1;
        bias_term = false;
        scale_dim = c_in;
        Shape scale_shape = Shape({1 ,c_in, 1, 1}, Layout_NCHW);
        std::vector<Shape> inputs;
        inputs.push_back(Shape({num_in, c_in, h_in, w_in}, Layout_NCHW));
        inputs.push_back(scale_shape);
        fill_vector_rand(scale_data);
        fill_vector_rand(bias_data);
        testbase2.set_param(param1);
        testbase2.set_input_shape(inputs, RANDOM, 1);
        testbase2.run_test(scale_cpu<float, X86, X86>);
        testbase2.set_input_shape(inputs);
        testbase2.run_test(scale_cpu<float, X86, X86>);
    } while (0);

#endif
#ifdef USE_ARM_PLACE
    LOG(INFO) << "ARM test......";

    do {
        TestSaberBase<ARM, ARM, AK_FLOAT, Scale, ScaleParam> testbase;
        //test1
        int num_in = 2;
        int c_in = 2;
        int h_in = 4;
        int w_in = 4;
        int axis = 1;
        int num_axes = 1;
        bool bias_term = true;
        int scale_dim = 2;
        std::vector<float> scale_data(scale_dim);
        std::vector<float> bias_data(scale_dim);
        fill_vector_rand(scale_data);
        fill_vector_rand(bias_data);
        ScaleParam<ARM> param1(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param1);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, ARM, ARM>);
        //test2
        bias_term = false;
        ScaleParam<ARM> param2(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param2);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, ARM, ARM>);
        //test3
        axis = 0;
        num_axes = -1;
        bias_term = true;
        scale_dim = 64;
        scale_data.resize(scale_dim);
        bias_data.resize(scale_dim);
        fill_vector_rand(scale_data);
        fill_vector_rand(bias_data);
        ScaleParam<ARM> param3(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param3);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, ARM, ARM>);
        //test4
        bias_term = false;
        ScaleParam<ARM> param4(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param4);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, ARM, ARM>);
        //test5
        axis = 0;
        num_axes = 0;
        bias_term = true;
        scale_dim = 1;
        scale_data.resize(scale_dim);
        bias_data.resize(scale_dim);
        fill_vector_rand(scale_data);
        fill_vector_rand(bias_data);
        ScaleParam<ARM> param5(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param5);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, ARM, ARM>);
        //test6
        bias_term = false;
        ScaleParam<ARM> param6(scale_data, bias_data, bias_term, axis, num_axes);
        testbase.set_param(param6);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(scale_cpu<float, ARM, ARM>);
    } while (0);

#endif

#ifdef USE_MLU
    LOG(INFO) << "MLU test......";
    Env<MLU>::env_init();
    TestSaberBase<MLU, MLUHX86, AK_FLOAT, Scale, ScaleParam> testbase;
    //test1
    int num_in = 2;
    int c_in = 2;
    int h_in = 4;
    int w_in = 4;
    int axis = 1;
    int num_axes = 1;
    bool bias_term = true;
    int scale_dim = 2;
    std::vector<float> scale_data(scale_dim);
    std::vector<float> bias_data(scale_dim);
    fill_vector_rand(scale_data);
    fill_vector_rand(bias_data);
    ScaleParam<MLU> param1(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param1);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, MLU, MLUHX86>, 0.02, true);
    //test2
    bias_term = false;
    ScaleParam<MLU> param2(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param2);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, MLU, MLUHX86>, 0.02, true);
    //test3
    bias_term = false;
    ScaleParam<MLU> param4(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param4);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, MLU, MLUHX86>, 0.02, true);
    //test4
    axis = 0;
    num_axes = 0;
    bias_term = true;
    scale_dim = 1;
    scale_data.resize(scale_dim);
    bias_data.resize(scale_dim);
    fill_vector_rand(scale_data);
    fill_vector_rand(bias_data);
    ScaleParam<MLU> param5(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param5);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, MLU, MLUHX86>, 0.02, true);
    //test5
    bias_term = false;
    ScaleParam<MLU> param6(scale_data, bias_data, bias_term, axis, num_axes);
    testbase.set_param(param6);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(scale_cpu<float, MLU, MLUHX86>, 0.02, true);
    LOG(INFO) << "MLU end.......";
#endif
}


int main(int argc, const char** argv) {
    // initial logger
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
