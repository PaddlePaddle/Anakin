#include "saber/core/context.h"
#include "saber/funcs/conv.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "conv_func_helper.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFunc, test_saber_depthwise_conv_results) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    TestSaberBase<NV,NVHX86,AK_FLOAT,Conv,ConvParam> testbase_nv;
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    TestSaberBase<X86,X86,AK_FLOAT,Conv,ConvParam> testbase_x86;
#endif
    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1};
    std::vector<int> dilation_w_v{1};
    std::vector<int> group_v{4, 7};
    std::vector<int> in_h_v{4, 7};
    std::vector<int> in_w_v{4, 7};
    std::vector<int> input_num_v{1,3};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{true, false};

    for (int bias_term : bias_term_v)
    for (int with_relu : with_relu_v)
    for (auto kernel_h : kernel_h_v)
    for (auto kernel_w : kernel_w_v)
    for (auto pad_h : pad_h_v)
    for (auto pad_w : pad_w_v)
    for (auto stride_h : stride_h_v)
    for (auto stride_w : stride_w_v)
    for (auto dilation_h : dilation_h_v)
    for (auto dilation_w : dilation_w_v)
    for (auto bias_term : bias_term_v)
    for (auto with_relu : with_relu_v)
    for (auto group : group_v) {
        int out_channels = group;
        int in_channels = group;
        Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
        Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
#ifdef USE_CUDA
        Tensor<NV> weights_dev;
        Tensor<NV> bias_dev;

        weights_dev.re_alloc(weights_s, AK_FLOAT);
        fill_tensor_rand(weights_dev, -10.f, 20.0f);
        if (bias_term) {
            bias_dev.re_alloc(bias_s, AK_FLOAT);
            fill_tensor_rand(bias_dev, -10.0f, 20.0f);
        }
        ConvParam<NV> param_nv(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);
#endif
#ifdef USE_X86_PLACE
        Tensor<X86> weights_x86;
        weights_x86.re_alloc(weights_s, AK_FLOAT);
        fill_tensor_rand(weights_x86, -10.f, 20.0f);

        Tensor<X86> bias_x86;
        if (bias_term) {
            bias_x86.re_alloc(bias_s, AK_FLOAT);
            fill_tensor_rand(bias_x86, -10.0f, 20.0f);
        }
        ConvParam<X86> param_x86(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_x86, &bias_x86);
#endif
        for (auto input_num : input_num_v)
        for (auto height : in_h_v)
        for (auto width : in_w_v) {
#ifdef USE_CUDA
            testbase_nv.set_param(param_nv);//set param
            testbase_nv.set_input_shape(Shape({input_num,in_channels,height,width},
                                              Layout_NCHW));//add some input shape
            testbase_nv.run_test(conv_cpu_func<float, NV, NVHX86>, 1e-3);//run test
#endif
#ifdef USE_X86_PLACE
            testbase_x86.set_param(param_x86);//set param
            testbase_x86.set_input_shape(Shape({input_num, in_channels, height, width},
                                               Layout_NCHW));//add some input shape
            testbase_x86.run_test(conv_cpu_func<float, X86, X86>);//run test
#endif
        }
     }
}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
