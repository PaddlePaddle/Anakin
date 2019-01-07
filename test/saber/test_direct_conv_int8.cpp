#include "anakin_config.h"
#include "core/context.h"
#include "test_saber_func.h"
#include "saber/core/tensor.h"
#include "saber/funcs/debug.h"
#include "saber/funcs/calibrate.h"
#include "tensor_op.h"
#include "saber_types.h"
#include "conv_func_helper.h"
#include <vector>
#ifdef USE_CUDA
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/impl/cuda/saber_conv_direct.h"
#include "saber/funcs/impl/cuda/saber_conv_gemmlike.h"
#endif

using namespace anakin::saber;
template <typename Dtype>
void transpose_filter_KCRS_2_CRSKC4(const Dtype *input, Dtype *temp, Dtype *output, \
    int K, int C, int R, int S) {
    const int CRS = C * R * S;
    for (int var_k = 0; var_k < K; var_k++) {
        for (int var_crs = 0; var_crs < CRS; var_crs++) {
            temp[var_crs * K + var_k] = input[var_k * CRS + var_crs];
        }
    }
    int read_in = 0;
    int write_out = 0;
    int out_loop = C / 4;
    int inner_loop =  K * R * S * 4;
    for (int i = 0; i < out_loop; ++i) {
        for (int j = 0; j < inner_loop; ++j) {
            write_out = i * inner_loop + j;
            read_in = ((i * 4) + (j % 4))  * (inner_loop / 4) + j / 4;
            output[write_out] = temp[read_in];
        }
    }
}

template <typename Dtype>
void transpose_img_NCHW_2_NCHWC4(const Dtype* input, Dtype *output,
                                 int N, int C, int H, int W) {
    int read_in = 0;
    int write_out = 0;
    int out_loop = N * C / 4;
    int inner_loop =  H * W * 4;
    for (int i = 0; i < out_loop; ++i) {
        for (int j = 0; j < inner_loop; ++j) {
            write_out = i * inner_loop + j;
            read_in = ((i * 4) + (j % 4))  * (inner_loop / 4) + j / 4;
            output[write_out] = input[read_in];
        }
    }
}

#ifdef USE_CUDA
TEST(TestSaberFunc, test_saber_conv_int8_results) {

    Env<NV>::env_init();
    Env<NVHX86>::env_init();

    bool with_relu = true;
    float alpha = 2.1f;
    int input_num = 1;
    int in_channels = 128;
    int out_channels = 256;
    int height = 64;
    int width = 64;

    int kernel_h = 3;
    int kernel_w = 3;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int group = 1;

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape output_s({input_num, out_channels, height, width}, Layout_NCHW);
    // trans to input_num, in_channels/4, height, width, inner_channels(4)
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    // trans to in_channels/4, kernel_h, kernel_w, out_channels, inner_channels(4);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    Tensor<NV> input_dev;
    Tensor<NV> weights_dev;
    Tensor<NV> bias_dev;
    Tensor<NV> output_dev;

    Tensor<NVHX86> input_host;
    Tensor<NVHX86> weights_host;
    Tensor<NVHX86> bias_host;
    Tensor<NVHX86> output_host;
    Tensor<NVHX86> check_output;

    input_dev.re_alloc(input_s, AK_INT8);
    input_host.re_alloc(input_s, AK_INT8);

    weights_dev.re_alloc(weights_s, AK_INT8);
    weights_host.re_alloc(weights_s, AK_INT8);

    output_dev.re_alloc(output_s, AK_FLOAT);
    output_host.re_alloc(output_s, AK_FLOAT);
    check_output.re_alloc(output_s, AK_FLOAT);

    bias_dev.re_alloc(bias_s, AK_FLOAT);
    bias_host.re_alloc(bias_s, AK_FLOAT);

    fill_tensor_rand(input_host, -10, 10);
    fill_tensor_rand(weights_host, -10, 10);
    fill_tensor_rand(bias_dev, -10, 10);
    bias_host.copy_from(bias_dev);

    Context<NV> ctx(0, 0, 1);
    int generate_arch = Env<NV>::cur_env()[ctx.get_device_id()]._info._generate_arch;
    // only support 61 arch for now.
    bool arch_check = (generate_arch == 61);
    if (!arch_check) {
                LOG(INFO) << "device not support int8 op!!";
        return;
    }
    auto stream = ctx.get_compute_stream();
    {
        Tensor<NVHX86> input_temp;
        input_temp.re_alloc(input_host.valid_shape(), AK_INT8);
        transpose_img_NCHW_2_NCHWC4((const char *) input_host.data(),
                (char *) input_temp.mutable_data(),
                input_host.num(),
                input_host.channel(),
                input_host.height(),
                input_host.width());
        input_dev.copy_from(input_temp);
    }
    bool use_1x1 = true;
    use_1x1 = use_1x1 && (kernel_h == 1);
    use_1x1 = use_1x1 && (kernel_w == 1);
    use_1x1 = use_1x1 && (dilation_h == 1);
    use_1x1 = use_1x1 && (dilation_w == 1);
    use_1x1 = use_1x1 && (stride_h == 1);
    use_1x1 = use_1x1 && (stride_w == 1);
    use_1x1 = use_1x1 && (pad_h == 0);
    use_1x1 = use_1x1 && (pad_w == 0);
    use_1x1 = use_1x1 && (group == 1);

    if (!use_1x1) {
        {
            Tensor<NVHX86> weight_temp;
            Tensor<NVHX86> weight_temp2;
            weight_temp.re_alloc(weights_host.valid_shape(), AK_INT8);
            weight_temp2.re_alloc(weights_host.valid_shape(), AK_INT8);
            transpose_filter_KCRS_2_CRSKC4(
                    (const char *) weights_host.data(),
                    (char *) weight_temp.mutable_data(),
                    (char *) weight_temp2.mutable_data(),
                    weights_host.num(),
                    weights_host.channel(),
                    weights_host.height(),
                    weights_host.width());
            weights_dev.copy_from(weight_temp2);
        }
        ConvParam<NV> param(group, pad_h, pad_w,
                stride_h, stride_w,
                dilation_h, dilation_w,
                &weights_dev, &bias_dev);
        param.activation_param.has_active = with_relu;
        param.alpha = alpha;
        SaberDirectConv<AK_INT8> conv_direct;
        std::vector<Tensor<NV>*> inputs;
        std::vector<Tensor<NV>*> outputs;
        inputs.push_back(&input_dev);
        outputs.push_back(&output_dev);
        conv_direct.init(inputs, outputs, param, ctx);
        conv_direct.dispatch(inputs, outputs, param);

    } else {
        {
            Tensor<NVHX86> weight_temp;
            weight_temp.re_alloc(weights_host.valid_shape(), AK_INT8);
            transpose_img_NCHW_2_NCHWC4((const char *) weights_host.data(),
                                        (char *) weight_temp.mutable_data(),
                                        weights_host.num(),
                                        weights_host.channel(),
                                        weights_host.height(),
                                        weights_host.width());

            weights_dev.copy_from(weight_temp);
        }
        ConvParam<NV> param(group, pad_h, pad_w,
                            stride_h, stride_w,
                            dilation_h, dilation_w,
                            &weights_dev, &bias_dev);
        param.activation_param.has_active = with_relu;
        param.alpha = alpha;
        SaberGemmLikeConv<AK_INT8> conv_gemm;
        std::vector<Tensor<NV>*> inputs;
        std::vector<Tensor<NV>*> outputs;
        inputs.push_back(&input_dev);
        outputs.push_back(&output_dev);
        conv_gemm.init(inputs, outputs, param, ctx);
        conv_gemm.dispatch(inputs, outputs, param);
    }
    cudaDeviceSynchronize();
    output_host.copy_from(output_dev);
    cudaDeviceSynchronize();
    conv_basic_check<NVHX86, float, char>(input_host, check_output,
            (const char*)weights_host.data(), (const float*)bias_host.data(), group,
            kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
            pad_w, pad_h, true, with_relu, 0.f, alpha);

    write_tensorfile(output_dev, "int8_output.txt");
    write_tensorfile(check_output, "fp32_output.txt");

    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)output_host.data(), (const float*)check_output.data(),
                    check_output.valid_size(), max_ratio, max_diff);
    LOG(INFO) << "ratio = " << max_ratio << " max_diff = " << max_diff;
}

TEST(TestSaberFunc, test_weights_calibrate) {
    Tensor<NVHX86> weights_host;
    Tensor<NVHX86> weights_temp;

    Shape weight_s({4, 4, 3, 3}, Layout_NCHW);
    Shape weight_t_s({4, 4, 3, 3}, Layout_NCHW);
    weights_host.re_alloc(weight_s, AK_FLOAT);
    weights_temp.re_alloc(weight_t_s, AK_INT8);
    Context<NV> ctx(0, 0, 1);
    fill_tensor_rand(weights_host, -10, 10);
    convert_weights_to_direct<NV, NVHX86> (weights_temp, weights_host, ctx);
//    print_tensor_valid(weights_host);
//    print_tensor_valid(weights_temp);
//    write_tensorfile(weights_host, "int8_output.txt");
//    write_tensorfile(weights_temp, "fp32_output.txt");
}

void test_saber_cudnn_speed(int input_num,
                            int in_channels,
                            int out_channels,
                            int height,
                            int width,
                            int kernel_h,
                            int kernel_w,
                            int pad_h,
                            int pad_w,
                            int stride_h,
                            int stride_w,
                            int dilation_h,
                            int dilation_w,
                            int group) {

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape output_s({input_num, out_channels, height, width}, Layout_NCHW);
    // trans to input_num, in_channels/4, height, width, inner_channels(4)
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    // trans to in_channels/4, kernel_h, kernel_w, out_channels, inner_channels(4);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    Tensor<NV> input_dev;
    Tensor<NV> weights_dev;
    Tensor<NV> bias_dev;
    Tensor<NV> output_dev;

    Tensor<NVHX86> input_host;
    Tensor<NVHX86> weights_host;
    Tensor<NVHX86> bias_host;
    Tensor<NVHX86> output_host;
    Tensor<NVHX86> check_output;

    input_dev.re_alloc(input_s, AK_INT8);
    input_host.re_alloc(input_s, AK_INT8);

    weights_dev.re_alloc(weights_s, AK_INT8);
    weights_host.re_alloc(weights_s, AK_INT8);

    output_dev.re_alloc(output_s, AK_FLOAT);
    output_host.re_alloc(output_s, AK_FLOAT);
    check_output.re_alloc(output_s, AK_FLOAT);

    bias_dev.re_alloc(bias_s, AK_FLOAT);
    bias_host.re_alloc(bias_s, AK_FLOAT);

    fill_tensor_rand(input_host, -10, 10);
    fill_tensor_rand(weights_host, -10, 10);
    fill_tensor_rand(bias_dev, -10, 10);
    bias_host.copy_from(bias_dev);

    Context<NV> ctx(0, 0, 1);
    auto stream = ctx.get_compute_stream();
    {
        Tensor<NVHX86> input_temp;
        input_temp.re_alloc(input_host.valid_shape(), AK_INT8);
        transpose_img_NCHW_2_NCHWC4((const char *) input_host.data(),
                                    (char *) input_temp.mutable_data(),
                                    input_host.num(),
                                    input_host.channel(),
                                    input_host.height(),
                                    input_host.width());

        input_dev.copy_from(input_temp);
    }
    bool use_1x1 = true;
    use_1x1 = use_1x1 && (kernel_h == 1);
    use_1x1 = use_1x1 && (kernel_w == 1);
    use_1x1 = use_1x1 && (dilation_h == 1);
    use_1x1 = use_1x1 && (dilation_w == 1);
    use_1x1 = use_1x1 && (stride_h == 1);
    use_1x1 = use_1x1 && (stride_w == 1);
    use_1x1 = use_1x1 && (pad_h == 0);
    use_1x1 = use_1x1 && (pad_w == 0);
    use_1x1 = use_1x1 && (group == 1);

    int ts = 100;
    SaberTimer<NV> timer;
    {
        {
            Tensor<NVHX86> weight_temp;
            weight_temp.re_alloc(weights_host.valid_shape(), AK_INT8);
            transpose_img_NCHW_2_NCHWC4((const char *) weights_host.data(),
                                        (char *) weight_temp.mutable_data(),
                                        weights_host.num(),
                                        weights_host.channel(),
                                        weights_host.height(),
                                        weights_host.width());

            weights_dev.copy_from(weight_temp);
        }
        ConvParam<NV> param(group, pad_h, pad_w,
                            stride_h, stride_w,
                            dilation_h, dilation_w,
                            &weights_dev, &bias_dev);

        VenderConv2D<NV, AK_INT8> conv_vender;
        std::vector<Tensor<NV>*> inputs;
        std::vector<Tensor<NV>*> outputs;
        inputs.push_back(&input_dev);
        outputs.push_back(&output_dev);
        conv_vender.init(inputs, outputs, param, ctx);
        conv_vender.dispatch(inputs, outputs, param);

        cudaDeviceSynchronize();
        for (int i = 0; i < ts; ++i) {
            timer.start(ctx);
            conv_vender.dispatch(inputs, outputs, param);
            output_dev.record_event(ctx.get_compute_stream());
            output_dev.sync();
            timer.end(ctx);
        }
        printf("cudnn,%lf\n", timer.get_average_ms());
    }
    cudaDeviceSynchronize();
}

void test_saber_direct_speed(int input_num, int in_channels,
                             int out_channels,
                             int height,
                             int width,
                             int kernel_h,
                             int kernel_w,
                             int pad_h,
                             int pad_w,
                             int stride_h,
                             int stride_w,
                             int dilation_h,
                             int dilation_w,
                             int group) {

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape output_s({input_num, out_channels, height, width}, Layout_NCHW);
    // trans to input_num, in_channels/4, height, width, inner_channels(4)
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    // trans to in_channels/4, kernel_h, kernel_w, out_channels, inner_channels(4);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    Tensor<NV> input_dev;
    Tensor<NV> weights_dev;
    Tensor<NV> bias_dev;
    Tensor<NV> output_dev;

    Tensor<NVHX86> input_host;
    Tensor<NVHX86> weights_host;
    Tensor<NVHX86> bias_host;
    Tensor<NVHX86> output_host;
    Tensor<NVHX86> check_output;

    input_dev.re_alloc(input_s, AK_INT8);
    input_host.re_alloc(input_s, AK_INT8);

    weights_dev.re_alloc(weights_s, AK_INT8);
    weights_host.re_alloc(weights_s, AK_INT8);

    output_dev.re_alloc(output_s, AK_FLOAT);
    output_host.re_alloc(output_s, AK_FLOAT);
    check_output.re_alloc(output_s, AK_FLOAT);

    bias_dev.re_alloc(bias_s, AK_FLOAT);
    bias_host.re_alloc(bias_s, AK_FLOAT);

    fill_tensor_rand(input_host, -10, 10);
    fill_tensor_rand(weights_host, -10, 10);
    fill_tensor_rand(bias_dev, -10, 10);
    bias_host.copy_from(bias_dev);

    Context<NV> ctx(0, 0, 1);
    auto stream = ctx.get_compute_stream();
    {
        Tensor<NVHX86> input_temp;
        input_temp.re_alloc(input_host.valid_shape(), AK_INT8);
        transpose_img_NCHW_2_NCHWC4((const char *) input_host.data(),
                                    (char *) input_temp.mutable_data(),
                                    input_host.num(),
                                    input_host.channel(),
                                    input_host.height(),
                                    input_host.width());

        input_dev.copy_from(input_temp);
    }
    bool use_1x1 = true;
    use_1x1 = use_1x1 && (kernel_h == 1);
    use_1x1 = use_1x1 && (kernel_w == 1);
    use_1x1 = use_1x1 && (dilation_h == 1);
    use_1x1 = use_1x1 && (dilation_w == 1);
    use_1x1 = use_1x1 && (stride_h == 1);
    use_1x1 = use_1x1 && (stride_w == 1);
    use_1x1 = use_1x1 && (pad_h == 0);
    use_1x1 = use_1x1 && (pad_w == 0);
    use_1x1 = use_1x1 && (group == 1);
    int ts = 100;
    SaberTimer<NV> timer;
    if (!use_1x1) {
        {
            Tensor<NVHX86> weight_temp;
            Tensor<NVHX86> weight_temp2;
            weight_temp.re_alloc(weights_host.valid_shape(), AK_INT8);
            weight_temp2.re_alloc(weights_host.valid_shape(), AK_INT8);
            transpose_filter_KCRS_2_CRSKC4(
                    (const char *) weights_host.data(),
                    (char *) weight_temp.mutable_data(),
                    (char *) weight_temp2.mutable_data(),
                    weights_host.num(),
                    weights_host.channel(),
                    weights_host.height(),
                    weights_host.width());
            weights_dev.copy_from(weight_temp2);
        }
        ConvParam<NV> param(group, pad_h, pad_w,
                            stride_h, stride_w,
                            dilation_h, dilation_w,
                            &weights_dev, &bias_dev);

        SaberDirectConv<AK_INT8> conv_direct;
        std::vector<Tensor<NV>*> inputs;
        std::vector<Tensor<NV>*> outputs;
        inputs.push_back(&input_dev);
        outputs.push_back(&output_dev);
        conv_direct.init(inputs, outputs, param, ctx);
        conv_direct.dispatch(inputs, outputs, param);

        cudaDeviceSynchronize();
        for (int i = 0; i < ts; ++i) {
            timer.start(ctx);
            conv_direct.dispatch(inputs, outputs, param);
            output_dev.record_event(ctx.get_compute_stream());
            output_dev.sync();
            timer.end(ctx);
        }
        printf("direct,%lf\n", timer.get_average_ms());

    } else {
        {
            Tensor<NVHX86> weight_temp;
            weight_temp.re_alloc(weights_host.valid_shape(), AK_INT8);
            transpose_img_NCHW_2_NCHWC4((const char *) weights_host.data(),
                                        (char *) weight_temp.mutable_data(),
                                        weights_host.num(),
                                        weights_host.channel(),
                                        weights_host.height(),
                                        weights_host.width());

            weights_dev.copy_from(weight_temp);
        }
        ConvParam<NV> param(group, pad_h, pad_w,
                            stride_h, stride_w,
                            dilation_h, dilation_w,
                            &weights_dev, &bias_dev);

        SaberGemmLikeConv<AK_INT8> conv_gemm;
        std::vector<Tensor<NV>*> inputs;
        std::vector<Tensor<NV>*> outputs;
        inputs.push_back(&input_dev);
        outputs.push_back(&output_dev);
        conv_gemm.init(inputs, outputs, param, ctx);
        conv_gemm.dispatch(inputs, outputs, param);

        cudaDeviceSynchronize();
        for (int i = 0; i < ts; ++i) {
            timer.start(ctx);
            conv_gemm.dispatch(inputs, outputs, param);
            output_dev.record_event(ctx.get_compute_stream());
            output_dev.sync();
            timer.end(ctx);
        }
        printf("gemm,%lf\n", timer.get_average_ms());
    }
    cudaDeviceSynchronize();
    output_host.copy_from(output_dev);
    cudaDeviceSynchronize();
}
#if 0
TEST(TestSaberFunc, test_saber_speed) {
    Env<NV>::env_init();
    Env<NVHX86>::env_init();

    std::vector<int> input_num_v{1};
    std::vector<int> in_channels_v{512};
    std::vector<int> out_channels_v{512};
    std::vector<int> height_v{14};
    std::vector<int> width_v{14};
    std::vector<int> kernel_h_v{3};
    std::vector<int> kernel_w_v{3};
    std::vector<int> pad_h_v{1};
    std::vector<int> pad_w_v{1};
    std::vector<int> stride_h_v{1};
    std::vector<int> stride_w_v{1};
    std::vector<int> dilation_h_v{1};
    std::vector<int> dilation_w_v{1};
    std::vector<int> group_v{1};
    printf("input_num,in_channels,out_channels,"
           "height,width,kernel_h,kernel_w,"
           "pad_h,pad_w,"
           "stride_h,stride_w,"
           "dilation_h,dilation_w,"
           "group,type,latency,\n");

    for (auto input_num : input_num_v)
    for (auto in_channels : in_channels_v)
    for (auto out_channels : out_channels_v)
    for (auto height : height_v)
    for (auto width : width_v)
    for (auto kernel_h: kernel_h_v)
    for (auto kernel_w: kernel_w_v)
    for (auto pad_h: pad_h_v)
    for (auto pad_w: pad_w_v)
    for (auto stride_h: stride_h_v)
    for (auto stride_w: stride_w_v)
    for (auto dilation_h: dilation_h_v)
    for (auto dilation_w: dilation_w_v)
    for (auto group: group_v) {
        printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
                input_num, in_channels, out_channels,
                height, width,
                kernel_h, kernel_w,
                pad_h, pad_w,
                stride_h, stride_w,
                dilation_h, dilation_w, group);

        test_saber_direct_speed(input_num,
                in_channels,
                out_channels,
                height,
                width,
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                group);

        printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
               input_num, in_channels, out_channels,
               height, width,
               kernel_h, kernel_w,
               pad_h, pad_w,
               stride_h, stride_w,
               dilation_h, dilation_w, group);

        test_saber_cudnn_speed(input_num,
                                in_channels,
                                out_channels,
                                height,
                                width,
                                kernel_h,
                                kernel_w,
                                pad_h,
                                pad_w,
                                stride_h,
                                stride_w,
                                dilation_h,
                                dilation_w,
                                group);
    }
}
#endif
#endif

int main(int argc, char* argv[]) {
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}