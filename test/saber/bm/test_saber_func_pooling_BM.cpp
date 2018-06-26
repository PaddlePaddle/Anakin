#include "core/context.h"
#include "funcs/pooling.h"
#include "test_saber_func_BM.h"
#include "tensor_op.h"
#include "saber_types.h"
#include "funcs/timer.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncBM, test_func_pooling) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<BM> BM_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 4;
    int img_h = 800;
    int img_w = 1440;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);

    TensorHf4 output_host;
    TensorDf4 output_dev;

    // start Reshape & doInfer

    Context<BM> ctx1(0, 1, 1);
    int window_h = 2;
    int window_w = 2;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    LOG(INFO) << " img_num: " << img_num;
    LOG(INFO) << " in_channels: " << in_channels;
    LOG(INFO) << " img_h: " << img_h;
    LOG(INFO) << " img_w: " << img_w;
    LOG(INFO) << " window_h: " << window_h;
    LOG(INFO) << " window_w: " << window_w;
    LOG(INFO) << " pad_h: " << pad_h;
    LOG(INFO) << " pad_w: " << pad_w;
    LOG(INFO) << " stride_h: " << stride_h;
    LOG(INFO) << " stride_w: " << stride_w;

    PoolingParam<TensorDf4> param(window_h, window_w, pad_h, pad_w
                                  , stride_h, stride_w, Pooling_max);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Pooling<BM, AK_BM, AK_BM, AK_BM, NCHW> pooling;
    pooling.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    pooling.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling(input, output, param, ctx1);

    SaberTimer<BM> t1;
    int ts = 1000;

    for (int i = 0; i < ts; ++i) {
        t1.start(ctx1);
        pooling(input, output, param, ctx1);
        output[0]->sync();
        t1.end(ctx1);
    }

    output_dev.sync();
    LOG(INFO) << " average time: " << t1.get_average_ms() << " ms";
    LOG(INFO) << " tile 10% time: " << t1.get_tile_time(10) << " ms";
    LOG(INFO) << " tile 50% time: " << t1.get_tile_time(50) << " ms";
    LOG(INFO) << " tile 90% time: " << t1.get_tile_time(90) << " ms";
    LOG(INFO) << " tile 95% time: " << t1.get_tile_time(95) << " ms";
    LOG(INFO) << " tile 99% time: " << t1.get_tile_time(99) << " ms";
}

TEST(TestSaberFuncBM, test_pooling_result) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<BM> BM_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);

    TensorDf4 output_dev;

    // start Reshape & doInfer

    Context<BM> ctx1(0, 1, 1);
    int window_h = 2;
    int window_w = 2;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;

    LOG(INFO) << " img_num: " << img_num;
    LOG(INFO) << " in_channels: " << in_channels;
    LOG(INFO) << " img_h: " << img_h;
    LOG(INFO) << " img_w: " << img_w;
    LOG(INFO) << " window_h: " << window_h;
    LOG(INFO) << " window_w: " << window_w;
    LOG(INFO) << " pad_h: " << pad_h;
    LOG(INFO) << " pad_w: " << pad_w;
    LOG(INFO) << " stride_h: " << stride_h;
    LOG(INFO) << " stride_w: " << stride_w;

    PoolingParam<TensorDf4> param(window_h, window_w, pad_h, pad_w
                                  , stride_h, stride_w, Pooling_max);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Pooling<BM, AK_BM> pooling;
    pooling.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    pooling.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling(input, output, param, ctx1);

    output_dev.sync();
    print_tensor_device(output_dev);
}

TEST(TestSaberFuncBM, test_pooling_shared_buffer) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<BM> BM_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);

    TensorDf4 t0;
    TensorDf4 t1;
    Shape img_s_sub(img_num, in_channels, img_h / 2, img_w / 2);

    t0.share_sub_buffer(img_dev, img_s_sub, {0, 0, 0, 0});
    t1.share_sub_buffer(img_dev, img_s_sub, {0, 0, 4, 4});

    TensorDf4 output_dev;

    TensorDf4 out0;
    TensorDf4 out1;

    // start Reshape & doInfer

    Context<BM> ctx1(0, 1, 1);
    int window_h = 2;
    int window_w = 2;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;

    LOG(INFO) << " img_num: " << img_num;
    LOG(INFO) << " in_channels: " << in_channels;
    LOG(INFO) << " img_h: " << img_h;
    LOG(INFO) << " img_w: " << img_w;
    LOG(INFO) << " window_h: " << window_h;
    LOG(INFO) << " window_w: " << window_w;
    LOG(INFO) << " pad_h: " << pad_h;
    LOG(INFO) << " pad_w: " << pad_w;
    LOG(INFO) << " stride_h: " << stride_h;
    LOG(INFO) << " stride_w: " << stride_w;

    PoolingParam<TensorDf4> param(window_h, window_w, pad_h, pad_w
                                  , stride_h, stride_w, Pooling_max);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Pooling<BM, AK_BM> pooling;
    Pooling<BM, AK_BM> pooling0;
    Pooling<BM, AK_BM> pooling1;

    pooling.compute_output_shape(input,output,  param);

    Shape total_shape = output[0]->shape();

    output_dev.re_alloc(total_shape);
    Shape out_sub_shape = {total_shape[0], total_shape[1], total_shape[2] / 2, total_shape[3] / 2};

    out0.share_sub_buffer(output_dev, out_sub_shape, {0, 0, 0, 0});
    out1.share_sub_buffer(output_dev, out_sub_shape, {0, 0, out_sub_shape[2], out_sub_shape[3]});

    std::vector<TensorDf4*> input0, input1;
    std::vector<TensorDf4*> output0, output1;

    input0.push_back(&t0);
    input1.push_back(&t1);
    output0.push_back(&out0);
    output1.push_back(&out1);

    // init assume output tensor has been reshpaed by user.
    pooling0.init(input0, output0, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling0(input0, output0, param, ctx1);

    pooling1.init(input1, output1, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling1(input1, output1, param, ctx1);

    out0.sync();
    out1.sync();

    print_tensor_device(output_dev);
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

