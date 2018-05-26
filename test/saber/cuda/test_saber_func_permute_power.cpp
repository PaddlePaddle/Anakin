
#include "core/context.h"
#include "test_saber_func_permute_power.h"
#include "funcs/permute_power.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncPermutePowerNV, test_func_constructor) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;


    //int img_num = 1;
    //int in_channels = 10;
    //int img_h = 480;
    //int img_w = 1440;
    int img_num = 1;
    int in_channels = 3;
    int img_h = 4;
    int img_w = 4;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape offset(0, 0, 0, 0);
    //Shape real_shape(img_num, img_h+1, img_w+1, in_channels + 1);
    Shape real_shape(img_num, in_channels + 1, img_h + 1, img_w + 1);
    TensorHf4 img_host_big;
    TensorDf4 img_dev_big;
    img_host_big.re_alloc(real_shape);
    img_dev_big.re_alloc(real_shape);
    auto data = img_host_big.mutable_data();

    for (int i = 0; i < img_host_big.size(); ++i) {
        data[i] = 0x7f & i;
    }

    img_dev_big.copy_from(img_host_big);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    //Shape valid_shape(img_num, img_h, img_w, in_channels);
    Shape valid_shape(img_num, in_channels, img_h, img_w);
    img_host.set_shape(valid_shape);
    img_dev.set_shape(valid_shape);
    //img_dev_big.copy_from(img_host_big);
    //img_dev.share_from(img_dev_big);
    //img_host.share_from(img_host_big);
    img_dev.share_sub_buffer(img_dev_big, \
                             valid_shape, offset);
    //img_host.re_alloc(valid_shape);


    TensorHf4 output_host_big;
    TensorDf4 output_dev_big;
    TensorHf4 output_host;
    TensorDf4 output_dev;

    // start Reshape & doInfer

    Context<NV> ctx1(0, 1, 1);
    std::vector<int> permute_order = {0, 2, 3, 1};
    PermuteParam<TensorDf4> permute_param(permute_order);
    //PowerParam<void> power_param(/*power*/1.0f, /*scale*/ float(1.0/255), /*shift*/0.0f);
    PowerParam<TensorDf4> power_param(/*power*/1.0f, /*scale*/ float(1), /*shift*/0.0f);

    PermutePowerParam<TensorDf4> param(permute_param, power_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    PermutePower<NV, AK_FLOAT> permute;
    permute.compute_output_shape(input, output, param);
    //output_dev.set_shape(output[0]->valid_shape());
    //output_host.set_shape(output[0]->valid_shape());
    //Shape diff(0, 1, 2, 3);
    //Shape out_real_shape = output[0]->valid_shape() + diff;
    //output_dev_big.re_alloc(out_real_shape);
    //output_host_big.re_alloc(out_real_shape);
    //output_dev.share_sub_buffer(output_dev_big, output[0]->valid_shape(), offset);
    //output_host.share_sub_buffer(output_host_big, output[0]->valid_shape(), offset);
    //output_dev.share_from(output_dev_big);
    output_host.re_alloc(output[0]->valid_shape());
    output_dev.re_alloc(output[0]->valid_shape());

    // init assume output tensor has been reshpaed by user.
    //permute.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    permute.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    permute(input, output, param, ctx1);
    output[0]->record_event(cuda_stream);
    output[0]->sync();
    //SaberTimer<NV> my_time;
    //my_time.start(ctx1);
    //for (int i = 0; i < 100; i++) {
    //    permute(input, output, param, ctx1);
    //    cudaEventSynchronize (event);
    //}
    //my_time.end(ctx1);
    //LOG(INFO)<<"permute_power cudnn aveage time"<<my_time.get_average_ms()/100;

    //PermutePower<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> permute_s;
    //permute_s.compute_output_shape(out_shape, input, param);
    //permute_s.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    ////permute.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);

    //permute_s(input, output, param, ctx1);
    //cudaEventSynchronize (event);
    //SaberTimer<NV> saber_time;
    //saber_time.start(ctx1);
    //for (int i = 0; i < 100; i++) {
    //    permute_s(input, output, param, ctx1);
    //    cudaEventSynchronize (event);
    //}
    //saber_time.end(ctx1);
    //LOG(INFO)<<"permute_power saber aveage time"<<saber_time.get_average_ms()/100;
    cudaDeviceSynchronize();
    output_host.copy_from(output_dev);
    print_tensor_host(img_host_big);
    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

