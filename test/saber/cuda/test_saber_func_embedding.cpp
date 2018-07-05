#include "core/context.h"
#include "funcs/embedding.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor& t0) {

    LOG(INFO) << name << " valid shape is ["
              << t0.valid_shape()[0] << ", "
              << t0.valid_shape()[1] << ", "
              << t0.valid_shape()[2] << ", "
              << t0.valid_shape()[3] << "].";

    LOG(INFO) << name << " real shape is ["
              << t0.shape()[0] << ", "
              << t0.shape()[1] << ", "
              << t0.shape()[2] << ", "
              << t0.shape()[3] << "].";

    LOG(INFO) << name << " offset is ["
              << t0.offset()[0] << ", "
              << t0.offset()[1] << ", "
              << t0.offset()[2] << ", "
              << t0.offset()[3] << "].";
}

TEST(TestSaberFuncNV, test_func_constructor) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int img_num = 10;
    int in_channels = 1;
    int img_h = 1;
    int img_w = 1;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = std::rand() % 128;
    }

    img_dev.copy_from(img_host);
    TensorDf4 output_dev;

    // start Reshape & doInfer


    Context<NV> ctx1(0, 1, 1);

    int num_word = 128;
    int emb_dim = 10;
    int padding_idx = -1;
    Shape weight_shape = {1, 1, num_word, emb_dim};
    TensorDf4 weight_d(weight_shape);
    TensorHf4 weight_h(weight_shape);
    fill_tensor_host_rand(weight_h, -0.5, 0.5);
    weight_d.copy_from(weight_h);
    cudaDeviceSynchronize();
    EmbeddingParam<TensorDf4> param(num_word, emb_dim, -1, &weight_d);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Embedding<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> emb;
    emb.compute_output_shape(input, output, param);
    output_dev.re_alloc(output[0]->valid_shape());

    // init assume output tensor has been reshpaed by user.
    emb.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    emb(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    output_dev.reshape({1, 1, -1, 10});
    print_tensor_device(output_dev);
    print_tensor_device(img_dev);
    print_tensor_device(weight_d);
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
}

int main(int argc, const char** argv) {
    Env<NV>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

