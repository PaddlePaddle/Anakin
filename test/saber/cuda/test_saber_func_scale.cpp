#include "core/context.h"
#include "funcs/scale.h"
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

template <typename Dtype>
void fill_vector_rand(std::vector<Dtype>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = rand() *1.0f/RAND_MAX - 0.5;
    }
}
template<typename Dtype>
void print_vector_data(std::vector<Dtype>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        printf("%d, %f\n", i, vec[i]);
    }
}


void test_scale(int n, int c, int h, int w, int axis, int num_axes, bool bias_term, int scale_dim) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int img_num = n;
    int in_channels = c;
    int img_h = h;
    int img_w = w;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);
    fill_tensor_host_rand(img_host, -0.5, 0.5);
    img_dev.copy_from(img_host);

    TensorDf4 output_dev;

    // start Reshape & doInfer


    Context<NV> ctx1(0, 1, 1);
    std::vector<float> scale_w;
    std::vector<float> scale_b;
    scale_w.resize(scale_dim);
    fill_vector_rand(scale_w);
    if (bias_term) {
        scale_b.resize(scale_dim);
        fill_vector_rand(scale_b);
    }

    ScaleParam<TensorDf4> param(scale_w, scale_b, bias_term, axis, num_axes);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Scale<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> scale;
    scale.compute_output_shape(input, output, param);
    output_dev.re_alloc(output[0]->valid_shape());

    // init assume output tensor has been reshpaed by user.
    scale.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    scale(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    print_tensor_device(output_dev);
    print_tensor_device(img_dev);
    print_vector_data(scale_w);
    if (bias_term) {
        print_vector_data(scale_b);
    }
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
}


TEST(TestSaberFuncNV, test_func_constructor_elt) {
 test_scale(2, 2, 4, 4, 1, 1, true, 2);
 test_scale(2, 2, 4, 4, 1, 1, false, 2);
 test_scale(2, 2, 4, 4, 0, -1, true, 64);
 test_scale(2, 2, 4, 4, 0, -1, false, 64);
 test_scale(2, 2, 4, 4, 0, 0, true, 1);
 test_scale(2, 2, 4, 4, 0, 0, false, 1);

}


int main(int argc, const char** argv) {
    Env<NV>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

