
#include <vector>
#include "saber/core/context.h"
#include "saber/funcs/activation.h"
#include "test_saber_func_activation_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "x86_test_common.h"

using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;
typedef Tensor<X86, AK_FLOAT, HW> Tensor2f;
typedef Tensor<X86, AK_FLOAT, W> Tensor1f;

void test(int n, int c, int h, int w) {
    int num_in = n;
    int ch_in = c;
    int h_in = h;
    int w_in = w;

    // LOG(INFO) << " input num:" << num_in << ", channel:" << ch_in << ", height:" << h_in << ", width:" << w_in;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out(num_in, ch_in, h_in, w_in);

    Tensor4f src_in, dst_saber, dst_ref;
    src_in.re_alloc(shape_in);
    fill_tensor_host_rand(src_in);

    dst_ref.re_alloc(shape_out);
    for (int i = 0; i < dst_ref.size(); ++i) {
        dst_ref.mutable_data()[i] = (src_in.data()[i] >= 0) ? src_in.data()[i] : 0;
    }

    Context<X86> ctx_host;

    std::vector<Tensor4f*> input_relu;
    std::vector<Tensor4f*> output_relu;

    input_relu.push_back(&src_in);

    dst_saber.re_alloc(shape_out);
    output_relu.push_back(&dst_saber);

    ActivationParam<Tensor4f> param_host(Active_relu);

    Activation<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_relu;

    op_relu.init(input_relu, output_relu, param_host, SPECIFY, SABER_IMPL, ctx_host);

    op_relu(input_relu, output_relu, param_host, ctx_host);

    bool pass = compare_tensor<Tensor4f>(dst_ref, dst_saber, 1e-6);
    if (pass) {
        LOG(INFO) << "Test Passed";
    }
    else {
        LOG(ERROR) << "Test Failed";
    }
}

void test_stanh(int n, int c, int h, int w){
    int n_in = n;
    int c_in = c;
    int h_in = h;
    int w_in = w;
    float scale_a = 2.0f / 3.0f;
    float scale_b = 1.7159f;

    Shape shape_in(n_in, c_in, h_in, w_in);
    Shape shape_out(n_in, c_in, h_in, w_in);

    Tensor4f src, dst, dst_host;
    src.re_alloc(shape_in);

    float *src_ptr = src.mutable_data();
    for (int i = 0; i<src.valid_size(); i++) {
        src_ptr[i] = 0.12345f + (float)i*1e-4;
    }

    dst_host.re_alloc(shape_in);
    float *dst_host_ptr = dst_host.mutable_data();
    for (int i = 0; i< dst_host.valid_size(); i++) {
        dst_host_ptr[i] = 0.12345f + (float)i*1e-4;
        dst_host_ptr[i] = scale_b * tanh(scale_a * dst_host_ptr[i]);
    }


    Context<X86> ctx_host;

    std::vector<Tensor4f*> input_stanh;
    std::vector<Tensor4f*> output_stanh;

    input_stanh.push_back(&src);

    dst.re_alloc(shape_out);
    output_stanh.push_back(&dst);

    ActivationParam<Tensor4f> param_host(Active_stanh, scale_a, scale_b);

    Activation<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_stanh;

    op_stanh.init(input_stanh, output_stanh, param_host, SPECIFY, SABER_IMPL, ctx_host);

    op_stanh(input_stanh, output_stanh, param_host, ctx_host);

    const float *dst_ptr = dst.data();
    std::cout<< std::endl;
    std::cout<< "This tensor size is:" << dst.size()<< std::endl;
    for (int i = 0; i < dst.size(); i++) {
        if(i%5==0 && i)
            std::cout << std::endl;
        std::cout << dst_ptr[i] <<"  ";

    }

    bool pass = compare_tensor<Tensor4f>(dst_host, dst, 1e-6);
    if (pass) {
        LOG(INFO) << "Test Passed";
    }
    else {
        LOG(ERROR) << "Test Failed";
    }
}

void test_sigmoid(int n, int c, int h, int w){
    int n_in = n;
    int c_in = c;
    int h_in = h;
    int w_in = w;

    Shape shape_in(n_in, c_in, h_in, w_in);
    Shape shape_out(n_in, c_in, h_in, w_in);

    Tensor4f src, dst, dst_host;
    src.re_alloc(shape_in);
    fill_tensor_host_rand(src);

    dst_host.re_alloc(shape_in);
    float *dst_host_ptr = dst_host.mutable_data();
    float *src_ptr = src.mutable_data();
    for (int i = 0; i< dst_host.valid_size(); i++) {
        dst_host_ptr[i] = 1.0f / (exp(-src_ptr[i]) + 1.0f);
    }


    Context<X86> ctx_host;

    std::vector<Tensor4f*> input;
    std::vector<Tensor4f*> output;

    input.push_back(&src);

    dst.re_alloc(shape_out);
    output.push_back(&dst);

    ActivationParam<Tensor4f> param_host(Active_sigmoid);

    Activation<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_stanh;

    op_stanh.init(input, output, param_host, SPECIFY, SABER_IMPL, ctx_host);

    op_stanh(input, output, param_host, ctx_host);

    bool pass = compare_tensor<Tensor4f>(dst_host, dst, 1e-6);
    if (pass) {
        LOG(INFO) << "Test Passed";
    }
    else {
        LOG(ERROR) << "Test Failed";
    }
}

void test_tanh(int n, int c, int h, int w){
    int n_in = n;
    int c_in = c;
    int h_in = h;
    int w_in = w;

    Shape shape_in(n_in, c_in, h_in, w_in);
    Shape shape_out(n_in, c_in, h_in, w_in);

    Tensor4f src, dst, dst_host;
    src.re_alloc(shape_in);
    fill_tensor_host_rand(src);

    dst_host.re_alloc(shape_in);
    float *dst_host_ptr = dst_host.mutable_data();
    float *src_ptr = src.mutable_data();
    for (int i = 0; i< dst_host.valid_size(); i++) {
        dst_host_ptr[i] = tanh(src_ptr[i]);
    }


    Context<X86> ctx_host;

    std::vector<Tensor4f*> input;
    std::vector<Tensor4f*> output;

    input.push_back(&src);

    dst.re_alloc(shape_out);
    output.push_back(&dst);

    ActivationParam<Tensor4f> param_host(Active_tanh);

    Activation<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_tanh;

    op_tanh.init(input, output, param_host, SPECIFY, SABER_IMPL, ctx_host);

    op_tanh(input, output, param_host, ctx_host);

    bool pass = compare_tensor<Tensor4f>(dst_host, dst, 1e-6);
    if (pass) {
        LOG(INFO) << "Test Passed";
    }
    else {
        LOG(ERROR) << "Test Failed";
    }
}

void test_clipped_relu(int n, int c, int h, int w){
    int n_in = n;
    int c_in = c;
    int h_in = h;
    int w_in = w;

    float threshold = 0.85f;

    Shape shape_in(n_in, c_in, h_in, w_in);
    Shape shape_out(n_in, c_in, h_in, w_in);

    Tensor4f src, dst, dst_host;
    src.re_alloc(shape_in);
    fill_tensor_host_rand(src);

    dst_host.re_alloc(shape_in);
    float *dst_host_ptr = dst_host.mutable_data();
    float *src_ptr = src.mutable_data();
    for (int i = 0; i< dst_host.valid_size(); i++) {
        src_ptr[i] = src_ptr[i] > 0 ? src_ptr[i] : 0;
        dst_host_ptr[i] = src_ptr[i] < threshold ? src_ptr[i] : threshold;
    }


    Context<X86> ctx_host;

    std::vector<Tensor4f*> input;
    std::vector<Tensor4f*> output;

    input.push_back(&src);

    dst.re_alloc(shape_out);
    output.push_back(&dst);

    ActivationParam<Tensor4f> param_host(Active_clipped_relu);
    param_host.coef = threshold;

    Activation<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_clipped_relu;

    op_clipped_relu.init(input, output, param_host, SPECIFY, SABER_IMPL, ctx_host);

    op_clipped_relu(input, output, param_host, ctx_host);

    bool pass = compare_tensor<Tensor4f>(dst_host, dst, 1e-6);
    if (pass) {
        LOG(INFO) << "Test Passed";
    }
    else {
        LOG(ERROR) << "Test Failed";
    }
}

void test_elu(int n, int c, int h, int w){
    int n_in = n;
    int c_in = c;
    int h_in = h;
    int w_in = w;

    float coef = 0.37f;

    Shape shape_in(n_in, c_in, h_in, w_in);
    Shape shape_out(n_in, c_in, h_in, w_in);

    Tensor4f src, dst, dst_host;
    src.re_alloc(shape_in);
    fill_tensor_host_rand(src);

    dst_host.re_alloc(shape_in);
    float *dst_host_ptr = dst_host.mutable_data();
    float *src_ptr = src.mutable_data();
    for (int i = 0; i< dst_host.valid_size(); i++) {
        dst_host_ptr[i] = src_ptr[i] > 0 ? src_ptr[i] : coef * (exp(src_ptr[i]) - 1);
    }


    Context<X86> ctx_host;

    std::vector<Tensor4f*> input;
    std::vector<Tensor4f*> output;

    input.push_back(&src);

    dst.re_alloc(shape_out);
    output.push_back(&dst);

    ActivationParam<Tensor4f> param_host(Active_elu);
    param_host.coef = coef;

    Activation<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_elu;

    op_elu.init(input, output, param_host, SPECIFY, SABER_IMPL, ctx_host);

    op_elu(input, output, param_host, ctx_host);

    bool pass = compare_tensor<Tensor4f>(dst_host, dst, 1e-6);
    if (pass) {
        LOG(INFO) << "Test Passed";
    }
    else {
        LOG(ERROR) << "Test Failed";
    }
}

TEST(TestSaberActivationX86, test_tensor_activation) {
    Env<X86>::env_init();

    LOG(INFO) << "case 1:";
    test(1, 1, 1, 1024);
    LOG(INFO) << "case 2:";
    test(1, 1, 1024, 1024);
    LOG(INFO) << "case 3:";
    test(2, 2, 32, 32);
    LOG(INFO) << "case 4:";
    test(2, 32, 512, 512);

    LOG(INFO) << "test for stanh:";

    std::cout << "case 1:" << std::endl;
    test_stanh(1, 1, 1, 4);
    std::cout << "case 2:" << std::endl;
    test_stanh(1, 1, 20, 2);
    std::cout << "case 3:" << std::endl;
    test_stanh(2, 2, 32, 1);
    std::cout << "case 4:" << std::endl;
    test_stanh(2, 32, 2, 2);

}

TEST(TestSaberActivationX86, test_activation_sigmoid) {
    Env<X86>::env_init();

    LOG(INFO) << "case 1:";
    test_sigmoid(1, 1, 1, 1024);
    LOG(INFO) << "case 2:";
    test_sigmoid(1, 1, 1024, 1024);
    LOG(INFO) << "case 3:";
    test_sigmoid(2, 2, 32, 32);
    LOG(INFO) << "case 4:";
    test_sigmoid(2, 32, 512, 512);

}

TEST(TestSaberActivationX86, test_activation_tanh) {
    Env<X86>::env_init();

    LOG(INFO) << "case 1:";
    test_tanh(1, 1, 1, 1024);
    LOG(INFO) << "case 2:";
    test_tanh(1, 1, 1024, 1024);
    LOG(INFO) << "case 3:";
    test_tanh(2, 2, 32, 32);
    LOG(INFO) << "case 4:";
    test_tanh(2, 32, 512, 512);

}

TEST(TestSaberActivationX86, test_activation_clipped_relu) {
    Env<X86>::env_init();

    LOG(INFO) << "case 1:";
    test_clipped_relu(1, 1, 1, 1024);
    LOG(INFO) << "case 2:";
    test_clipped_relu(1, 1, 1024, 1024);
    LOG(INFO) << "case 3:";
    test_clipped_relu(2, 2, 32, 32);
    LOG(INFO) << "case 4:";
    test_clipped_relu(2, 32, 512, 512);

}

TEST(TestSaberActivationX86, test_activation_elu) {
    Env<X86>::env_init();

    LOG(INFO) << "case 1:";
    test_elu(1, 1, 1, 1024);
    LOG(INFO) << "case 2:";
    test_elu(1, 1, 1024, 1024);
    LOG(INFO) << "case 3:";
    test_elu(2, 2, 32, 32);
    LOG(INFO) << "case 4:";
    test_elu(2, 32, 512, 512);

}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
