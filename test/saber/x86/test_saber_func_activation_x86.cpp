
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

    Tensor4f src, dst;
    src.re_alloc(shape_in);

    float *ptr = src.mutable_data();
    for(int i = 0; i<src.size(); i++){
        ptr[i] = 3.27f + (float)i; 
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
    for(int i = 0; i < dst.size(); i++){
        std::cout << dst_ptr[i] <<"  ";
        if(i%5==0 && i)
            std::cout << std::endl;
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

    LOG(INFO) << "test for stanh:" 

    LOG(INFO) << "case 1:"; 
    test_stanh(1, 1, 1, 4);
    LOG(INFO) << "case 2:"; 
    test_stanh(1, 1, 20, 2);
    LOG(INFO) << "case 3:"; 
    test_stanh(2, 2, 32, 1);
    LOG(INFO) << "case 4:"; 
    test_stanh(2, 32, 2, 2);

}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

