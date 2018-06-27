#include <time.h>
#include <stdio.h>
#include <vector>

#include "saber/core/context.h"
#include "saber/funcs/softmax.h"
#include "test_saber_func_softmax_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "x86_test_common.h"
using namespace anakin::saber;
using namespace std;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;

void test(int num, int channel) {
    int num_in = num;
    int ch_in = channel;
    int h_in = 1;
    int w_in = 1;
    int d = ch_in * h_in * w_in;

    LOG(INFO) << " input size, num=" << num_in
              << ", channel=" << ch_in
              << ", height=" << h_in
              << ", width=" << w_in;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out(num_in, ch_in, h_in, w_in);

    Tensor4f src_in, dst_saber, dst_ref;
    src_in.re_alloc(shape_in);
    fill_tensor_host_rand(src_in, 0.1f, 10.f);

    std::vector<float> ref_sum(num_in, 0.f);
    for (int i = 0; i < num_in; i++) {
        for (int j = 0; j < d; j++) {
            ref_sum[i] += expf(src_in.data()[i * d + j]);
        }
    }

    LOG(INFO) << "reference src_data: ";
//    print_tensor_host(src_in);
    dst_ref.re_alloc(shape_out);
    for (int i = 0; i < num_in; ++i) {
        for (int j = 0; j < d; j++) {
            dst_ref.mutable_data()[i * d + j]
                    = expf(src_in.data()[i * d + j]) / ref_sum[i];
        }
    }
     LOG(INFO) << "reference dst_data: ";
//     print_tensor_host(dst_ref);

    // saber dst
    Context<X86> ctx_host;

    std::vector<Tensor4f*> input_softmax;
    std::vector<Tensor4f*> output_softmax;

    input_softmax.push_back(&src_in);

    dst_saber.re_alloc(shape_out);
    output_softmax.push_back(&dst_saber);

    Softmax<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_softmax;
    SoftmaxParam<Tensor4f> smx_pm;

    op_softmax.init(input_softmax, output_softmax, smx_pm, SPECIFY, SABER_IMPL, ctx_host);

    op_softmax(input_softmax, output_softmax, smx_pm, ctx_host);
    // LOG(INFO) << "saber dst_data: ";
    // print_tensor_host(*output_softmax[0]);

    bool flag = compare_tensor(dst_saber, dst_ref, 1e-6);
    if (flag == true) {
        LOG(INFO) << "Test Passed";
    }
    else {
        LOG(ERROR) << "Test Failed";
    }
}


TEST(TestSaberSoftmaxX86, test_tensor_softmax) {
    Env<X86>::env_init();

    LOG(INFO) << "case 1:";
    test(1, 16);
    LOG(INFO) << "case 2:";
    test(1, 25);
    LOG(INFO) << "case 3:";
    test(1, 1000);
    LOG(INFO) << "case 4:";
    test(2, 1000);
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

