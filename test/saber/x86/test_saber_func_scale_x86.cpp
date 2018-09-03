
#include <vector>
#include "saber/core/context.h"
#include "saber/funcs/scale.h"
#include "test_saber_func_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "x86_test_common.h"

using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;
typedef Tensor<X86, AK_FLOAT, HW> Tensor2f;
typedef Tensor<X86, AK_FLOAT, W> Tensor1f;

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

void test(int n, int c, int h, int w, int axis, int num_axes, bool bias_term, int scale_dim) {
    int num_in = n;
    int ch_in = c;
    int h_in = h;
    int w_in = w;

    // LOG(INFO) << " input num:" << num_in << ", channel:" << ch_in << ", height:" << h_in << ", width:" << w_in;

    Shape shape_in(num_in, ch_in, h_in, w_in);

    Tensor4f src_in, dst_saber;
    src_in.re_alloc(shape_in);
    fill_tensor_host_rand(src_in, -0.5, 0.5);

    Context<X86> ctx_host;

    std::vector<Tensor4f*> input;
    std::vector<Tensor4f*> output;

    input.push_back(&src_in);
    output.push_back(&dst_saber);
    std::vector<float> scale_w;
    scale_w.resize(scale_dim);
    fill_vector_rand(scale_w);
    std::vector<float> scale_b;
    if (bias_term) {
        scale_b.resize(scale_dim);
        fill_vector_rand(scale_b);
    }

    ScaleParam<Tensor4f> param(scale_w, scale_b, bias_term, axis, num_axes);

    Scale<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> scale;
    scale.compute_output_shape(input, output, param);
    output[0]->re_alloc(output[0]->valid_shape());

    scale.init(input, output, param, SPECIFY, SABER_IMPL, ctx_host);

    scale(input, output, param, ctx_host);
    print_tensor_host(*output[0]);
    print_vector_data(scale_w);
    if (bias_term) {
        print_vector_data(scale_b);
    }
    print_tensor_host(*input[0]);
}


TEST(TestSaberFuncX86, test_tensor_scaleedding) {
    Env<X86>::env_init();

    LOG(INFO) << "case 1:"; 
    test(2, 2, 4, 4, 1, 1, true, 2);
    test(2, 2, 4, 4, 1, 1, false, 2);
    test(2, 2, 4, 4, 0, -1, true, 64);
    test(2, 2, 4, 4, 0, -1, false, 64);
    test(2, 2, 4, 4, 0, 0, true, 1);
    test(2, 2, 4, 4, 0, 0, false, 1);
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

