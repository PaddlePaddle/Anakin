
#include <vector>
#include "saber/core/context.h"
#include "saber/funcs/crf_decoding.h"
#include "test_saber_func_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/timer.h"
#include "x86_test_common.h"

using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;

void test(Tensor4f &src_in, Tensor4f &weights, int test) {
    Tensor4f dst_saber;

    Context<X86> ctx_host;
    std::vector<Tensor4f*> input_v;
    std::vector<Tensor4f*> output_v;
    input_v.push_back(&src_in);
    output_v.push_back(&dst_saber);

    CrfDecodingParam<Tensor4f> param_host(&weights);
    CrfDecoding<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_crf_decoding;
    op_crf_decoding.compute_output_shape(input_v, output_v, param_host);
    dst_saber.re_alloc(output_v[0]->valid_shape());
    fill_tensor_host_const(dst_saber, 0.f);
    op_crf_decoding.init(input_v, output_v, param_host, SPECIFY, SABER_IMPL, ctx_host);
    SaberTimer<X86> timer;
    timer.start(ctx_host);
    op_crf_decoding(input_v, output_v, param_host, ctx_host);
    timer.end(ctx_host);
    LOG(INFO) << "elapse time: " << timer.get_average_ms() << " ms";
//    print_tensor_host(dst_saber);
}

TEST(TestSaberFuncX86, test_crf_decoding) {
    int n = 400;
    int d = 400;

    std::vector<int> lod = {0, 2, 400};
    int num_in = n;
    int ch_in = d;
    int h_in = 1;
    int w_in = 1;
    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape weight_shape(d+2, d, 1, 1);
    Tensor4f src_in;
    Tensor4f weight_host;
    src_in.re_alloc(shape_in);
    weight_host.re_alloc(weight_shape);
    fill_tensor_host_rand(src_in, 1.f, 2.f);
    src_in.set_seq_offset(lod);

    fill_tensor_host_rand(weight_host, 1.f, 2.f);
//    print_tensor_host(src_in);
    LOG(INFO) << "crf decoding:";
    test(src_in,  weight_host, 0);

}

int main(int argc, const char** argv) {
    Env<X86>::env_init();
//    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

