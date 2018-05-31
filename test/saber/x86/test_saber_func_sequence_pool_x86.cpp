
#include <vector>
#include "saber/core/context.h"
#include "saber/funcs/sequence_pool.h"
#include "test_saber_func_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "x86_test_common.h"

using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;

void test(Tensor4f &src_in, SequencePoolType pool_type) {
    Tensor4f dst_saber;

    Context<X86> ctx_host;
    std::vector<Tensor4f*> input_v;
    std::vector<Tensor4f*> output_v;
    input_v.push_back(&src_in);
    output_v.push_back(&dst_saber);

    SequencePoolParam<Tensor4f> param_host(pool_type);
    SequencePool<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op_sequence_pool;
    op_sequence_pool.compute_output_shape(input_v, output_v, param_host);
    dst_saber.re_alloc(output_v[0]->valid_shape());
    fill_tensor_host_const(dst_saber, 0.f);
    op_sequence_pool.init(input_v, output_v, param_host, SPECIFY, SABER_IMPL, ctx_host);
    op_sequence_pool(input_v, output_v, param_host, ctx_host);

    print_tensor_host(dst_saber);
}

TEST(TestSaberFuncX86, test_sequence_pool) {
    int batch_num = 7;
    int m = 4;
    int n = 3;

    std::vector<int> lod = {0, 2, 5, 7};
    int num_in = batch_num;
    int ch_in = 1;
    int h_in = m;
    int w_in = n;
    Shape shape_in(num_in, ch_in, h_in, w_in);
    Tensor4f src_in;
    src_in.re_alloc(shape_in);
    fill_tensor_host_rand(src_in, 1.f, 2.f);
    src_in.set_seq_offset(lod);

    print_tensor_host(src_in);
    LOG(INFO) << "Average:";
    test(src_in, Sequence_pool_average);
    LOG(INFO) << "Sum:";
    test(src_in, Sequence_pool_sum);
    LOG(INFO) << "Sqrt:";
    test(src_in, Sequence_pool_sqrt);
    LOG(INFO) << "Max:";
    test(src_in, Sequence_pool_max);
    LOG(INFO) << "Last:";
    test(src_in, Sequence_pool_last);
    LOG(INFO) << "First:";
    test(src_in, Sequence_pool_first);


}

int main(int argc, const char** argv) {
    Env<X86>::env_init();
//    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

