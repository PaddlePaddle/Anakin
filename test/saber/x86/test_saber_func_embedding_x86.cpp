
#include <vector>
#include "saber/core/context.h"
#include "saber/funcs/embedding.h"
#include "test_saber_func_x86.h"
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

    Tensor4f src_in, dst_saber, weight;
    src_in.re_alloc(shape_in);
    auto data = src_in.mutable_data();
    for (int i = 0; i < src_in.size(); ++i) {
        data[i] = std::rand() % 128;
    }

    Context<X86> ctx_host;

    std::vector<Tensor4f*> input;
    std::vector<Tensor4f*> output;

    input.push_back(&src_in);

    int word_num = 128;
    int emb_dim = 10;
    int padding_idx = -1;
    weight.re_alloc({128, 1, 1, 10});
    fill_tensor_host_rand(weight, -0.5, 0.5);
    output.push_back(&dst_saber);
    

    EmbeddingParam<Tensor4f> param(word_num, emb_dim, padding_idx, &weight);

    Embedding<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> emb;
    emb.compute_output_shape(input, output, param);
     output[0]->set_shape({input[0]->valid_size(), emb_dim, 1, 1});
    output[0]->re_alloc(output[0]->valid_shape());

    emb.init(input, output, param, SPECIFY, SABER_IMPL, ctx_host);

    emb(input, output, param, ctx_host);
    output[0]->reshape({1, 1, -1, emb_dim});
    print_tensor_host(*output[0]);
    print_tensor_host(weight);
    print_tensor_host(*input[0]);

}


TEST(TestSaberFuncX86, test_tensor_embedding) {
    Env<X86>::env_init();

    LOG(INFO) << "case 1:"; 
    test(10, 1, 1, 1);
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

