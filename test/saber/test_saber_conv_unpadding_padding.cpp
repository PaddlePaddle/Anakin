#include "core/context.h"
#include "funcs/concat.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>
#include "funcs/conv_unpadding_padding.h"
using namespace anakin::saber;

template <typename Htype, typename Ttype>
void test_model() {
    Env<Htype>::env_init();
    Env<Ttype>::env_init();
    Shape shape_in({5, 1, 3, 5});
    std::vector<int> offset = {0, 1, 3, 6, 10, 15};
    Tensor<Htype> host_in(shape_in);
    Tensor<Ttype> target_in(shape_in);
    Tensor<Ttype> target_out(shape_in);
    fill_tensor_const(host_in, 1);
    target_in.copy_from(host_in);
    target_in.set_seq_offset({offset});
    std::vector<Tensor<Ttype>*> inputs;
    std::vector<Tensor<Ttype>*> outputs;
    inputs.push_back(&target_in);
    outputs.push_back(&target_out);

    ConvUnpaddingPaddingParam<Ttype> param(1, 1);
    ConvUnpaddingPadding<Ttype, AK_FLOAT> op;
    Context<Ttype> ctx_dev(0, 0, 1);
    SABER_CHECK(op.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx_dev));
    SABER_CHECK(op.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape(), outputs[0]->get_dtype());
    SABER_CHECK(op(inputs, outputs, param, ctx_dev));
    outputs[0]->record_event(ctx_dev.get_compute_stream());
    outputs[0]->sync();
    print_tensor(target_out);

}

TEST(TestSaberFunc, test_func_concat) {

#ifdef USE_CUDA
    //Init the test_base
    test_model<NVHX86, NV >();
#endif


}

int main(int argc, const char** argv) {

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

