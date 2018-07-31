
#include <vector>
#include "tensor_op.h"
#include "funcs/sequence_conv.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"
#include "stdio.h"
#include "x86_test_common.h"
#include "test_saber_func_x86.h"



using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
void test_func_saber_sequence_conv_x86() {


    Context<X86> ctx_dev(0, 1, 1);
    std::vector<int> offsets = {0, 3, 7};

    int hidden_size = 2;
    int context_length = 3; //kerner size
    int feature_size = 5;
    int word_num = offsets[offsets.size() - 1];
    Shape shape_in(word_num, hidden_size, 1, 1);
    Shape shape_filter(1, 1, context_length * hidden_size, feature_size);

    TensorHf4 data_in;
    TensorHf4 data_out;
    TensorHf4 data_filter;
    data_filter.re_alloc(shape_filter);
    data_in.re_alloc(shape_in);
    fill_tensor_host_seq(data_filter);
    fill_tensor_host_seq(data_in);
    data_in.set_seq_offset(offsets);

    SequenceConvParam<TensorHf4> param(&data_filter, 3, -1);
    SequenceConv<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> dev_seq_conv;
    std::vector<TensorHf4*> input_dev_4d;
    std::vector<TensorHf4*> output_dev_4d;
    input_dev_4d.push_back(&data_in);
    output_dev_4d.push_back(&data_out);

    dev_seq_conv.compute_output_shape(input_dev_4d, output_dev_4d, param);
    LOG(INFO) << "shape of output =" << data_out.valid_shape().data()[0] << ","
              << data_out.valid_shape().data()[1] << ","
              << data_out.valid_shape().data()[2] << ","
              << data_out.valid_shape().data()[3];
    data_out.re_alloc(data_out.valid_shape());
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());

    SABER_CHECK(dev_seq_conv.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev));
    dev_seq_conv(input_dev_4d, output_dev_4d, param, ctx_dev);

    for (int i = 0; i < word_num * feature_size; i++) {
        printf("[%d] = %f\n", i, data_out.data()[i]);
    }

    //    return;
}

TEST(TestSaberFuncX86, test_func_saber_sequence_conv) {

    test_func_saber_sequence_conv_x86();

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    Env<X86>::env_init();

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
