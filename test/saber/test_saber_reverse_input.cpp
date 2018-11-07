#include "saber/core/context.h"
#include "saber/funcs/reverse_input.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_base.h"
#include "test_saber_func.h"
#include <vector>

using namespace anakin::saber;



//native cpu version
template <typename dtype, typename TargetType_D, typename TargetType_H>
void reverse_sequence_cpu_base(const std::vector<Tensor<TargetType_H> *>& inputs,
                               std::vector<Tensor<TargetType_H> *>& outputs, EmptyParam<TargetType_D>& param) {

    int input_size = inputs.size();

    for (int input_id = 0; input_id < input_size; ++input_id) {
        std::vector<std::vector<int>> offset_vec = inputs[input_id]->get_seq_offset();
        std::vector<int> offset = offset_vec[offset_vec.size() - 1];
        const dtype* in = static_cast<const dtype*>(inputs[input_id]->data());
        dtype* out = static_cast<dtype*>(outputs[input_id]->mutable_data());

        for (int sequence_id = 0; sequence_id < offset.size() - 1; sequence_id++) {
            int start = offset[sequence_id];
            int end = offset[sequence_id + 1] - 1;

            for (int index = 0; index <= end - start; index++) {
                out[end - index] = in[start + index];
            }
        }
    }
}

template <typename Ttype>
void test_reverse_sequence() {
    Env<Ttype>::env_init();
    Context<Ttype> ctx_dev(0, 1, 1);
    EmptyParam<Ttype> param;
    Shape input_shape({5, 1, 1, 1});
    std::vector<int>offset = {0, 2, 5};
    Tensor<Ttype> input_tensor(input_shape);
    Tensor<Ttype> output_tensor(input_shape);
    fill_tensor_rand(input_tensor);
    input_tensor.set_seq_offset({offset});
    std::vector<Tensor<Ttype>*> inputs = {&input_tensor};
    std::vector<Tensor<Ttype>*> outputs = {&output_tensor};
    ReverseInput<Ttype, AK_FLOAT> reverse_input;
    reverse_input.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx_dev);
    SABER_CHECK(reverse_input.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape(), outputs[0]->get_dtype());
    SABER_CHECK(reverse_input(inputs, outputs, param, ctx_dev));
    outputs[0]->record_event(ctx_dev.get_compute_stream());
    outputs[0]->sync();
    print_tensor(input_tensor);
    print_tensor(output_tensor);
}

template <typename Ttype, typename Htype>
void test_base_reverse_sequence() {
    TestSaberBase<Ttype, Htype, AK_FLOAT, ReverseInput, EmptyParam> testbase;
    EmptyParam<Ttype> param;
    testbase.set_param(param);
    testbase.set_rand_limit(1, 12);
    Shape input_shape({5, 1, 1, 1});
    std::vector<int>offset = {0, 2, 5};

    Tensor<Ttype> input_tensor(input_shape);
    fill_tensor_rand(input_tensor);
    input_tensor.set_seq_offset({offset});
    std::vector<Tensor<Ttype>*> inputs = {&input_tensor};
    testbase.add_custom_input(inputs);
    testbase.run_test(reverse_sequence_cpu_base<float, Ttype, Htype>);
}

TEST(TestSaberFunc, test_op_reverse_input) {

#ifdef USE_X86_PLACE
    test_reverse_sequence<X86>();
    test_base_reverse_sequence<X86, X86>();
#endif
#ifdef NVIDIA_GPU
    test_reverse_sequence<NV>();
    test_base_reverse_sequence<NV, NVHX86>();
#endif


}


int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

