#include "saber/core/context.h"
#include "saber/funcs/reverse_sequence.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_base.h"
#include "test_saber_func.h"
#include <vector>

using namespace anakin::saber;



//native cpu version
template <typename dtype, typename TargetType_D, typename TargetType_H>
static void reverse_sequence_cpu_base(const std::vector<Tensor<TargetType_H> *> &inputs,
                                      std::vector<Tensor<TargetType_H> *> &outputs, EmptyParam<TargetType_D> &param) {

    int input_size=inputs.size();
    CHECK_EQ(input_size,1)<<"only support one input now";

    std::vector<std::vector<int>> offset_vec=inputs[0]->get_seq_offset();
    std::vector<int> offset=offset_vec[offset_vec.size()-1];
    const float* in= static_cast<const float*>(inputs[0]->data());
    float* out=static_cast<float*>(outputs[0]->mutable_data());
    int batch_size=offset.size()-1;
    int word_size=inputs[0]->valid_shape()[1];
    for (int i = 0; i < batch_size; i++) {
        int seq_len = offset[i + 1] - offset[i];
        int start_word_id=offset[i];
        for (int j = 0; j < seq_len; j++) {
            int output_offset = word_size * (start_word_id + seq_len - j - 1);
            int input_offset = word_size * (start_word_id + j);
            memcpy(out + output_offset, in + input_offset, word_size * sizeof(float));
        }
    }
}

template <typename Ttype>
void test_reverse_sequence(){
    Env<Ttype>::env_init();
    Context<Ttype> ctx_dev(0, 1, 1);
    EmptyParam<Ttype> param;
    Shape input_shape({5,3,1,1});
    std::vector<int>offset={0,2,5};
    Tensor<Ttype> input_tensor(input_shape);
    Tensor<Ttype> output_tensor(input_shape);
    fill_tensor_rand(input_tensor);
    input_tensor.set_seq_offset({offset});
    std::vector<Tensor<Ttype>*> inputs={&input_tensor};
    std::vector<Tensor<Ttype>*> outputs={&output_tensor};
    ReverseSequence<Ttype,AK_FLOAT> reverse_sequence;
    reverse_sequence.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx_dev);
    SABER_CHECK(reverse_sequence.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape(),outputs[0]->get_dtype());
    SABER_CHECK(reverse_sequence(inputs, outputs, param, ctx_dev));
    outputs[0]->record_event(ctx_dev.get_compute_stream());
    outputs[0]->sync();
    print_tensor(input_tensor);
    print_tensor(output_tensor);
}

template <typename Ttype,typename Htype>
void test_base_reverse_sequence(){
    TestSaberBase<Ttype, Htype, AK_FLOAT, ReverseSequence, EmptyParam> testbase;
    EmptyParam<Ttype> param;
    testbase.set_param(param);
    testbase.set_rand_limit(1, 12);
    Shape input_shape({5,3,1,1});
    std::vector<int>offset={0,2,5};

    Tensor<Ttype> input_tensor(input_shape);
    fill_tensor_rand(input_tensor);
    input_tensor.set_seq_offset({offset});
    std::vector<Tensor<Ttype>*> inputs={&input_tensor};
    testbase.add_custom_input(inputs);
    testbase.run_test(reverse_sequence_cpu_base<float, Ttype, Htype>);
}

TEST(TestSaberFunc, test_op_reverse_sequence) {

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

