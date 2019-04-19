#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/sequence_depadding.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
template <typename dtype, typename TargetType_D, typename TargetType_H>
void sequence_depadding_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                      std::vector<Tensor<TargetType_H>*>& outputs,
                      SequenceDePaddingParam<TargetType_D>& param) {
    dtype *input_data = (dtype*)inputs[0]->mutable_data();
    dtype *output_data = (dtype*)outputs[0]->mutable_data();
    auto pad_offset = inputs[0]->get_seq_offset()[0];
    auto src_offset = inputs[1]->get_seq_offset()[0];
    int seq_num = src_offset.size() - 1;
    int emb_size = inputs[0]->count_valid(1, inputs[0]->dims());

    for (size_t i = 0; i < seq_num; i++) {
        int src_len_i = src_offset[i+1] - src_offset[i];
        int pad_len_i = pad_offset[i+1] - pad_offset[i];
        CHECK_LE(src_len_i, pad_len_i) << "pad sequence length is bigger than source sequence length";
        memcpy(output_data + src_offset[i] * emb_size, input_data + i * pad_len_i * emb_size, src_len_i * emb_size * sizeof(dtype));
    }
}


void generate_sequence_offset(int seq_num, int max_seq_len,
    std::vector<int>& offset) {
    offset.clear();
    int cumsum = 0;
	offset.push_back(cumsum);
    for (int i = 0; i < seq_num; i++){
        int cur_len = rand() % max_seq_len + 1;
        cumsum += cur_len;
        offset.push_back(cumsum);
    }
}

int get_max_len(std::vector<int>& offset) {
    int max_len = 0;
    for (int i = 0; i < offset.size() - 1; i++) {
        int cur_len = offset[i+1] - offset[i];
        max_len = max_len < cur_len ? cur_len : max_len;
    }
    return max_len;
}

void generate_equal_step_offset(int seq_num, int max_seq_len, std::vector<int>& offset) {
    offset.clear();
    offset.push_back(0);
    for (int i = 0; i < seq_num; i++){
        offset.push_back((i+1)* max_seq_len);
    }
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {
    //test example
    TestSaberBase<TargetType_D, TargetType_H, Dtype, SequenceDePadding, SequenceDePaddingParam> testbase(2, 1);
    for (auto seq_num : {1, 3, 8}) {
        for (auto max_seq_len: {3, 30}) {
            for (auto emb_size: {5, 128, 256}) {
                std::vector<Tensor<TargetType_D>*> inputs;
                std::vector<int> seq_offset_1;
                std::vector<int> seq_offset_0;
                generate_sequence_offset(seq_num, max_seq_len, seq_offset_1);
                int max_len = get_max_len(seq_offset_1);
                generate_equal_step_offset(seq_num, max_len, seq_offset_0);
                int word_num_0 = seq_offset_1.back();
                Tensor<TargetType_D>* input_0 = new Tensor<TargetType_D>(Shape({seq_num * max_len, emb_size, 1, 1}), AK_FLOAT);
                Tensor<TargetType_D>* input_1 = new Tensor<TargetType_D>(Shape({word_num_0, emb_size, 1, 1}), AK_FLOAT);
                fill_tensor_rand(*input_0, -1.f, 1.f);
                std::vector<std::vector<int>> vseq_offset_0 = {seq_offset_0};
				input_0->set_seq_offset(vseq_offset_0);

                fill_tensor_rand(*input_1, -1.f, 1.f);
                std::vector<std::vector<int>> vseq_offset_1 = {seq_offset_1};
				input_1->set_seq_offset(vseq_offset_1);

                inputs.push_back(input_0);
                inputs.push_back(input_1);
                testbase.add_custom_input(inputs);
                SequenceDePaddingParam<TargetType_D> param;
                testbase.set_param(param);
                testbase.run_test(sequence_depadding_basic<float, TargetType_D, TargetType_H>, 0.00001, true, true);
                for (auto input: inputs) {
                    delete input;
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_sequence_depadding) {

#ifdef USE_CUDA
    //Init the test_base
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_model<AK_FLOAT, ARM, ARM>();
#endif
#ifdef AMD_GPU
    //    Env<AMD>::env_init();
    //    test_model<AK_FLOAT, AMD, AMDHX86>();
#endif
#ifdef USE_BM_PLACE
    //    Env<BM>::env_init();
    //    test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

