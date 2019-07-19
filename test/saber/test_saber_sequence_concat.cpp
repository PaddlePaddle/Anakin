#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/sequence_concat.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
template <typename dtype, typename TargetType_D, typename TargetType_H>
void sequence_concat_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                      std::vector<Tensor<TargetType_H>*>& outputs,
                      SequenceConcatParam<TargetType_D>& param) {
    dtype *output_data = (dtype*)outputs[0]->mutable_data();
    int emb_size = inputs[0]->valid_size() / inputs[0]->num();
    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    for (int i = 1; i < inputs.size(); i++) {
        int cur_emb_size = inputs[i]->valid_size() / inputs[i]->num();
        int cur_seq_num  = inputs[i]->get_seq_offset()[0].size() - 1;
        CHECK_EQ(emb_size, cur_emb_size) << "sequence concat emb size must be the same";
        CHECK_EQ(seq_num, cur_seq_num) << "sequence concat seq num must be the same";
    }

    for (int i = 0; i < seq_num; i++) {
        for (int j = 0; j < inputs.size(); j++) {
            size_t cur_len = inputs[j]->get_seq_offset()[0][i+1] - inputs[j]->get_seq_offset()[0][i];

            const dtype *input_data = (const dtype*)inputs[j]->data() + inputs[j]->get_seq_offset()[0][i] * emb_size;
            memcpy(output_data, input_data, sizeof(dtype) * cur_len * emb_size);
            output_data += cur_len * emb_size;
        }
    }

    std::vector<std::vector<int>> out_offset;
    out_offset.resize(1);
    int seq_len = inputs[0]->get_seq_offset()[0].size() - 1;
    out_offset[0].push_back(0);
    int cur_off = 0;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < inputs.size(); j++) {
            cur_off += inputs[j]->get_seq_offset()[0][i + 1];
        }
        out_offset[0].push_back(cur_off);
    }
    outputs[0]->set_seq_offset(out_offset);
}

std::vector<int> generate_sequence_offset(int seq_num, int max_seq_len) {
    std::vector<int> offset;
    int cumsum = 0;
	offset.push_back(cumsum);
    for (int i = 0; i < seq_num; i++){
        int cur_len = rand() % max_seq_len + 1;
        cumsum += cur_len;
        offset.push_back(cumsum);
    }
    return offset;
}



template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {
    //test example
    //for (auto seq_num : {1, 2, 8}) {
    //    for (auto max_seq_len: {10, 16, 30}) {
    //        for (auto emb_size: {32, 128, 61}) {
    for (auto seq_num : {4, 40}) {
        for (auto max_seq_len: {50}) {
            for (auto emb_size: {128, 256}) {
                for (auto in_num: {2, 5}) {
                    TestSaberBase<TargetType_D, TargetType_H, Dtype, SequenceConcat, SequenceConcatParam> testbase(in_num, 1);
                    std::vector<Tensor<TargetType_D>*> inputs;
                    for (int i = 0; i < in_num; i++) {
                        std::vector<int> seq_offset_0 = generate_sequence_offset(seq_num, max_seq_len);
                        int word_num_0 = seq_offset_0.back();
                        Tensor<TargetType_D>* input_0 = new Tensor<TargetType_D>(Shape({word_num_0, emb_size, 1, 1}), AK_FLOAT);
                        //input_0.re_alloc(Shape({word_num_0, emb_size, 1, 1}), AK_FLOAT);
                        fill_tensor_rand(*input_0, -1.f, 1.f);
                        std::vector<std::vector<int>> vseq_offset_0 = {seq_offset_0};
				        input_0->set_seq_offset(vseq_offset_0);
                        inputs.push_back(input_0);
                    }
                    testbase.add_custom_input(inputs);
                    SequenceConcatParam<TargetType_D> param;
                    testbase.set_param(param);
                    testbase.run_test(sequence_concat_basic<float, TargetType_D, TargetType_H>, 0.00001, true, true);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_sequence_concat) {

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

