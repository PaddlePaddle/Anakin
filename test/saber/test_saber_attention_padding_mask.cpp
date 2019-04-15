#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/attention_padding_mask.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
template <typename dtype, typename TargetType_D, typename TargetType_H>
void attention_padding_mask_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                      std::vector<Tensor<TargetType_H>*>& outputs,
                      AttentionPaddingMaskParam<TargetType_D>& param) {

    auto src_offset = inputs[1]->get_seq_offset()[0];
    auto attn_offset = inputs[0]->get_seq_offset()[0];
    int src_len = inputs[1]->count_valid(1, inputs[1]->dims());
    int attn_seq_num = attn_offset.size() - 1;
    int src_seq_num = src_offset.size() - 1;
    int attn_seq_len = attn_offset[1];
    int src_seq_len = src_offset[1];
    CHECK_EQ(attn_seq_num % src_seq_num, 0) << "Missmatch batch size";

    size_t count = inputs[0]->valid_size();
    dtype *attn_data = (dtype*)inputs[0]->mutable_data();
    dtype *output_data = (dtype*)outputs[0]->mutable_data();
    memcpy(output_data, attn_data, count * sizeof(dtype));
    for (int i = 0; i < attn_seq_num; ++i) {
        for (int j = 0; j < attn_seq_len; ++j) {
            auto tmp_output_data = output_data + src_seq_len * (attn_seq_len * i + j);
            int src_seq_idx = i % src_seq_num;
            int cur_len = src_offset[src_seq_idx+1]-src_offset[src_seq_idx];
            for (int k = cur_len; k < src_seq_len; k++) {
                tmp_output_data[k] = param.mask;
            }
        }
    }
    //print_tensor(*inputs[0]);
    //print_tensor(*outputs[0]);
}

void generate_equal_step_offset(int seq_num, int max_seq_len, std::vector<int>& offset) {
    offset.clear();
    offset.push_back(0);
    for (int i = 0; i < seq_num; i++){
        offset.push_back((i+1)* max_seq_len);
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
        //printf("offset:%d, %d\n", i, cumsum);
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



template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {
    //test example
    TestSaberBase<TargetType_D, TargetType_H, Dtype, AttentionPaddingMask, AttentionPaddingMaskParam> testbase(2, 1);
    float scale = 0.8;
    for (auto seq_num : {1, 3}) {
        for (auto left_seq_len: {2}) {
            for (auto right_seq_len: {3}) {
                for (auto trans_a : {false}) {
                    for (auto trans_b: {true}) {
                        for (auto emb_size: {5}) {
                            std::vector<Tensor<TargetType_D>*> inputs;
                            std::vector<int> seq_offset_0;
                            std::vector<int> seq_offset_1;
                            generate_sequence_offset(seq_num, left_seq_len, seq_offset_1);
                            int max_len = get_max_len(seq_offset_1);
                            generate_equal_step_offset(seq_num, right_seq_len, seq_offset_0);
                            int word_num_0 = seq_offset_0.back();
                            int word_num_1 = seq_offset_1.back();
                            Tensor<TargetType_D>* input_0 = new Tensor<TargetType_D>(Shape({word_num_0, max_len, 1, 1}), AK_FLOAT);
                            Tensor<TargetType_D>* input_1 = new Tensor<TargetType_D>(Shape({word_num_1, emb_size, 1, 1}), AK_FLOAT);
                            fill_tensor_rand(*input_0, -1.f, 1.f);
                            fill_tensor_rand(*input_1, -1.f, 1.f);
                            std::vector<std::vector<int>> vseq_offset_0 = {seq_offset_0};
		    	            input_0->set_seq_offset(vseq_offset_0);
                            std::vector<std::vector<int>> vseq_offset_1 = {seq_offset_1};
		    	            input_1->set_seq_offset(vseq_offset_1);
                            inputs.push_back(input_0);
                            inputs.push_back(input_1);
                            testbase.add_custom_input(inputs);
                            AttentionPaddingMaskParam<TargetType_D> param(-900000000.f, 12800001);
                            testbase.set_param(param);
                            testbase.run_test(attention_padding_mask_basic<float, TargetType_D, TargetType_H>, 0.00001, true, true);
                            for (auto input: inputs) {
                                delete input;
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_attention_padding_mask) {

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

