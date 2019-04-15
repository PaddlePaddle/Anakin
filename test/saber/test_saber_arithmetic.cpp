#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/arithmetic.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
int active = 1;
int num_in = 1;
int ch_in = 2;
int h_in = 3;
int w_in = 5;
template <typename dtype, typename TargetType_D, typename TargetType_H>
void arithmetic_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                      std::vector<Tensor<TargetType_H>*>& outputs, ArithmeticParam<TargetType_D>& param) {
    const dtype *input_data_0 = (const dtype*)inputs[0]->data();
    const dtype *input_data_1 = (const dtype*)inputs[1]->data();
    dtype *output_data = (dtype*)outputs[0]->mutable_data();
    auto seq_offset_0 = inputs[0]->get_seq_offset()[0];
    auto seq_offset_1 = inputs[1]->get_seq_offset()[0];
    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    int inner_size = inputs[0]->count_valid(1, inputs[0]->dims());
    

    // out[j] = input_0[j] + input_1[j] if j < count_0 && j < count_1;
    // out[j] = input_0[j] if j < count_0 && j >= count_1;
    if (param.op_type == SUM) {
        size_t len = inputs[0]->valid_size();
        for (int i = 0; i < seq_num; i++) {
            int len_0 = (seq_offset_0[i+1] - seq_offset_0[i]) * inner_size;
            int len_1 = (seq_offset_1[i+1] - seq_offset_1[i]) * inner_size; 
            auto input_0 = input_data_0 + seq_offset_0[i] * inner_size;
            auto input_1 = input_data_1 + seq_offset_1[i] * inner_size;
            auto out = output_data + seq_offset_0[i] * inner_size;
            if (len_0 > len_1) {
                for (int j = 0; j < len_1; j++) {
                    out[j] = input_0[j] + input_1[j];
                }
                for (int j = len_1; j < len_0; j++) {
                    out[j] = input_0[j];
                }
            } else {
                for (int j = 0; j < len_0; j++) {
                    out[j] = input_0[j] + input_1[j];
                }
            }
            
        }
    }

    // out[j] = input_0[j] - input_1[j] if j < count_0 && j < count_1;
    // out[j] = input_0[j] if j < count_0 && j >= count_1;
    if (param.op_type == SUB) {
        size_t len = inputs[0]->valid_size();
        for (int i = 0; i < seq_num; i++) {
            int len_0 = (seq_offset_0[i+1] - seq_offset_0[i]) * inner_size;
            int len_1 = (seq_offset_1[i+1] - seq_offset_1[i]) * inner_size;
            auto input_0 = input_data_0 + seq_offset_0[i] * inner_size;
            auto input_1 = input_data_1 + seq_offset_1[i] * inner_size;
            auto out = output_data + seq_offset_0[i] * inner_size;
            if (len_0 > len_1) {
                for (int j = 0; j < len_1; j++) {
                    out[j] = input_0[j] - input_1[j];
                }
                for (int j = len_1; j < len_0; j++) {
                    out[j] = input_0[j];
                }
            } else {
                for (int j = 0; j < len_0; j++) {
                    out[j] = input_0[j] - input_1[j];
                }
            }
        }
    }
    // out[j] = input_0[j] * input_1[j] if j < count_0 && j < count_1;
    // out[j] = input_0[j] if j < count_0 && j >= count_1;
    if (param.op_type == MUL) {
        size_t len = inputs[0]->valid_size();
        for (int i = 0; i < seq_num; i++) {
            int len_0 = (seq_offset_0[i+1] - seq_offset_0[i]) * inner_size;
            int len_1 = (seq_offset_1[i+1] - seq_offset_1[i]) * inner_size;
            auto input_0 = input_data_0 + seq_offset_0[i] * inner_size;
            auto input_1 = input_data_1 + seq_offset_1[i] * inner_size;
            auto out = output_data + seq_offset_0[i] * inner_size;
            if (len_0 > len_1) {
                for (int j = 0; j < len_1; j++) {
                    out[j] = input_0[j] * input_1[j];
                }
                for (int j = len_1; j < len_0; j++) {
                    out[j] = input_0[j];
                }
            } else {
                for (int j = 0; j < len_0; j++) {
                    out[j] = input_0[j] * input_1[j];
                }
            }
        }
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
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
    TestSaberBase<TargetType_D, TargetType_H, Dtype, Arithmetic, ArithmeticParam> testbase(2, 1);
    //test example
    for (auto seq_num : {1, 2, 8}) {
        for (auto max_seq_len: {10, 16, 30}) {
            for (auto emb_size: {32, 128, 61}) {
                for (auto op_type : {SUM, SUB, MUL}) {
                    std::vector<int> seq_offset_0 = generate_sequence_offset(seq_num, max_seq_len);
                    std::vector<int> seq_offset_1 = generate_sequence_offset(seq_num, max_seq_len);
                    int word_num_0 = seq_offset_0.back();
                    int word_num_1 = seq_offset_1.back();
                    Tensor<TargetType_D> input_0;
                    Tensor<TargetType_D> input_1;
                    input_0.re_alloc(Shape({word_num_0, emb_size, 1, 1}), AK_FLOAT);
                    input_1.re_alloc(Shape({word_num_1, emb_size, 1, 1}), AK_FLOAT);
                    fill_tensor_rand(input_0, -1.f, 1.f);
                    fill_tensor_rand(input_1, -1.f, 1.f);

                    std::vector<std::vector<int>> vseq_offset_0 = {seq_offset_0};
                    std::vector<std::vector<int>> vseq_offset_1 = {seq_offset_1};
					input_0.set_seq_offset(vseq_offset_0);
                    input_1.set_seq_offset(vseq_offset_1);
                    std::vector<Tensor<TargetType_D>*> inputs;
                    inputs.push_back(&input_0);
                    inputs.push_back(&input_1);
                    testbase.add_custom_input(inputs);
                    ArithmeticParam<TargetType_D> param(op_type);
                    testbase.set_param(param);
                    testbase.run_test(arithmetic_basic<float, TargetType_D, TargetType_H>, 0.00001, true, true);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_arithmetic) {

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

