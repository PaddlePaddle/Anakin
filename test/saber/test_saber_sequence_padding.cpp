#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/sequence_padding.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
template <typename dtype, typename TargetType_D, typename TargetType_H>
void sequence_padding_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                      std::vector<Tensor<TargetType_H>*>& outputs,
                      SequencePaddingParam<TargetType_D>& param) {
    size_t len = inputs[0]->valid_size();
    dtype *input_data = (dtype*)inputs[0]->mutable_data();
    dtype *output_data = (dtype*)outputs[0]->mutable_data();
    int max_len = 0;
    auto seq_offset = inputs[0]->get_seq_offset()[0];
    int seq_num = seq_offset.size() - 1;
    int emb_size = inputs[0]->count_valid(1, inputs[0]->dims());
    for (int i = 0; i < seq_num; i++) {
        int cur_len = seq_offset[i+1] - seq_offset[i];
        max_len = cur_len > max_len ? cur_len : max_len;
    }

    Shape out_shape = inputs[0]->valid_shape();
    out_shape[0] = seq_num * max_len;
    outputs[0]->reshape(out_shape);
    for (size_t i = 0; i < seq_num; i++) {
        int start = i * max_len * emb_size;
        int cur_len = seq_offset[i+1] - seq_offset[i];
        int pad_start =  start + cur_len * emb_size;
        int pad_num = max_len - cur_len;
        memcpy(output_data + start, input_data + seq_offset[i] * emb_size, cur_len * emb_size * sizeof(dtype));
        if (pad_num > 0) {
            memset(output_data + pad_start, 0, pad_num * emb_size * sizeof(dtype));
        }
    }
    
    std::vector<int> out_offset;
    for (int i = 0; i < seq_num + 1; i++) {
        out_offset.push_back(i * max_len);
    }
    outputs[0]->set_seq_offset({out_offset});
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
    TestSaberBase<TargetType_D, TargetType_H, Dtype, SequencePadding, SequencePaddingParam> testbase(1, 1);
    for (auto seq_num : {4, 40}) {
        for (auto max_seq_len: {50}) {
            for (auto emb_size: {128, 256}) {
                std::vector<Tensor<TargetType_D>*> inputs;
                std::vector<int> seq_offset_0 = generate_sequence_offset(seq_num, max_seq_len);
                int word_num_0 = seq_offset_0.back();
                Tensor<TargetType_D>* input_0 = new Tensor<TargetType_D>(Shape({word_num_0, emb_size, 1, 1}), AK_FLOAT);
                fill_tensor_rand(*input_0, -1.f, 1.f);
                std::vector<std::vector<int>> vseq_offset_0 = {seq_offset_0};
				input_0->set_seq_offset(vseq_offset_0);
                inputs.push_back(input_0);
                testbase.add_custom_input(inputs);
                SequencePaddingParam<TargetType_D> param;
                testbase.set_param(param);
                testbase.run_test(sequence_padding_basic<float, TargetType_D, TargetType_H>, 0.00001, true, true);
                for (auto input: inputs) {
                    delete input;
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_sequence_padding) {

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

