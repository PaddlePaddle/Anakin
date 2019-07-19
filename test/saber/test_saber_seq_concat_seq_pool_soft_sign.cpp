#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/seq_concat_seq_pool_soft_sign.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;

template <typename dtype, typename TargetType_D, typename TargetType_H>
void seq_concat_seq_pool_soft_sign_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                    std::vector<Tensor<TargetType_H>*>& outputs,
                    SeqConcatSeqPoolSoftSignParam<TargetType_D>& param) {

    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    int emb_size = inputs[0]->valid_size() / inputs[0]->num();
    for (int i = 1; i < inputs.size(); i++) {
        int cur_emb_size = inputs[i]->valid_size() / inputs[i]->num();
        int cur_seq_num = inputs[i]->get_seq_offset()[0].size() - 1 ;
        CHECK_EQ(emb_size, cur_emb_size) << "emb size must be the same";
        CHECK_EQ(seq_num, cur_seq_num) << "seq num  must be the same";
    }

    outputs[0]->reshape(Shape({seq_num, emb_size, 1, 1}, Layout_NCHW));
    dtype *output_data = (dtype*)outputs[0]->mutable_data();
    std::vector<std::vector<int>> offset_vecs;
    for (int i = 0; i < inputs.size(); i++) {
        offset_vecs.push_back(inputs[i]->get_seq_offset()[0]);
    }
    dtype buf[emb_size];
    for (size_t i = 0; i < seq_num; i++) {
        memset(buf, 0, sizeof(dtype) * emb_size);
        for (int j = 0; j < inputs.size(); j++) {
            const dtype *in_data = (const dtype*)inputs[j]->data();
            for (int k = offset_vecs[j][i]; k < offset_vecs[j][i + 1]; k++) {
                int start = k * emb_size;
                for (int m = 0; m < emb_size; m++) {
                    buf[m] += in_data[k * emb_size + m];
                }
            }
        }

        for (int m = 0; m < emb_size; m++) {
            auto tmp = buf[m] > 0 ? buf[m] : -buf[m];
            output_data[i * emb_size + m]  = buf[m] / (1 + tmp);
        }
    }
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {
    int max_seq_len = 1;
    int emb_size = 256;
    for (auto input_size : {4}) {
        TestSaberBase<TargetType_D, TargetType_H, Dtype, SeqConcatSeqPoolSoftSign, SeqConcatSeqPoolSoftSignParam> testbase(input_size, 1);
        for (auto seq_num: {1}) {
            std::vector<std::vector<int>> seq_offset_vec;
            seq_offset_vec.resize(input_size);
            std::vector<Tensor<TargetType_D>*> input_vec;
            for (int i = 0; i < input_size; i++) {
                int num = 0;
                seq_offset_vec[i].push_back(num);
                for (int j = 0; j < seq_num; j++) {
                    //int len = std::rand() % max_seq_len;
                    int len = 1;
                    num += len;
                    seq_offset_vec[i].push_back(num);
                }
                std::vector<std::vector<int>> cur_seq_offset = {seq_offset_vec[i]};
                Shape shape({num, emb_size, 1, 1}, Layout_NCHW);
                Tensor<TargetType_D>* input = new Tensor<TargetType_D>(shape);
                input->set_seq_offset(cur_seq_offset);
                fill_tensor_rand(*input);
                input_vec.push_back(input);
            }
        //test example
            SoftSignParam<TargetType_D> soft_sign_param;
            SequenceConcatParam<TargetType_D> seq_concat_param;
            SequencePoolParam<TargetType_D> seq_pool_param(Sequence_pool_sum);
            SeqConcatSeqPoolSoftSignParam<TargetType_D> param(seq_concat_param, seq_pool_param, soft_sign_param);
            testbase.set_param(param);//set param
            testbase.add_custom_input(input_vec);
            testbase.run_test(seq_concat_seq_pool_soft_sign_basic<float, TargetType_D, TargetType_H>, 0.00001, false, true);//run test
            for (int i = 0; i < input_size; i++) {
                delete input_vec[i];
            }
        }
    }
}
TEST(TestSaberFunc, test_func_soft_sign) {

#ifdef USE_CUDA
    //Init the test_base
    //Env<NV>::env_init();
    //test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
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

