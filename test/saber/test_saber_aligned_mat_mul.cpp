#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/aligned_mat_mul.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>
using namespace anakin::saber;

template<typename dtype>
void gemm(const dtype* data_A, const dtype* data_B, int M, int N, int K,
         bool trans_A, bool trans_B, dtype alpha, dtype beta, dtype* data_C) {
    if (trans_A && trans_B) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                dtype result = (dtype) 0;
                for (int k = 0; k < K; k++) {
                    result += data_A[k * M + m] * data_B[n * K  + k];
                }
                data_C[m * N + n] = alpha * result + beta * data_C[m * N + n];
            }
        }
    } else if (!trans_A && trans_B) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                dtype result = (dtype) 0;
                for (int k = 0; k < K; k++) {
                    result += data_A[m * K + k] * data_B[n * K  + k];
                }
                data_C[m * N + n] = alpha * result + beta * data_C[m * N + n];
            }
        }
    }
}

template <typename dtype, typename TargetType_D, typename TargetType_H>
void aligned_mat_mul_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                      std::vector<Tensor<TargetType_H>*>& outputs,
                      AlignedMatMulParam<TargetType_D>& param) {
    float alpha = param.scale;
    float beta = 0.f;
    bool trans_A = param.is_transpose_X;
    bool trans_B = param.is_transpose_Y;
    const dtype* src0 = (dtype*)inputs[0]->data();
    const dtype* src1 = (dtype*)inputs[1]->data();
    dtype* dst = (dtype*)outputs[0]->mutable_data();
    auto seq_offset_0 = inputs[0]->get_seq_offset()[0];
    auto seq_offset_1 = inputs[1]->get_seq_offset()[0];
    int inner_A = inputs[0]->count_valid(1, inputs[0]->dims());
    int inner_B = inputs[1]->count_valid(1, inputs[1]->dims());
    int batch_A = seq_offset_0[1];
    int batch_B = seq_offset_1[1];
    int M = param.is_transpose_X ? inner_A : batch_A;
    int N = param.is_transpose_Y ? batch_B: inner_B;
    int K_A = param.is_transpose_X ? batch_A : inner_A;
    int K_B = param.is_transpose_Y ? inner_B : batch_B;
    CHECK_EQ(K_A, K_B) << "mat mul two inputs K is not equal";
    int K = K_A;
    int seq_num = seq_offset_0.size() - 1;
    for (int i = 0; i < seq_num; i++) {
        gemm(src0 + i * batch_A * inner_A,  src1 + i * batch_B * inner_B, M, N,  K,
                trans_A, trans_B, alpha, beta, dst + i * M * N);
    }
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
    TestSaberBase<TargetType_D, TargetType_H, Dtype, AlignedMatMul, AlignedMatMulParam> testbase(2, 1);
    float scale = 0.8;
    for (auto seq_num : {1}) {
        for (auto left_seq_len: {2}) {
            for (auto right_seq_len: {3}) {
                for (auto trans_a : {false}) {
                    for (auto trans_b: {true}) {
                        for (auto emb_size: {5}) {
                            std::vector<Tensor<TargetType_D>*> inputs;
                            std::vector<int> seq_offset_0;
                            std::vector<int> seq_offset_1;
                            generate_equal_step_offset(seq_num, left_seq_len, seq_offset_0);
                            generate_equal_step_offset(seq_num, right_seq_len, seq_offset_1);
                            int word_num_0 = seq_offset_0.back();
                            int word_num_1 = seq_offset_1.back();
                            Tensor<TargetType_D>* input_0 = new Tensor<TargetType_D>(Shape({word_num_0, emb_size, 1, 1}), AK_FLOAT);
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
                            AlignedMatMulParam<TargetType_D> param(trans_a, trans_b, scale);
                            testbase.set_param(param);
                            testbase.run_test(aligned_mat_mul_basic<float, TargetType_D, TargetType_H>, 0.00001, true, true);
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

TEST(TestSaberFunc, test_func_aligned_mat_mul) {

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

