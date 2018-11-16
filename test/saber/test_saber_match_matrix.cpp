#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/match_matrix.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/funcs/debug.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;

#if defined(BUILD_LITE) || defined(USE_X86_PLACE) || defined(USE_CUDA)

template<typename dtype>
void transpose(const dtype* in, int M, int N, dtype* out) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[j * M + i] = in[i * N + j];
        }
    }
}

/*(total_len_r, dim_t, len_l)->(batch, dim_t, len_l, max_len_r)*/
template<typename dtype>
void padding_out(const dtype* src, std::vector<int>& offset_r, int dim_t, int len_l, dtype* dst) {
    int max_len_r = 0;
    for (int i = 0; i < offset_r.size() - 1; i++) {
        int cur_len = offset_r[i+1] - offset_r[i];
        max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
    }
    int seq_num  = offset_r.size() - 1;
    int tl = dim_t * len_l;
    for (int i = 0; i < seq_num; i++) {
        dtype* dst_tmp = dst + i * tl * max_len_r;
        const dtype* src_tmp = src + offset_r[i] *  tl;
        int cur_len = offset_r[i+1] - offset_r[i];
        for (int j = 0; j < cur_len; j++) {
            for (int k = 0; k < tl; k++) {
                dst_tmp[k * max_len_r + j] = src_tmp[j * tl + k];
            }
        }
        for (int k = 0; k < tl; k++) {
            memset(dst_tmp + k * max_len_r + cur_len, 0, sizeof(dtype) * (max_len_r - cur_len));
        }
    }
}

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
                data_C[m * N + n] = result;
            }
        }
    } else if (!trans_A && trans_B) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                dtype result = (dtype) 0;
                for (int k = 0; k < K; k++) {
                    result += data_A[m * K + k] * data_B[n * K  + k];
                }
                data_C[m * N + n] = result;
            }
        }
    }
}




template <typename dtype,typename TargetType_D,typename TargetType_H>
void match_matrix_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                       std::vector<Tensor<TargetType_H>*>& outputs, 
                       MatchMatrixParam<TargetType_D>& param) {

    CHECK_EQ(inputs.size(), 2) <<"topk pooling need two inputs";
    int dim_t = param.dim_t;
    int dim_in = param.dim_in;
    auto offset_l = inputs[0]->get_seq_offset()[0];
    auto offset_r = inputs[1]->get_seq_offset()[0];
    int len_l = offset_l[1] - offset_l[0];
    int len_r = offset_r[offset_r.size() - 1];
    Tensor<TargetType_H> _input_l_transform;
    Tensor<TargetType_H> _input_l_transform_reorganize;
    Tensor<TargetType_H> _output_tmp;
    Tensor<TargetType_H> weight;
    weight.reshape(param.weight()->valid_shape());
    weight.copy_from(*(param.weight()));
    Gemm<TargetType_H, VENDER_IMPL, float> _gemm_l_transform;
    Gemm<TargetType_H, VENDER_IMPL, float> _gemm_r_transform;
    int max_len_r = 0;
    for (int i = 0; i < offset_r.size() - 1; i++) {
        int cur_len = offset_r[i+1] - offset_r[i];
        max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
    }
    int batch = offset_r.size() - 1;
    
     _input_l_transform.reshape(std::vector<int>{1, dim_t, dim_in, len_l});
    _input_l_transform_reorganize.reshape(std::vector<int>{1, dim_t, len_l, dim_in});
    _output_tmp.reshape(std::vector<int>{1, offset_r[offset_r.size() - 1], dim_t, len_l});
    outputs[0]->reshape(std::vector<int>{batch, dim_t, len_l, max_len_r});

    const dtype* weight_data =  (const dtype*) weight.data();
    const dtype* input_l = (const dtype*)inputs[0]->data();
    const dtype* input_r = (const dtype*)inputs[1]->data();
    dtype* input_l_transform = (dtype*)_input_l_transform.mutable_data();
    dtype* input_l_transform_reorganize = (dtype*)_input_l_transform_reorganize.mutable_data();
    dtype* output_tmp = (dtype*)_output_tmp.mutable_data();
    dtype* output_data = (dtype*) outputs[0]->mutable_data();
    gemm(weight_data,
         input_l, 
         dim_t * dim_in, len_l, dim_in,
         true, true, 
         1.0f, 0.0f, input_l_transform);
    for (int i = 0; i < dim_t; i++) {
        int offset =  i * dim_in * len_l;
        transpose<dtype>(input_l_transform + offset, dim_in, len_l, input_l_transform_reorganize +  offset);
    }
    gemm(input_r,
         input_l_transform_reorganize, 
         len_r, dim_t*len_l, dim_in,
         false, true, 
         1.0f, 0.0f, output_tmp);

    padding_out(output_tmp, offset_r, dim_t, len_l, output_data);
     LOG(INFO )<< "*******************************";
     write_tensorfile(_input_l_transform, "./_input_l_transform");
 //    record_dev_tensorfile(input_l_transform_reorganize, _input_l_transform_reorganize.valid_size(),  ("_input_l_transform_reorganize").c_str());
 //    record_dev_tensorfile(output_tmp, _output_tmp.valid_size(),  ("_output_tmp").c_str());
 //    record_dev_tensorfile(output_data, outputs[0]->valid_size(), ("output").c_str());
    outputs[0]->set_seq_offset(inputs[1]->get_seq_offset());
}

template <DataType Dtype,typename TargetType_D,typename TargetType_H>
void test_model(){
    int seq_num = 1;
    int dim_t = 5;
    int dim_in = 128;
    int left_seq_len = 10;
    int right_max_seq_len = 10;
    Env<TargetType_D>::env_init();
    Env<TargetType_H>::env_init();

    TestSaberBase<TargetType_D, TargetType_H, Dtype, MatchMatrix, MatchMatrixParam> testbase(2,1);
    
    //test example
        for (auto dim_t: {1, 3, 5}) {
            Shape weight_shape = std::vector<int>{dim_in*dim_t*dim_in, 1, 1, 1};
            Tensor<TargetType_D> weight(weight_shape);
            fill_tensor_rand(weight, -1, 1);
            
            MatchMatrixParam<TargetType_D> param(dim_in, dim_t, &weight);
            testbase.set_param(param);//set param
            std::vector<std::vector<int>> left_seq_offset;
            std::vector<std::vector<int>> right_seq_offset;
            left_seq_offset.resize(1);
            right_seq_offset.resize(1);
            int cumsum_right = 0;
            int cumsum_left = 0;
            left_seq_offset[0].push_back(cumsum_left);
            right_seq_offset[0].push_back(cumsum_right);
            for (int i = 0; i < seq_num; i++) {
                int len_r = std::rand() % right_max_seq_len + 1;
                cumsum_left += left_seq_len;
                cumsum_right += len_r; 
                right_seq_offset[0].push_back(cumsum_right);
                left_seq_offset[0].push_back(cumsum_left);
            }
               
            Shape shape_0 = std::vector<int>{cumsum_left, dim_in, 1, 1};
            Shape shape_1 = std::vector<int>{cumsum_right, dim_in, 1, 1};
            Tensor<TargetType_D> input_0(shape_0);
            Tensor<TargetType_D> input_1(shape_1);
            fill_tensor_rand(input_0, -1, 1);
            fill_tensor_rand(input_1, -1, 1);
            input_0.set_seq_offset(left_seq_offset);
            input_1.set_seq_offset(right_seq_offset);
            std::vector<Tensor<TargetType_D>*> input_vec;
            input_vec.push_back(&input_0);
            input_vec.push_back(&input_1);
            testbase.add_custom_input (input_vec);
            testbase.run_test(match_matrix_basic<float, TargetType_D, TargetType_H>, 5e-5);//run test
        }
}

#endif

TEST(TestSaberFunc, test_func_activation) {

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
#ifdef USE_BM
   // Env<BM>::env_init();
    //test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

