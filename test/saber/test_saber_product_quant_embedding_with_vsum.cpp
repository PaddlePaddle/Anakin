#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/product_quant_embedding_with_vsum.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
bool decode_4d12b( const unsigned char *in,
                   unsigned int ilen,
                   unsigned int *out,
                   unsigned int olen) {
    if (ilen % 3 != 0) {
        LOG(INFO) << "error, ilen mod 3 != 0";
        return false;
    }
    if (ilen * 2 != olen * 3) {
        LOG(INFO) << "error, ilen * 2 != olen * 3";
        return false;
    }
    memset(out, 0, olen * sizeof(unsigned int));
    for (unsigned int i = 0; i < ilen / 3; i++) {
        unsigned char *raw_ptr = (unsigned char *)(out + i * 2);
        raw_ptr[0] = in[3 * i];
        raw_ptr[1] = in[3 * i + 1] & 0x0f;
        raw_ptr[4] = in[3 * i + 2];
        raw_ptr[5] = in[3 * i + 1] >> 4;
    }
    return true;
}

void get_cur_idx(size_t word_idx, const size_t* word_offset, int offset_len, size_t* real_idx, int* case_idx) {
    CHECK_EQ(offset_len, 9);
    if (word_idx <  word_offset[0]) {
        *case_idx = 0;
        *real_idx = word_idx;
    } else if (word_idx <  word_offset[1]) {
        *case_idx = 1;
        *real_idx = word_idx - word_offset[0];
    } else if (word_idx <  word_offset[2]) {
        *case_idx = 2;
        *real_idx = word_idx - word_offset[1];
    } else if (word_idx <  word_offset[3]) {
        *case_idx = 0;
        *real_idx = word_idx - word_offset[2] + word_offset[0];
    } else if (word_idx <  word_offset[4]) {
        *case_idx = 1;
        *real_idx = word_idx - word_offset[3] + word_offset[1] - word_offset[0];
    } else if (word_idx <  word_offset[5]) {
        *case_idx = 2;
        *real_idx = word_idx - word_offset[4] + word_offset[2] - word_offset[1];
    } else if (word_idx <  word_offset[6]) {
        *case_idx = 0;
        *real_idx = word_idx - word_offset[5] + word_offset[0] + word_offset[3] - word_offset[2];
    } else if (word_idx <  word_offset[7]) {
        *case_idx = 1;
        *real_idx = word_idx - word_offset[6] + word_offset[1] - word_offset[0] + word_offset[4] - word_offset[3];
    } else if (word_idx <  word_offset[8]) {
        *case_idx = 2;
        *real_idx = word_idx - word_offset[7] + word_offset[2] - word_offset[1] + word_offset[5] - word_offset[4];
    }
}

template <typename dtype, typename TargetType_D, typename TargetType_H>
void product_quant_embedding_with_vsum_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                    std::vector<Tensor<TargetType_H>*>& outputs,
                    ProductQuantEmbeddingWithVsumParam<TargetType_D>& param) {
    size_t voc_size;
    size_t emb_size;
    size_t max_seq_len;
    size_t unigram_num[3];
    size_t bigram_num[3];
    size_t collocation_num[3];
    size_t chnl_num[3];
    size_t word_len[3];
    size_t word_num[3];
    size_t dict_size[3];
    size_t word_offset[9];
    const unsigned char* weights[3];
    const float* quant_dict[3];
    voc_size = param.word_voc;
    emb_size = param.word_emb;
    max_seq_len = param.max_seq_len;
    
    unigram_num[0] = param.top_unigram;
    unigram_num[1] = param.sec_unigram;
    unigram_num[2] = param.thd_unigram;
    
    bigram_num[0] = param.top_bigram;
    bigram_num[1] = param.sec_bigram;
    bigram_num[2] = param.thd_bigram;

    collocation_num[0] = param.top_collocation;
    collocation_num[1] = param.sec_collocation;
    collocation_num[2] = param.thd_collocation;
    int level_num = 3;
    for (unsigned int i = 0; i < level_num; i++) {
        word_num[i] = unigram_num[i] + bigram_num[i] + collocation_num[i];
        quant_dict[i] = NULL;
    }

    chnl_num[0] = 1;                 // log quant
    chnl_num[1] = emb_size / 2;     // 2d8b product quant
    chnl_num[2] = emb_size / 4;     // 4d12b product quant
    
    word_len[0] = emb_size;
    word_len[1] = chnl_num[1];
    word_len[2] = chnl_num[2] / 2 * 3;
    
    dict_size[0] = 256;
    dict_size[1] = 2 * 256;
    dict_size[2] = 4 * 4096;
    word_offset[0] = unigram_num[0];
    word_offset[1] = word_offset[0] + unigram_num[1];
    word_offset[2] = word_offset[1] + unigram_num[2];
    
    word_offset[3] = word_offset[2] + bigram_num[0];
    word_offset[4] = word_offset[3] + bigram_num[1];
    word_offset[5] = word_offset[4] + bigram_num[2];
    
    word_offset[6] = word_offset[5] + collocation_num[0];
    word_offset[7] = word_offset[6] + collocation_num[1];
    word_offset[8] = word_offset[7] + collocation_num[2];

    unsigned int* buf = new unsigned int[chnl_num[2]];
    float* top_pos = new float[emb_size];

    weights[0] = (const unsigned char*)param.embedding_0->data();
    weights[1] = (const unsigned char*)param.embedding_1->data();
    weights[2] = (const unsigned char*)param.embedding_2->data();

    //CHECK_NE(weights[0],  NULL) << "embedding  weights 0 is NULL";
    //CHECK_NE(weights[1],  NULL) << "embedding  weights 1 is NULL";
    //CHECK_NE(weights[2],  NULL) << "embedding  weights 2 is NULL";
    quant_dict[0] = (const float*)param.quant_dict_0->data();
    quant_dict[1] = (const float*)param.quant_dict_1->data();
    quant_dict[2] = (const float*)param.quant_dict_2->data();
    //CHECK_NE(quant_dict[0],  NULL) << "quant dict 0 is NULL";
    //CHECK_NE(quant_dict[1],  NULL) << "quant dict 1 is NULL";
    //CHECK_NE(quant_dict[2],  NULL) << "quant dict 2 is NULL";


    auto offset = inputs[0]->get_seq_offset()[0];
    int seq_num =  offset.size() - 1;

    outputs[0]->reshape(Shape({seq_num, emb_size, 1, 1}, Layout_NCHW));

    const dtype *input_data = (const dtype*)inputs[0]->data();
    dtype *output_data = (dtype*)outputs[0]->mutable_data();
    memset(output_data, 0, sizeof(dtype) * outputs[0]->valid_size());
    for (int seq_id = 0; seq_id  < seq_num; seq_id++) {
        size_t cur_len = offset[seq_id+1] - offset[seq_id];
        size_t len = max_seq_len == -1 ? cur_len : std::min(cur_len, max_seq_len);
        auto tmp_out_data = output_data + seq_id * emb_size;
        for (size_t i = 0; i < len; i++) {
            size_t word_idx = static_cast<size_t>(input_data[offset[seq_id] + i]);
            size_t real_idx = 0;
            int case_idx = 0;
            get_cur_idx(word_idx, word_offset, 9, &real_idx, &case_idx);
            
            if (case_idx == 0) {
                const unsigned char* word_pos = weights[0] + real_idx * word_len[0];
                for (size_t j = 0; j < word_len[0]; j++) {
                    top_pos[j] = quant_dict[0][word_pos[j]];
                }
            } else if (case_idx == 1) {
                const unsigned char* word_pos = weights[1] + real_idx * word_len[1];
                for (size_t j = 0; j < chnl_num[1]; j++) {
                    const float *curr_dict = quant_dict[1] + j * dict_size[1]; 
                    memcpy(top_pos + j * 2,
                        curr_dict + word_pos[j] * 2, 2 * sizeof(float));
                }
            } else {
                const unsigned char* word_pos = weights[2] + real_idx * word_len[2];
                decode_4d12b(word_pos, word_len[2], buf, chnl_num[2]);
                for (size_t j = 0; j < chnl_num[2]; j++) {
                    const float *curr_dict = quant_dict[2] + j * dict_size[2];
                    memcpy(top_pos + j * 4, 
                        curr_dict + buf[j] * 4, 4 * sizeof(float));
                }
            }
            for (size_t i = 0; i < emb_size; i++) {
                tmp_out_data[i] +=  top_pos[i];
            }
        }
    }

    delete [] buf;
    delete [] top_pos;

}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {

    TestSaberBase<TargetType_D, TargetType_H, Dtype, ProductQuantEmbeddingWithVsum, ProductQuantEmbeddingWithVsumParam> testbase(1, 1);
    size_t word_emb = 256;
    size_t word_voc = 10000;
    size_t top_unigram = 1000;
    size_t top_bigram = 500;
    size_t top_collocation = 500;
    size_t sec_unigram = 2000;
    size_t sec_bigram = 500;
    size_t sec_collocation = 500;
    size_t thd_unigram = 3000;
    size_t thd_bigram = 1000;
    size_t thd_collocation = 1000;
    int max_seq_len{512};
    int word_num[3];
    int word_len[3];
    int dict_size[3];
    int chnl_num[3];

    int level_num = 3;
    word_num[0] = top_unigram + top_bigram + top_collocation;
    word_num[1] = sec_unigram + sec_bigram + sec_collocation;
    word_num[2] = thd_unigram + thd_bigram + thd_collocation;

    chnl_num[0] = 1;                 // log quant
    chnl_num[1] = word_emb / 2;     // 2d8b product quant
    chnl_num[2] = word_emb / 4;     // 4d12b product quant

    word_len[0] = word_emb;
    word_len[1] = chnl_num[1];
    word_len[2] = chnl_num[2] / 2 * 3;

    dict_size[0] = 256;
    dict_size[1] = 2 * 256;
    dict_size[2] = 4 * 4096;

    Shape embedding_shape_0(std::vector<int>{word_num[0], word_len[0], 1, 1}, Layout_NCHW);
    Shape embedding_shape_1(std::vector<int>{word_num[1], word_len[1], 1, 1}, Layout_NCHW);
    Shape embedding_shape_2(std::vector<int>{word_num[2], word_len[2], 1, 1}, Layout_NCHW);
    Tensor<TargetType_D> embedding_0(embedding_shape_0, AK_UINT8);
    Tensor<TargetType_D> embedding_1(embedding_shape_1, AK_UINT8);
    Tensor<TargetType_D> embedding_2(embedding_shape_2, AK_UINT8);

    Shape quant_dict_shape_0(std::vector<int>{dict_size[0], chnl_num[0], 1, 1}, Layout_NCHW);
    Shape quant_dict_shape_1(std::vector<int>{dict_size[1], chnl_num[1], 1, 1}, Layout_NCHW);
    Shape quant_dict_shape_2(std::vector<int>{dict_size[2], chnl_num[2], 1, 1}, Layout_NCHW);
    Tensor<TargetType_D> quant_dict_0(quant_dict_shape_0);
    Tensor<TargetType_D> quant_dict_1(quant_dict_shape_1);
    Tensor<TargetType_D> quant_dict_2(quant_dict_shape_2); 
    //test example
    //
    for (auto seq_num : {1, 2, 16, 40}) {
        for (auto seq_len : {10, 16, 32}) {
            //fill_tensor_rand(embedding_0, 0, 128);
            //fill_tensor_rand(embedding_1, 0, 128);
            //fill_tensor_rand(embedding_2, 0, 128);
            //fill_tensor_rand(quant_dict_0, -1, 1);
            //fill_tensor_rand(quant_dict_1, -1, 1);
            //fill_tensor_rand(quant_dict_2, -1, 1);
            
            ProductQuantEmbeddingWithVsumParam<TargetType_D> param(word_emb, word_voc, 
                   top_unigram, top_bigram, top_collocation,
                   sec_unigram, sec_bigram, sec_collocation,
                   thd_unigram, thd_bigram, thd_collocation,
                   max_seq_len, &embedding_0, &embedding_1, &embedding_2, 
                   &quant_dict_0, &quant_dict_1, &quant_dict_2);

            testbase.set_param(param);//set param
            std::vector<std::vector<int>> seq_offset;
            seq_offset.resize(1);
            int cumsum = 0;
            seq_offset[0].push_back(cumsum);
            for (int i = 0; i < seq_num; i++) {
                int len = std::rand() % seq_len + 1;
                cumsum += len;
                seq_offset[0].push_back(cumsum);
            }

            Shape shape_0 = std::vector<int>{cumsum, 1, 1, 1};
            std::vector<Tensor<TargetType_D>*> input_vec;
            Tensor<TargetType_D> input_0(shape_0);
            fill_tensor_rand(input_0, 0, word_voc);
            input_0.set_seq_offset(seq_offset);
            input_vec.push_back(&input_0);
            testbase.add_custom_input(input_vec);
            testbase.run_test(product_quant_embedding_with_vsum_basic<float, TargetType_D, TargetType_H>);//run test
        }
    }
}

TEST(TestSaberFunc, test_func_product_quant_embedding_with_vsum) {

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

