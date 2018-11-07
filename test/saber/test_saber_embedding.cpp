#include "saber/core/context.h"
#include "saber/funcs/embedding.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_base.h"
#include "test_saber_func.h"
#include <vector>

using namespace anakin::saber;

//native cpu version
template <typename dtype, typename TargetType_D, typename TargetType_H>
void embedding_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                        std::vector<Tensor<TargetType_H>* >& output,
                        EmbeddingParam<TargetType_D>& param) {

    const dtype* in_data = (const dtype*)input[0]->data();
    int num = input[0]->valid_size();
    dtype* out_data = (dtype*)output[0]->mutable_data();
    //host weight
    Tensor<TargetType_H> weight_h(param.weight()->valid_shape());
    weight_h.copy_from(*param.weight());
    auto weight_data = (const dtype*)weight_h.data();

    for (int i = 0; i < num; i++) {
        if (in_data[i] == param.padding_idx) {
            memset(out_data + i * param.emb_dim, 0, sizeof(dtype) * param.emb_dim);
        } else {
            CHECK_GE(int(in_data[i]), 0);
            CHECK_LT(int(in_data[i]), param.word_num);
            memcpy(out_data + i * param.emb_dim, \
                   weight_data + int(in_data[i]) * param.emb_dim, \
                   sizeof(dtype) * param.emb_dim);
        }
    }

    if (param.num_direct == 2) {
        dtype* out_data = (dtype*)output[1]->mutable_data();
        auto seq_offset = input[0]->get_seq_offset();
        CHECK_GE(seq_offset.size(), 1) << "embedding seq offset is not null";
        auto cur_seq_offset = seq_offset[0];

        for (int i = 0; i < cur_seq_offset.size() - 1; i++) {
            int cur_len = cur_seq_offset[i + 1] - cur_seq_offset[i];

            for (int j = 0; j < cur_len; j++) {
                int src_index = cur_seq_offset[i] + j;
                int dst_index = cur_seq_offset[i + 1] - 1 - j;
                int index = in_data[src_index];

                if (index == param.padding_idx) {
                    memset(out_data +  dst_index * param.emb_dim, 0, sizeof(dtype) * param.emb_dim);
                } else {
                    CHECK_GE(index, 0);
                    CHECK_LT(index, param.word_num);
                    memcpy(out_data + dst_index * param.emb_dim, weight_data + int(index) * param.emb_dim,
                           sizeof(dtype) * param.emb_dim);
                }
            }
        }
    }
}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_embedding() {
    int word_num = 128;
    int emb_dim = 10;
    int padding_idx = -1;

    Shape weights_s({1, 1, word_num, emb_dim});
    typedef typename DataTrait<TargetType_D, OpDtype> :: Dtype dtype;
    Tensor<TargetType_D> weight_h(weights_s);
    fill_tensor_rand(weight_h, -0.5, 0.5);

    for (auto num_direct : {
                1, 2
            }) {
        TestSaberBase<TargetType_D, TargetType_H, OpDtype, Embedding, EmbeddingParam> testbase(1,
                num_direct);
        EmbeddingParam<TargetType_D> param(word_num, emb_dim, padding_idx, num_direct, &weight_h);
        testbase.set_param(param);

        for (auto seq_num : {
                    1, 32, 64
                }) {
            std::vector<int> vec;
            int cumsum_num = 0;
            vec.push_back(cumsum_num);

            for (int i = 0; i < seq_num; i++) {
                cumsum_num += std::rand() % 10 + 1;
                vec.push_back(cumsum_num);
            }

            std::vector<Tensor<TargetType_D>*> input_vec;
            Shape shape = Shape({cumsum_num, 1, 1, 1}, Layout_NCHW);
            Tensor<TargetType_D> input_0(shape);
            fill_tensor_rand(input_0, 1, 128);
            input_vec.push_back(&input_0);
            std::vector<std::vector<int>> seq_offset;
            seq_offset.push_back(vec);
            input_0.set_seq_offset(seq_offset);
            testbase.add_custom_input(input_vec);
            testbase.run_test(embedding_cpu_base<dtype, TargetType_D, TargetType_H>);//run test
        }
    }

    ////test for nc
    //for(int ch_in : {3, 8, 16, 64}) {
    //    for(int num_in:{1, 2, 32, 64}) {
    //        testbase.set_rand_limit(1, 128);
    //        testbase.set_input_shape(Shape({num_in, ch_in}, Layout_HW));
    //        testbase.run_test(embedding_cpu_base<float, X86, X86>);//run test
    //    }
    //}

}


TEST(TestSaberFunc, test_op_embedding) {
    //#ifdef USE_X86_PLACE
    //    test_embedding<X86, X86, AK_FLOAT>();
    //#endif

#ifdef USE_CUDA
    test_embedding<NV, NVHX86, AK_FLOAT>();
#endif

}


int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

