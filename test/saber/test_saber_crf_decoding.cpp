
#include "core/context.h"
#include "funcs/crf_decoding.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "saber_types.h"
#include "saber/core/tensor_op.h"
#include <vector>

using namespace anakin::saber;

int num_in = 7;
int ch_in = 7;
int h_in = 1;
int w_in = 1;
int GLB_flag = 0; //NV

template<typename dtype>
void decoding(dtype* path, const dtype* emission, const dtype* transition,
              dtype* alpha_value, int* track_value, int aligned_tag_num, int seq_len, int tag_num) {

    const dtype* x = emission;
    const dtype* w = transition;
    const int state_trans_base_idx = 2;

    for (int i = 0; i < tag_num; ++i) {
        alpha_value[i] = w[i] + x[i];
    }

    for (int k = 1; k < seq_len; ++k) {
        for (int i = 0; i < tag_num; ++i) {
            dtype max_score = -std::numeric_limits<dtype>::max();
            int max_j = 0;
#ifdef __AVX2__

            if (tag_num % 8 == 0 && GLB_flag) {
                for (size_t j = 0; j < tag_num; ++j) {
                    dtype score = alpha_value[(k - 1) * tag_num + j] +
                                  w[(i + state_trans_base_idx) * tag_num + j];

                    if (score > max_score) {
                        max_score = score;
                        max_j = j;
                    }
                }
            } else {
                for (size_t j = 0; j < tag_num; ++j) {
                    dtype score = alpha_value[(k - 1) * tag_num + j] +
                                  w[(j + state_trans_base_idx) * tag_num + i];

                    if (score > max_score) {
                        max_score = score;
                        max_j = j;
                    }
                }
            }

#else

            for (size_t j = 0; j < tag_num; ++j) {
                dtype score = alpha_value[(k - 1) * tag_num + j] +
                              w[(j + state_trans_base_idx) * tag_num + i];

                if (score > max_score) {
                    max_score = score;
                    max_j = j;
                }
            }

#endif
            alpha_value[k * tag_num + i] = max_score + x[k * tag_num + i];
            track_value[k * tag_num + i] = max_j;
        }
    }

    dtype max_score = -std::numeric_limits<dtype>::max();
    int max_i = 0;

    for (size_t i = 0; i < tag_num; ++i) {
        dtype score = alpha_value[(seq_len - 1) * tag_num + i] + w[tag_num + i];

        if (score > max_score) {
            max_score = score;
            max_i = i;
        }
    }

    path[seq_len - 1] = max_i;

    for (int k = seq_len - 1; k >= 1; --k) {
        path[k - 1] = max_i = track_value[k * tag_num + max_i];
    }

}

template <typename dtype, typename TargetType_D, typename TargetType_H>
void crfdecoding_nv_basic(const std::vector<Tensor<TargetType_H>*>& inputs, \
                          std::vector<Tensor<TargetType_H>*>& outputs, \
                          CrfDecodingParam<TargetType_D>& param) {

    typedef Tensor<TargetType_H> TensorH;
    typedef Tensor<TargetType_D> TensorD;
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int slice_size = channel * height * width;
    //  LOG(INFO) << "channel: " << channel << ", heigth: " << height << ", width: " << inputs[0]->width();
    // LOG(INFO) << "slice_size: " << slice_size;

    const dtype* emission_ptr = (const dtype*)inputs[0]->data();
    TensorD* trans = param.mutable_transition_weight();
    TensorH* trans_host;
    trans_host = new TensorH(trans->valid_shape());
    // LOG(INFO) << "shape: " << trans->valid_size();
    trans_host->copy_from(*trans);
    const dtype* transition_ptr = (const dtype*)trans_host->data();
    dtype* decode_path = (dtype*)outputs[0]->mutable_data();

    // LOG(INFO) << "transition_ptr: " << transition_ptr;
    std::vector<std::vector<int>> seq_offset = inputs[0]->get_seq_offset();
    int seq_num = seq_offset[0].size() - 1;
    // LOG(INFO) << "seq_num: " << seq_num;

    TensorH* alpha;
    TensorH* track;
    alpha = new TensorH(inputs[0]->valid_shape());
    track = new TensorH(inputs[0]->valid_shape());

    fill_tensor_const(*alpha, 0.f);
    fill_tensor_const(*track, 0);

    for (int i = 0; i < seq_num; i++) {
        int seq_len = seq_offset[0][i + 1] - seq_offset[0][i];
        decoding<dtype>(decode_path, emission_ptr, transition_ptr,
                        (dtype*)alpha->mutable_data(), (int*)track->mutable_data(), slice_size, seq_len, channel);
        //LOG(INFO) << "slice_size: " << slice_size << ", seq_num: " << seq_num << ", seq_len: " << seq_len;
        emission_ptr += slice_size * seq_len;
        decode_path += seq_len;
    }

    delete trans_host;
    delete alpha;
    delete track;
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model(int flag) {
    typedef Tensor<TargetType_H> TensorH;
    typedef Tensor<TargetType_D> TensorD;
    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;

    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_shape2({100, 100, 1, 1}, Layout_NCHW);

    if (flag == 1) {
        GLB_flag = 1;    //X86
    }

    TestSaberBase<TargetType_D, TargetType_H, Dtype, CrfDecoding, CrfDecodingParam> testbase(1, 1);

    for (auto shape : {
                input_shape, input_shape2
            }) {
        for (auto num_tag : {
                    0
                }) {
            TensorD weights;
            float min = -1.f;
            float max = 1.f;
            std::vector<std::vector<int>> lod = {{0, 2, shape[1]}};
            Shape wei_shape({shape[1] + 2, shape[1], shape[2], shape[3]}, Layout_NCHW);
            weights.re_alloc(wei_shape, Dtype);
            fill_tensor_rand(weights, min, max);
            CrfDecodingParam<TargetType_D> param(&weights, num_tag);

            TensorD in;
            in.re_alloc(shape, Dtype);
            fill_tensor_rand(in, min, max);
            in.set_seq_offset(lod);

            // LOG(INFO) << "set_seq_offset size: " << in.get_seq_offset().size();

            std::vector<TensorD*> inputs;
            inputs.push_back(&in);

            //set param
            testbase.set_param(param);
            testbase.add_custom_input(inputs);
            // testbase.set_input_shape(shape);
            //run test
            testbase.run_test(crfdecoding_nv_basic<float, TargetType_D, TargetType_H>);
        }
    }
}

TEST(TestSaberFunc, test_func_normalize) {
    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;
#ifdef USE_CUDA
    //Init the test_base
    test_model<AK_FLOAT, NV, NVHX86>(0);
#endif

#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>(1);
#endif

#ifdef USE_ARM_PLACE
    //Init the test_base
    //  test_model<AK_FLOAT, ARM, ARM>();
#endif
}



int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    if (argc >= 2) {
        if (argc < 5) {
            LOG(ERROR) << "usage: ./" << argv[0] << "axis " << \
                       " num ch_in h_in w_in" ;
            return 0;
        }

        num_in = atoi(argv[1]);
        ch_in = atoi(argv[2]);
        h_in = atoi(argv[3]);
        w_in = atoi(argv[4]);
    }

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}
