
#ifndef SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_CUDA_UTILS_H
#define SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_CUDA_UTILS_H

#include <vector>
#include <algorithm>
#include "core/common.h"
#include "core/tensor.h"
#include "cuda.h"
#include "saber_util.h"

namespace anakin {

namespace saber {



template<typename Dtype>
extern void trans_map2out_cfunc(const Dtype* input, Dtype* output, int word_size, int seq_sum,
                         cudaStream_t stream,
                         int* dev_map_vec);

template<typename Dtype>
extern void trans_map2in_cfunc(const Dtype* input, Dtype* output, int hidden_size, int seq_sum,
                        cudaStream_t stream,
                        int* dev_map_vec);


class SeqSortedseqTranseUtil {

public:

    SeqSortedseqTranseUtil(bool is_reverse = false, bool is_bi = false)
            : _is_reverse(is_reverse),
              _is_bi(is_bi),
              _dev_map_vec(nullptr),
              _dev_map_vec_length(0)

    {};

    ~SeqSortedseqTranseUtil() {
        if (_dev_map_vec != nullptr) {
            CUDA_CHECK(cudaFree(_dev_map_vec));
        }
    };
    std::vector<int>& get_length_index() {return _length_index;}
    std::vector<int>& get_emit_offset_vec() {return _emit_offset_vec;}
    std::vector<int>& get_map_vec() {return _map_vec;}
    int* get_dev_map_vec() {return _dev_map_vec;}
    int get_emit_length() {return _emit_length;}

    void print_vec(const float* in, int size, const char* perfix) {
        for (int i = 0; i < size; i++) {
            printf("[%s] %d = %f\n", perfix, i, in[i]);
        }
    }
    void print_vec(const int* in, int size, const char* perfix) {
        for (int i = 0; i < size; i++) {
            printf("[%s] %d = %d\n", perfix, i, in[i]);
        }
    }
    template <typename Dtype>
    void seq_2_sorted_seq(const Dtype*  input, Dtype* output, int word_size, cudaStream_t stream) {
        int seq_sum = _map_vec.size();
        trans_map2out_cfunc(input, output, word_size, seq_sum, stream, _dev_map_vec);
    }
    template <typename Dtype>
    void hidden_2_sorted_hidden(const Dtype*  input, Dtype* output, int hidden_size) {
        int batch_size = _length_index.size();

        for (int ori_word_id = 0; ori_word_id < batch_size; ++ori_word_id) {
            //can param
            int word_start = ori_word_id * hidden_size;
            int maped_id = _length_index[ori_word_id];
            int maped_start = maped_id * hidden_size;

            for (int word_vec_offset = 0; word_vec_offset < hidden_size; ++word_vec_offset) {

                output[word_start + word_vec_offset] = input[maped_start + word_vec_offset];

            }
        }
    }
    template <typename Dtype>
    void sorted_seq_2_seq(const Dtype* input, Dtype* output, int hidden_size, cudaStream_t stream) {
        int seq_sum = _map_vec.size();
        trans_map2in_cfunc(input, output, hidden_size, seq_sum, stream, _dev_map_vec);
    }

    bool get_sorted_map(std::vector<int>& offset_vec, cudaStream_t stream_id) {
        int batch_size = offset_vec.size() - 1;
        int word_sum = offset_vec[offset_vec.size() - 1];
        std::vector<int> length_vec(batch_size);
        _length_index.resize(batch_size);
        int emit_length = 0;

        if (batch_size == 1) {
            emit_length = offset_vec[1] - offset_vec[0];
            _emit_offset_vec.resize(emit_length + 1);

            for (int i = 0; i <= emit_length; i++) {
                _emit_offset_vec[i] = i;
            }

            return false;
        }

        int max_len = 0;

        for (int i = 0; i < offset_vec.size() - 1; ++i) {
            int len = offset_vec[i + 1] - offset_vec[i];
            max_len = max_len > len ? max_len : len;
            length_vec[i] = len;
            _length_index[i] = i;
        }

        emit_length = max_len;

        if (max_len == 1) {
            _emit_offset_vec.resize(2);
            _emit_offset_vec[0] = 0;
            _emit_offset_vec[1] = emit_length * batch_size;
            return false;
        }

        std::sort(_length_index.begin(), _length_index.end(), [&length_vec](int i1, int i2) {
            return length_vec[i1] > length_vec[i2];
        });

        _emit_offset_vec.resize(max_len + 1);
        _map_vec.resize(word_sum);

        if (word_sum > _dev_map_vec_length) {
            if (_dev_map_vec != nullptr) {
                CUDA_CHECK(cudaFree(_dev_map_vec));
            }

            CUDA_CHECK(cudaMalloc(&_dev_map_vec, sizeof(int)*word_sum));
            _dev_map_vec_length = word_sum;
        }

        int target_word_id = 0;
        std::vector<int> length_vec_cnt = length_vec;
        int last_batch_size = batch_size;
        for (int word_id_in_seq = 0; word_id_in_seq < max_len; word_id_in_seq++) {
            _emit_offset_vec[word_id_in_seq] = target_word_id;

            for (int batch_id = 0; batch_id < last_batch_size; batch_id++) {
                int old_batch_id = _length_index[batch_id];

                if (length_vec_cnt[old_batch_id] > 0) {
                    int inner_word_id_in_seq = word_id_in_seq;

                    if (_is_reverse) {
                        inner_word_id_in_seq = length_vec[old_batch_id] - 1 - word_id_in_seq;
                    }

                    int old_word_id = offset_vec[old_batch_id] + inner_word_id_in_seq;
                    _map_vec[old_word_id] = target_word_id;
                    //                    printf("map %d -> %d\n",old_word_id,target_word_id);
                    length_vec_cnt[old_batch_id]--;
                    target_word_id++;
                } else {
                    last_batch_size--;
                    break;
                }
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(_dev_map_vec, _map_vec.data(), sizeof(int)*word_sum,
                                   cudaMemcpyHostToDevice, stream_id));
        _emit_offset_vec[max_len] = word_sum;
        _emit_length = emit_length;
        return true;
    }

private:

    std::vector<int> _length_index;
    std::vector<int> _emit_offset_vec;
    std::vector<int> _map_vec;
    int _emit_length;

    int* _dev_map_vec;
    int _dev_map_vec_length;
    bool _is_reverse;
    bool _is_bi;

};

template <typename Dtype>
extern void  get_sub_tensor(const Dtype* in, Dtype* out, int h, int w, int stride_w, cudaStream_t stream);


}
}
#endif //SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_CUDA_UTILS_H
