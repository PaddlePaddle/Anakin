
#include "saber/funcs/impl/x86/saber_crf_decoding.h"
#include "saber/saber_funcs_param.h"
#include "x86_utils.h"
#include <cstring>
#include <limits>
#include <cmath>
#include <immintrin.h>

namespace anakin {
namespace saber {

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberCrfDecoding<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        CrfDecodingParam<OpTensor> &param, Context<X86> &ctx) {

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberCrfDecoding<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        CrfDecodingParam<OpTensor> &param,
        Context<X86> &ctx) {
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    this->_ctx = &ctx;
    _track.re_alloc(inputs[0]->valid_shape());

#ifdef __AVX2__
    int tag_num = inputs[0]->channel();
    _aligned_tag_num = (tag_num % 8) ? (tag_num / 8 + 1) * 8 : tag_num;
    // get transposed transition weight
    const DataType_op *transition_ptr = param.transition_weight()->data();
    Shape trans_shape(tag_num + 2, _aligned_tag_num, 1, 1);
    _trans.re_alloc(trans_shape);
    DataType_op *transition = _trans.mutable_data();
    memcpy(transition, transition_ptr, sizeof(DataType_op) * tag_num);
    memcpy(transition + _aligned_tag_num, transition_ptr + tag_num, sizeof(DataType_op) * tag_num);
    for (int i = 0; i < tag_num; i++) {
        for (int j = 0; j < tag_num; j++) {
            transition[(i + 2) * _aligned_tag_num + j] = transition_ptr[(j + 2) * tag_num + i];
        }
        for (int j = tag_num; j < _aligned_tag_num; j++) {
            transition[(i + 2) * _aligned_tag_num + j] = 0;
        }
    }

    Shape emis_shape(inputs[0]->num(), _aligned_tag_num, 1, 1);
    _emis.re_alloc(emis_shape);
    _alpha.re_alloc(emis_shape);
#else
    _alpha.re_alloc(inputs[0]->valid_shape());
#endif


    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberCrfDecoding<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::decoding(
                        DataType_in* path, const DataType_in* emission, const DataType_in* transition,
                        DataType_in* alpha_value, int* track_value, int seq_len, int tag_num) {
#ifdef __AVX2__
    const DataType_in* x = emission;
    const DataType_in* w = transition;
    const int state_trans_base_idx = 2;

    {
        __m256 *ww = (__m256*)w;
        __m256 *xx = (__m256*)x;
        __m256 *aa = (__m256*)alpha_value;
        for (int i = 0; i < _aligned_tag_num / 8; ++i) {
            aa[i] = ww[i] + xx[i];
        }
    }

    int tail = ((_aligned_tag_num == tag_num) ? 8 : tag_num % 8); 

    for (int k = 1; k < seq_len; ++k) {
        for (int i = 0; i < tag_num; ++i) {
            DataType_in max_score = -std::numeric_limits<DataType_in>::max();
            int max_j = 0;

            __m256 *aa = (__m256*)(alpha_value + (k - 1) * _aligned_tag_num);
            __m256 *ww = (__m256*)(w + (i + state_trans_base_idx) * _aligned_tag_num);
            __m256 score_v;
            DataType_in *score = (DataType_in*)(&score_v);
            for (size_t j = 0; j < _aligned_tag_num / 8 - 1; ++j) {
                score_v = aa[j] + ww[j];
                for (int m = 0; m < 8; m++) {
                    if (score[m] > max_score) {
                        max_score = score[m];
                        max_j = j * 8 + m;
                    }
                }
            }
            int tail_idx = _aligned_tag_num / 8 - 1;
            score_v = aa[tail_idx] + ww[tail_idx];
            for (int m = 0; m < tail; m++) {
                if (score[m] > max_score) {
                    max_score = score[m];
                    max_j = tail_idx * 8 + m;
                }
            }

            alpha_value[k * _aligned_tag_num + i] = max_score + x[k * _aligned_tag_num + i];
            track_value[k * tag_num + i] = max_j;
        }
    }

    DataType_in max_score = -std::numeric_limits<DataType_in>::max();
    int max_i = 0;
    __m256* aa = (__m256*)(alpha_value + (seq_len - 1) * _aligned_tag_num);
    __m256* ww = (__m256*)(w + _aligned_tag_num);
    __m256 score_v;
    DataType_in *score = (DataType_in*)(&score_v);
    for (size_t i = 0; i < _aligned_tag_num / 8 - 1; ++i) {
        score_v = aa[i] + ww[i];
        for (int m = 0; m < 8; m++) {
            if (score[m] > max_score) {
                max_score = score[m];
                max_i = i * 8 + m;
            }
        }
    }
    int tail_idx = _aligned_tag_num / 8 - 1;
    score_v = aa[tail_idx] + ww[tail_idx];
    for (int m = 0; m < tail; m++) {
        if (score[m] > max_score) {
            max_score = score[m];
            max_i = tail_idx * 8 + m;
        }
    }

    path[seq_len - 1] = max_i;
    for (int k = seq_len - 1; k >= 1; --k) {
        path[k - 1] = max_i = track_value[k * tag_num + max_i];
    }
#else
    const DataType_in* x = emission;
    const DataType_in* w = transition;
    const int state_trans_base_idx = 2;

    for (int i = 0; i < tag_num; ++i) alpha_value[i] = w[i] + x[i];

    for (int k = 1; k < seq_len; ++k) {
        for (int i = 0; i < tag_num; ++i) {
            DataType_in max_score = -std::numeric_limits<DataType_in>::max();
            int max_j = 0;
            for (size_t j = 0; j < tag_num; ++j) {
                DataType_in score = alpha_value[(k - 1) * tag_num + j] +
                          w[(j + state_trans_base_idx) * tag_num + i];
                if (score > max_score) {
                    max_score = score;
                    max_j = j;
                }
            }
            alpha_value[k * tag_num + i] = max_score + x[k * tag_num + i];
            track_value[k * tag_num + i] = max_j;
        }
    }
    DataType_in max_score = -std::numeric_limits<DataType_in>::max();
    int max_i = 0;
    for (size_t i = 0; i < tag_num; ++i) {
        DataType_in score = alpha_value[(seq_len - 1) * tag_num + i] + w[tag_num + i];
        if (score > max_score) {
            max_score = score;
            max_i = i;
        }
    }
    path[seq_len - 1] = max_i;
    for (int k = seq_len - 1; k >= 1; --k) {
        path[k - 1] = max_i = track_value[k * tag_num + max_i];
    }
#endif
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberCrfDecoding<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        CrfDecodingParam<OpTensor> &param) {
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    std::vector<int> seq_offset = inputs[0]->get_seq_offset();

    const DataType_in *emission_ptr = inputs[0]->data();
    int tag_num = inputs[0]->channel();
    const DataType_op *transition_ptr = param.transition_weight()->data();
    int slice_size = inputs[0]->channel()
                     * inputs[0]->height()
                     * inputs[0]->width();
#ifdef __AVX2__
    if (tag_num % 8) {
        transition_ptr = _trans.data();

        // align emission to AVX2 register width
        DataType_in *emission = _emis.mutable_data();
        for (int i = 0; i < inputs[0]->num(); i++) {
          DataType_in* to = emission + i * _aligned_tag_num;
          DataType_in* from = emission_ptr + i * tag_num;
          memcpy(to, from, tag_num * sizeof(DataType_in));
          for (int j = tag_num; j < _aligned_tag_num; j++) {
              to[j] = 0;
          }
        }
        emission_ptr = emission;
        slice_size = _aligned_tag_num;
    }
#endif
    DataType_out *decoded_path = outputs[0]->mutable_data();

    int seq_num = seq_offset.size() - 1;
    int nthreads = omp_get_max_threads();
    if (nthreads > seq_num) {
        nthreads = seq_num;
    }

    #pragma omp parallel for num_threads(nthreads) if(seq_num > 1)
    for (int i = 0; i < seq_num; ++i) {
        int seq_len = seq_offset[i+1] - seq_offset[i];
        decoding(decoded_path, emission_ptr, transition_ptr,
                 _alpha.mutable_data(), _track.mutable_data(),
                 seq_len, tag_num);

        decoded_path += seq_len;
        emission_ptr += slice_size * seq_len;
    }
    return SaberSuccess;
}

template class SaberCrfDecoding<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}
} // namespace anakin
