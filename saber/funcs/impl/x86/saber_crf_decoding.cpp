
#include "saber/funcs/impl/x86/saber_crf_decoding.h"
#include "saber/saber_funcs_param.h"
#include <cstring>
#include <limits>
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype ,
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

template <DataType OpDtype ,
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
    _alpha.re_alloc(inputs[0]->valid_shape());
    _track.re_alloc(inputs[0]->valid_shape());
    return SaberSuccess;
}
template <typename dtype>
void decoding(dtype* path, const dtype* emission, const dtype* transition,
              dtype* alpha_value, int* track_value, int seq_len, int tag_num) {
    const dtype* x = emission;
    const dtype* w = transition;
    const int state_trans_base_idx = 2;

    for (int i = 0; i < tag_num; ++i) alpha_value[i] = w[i] + x[i];

    for (int k = 1; k < seq_len; ++k) {
        for (int i = 0; i < tag_num; ++i) {
            dtype max_score = -std::numeric_limits<dtype>::max();
            int max_j = 0;
            for (size_t j = 0; j < tag_num; ++j) {
                dtype score = alpha_value[(k - 1) * tag_num + j] +
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

template <DataType OpDtype ,
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
    const DataType_op *transition_ptr = param.transition_weight()->data();
    DataType_out *decoded_path = outputs[0]->mutable_data();

    int seq_num = seq_offset.size() - 1;
    int slice_size = inputs[0]->channel()
                     * inputs[0]->height()
                     * inputs[0]->width();

    for (int i = 0; i < seq_num; ++i) {
        int seq_len = seq_offset[i+1] - seq_offset[i];
        decoding(decoded_path, emission_ptr, transition_ptr,
                 _alpha.mutable_data(), _track.mutable_data(),
                 seq_len, inputs[0]->channel());

        decoded_path += seq_len;
        emission_ptr += slice_size * seq_len;
    }
    return SaberSuccess;

}
template class SaberCrfDecoding<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
} // namespace anakin
