#include <cmath>
#include "saber_types.h"
#include "saber/funcs/impl/x86/saber_attension_lstm.h"
#include "saber/core/tensor_op.h"
#include "mkl_cblas.h"
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#include "saber/funcs/impl/x86/saber_normal_activation.h"

namespace anakin {

namespace saber {

//inline
static void gemm(const bool TransA, const bool TransB, int m, int n, int k, const float alpha,
                 const float* a, const float* b, const float beta, float* c) {
    //    cout << "(" << m << "," << n << "," << k << ")" << endl;
    int lda = (!TransA/* == CblasNoTrans*/) ? k : m;
    int ldb = (!TransB/* == CblasNoTrans*/) ? n : k;
    CBLAS_TRANSPOSE cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    cblas_sgemm(CblasRowMajor, cuTransA, cuTransB, m, n, k, alpha, a, k, b, n, beta, c, n);
};

template <typename Dtype>
void sequence_bias_relu(const Dtype* input_0,
                        const Dtype* input_1,
                        const Dtype* bias,
                        std::vector<int>& seq_offset,
                        const int dim,
                        Dtype* out) {
    int seq_num = seq_offset.size() - 1;
    int count = 0;

    for (int i = 0; i < seq_num; i++) {
        const Dtype* tmp_input_1 = input_1 + i * dim;

        for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
            for (int k = 0; k < dim; k++) {
                auto data = input_0[count] + bias[k]  + tmp_input_1[k];
                data = data > 0 ? data : 0;
                out[count] = data;
                count++;
            }
        }
    }
}

template <typename Dtype>
void bias_relu(Dtype* input,
               const Dtype* bias,
               const int n,
               const int dim
              ) {
    int count = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            auto data = input[count] + bias[j];
            data = data > 0 ? data : 0;
            input[count] = data;
            count++;
        }
    }
}

template <typename Dtype>
void sequence_softmax(const Dtype* data, std::vector<int>& seq_offset, Dtype* out) {
    for (int i = 0; i < seq_offset.size() - 1; i++) {
        Dtype max = -1e32;
        for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
            max = data[j] > max ? data[j] : max;
        }

        Dtype cumsum = 0.f;
        for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
            cumsum += expf(data[j] - max);
        }

        for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
            out[j] = expf(data[j] - max) / cumsum;

        }
    }
}


template <typename Dtype>
void sequence_pool(const Dtype* data, const Dtype* weight, std::vector<int>& seq_offset, int dim,
                   Dtype* out) {
    for (int i = 0; i < seq_offset.size() - 1; i++) {
        Dtype* tmp_out = out +  i * dim;
        memset(tmp_out, 0, sizeof(Dtype) * dim);

        for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
            Dtype scale = weight[j];
            const Dtype* tmp_data = data + j * dim;

            for (int k = 0; k < dim; k++) {
                tmp_out[k] += scale * tmp_data[k];
            }
        }
    }
}




template <typename Dtype>
void lstm_bias_and_act(const Dtype* hidden_in, const Dtype* bias_data, Dtype* out,
                       Dtype* cell_data, const int seq_num, const int hidden_size, const int with_peephole) {
    const Dtype* bias_i = bias_data;
    const Dtype* bias_f = bias_i + hidden_size;
    const Dtype* bias_c = bias_f + hidden_size;
    const Dtype* bias_o = bias_c + hidden_size;
    if (with_peephole) {
        const Dtype* w_ci = bias_o + hidden_size;
        const Dtype* w_cf = w_ci + hidden_size;
        const Dtype* w_co = w_cf + hidden_size;

        for (int i = 0; i < seq_num; i++) {
            const Dtype*  tmp_hidden_i = hidden_in + i * 4 * hidden_size;
            const Dtype*  tmp_hidden_f = tmp_hidden_i + hidden_size;
            const Dtype*  tmp_hidden_c = tmp_hidden_f + hidden_size;
            const Dtype*  tmp_hidden_o = tmp_hidden_c + hidden_size;
            Dtype*  tmp_cell_data = cell_data + i * hidden_size;
            Dtype*  tmp_out = out + i * hidden_size;

            for (int j = 0; j < hidden_size; j++) {
                Dtype ig = Sigmoid(tmp_hidden_i[j] + bias_i[j] + tmp_cell_data[j] * w_ci[j]);
                Dtype fg = Sigmoid(tmp_hidden_f[j] + bias_f[j] + tmp_cell_data[j] * w_cf[j]);
                Dtype c_t_0 = Tanh(tmp_hidden_c[j] + bias_c[j]);
                tmp_cell_data[j] = ig * c_t_0 + fg * tmp_cell_data[j];
                Dtype og = Sigmoid(tmp_hidden_o[j] + bias_o[j] + tmp_cell_data[j] * w_co[j]);
                tmp_out[j] = og * Tanh(tmp_cell_data[j]);
            }
        }
    } else {
        for (int i = 0; i < seq_num; i++) {
            const Dtype* tmp_hidden_i = hidden_in + i * 4 * hidden_size;
            const Dtype* tmp_hidden_f = tmp_hidden_i + hidden_size;
            const Dtype* tmp_hidden_c = tmp_hidden_f + hidden_size;
            const Dtype* tmp_hidden_o = tmp_hidden_c + hidden_size;
            Dtype* tmp_cell_data = cell_data + i * hidden_size;
            Dtype* tmp_out = out + i * hidden_size;

            for (int j = 0; j < hidden_size; j++) {
                Dtype ig = Sigmoid(tmp_hidden_i[j] + bias_i[j]);
                Dtype fg = Sigmoid(tmp_hidden_f[j] + bias_f[j]);
                Dtype og = Sigmoid(tmp_hidden_o[j] + bias_o[j]);
                Dtype c_t_0 = Tanh(tmp_hidden_c[j] + bias_c[j]);
                tmp_cell_data[j] = ig * c_t_0 + fg * tmp_cell_data[j];
                tmp_out[j] = og * Tanh(tmp_cell_data[j]);
            }
        }
    }
}


template <typename Dtype>
void lstm_result_to_sequence(const Dtype* in, int hidden_size, std::vector<int>& seq_offset,
                             Dtype* out) {
    int seq_num = seq_offset.size() - 1;

    for (int  i = 0; i < seq_num; i++) {
        for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
            int k = j - seq_offset[i];
            int offset = (k * seq_num  + i) * hidden_size;
            memcpy(out + j * hidden_size, in + offset, sizeof(Dtype) * hidden_size);
        }
    }
}

template<>
SaberStatus SaberAttensionLstm<X86, AK_FLOAT>::
cpu_dispatch(const std::vector<OpTensor*>& inputs,
             std::vector<OpTensor*>& outputs,
             AttensionLstmParam<X86>& param) {

    auto attn_param = param.attension_param;
    auto lstm_param = param.lstm_param;
    int word_num = inputs[0]->num();
    auto seq_offset = inputs[0]->get_seq_offset()[0];
    int seq_num = seq_offset.size() - 1;

    int max_len = 0;

    for (int i = 0; i < seq_num; i++) {
        int cur_len = seq_offset[i + 1] - seq_offset[i];
        max_len =  max_len < cur_len ? cur_len : max_len;
    }

    utils::try_expand_tensor(_cell_out,seq_num* _hidden_size);
    utils::try_expand_tensor(_hidden_out,seq_num* 4 * _hidden_size);
    utils::try_expand_tensor(_lstm_out,max_len * seq_num* _hidden_size);
    utils::try_expand_tensor(_pool_out,seq_num*_word_size);
    utils::try_expand_tensor(_softmax_out,word_num);

    utils::try_expand_tensor(_first_fc_out_0,word_num* _attn_fc_size[0]);

    memset(_cell_out.mutable_data(), 0, sizeof(float) * seq_num* _hidden_size);
    //first fc
    int input_dim = inputs[0]->valid_size() / word_num;
    gemm(false, false, word_num, _attn_fc_size[0], _word_size,
         1.f, static_cast<const OpDataType*>(inputs[0]->data()),static_cast<const OpDataType*>( _attn_fc_weights[0]->data()),
         0.f, static_cast<OpDataType*>(_first_fc_out_0.mutable_data()));

    utils::try_expand_tensor(*_attn_outs[0],word_num* _attn_fc_size[0]);
    for (int word_id = 0; word_id < max_len; word_id++) {
        if (word_id > 0) {
            utils::try_expand_tensor(_first_fc_out_1,seq_num* _attn_fc_size[0]);
            /*there may be some danger*/
            gemm(false, false, seq_num, _attn_fc_size[0], _hidden_size,
                 1.f, static_cast<const OpDataType*>(_cell_out.data()),
                 static_cast<const OpDataType*>(_attn_fc_weights[0]->data()) + input_dim * _attn_fc_size[0],
                 0.f, static_cast<OpDataType*>(_first_fc_out_1.mutable_data()));
            sequence_bias_relu(static_cast<const OpDataType*>(_first_fc_out_0.data()),
                               static_cast<const OpDataType*>(_first_fc_out_1.data()),
                               static_cast<const OpDataType*>( _attn_fc_bias[0]->data()),
                               seq_offset,
                               _attn_fc_size[0],
                               static_cast<OpDataType*>(_attn_outs[0]->mutable_data()));
        } else {
            memcpy(_attn_outs[0]->mutable_data(), _first_fc_out_0.data(),
                   sizeof(float) *word_num* _attn_fc_size[0]);
            bias_relu(static_cast<OpDataType*>(_attn_outs[0]->mutable_data()),static_cast<const OpDataType*>( _attn_fc_bias[0]->data()), word_num,
                      _attn_fc_bias[0]->valid_size());
        }

        for (int i = 1; i < attn_param.fc_vec.size(); i++) {
            utils::try_expand_tensor(*_attn_outs[i],word_num* _attn_fc_size[i]);
            gemm(false, false, word_num, _attn_fc_size[i],
                 _attn_fc_size[i - 1],
                 1.f, static_cast<const OpDataType*>(_attn_outs[i - 1]->data()),
                 static_cast<const OpDataType*>(_attn_fc_weights[i]->data()),
                 0.f, static_cast<OpDataType*>(_attn_outs[i]->mutable_data()));
            bias_relu(static_cast<OpDataType*>(_attn_outs[i]->mutable_data()), static_cast<const OpDataType*>(_attn_fc_bias[i]->data()),word_num,
                      _attn_fc_bias[i]->valid_size());
        }

        int fc_num = attn_param.fc_vec.size();
#if defined(__AVX2__) and defined(__FMA__)
        avx2_sequence_softmax(static_cast<OpDataType*>(_attn_outs[fc_num - 1]->mutable_data()), seq_offset, static_cast<OpDataType*>(_softmax_out.mutable_data()));
        avx2_sequence_pool(static_cast<const OpDataType*>(inputs[0]->data()), static_cast<const OpDataType*>(_softmax_out.data()), seq_offset,
                           inputs[0]->valid_size() / word_num, static_cast<OpDataType*>(_pool_out.mutable_data()));
#else
        sequence_softmax(static_cast<OpDataType*>(_attn_outs[fc_num - 1]->mutable_data()), seq_offset, static_cast<OpDataType*>(_softmax_out.mutable_data()));
        sequence_pool(static_cast<const OpDataType*>(inputs[0]->data()), static_cast<const OpDataType*>(_softmax_out.data()), seq_offset,
                      inputs[0]->valid_size() / word_num, static_cast<OpDataType*>(_pool_out.mutable_data()));
#endif


        utils::try_expand_tensor(_hidden_out,seq_num* 4 * _hidden_size);
        //LOG(INFO)<<"hidden_size" << _hidden_size;
        gemm(false, false, seq_num, 4 * _hidden_size, _word_size,
             1.f, static_cast<const OpDataType*>(_pool_out.data()), _weights_i2h, 0.f, static_cast<OpDataType*>(_hidden_out.mutable_data()));

        if (word_id > 0) {
            gemm(false, false, seq_num, 4 * _hidden_size, _hidden_size,
                 1.f, static_cast<const OpDataType*>(_lstm_out.data()) + (word_id - 1) * seq_num * _hidden_size, _weights_h2h, 1.f,
                 static_cast<OpDataType*>(_hidden_out.mutable_data()));
        }

#if defined(__AVX2__) and defined(__FMA__)
        avx2_lstm_bias_and_act(static_cast<const OpDataType*>(_hidden_out.data()), _weights_bias,
                          static_cast<OpDataType*>(_lstm_out.mutable_data()) + word_id * seq_num * _hidden_size,
                          static_cast<OpDataType*>(_cell_out.mutable_data()), seq_num, _hidden_size, false);
#else
        lstm_bias_and_act(static_cast<const OpDataType*>(_hidden_out.data()), _weights_bias,
                              static_cast<OpDataType*>(_lstm_out.mutable_data()) + word_id * seq_num * _hidden_size,
                              static_cast<OpDataType*>(_cell_out.mutable_data()), seq_num, _hidden_size, false);
#endif

    }

    lstm_result_to_sequence(static_cast<const OpDataType*>(_lstm_out.data()), _hidden_size, seq_offset, static_cast<OpDataType*>(outputs[0]->mutable_data()));
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    return SaberSuccess;
}
template<>
SaberStatus SaberAttensionLstm<X86, AK_FLOAT>::init(const std::vector<OpTensor*>& inputs,
                 std::vector<OpTensor*>& outputs,
                 AttensionLstmParam<X86>& param,
                 Context<X86>& ctx) {
    utils::AlignedUtils aligned_tool;
    auto lstm_param = param.lstm_param;
    auto attn_param = param.attension_param;

    _hidden_size = lstm_param.bias()->valid_size() / 4;
    _word_size = lstm_param.weight()->valid_size() / (4 * _hidden_size)  -  _hidden_size;

    int weights_i2h_size = 4 * _hidden_size * _word_size;
    _weights_i2h = static_cast<const OpDataType*>(lstm_param.weight()->data());
    _weights_h2h = static_cast<const OpDataType*>(lstm_param.weight()->data()) + weights_i2h_size;
    _weights_bias = static_cast<const OpDataType*>(lstm_param.bias()->data());

    int input_dim = inputs[0]->valid_size() / inputs[0]->num();
    int fc_num = attn_param.fc_vec.size();
    _attn_fc_weights.resize(fc_num);
    _attn_fc_bias.resize(fc_num);
    _attn_fc_size.resize(fc_num);
    _attn_outs.resize(fc_num);

    for (int i = 0; i < fc_num; i++) {
        int N = attn_param.fc_vec[i].num_output;
        auto fc_param = (attn_param.fc_vec[i]);
        _attn_fc_weights[i] = fc_param.weights;
        _attn_fc_bias[i] = fc_param.bias;
        _attn_fc_size[i] = N;
        Shape shape( {1, 1, 1, 1});
        _attn_outs[i] = new OpTensor(shape);
    }

    return SaberSuccess;
};

template<>
SaberStatus SaberAttensionLstm<X86, AK_FLOAT>::
dispatch(const std::vector<OpTensor*>& inputs,
         std::vector<OpTensor*>& outputs,
         AttensionLstmParam<X86>& param) {
    CHECK_EQ(inputs.size(), 1) << "only support input size = 1";
    CHECK_EQ(outputs.size(), 1) << "only support outputs size = 1";
    CHECK_EQ(param.lstm_param.init_hidden() == nullptr,
             true) << "only support param.init_hidden() == nullptr";
    cpu_dispatch(inputs, outputs, param);

    return SaberSuccess;
}


DEFINE_OP_TEMPLATE(SaberAttensionLstm, AttensionLstmParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAttensionLstm, AttensionLstmParam, X86, AK_INT8);

}
}
