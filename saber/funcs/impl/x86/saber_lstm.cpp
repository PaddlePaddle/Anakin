#include "saber/funcs/impl/x86/saber_lstm.h"
#include "saber/funcs/impl/x86/activation_functions.h"
#include "saber/funcs/impl/x86/sequence2batch.h"
#include "saber/funcs/impl/x86/kernel/jit_generator.h"

namespace anakin {
namespace saber {

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberLstm<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::compute_with_avx(LstmMetaValue<DataType_in> value,
                                                                        int hidden_size, int batch_size,
                                                                        const ActiveType &gate_act,
                                                                        const ActiveType &cell_act,
                                                                        const ActiveType &cand_act) {
#ifdef __AVX__
#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < hidden_size/8; i++) {
            __m256 r_checkI = _mm256_set1_ps(0.0f);
            __m256 r_checkF = _mm256_set1_ps(0.0f);
            __m256 r_checkO = _mm256_set1_ps(0.0f);
            __m256 prev_state_v = _mm256_set1_ps(0.0f);
            int batch_offset = b * hidden_size;
            __m256 *value_ig = reinterpret_cast<__m256 *>(value.gate_value + batch_offset * 4);
            __m256 *value_fg = reinterpret_cast<__m256 *>(value.gate_value + batch_offset * 4 + hidden_size);
            __m256 *value_in = reinterpret_cast<__m256 *>(value.gate_value + batch_offset * 4 + hidden_size * 2);
            __m256 *value_og = reinterpret_cast<__m256 *>(value.gate_value + batch_offset * 4 + hidden_size * 3);

            __m256 *state_active = reinterpret_cast<__m256 *>(value.state_active_value + batch_offset);
            __m256 *state = reinterpret_cast<__m256 *>(value.state_value + batch_offset);
            __m256 *output = reinterpret_cast<__m256 *>(value.output_value + batch_offset);

            if (value.prev_state_value) {
                prev_state_v = (reinterpret_cast<__m256 *>(value.prev_state_value + batch_offset))[i];
            }
            if (value.check_ig) {
                r_checkI = (reinterpret_cast<const __m256 *>(value.check_ig))[i];
                r_checkF = (reinterpret_cast<const __m256 *>(value.check_fg))[i];
                r_checkO = (reinterpret_cast<const __m256 *>(value.check_og))[i];
            }

            value_in[i] = math::avx_activation(value_in[i], cand_act);
            value_ig[i] = math::avx_activation(_mm256_add_ps(value_ig[i], _mm256_mul_ps(prev_state_v, r_checkI)), gate_act);
            value_fg[i] = math::avx_activation(_mm256_add_ps(value_fg[i], _mm256_mul_ps(prev_state_v, r_checkF)), gate_act);
            state[i] = _mm256_add_ps(_mm256_mul_ps(value_in[i],value_ig[i]), _mm256_mul_ps(prev_state_v, value_fg[i]));
            value_og[i] = math::avx_activation(_mm256_add_ps(value_og[i], _mm256_mul_ps(state[i], r_checkO)), gate_act);
            state_active[i] = math::avx_activation(state[i], cell_act);
            output[i] = _mm256_mul_ps(value_og[i], state_active[i]);
        }
    }
#endif
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberLstm<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::compute(LstmMetaValue<DataType_in> value,
                                                               int hidden_size, int batch_size,
                                                               const ActiveType &gate_act,
                                                               const ActiveType &cell_act,
                                                               const ActiveType &cand_act) {
#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < hidden_size; i++) {
            DataType_in *value_ig = value.gate_value + b * hidden_size * 4;
            DataType_in *value_fg = value_ig + hidden_size;
            DataType_in *value_in = value_ig + hidden_size * 2;
            DataType_in *value_og = value_ig + hidden_size * 3;
            DataType_in *state_active = value.state_active_value + b * hidden_size;
            DataType_in *state = value.state_value + b * hidden_size;
            DataType_in *output = value.output_value + b * hidden_size;
            DataType_in prev_state_v = 0;
            if (value.prev_state_value) {
                prev_state_v = *(value.prev_state_value + b * hidden_size + i);
            }

            DataType_in r_checkI = value.check_ig ? value.check_ig[i] : 0;
            DataType_in r_checkF = value.check_fg ? value.check_fg[i] : 0;
            DataType_in r_checkO = value.check_og ? value.check_og[i] : 0;

            math::activation(1, value_in + i, value_in + i, cand_act);
            DataType_in tmp = value_ig[i] + prev_state_v * r_checkI;
            math::activation(1, &tmp, value_ig + i, gate_act);
            tmp = value_fg[i] + prev_state_v * r_checkF;
            math::activation(1, &tmp, value_fg + i, gate_act);
            state[i] = value_in[i] * value_ig[i] + prev_state_v * value_fg[i];
            tmp = value_og[i] + state[i] * r_checkO;
            math::activation(1, &tmp, value_og + i, gate_act);
            math::activation(1, state + i, state_active + i, cell_act);
            output[i] = value_og[i] * state_active[i];
        }
    }
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberLstm<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        LstmParam<OpTensor> &param, Context<X86> &ctx) {
    avx2_available_ = jit::mayiuse(jit::avx2);

    DataTensor_in *input = inputs[0];

    const OpTensor *bias = param.bias();
    int frame_size = input->channel();
    int hidden_size = outputs[0]->channel();

    // aligned hidden_size with 8 float
    int aligned_size = 8;
    this->aligned_hidden_size_ = (hidden_size % aligned_size) ? ((hidden_size / aligned_size) + 1) * aligned_size : hidden_size;

    OpTensor *aligned_weights_data_h = nullptr;
    if (this->aligned_hidden_size_ != hidden_size) {
        Shape aligned_w_shape(this->aligned_hidden_size_, this->aligned_hidden_size_ * 4, 1, 1);
        aligned_weights_data_h = new OpTensor(aligned_w_shape);
    }

    DataType_op *weights_data = const_cast<DataType_op *>(param.weight()->data());
    MatrixInfo<DataType_op> *weight_x = nullptr;
    MatrixInfo<DataType_op> *weight_h = nullptr;
    MatrixInfo<DataType_op> *weight_h_tmp = nullptr;
    if (param.skip_input) {
        // if skip_input is true, the weights just includes [Wih, Wfh, Wch, Wph]
        weight_h = new MatrixInfo<DataType_op>(weights_data, hidden_size, (hidden_size * 4));
    }
    else {
        // split the weight to two parts: [Wix, Wfx, Wcx, Wox], [Wih, Wfh, Wch, Woh]
        weight_x = new MatrixInfo<DataType_op>(weights_data, frame_size, (hidden_size * 4));
        weight_h = new MatrixInfo<DataType_op>((weights_data + frame_size * hidden_size * 4), hidden_size, (hidden_size * 4));
    }

    if (this->aligned_hidden_size_ != hidden_size) {
        weight_h_tmp = weight_h;
        weight_h = new MatrixInfo<DataType_op>(aligned_weights_data_h->mutable_data(), this->aligned_hidden_size_, (this->aligned_hidden_size_ * 4));
        // do weight align
        int stride = 0;
        int diff = this->aligned_hidden_size_ - hidden_size;
        DataType_op *src = nullptr;
        DataType_op *dst = nullptr;
        for (int i = 0; i < this->aligned_hidden_size_; i++) {
            stride = 4 * (this->aligned_hidden_size_);

            dst = weight_h->buf() + i * stride;
            if (i >= hidden_size) {
                memset(dst, 0, stride * sizeof(DataType_op));
            } else {
                src = weight_h_tmp->buf() + i * 4 * hidden_size;
                for (int j = 0; j < 4; j++) {
                    memcpy(dst + j * this->aligned_hidden_size_,
                           src + j * hidden_size, hidden_size * sizeof(DataType_op));
                    memset(dst + j * this->aligned_hidden_size_ + hidden_size,
                           0, diff * sizeof(DataType_op));
                }
            }
        }

        delete weight_h_tmp;
    }

    // clean the packed weight
    safe_free(&(this->packed_w_x_));
    safe_free(&(this->packed_w_h_));
    // pack weights for Wix, Wfx, Wcx, Wox] and [Wih, Wfh, Wch, Woh]
    if (weight_x) {
        int m = input->num();
        this->packed_w_x_ = new mkl_packed_weight<OpDtype, LayOutType_op>(weight_x, m);
        this->packed_w_x_->pack();
    }
    this->packed_w_h_ = new mkl_packed_weight<OpDtype, LayOutType_op>(weight_h);
    this->packed_w_h_->pack();

    const OpTensor *init_t0 = param.init_hidden();
    safe_free(&batch_c0_);
    safe_free(&batch_h0_);
    // tensor for batched init cell and batched init hidden, they are both with size batch_size * hidden_size
    if (init_t0) {
        int batch_size = input->get_seq_offset().size() - 1;
        Shape batched_state_shape(batch_size, this->aligned_hidden_size_, 1 , 1);

        // create buf in create func, batch_size * hidden_size
        batch_c0_ = new OpTensor(batched_state_shape);

        // create buf in create func, batch_size * hidden_size
        batch_h0_ = new OpTensor(batched_state_shape);
    }

    bool with_peephole = param.with_peephole;
    if (bias && with_peephole) {
        const DataType_op *bias_data = bias->data();
        // shape for Wic, Wfc, Woc
        Shape weights_c_shape(1, this->aligned_hidden_size_, 1, 1);
        safe_free(&(this->check_ig_));
        safe_free(&(this->check_fg_));
        safe_free(&(this->check_og_));
        this->check_ig_ = new OpTensor(weights_c_shape);
        this->check_fg_ = new OpTensor(weights_c_shape);
        this->check_og_ = new OpTensor(weights_c_shape);
        memcpy(this->check_ig_->mutable_data(), bias_data + 4 * hidden_size, hidden_size * sizeof(DataType_op));
        memcpy(this->check_fg_->mutable_data(), bias_data + 5 * hidden_size, hidden_size * sizeof(DataType_op));
        memcpy(this->check_og_->mutable_data(), bias_data + 6 * hidden_size, hidden_size * sizeof(DataType_op));
    }

    safe_free(&weight_x);
    safe_free(&weight_h);
    safe_free(&aligned_weights_data_h);

    this->_ctx = ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberLstm<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        LstmParam<OpTensor> &param,
        Context<X86> &ctx) {
    DataTensor_in *input = inputs[0];
    DataTensor_out *hidden_out = outputs[0];
    int hidden_size = hidden_out->channel();

    // aligned hidden_size with AVX-512
    int aligned_size = 8;
    this->aligned_hidden_size_ = (hidden_size % aligned_size) ? ((hidden_size / aligned_size) + 1) * aligned_size : hidden_size;
    Shape aligned_output_shape(hidden_out->num(), this->aligned_hidden_size_, 1, 1);

    // xx = x * [Wix, Wfx, Wcx, Wox]
    Shape xx_shape(input->num(), hidden_size * 4, 1, 1);
    Shape aligned_xx_shape(input->num(), this->aligned_hidden_size_ * 4, 1, 1);
    // if current size < request size, realloc a buf
    this->xx_ = request_buf_for_input(this->xx_, xx_shape);
    this->batch_xx_ = request_buf_for_input(this->batch_xx_, aligned_xx_shape);
    this->batch_hidden_ = request_buf_for_input(this->batch_hidden_, aligned_output_shape);
    this->batch_cell_ = request_buf_for_input(this->batch_cell_, aligned_output_shape);
    this->batch_cell_act_ = request_buf_for_input(this->batch_cell_act_, aligned_output_shape);

    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberLstm<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        LstmParam<OpTensor> &param) {
    DataTensor_in *input = inputs[0];
    DataTensor_out *hidden_out = outputs[0];
    DataTensor_out *cell_out = nullptr;
    if (outputs.size() >= 2) {
        cell_out = outputs[1];
    }
    const OpTensor *bias = param.bias();
    const OpTensor *init_t0 = param.init_hidden();

    int hidden_size = hidden_out->channel();
    int batch_size = input->get_seq_offset().size() - 1;
    Shape offset(0, 0, 0, 0);

    // init state shape
    Shape init_state_shape(batch_size, hidden_size, 1 , 1);
    math::ReorderInitState<OpDtype, LayOutType_op> reorder;

    DataTensor_in *xx = nullptr;

    if (param.skip_input) {
        // if skip_input is true, the input memory layout should be
        // total_seq_len * (4 * hidden_size)
        xx = input;
    } else {
        // if skip_input is false, the input memory layout should be
        // total_seq_len * input_size
        // xx = x * [Wix, Wfx, Wcx, Wox]
        Shape xx_shape(input->num(), hidden_size * 4, 1, 1);

        // if current size < request size, realloc a buf for using
        xx = new DataTensor_in();
        this->xx_ = request_buf_for_input(this->xx_, xx_shape);
        xx->share_sub_buffer(*(this->xx_), xx_shape, offset);

        MatrixInfo<DataType_in> src((input->mutable_data()), input->num(), input->channel());
        MatrixInfo<DataType_in> dst((xx->mutable_data()), xx->num(), xx->channel());
        packed_w_x_->gemm_compute(src, &dst, 0.0f);

        // input activation
        int cnt = xx->size();
        DataType_out *p = xx->mutable_data();
        switch (param.input_activity) {
            case Active_stanh:
            case Active_tanh:
                math::parallel_activation(cnt, p, p, param.input_activity);
                break;
            case Active_unknow:
                break;
            default:
                        LOG(ERROR) << "not supported input activation";
                return SaberUnImplError;
        }
    }

    DataTensor_in batch_xx;
    Shape aligned_xx_shape(xx->num(), this->aligned_hidden_size_ * 4, 1 , 1);
    batch_xx.share_sub_buffer(*(this->batch_xx_), aligned_xx_shape, offset);

    DataTensor_out batch_hidden;
    Shape aligned_output_shape(hidden_out->num(), this->aligned_hidden_size_, 1 , 1);
    batch_hidden.share_sub_buffer(*(this->batch_hidden_), aligned_output_shape, offset);

    DataTensor_out batch_cell;
    batch_cell.share_sub_buffer(*(this->batch_cell_), aligned_output_shape, offset);

    DataTensor_out batch_cell_act;
    batch_cell_act.share_sub_buffer(*(this->batch_cell_act_), aligned_output_shape, offset);

    MatrixInfo<DataType_in> xx_matrix(xx->mutable_data(), xx->num(), xx->channel());
    MatrixInfo<DataType_in> batch_xx_matrix(batch_xx.mutable_data(), batch_xx.num(), batch_xx.channel());
    MatrixInfo<DataType_out> batch_hidden_matrix(batch_hidden.mutable_data(), batch_hidden.num(), batch_hidden.channel());
    MatrixInfo<DataType_out> batch_cell_matrix(batch_cell.mutable_data(), batch_cell.num(), batch_cell.channel());
    MatrixInfo<DataType_out> batch_cell_act_matrix(batch_cell_act.mutable_data(), batch_cell_act.num(), batch_cell_act.channel());

    // handle bias info
    if (bias) {
        // row-wise-add bias to batch_xx, the layout of bias [bi, bf, bc, bo]
        const DataType_op *bias_data = bias->data();
        for (int i = 0; i < input->num(); i++) {
            int row_size = 4 * hidden_size;
            cblas_saxpby(row_size, 1, bias_data, 1, 1, (xx_matrix.buf() + i * row_size), 1);
        }
    }

    // seq to batch meta data
    std::vector<std::vector<int>> seq_to_batch_meta;
    seq_to_batch_meta.push_back(input->get_seq_offset());

    // sequence to batch
    bool is_reverse = param.is_reverse;
    math::Seq2BatchFunctor<inDtype, LayOutType_in> to_batch;
    to_batch(xx, &batch_xx, seq_to_batch_meta, true, is_reverse, 4);

    std::vector<int> order(seq_to_batch_meta[2]);
    LstmMetaValue<DataType_in> lstm_value;
    bool with_peephole = param.with_peephole;
    if (bias && with_peephole) {
        // with peephole enable, [Wic, Wfc, Woc] is at the behind of bias
        const DataType_op *bias_data = bias->data();
        lstm_value.check_ig = this->check_ig_->data();
        lstm_value.check_fg = this->check_fg_->data();
        lstm_value.check_og = this->check_og_->data();
    } else {
        lstm_value.check_ig = nullptr;
        lstm_value.check_fg = nullptr;
        lstm_value.check_og = nullptr;
    }
    lstm_value.prev_state_value = nullptr;
    auto gate_act = param.gate_activity;
    auto cell_act = param.cell_activity;
    auto cand_act = param.candidate_activity;

    if (init_t0) {
        // if have init cell info, fill it to lstm value
        // get init_c0 from init_t0 and reorder it
        Shape offset(batch_size, 0, 0, 0);
        OpTensor init_c0;
        init_c0.share_sub_buffer(*init_t0, init_state_shape, offset);
        reorder(&init_c0, order, batch_c0_, true);

        lstm_value.prev_state_value = batch_c0_->mutable_data();
    }

    auto batch_starts = seq_to_batch_meta[0];
    size_t num_batch = batch_starts.size() - 1;
    for (size_t n = 0; n < num_batch; n++) {
        int bstart = batch_starts[n];
        int bend = batch_starts[n + 1];
        int cur_batch_size = bend - bstart;

        // xx += Ht-1 * [Wih, Wfh, Wch, Woh] according to batch number
        MatrixInfo<DataType_in> dst = batch_xx_matrix.subMatrixInfo(bstart, bend);
        if (n > 0) {
            // if n > 0, get Ht-1 information from last calc, and convert it to src
            int pre_h_start = batch_starts[n - 1];
            int pre_h_end = pre_h_start + cur_batch_size;
            MatrixInfo<DataType_in> src = batch_hidden_matrix.subMatrixInfo(pre_h_start, pre_h_end);
            packed_w_h_->gemm_compute(src, &dst);
        } else if (init_t0) {
            // if this is the fisrt time calc and the batch_h0_ is not NULL, then using the init hidden value as src
            // get init_h0 from init_t0 and reorder it
            Shape offset(0, 0, 0, 0);
            OpTensor init_h0;
            init_h0.share_sub_buffer(*init_t0, init_state_shape, offset);
            reorder(&init_h0, order, batch_h0_, true);

            MatrixInfo<DataType_in> src((batch_h0_->mutable_data()), batch_h0_->num(), batch_h0_->channel());
            packed_w_h_->gemm_compute(src, &dst);
        }

        // calc [Wic*Ct-1, Wfc*Ct-1, WocCt] and activation
        // fill lstm value with the calc result before and the output buf
        lstm_value.gate_value = dst.buf();
        lstm_value.output_value = batch_hidden_matrix.subMatrixInfo(bstart, bend).buf();
        lstm_value.state_value = batch_cell_matrix.subMatrixInfo(bstart, bend).buf();
        lstm_value.state_active_value = batch_cell_act_matrix.subMatrixInfo(bstart, bend).buf();
        if (avx2_available_) {
            compute_with_avx(lstm_value, this->aligned_hidden_size_, cur_batch_size, gate_act, cell_act, cand_act);
        } else {
            compute(lstm_value, this->aligned_hidden_size_, cur_batch_size, gate_act, cell_act, cand_act);
        }
        lstm_value.prev_state_value = lstm_value.state_value;
    }

    // batch to sequence
    math::Batch2SeqFunctor<outDtype, LayOutType_out> to_seq;
    to_seq(&batch_hidden, hidden_out, seq_to_batch_meta);

    if (cell_out) {
        to_seq(&batch_cell, cell_out, seq_to_batch_meta);
    }

    if (!param.skip_input && xx) {
        delete xx;
        xx = nullptr;
    }

    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberLstm<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init_conf(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        LstmParam<OpTensor> &param) {
    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberLstm<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::check_conf(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        LstmParam<OpTensor> &param) {
    return SaberSuccess;
}

template class SaberLstm<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} // namespace saber
} // namespace anakin
