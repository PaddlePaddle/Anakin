#include "saber/funcs/impl/cuda/saber_lstm.h"
#include "saber/core/tensor_op.h"
#include "cuda_inline_activation.h"
namespace anakin {

namespace saber {

template <typename Dtype>
__global__ void cal_lstm_kernel_batch_with_peephole_anyactivate(
        const Dtype* w_x,  const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o,
        const Dtype* w_ci, const Dtype* w_cf, const Dtype* w_co, Dtype* cell,const int hidden_size, const int word_start_id,
        const ActiveType gate_activity, const ActiveType cell_activity,const ActiveType candidate_activity,Dtype* output,
        const int i_offset = 0, const int f_offset = 1, const int c_offset = 2,const int o_offset = 3) {

    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid < hidden_size) {

        const int emit_wx_offset = (word_start_id + batch_id) * hidden_size * 4;
        const Dtype* w_x_i = w_x + i_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_f = w_x + f_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_c = w_x + c_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_o = w_x + o_offset * hidden_size + emit_wx_offset;

        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;

        const Dtype c_1 = gate_c_p[tid];
        const Dtype gate_i = activate_cuda_float(w_x_i[tid] + b_i[tid] + w_ci[tid] * c_1,gate_activity);
        const Dtype gate_f = activate_cuda_float(w_x_f[tid] + b_f[tid] + w_cf[tid] * c_1,gate_activity);

        const Dtype gate_c_s = activate_cuda_float(w_x_c[tid]  + b_c[tid],cell_activity);
        const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const Dtype gate_o = activate_cuda_float(w_x_o[tid] + b_o[tid] + gate_c * w_co[tid],gate_activity);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * activate_cuda_float(gate_c,candidate_activity);
    }
}

template <typename Dtype>
__global__ void cal_lstm_kernel_batch_without_peephole_anyactivate(
        const Dtype* w_x,const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o, Dtype* cell,
        const int hidden_size, const int word_start_id, const ActiveType gate_activity,const ActiveType cell_activity,const ActiveType candidate_activity,
        Dtype* output, const int i_offset = 0,const int f_offset = 1, const int c_offset = 2,const int o_offset = 3) {

    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid < hidden_size) {

        const int emit_wx_offset = (word_start_id + batch_id) * hidden_size * 4;
        const Dtype* w_x_i = w_x + i_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_f = w_x + f_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_c = w_x + c_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_o = w_x + o_offset * hidden_size + emit_wx_offset;

        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;

        const Dtype c_1 = gate_c_p[tid];
        const Dtype gate_i = activate_cuda_float(w_x_i[tid]  + b_i[tid],gate_activity);
        const Dtype gate_f = activate_cuda_float(w_x_f[tid]  + b_f[tid],gate_activity);

        const Dtype gate_c_s = activate_cuda_float(w_x_c[tid]  + b_c[tid],cell_activity);
        const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const Dtype gate_o = activate_cuda_float(w_x_o[tid]  + b_o[tid],gate_activity);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * activate_cuda_float(gate_c,candidate_activity);
    }
}


template <typename Dtype>
__global__ void cal_lstm_kernel_batch_with_peephole(
        const Dtype* w_x,  const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o,
        const Dtype* w_ci, const Dtype* w_cf, const Dtype* w_co, Dtype* cell,const int hidden_size, const int word_start_id,
        Dtype* output, const int i_offset = 0, const int f_offset = 1, const int c_offset = 2,const int o_offset = 3) {

    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid < hidden_size) {

        const int emit_wx_offset = (word_start_id + batch_id) * hidden_size * 4;
        const Dtype* w_x_i = w_x + i_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_f = w_x + f_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_c = w_x + c_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_o = w_x + o_offset * hidden_size + emit_wx_offset;

        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;

        const Dtype c_1 = gate_c_p[tid];
        const Dtype gate_i = sigmoid_fluid(w_x_i[tid] + b_i[tid] + w_ci[tid] * c_1);
        const Dtype gate_f = sigmoid_fluid(w_x_f[tid] + b_f[tid] + w_cf[tid] * c_1);

        const Dtype gate_c_s = tanh_fluid(w_x_c[tid]  + b_c[tid]);
        const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const Dtype gate_o = sigmoid_fluid(w_x_o[tid] + b_o[tid] + gate_c * w_co[tid]);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * tanh_fluid(gate_c);
        //        printf("tid = %d, f = %f, i = %f, o = %f, hout = %f, w_x_i = %f, c_i = %f,c_out = %f, batch_id = %d\n",tid,gate_f,gate_i,gate_o,gate_h_p[tid],w_x_i[tid],c_1,gate_c,batch_id);
    }
}

template <typename Dtype>
__global__ void cal_lstm_kernel_batch_without_peephole(
        const Dtype* w_x,const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o, Dtype* cell,
        const int hidden_size, const int word_start_id, Dtype* output, const int i_offset = 0,
        const int f_offset = 1, const int c_offset = 2,const int o_offset = 3) {

    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid < hidden_size) {

        const int emit_wx_offset = (word_start_id + batch_id) * hidden_size * 4;
        const Dtype* w_x_i = w_x + i_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_f = w_x + f_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_c = w_x + c_offset * hidden_size + emit_wx_offset;
        const Dtype* w_x_o = w_x + o_offset * hidden_size + emit_wx_offset;


        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;

        const Dtype c_1 = gate_c_p[tid];
        const Dtype gate_i = sigmoid_fluid(w_x_i[tid]  + b_i[tid]);
        const Dtype gate_f = sigmoid_fluid(w_x_f[tid]  + b_f[tid]);

        const Dtype gate_c_s = tanh_fluid(w_x_c[tid]  + b_c[tid]);
        const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const Dtype gate_o = sigmoid_fluid(w_x_o[tid]  + b_o[tid]);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * tanh_fluid(gate_c);
    }
}

template<>
SaberStatus
SaberLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch_batch(
    const std::vector < DataTensor_in* >& inputs,
    std::vector < DataTensor_out* >& outputs,
    LstmParam < OpTensor >& param) {
    DataTensor_in* x = inputs[0];
    std::vector<int> offset_vec = x->get_seq_offset();
    int seq_sum = x->num();
    int batch_size = offset_vec.size() - 1;
    const InDataType* x_data = x->data();

    const OpDataType *weight_h = param.weight()->data()+4*_hidden_size*_word_size;
    const OpDataType *weight_w = param.weight()->data();
    const OpDataType *bias = param.bias()->data();
    const OpDataType *weight_peephole = param.bias()->data()+4*_hidden_size;
    const OutDataType* h_init = nullptr;
    const OutDataType* cell_init = nullptr;
    const InDataType* inner_x = inputs[0]->data();
    OutDataType* inner_h_out = outputs[0]->mutable_data();
    OutDataType* inner_cell = nullptr;


    std::vector<int> emit_offset_vec;
    int emit_length = 0;
    _temp_map_dev.try_expand_size(seq_sum);
    bool transform = _seq_util.get_sorted_map(offset_vec, emit_offset_vec, emit_length,
                     _ctx->get_compute_stream());
    bool is_reverse = param.is_reverse;

    if (inputs.size() > 1) {
        h_init = inputs[1]->data();
        _init_hidden.try_expand_size(batch_size * _hidden_size);
        h_init = _init_hidden.data();
    } else if (param.init_hidden() != nullptr) {
        h_init = param.init_hidden()->data();
        //FIXME:is it correct?
    } else {
        if (_temp_zero.valid_size() < batch_size * _hidden_size) {
            _temp_zero.try_expand_size(batch_size * _hidden_size);
            CUDA_CHECK(cudaMemsetAsync(_temp_zero.mutable_data(), 0,
                                       sizeof(OutDataType)*batch_size * _hidden_size,
                                       _ctx->get_compute_stream()));
        }

        h_init = _temp_zero.data();
    }

    _temp_wx.try_expand_size(seq_sum * 4 * _hidden_size);
    _temp_wh.try_expand_size(batch_size * 4 * _hidden_size);
    _temp_out.try_expand_size(seq_sum * _hidden_size * param.num_direction);
    _temp_cell.try_expand_size(batch_size * _hidden_size);

    if (transform) {
        _temp_x.try_expand_size(seq_sum * _word_size);
        _seq_util.seq_2_sorted_seq(x_data, _temp_x.mutable_data(), _word_size, _ctx->get_compute_stream());

        inner_h_out = _temp_out.mutable_data();
        inner_x = _temp_x.mutable_data();

        if (inputs.size() > 1 || param.init_hidden() != nullptr) {
            CHECK(false) << "not support inner_h_init != nullptr";
        }
    }

    inner_cell = _temp_cell.mutable_data();
    CUDA_CHECK(cudaMemsetAsync(inner_cell, 0, sizeof(OutDataType)*batch_size * _hidden_size,
                               _ctx->get_compute_stream()));

    OutDataType* temp_wh = _temp_wh.mutable_data();
    OutDataType* temp_wx = _temp_wx.mutable_data();

    _gemm_wx(seq_sum, 4 * _hidden_size, _word_size, 1.0, inner_x, 0.0, weight_w, temp_wx,
             _ctx->get_compute_stream());


    const int i_offset = 0;
    const int f_offset = 1;
    const int c_offset = 2;
    const int o_offset = 3;
    const OpDataType* b_i =  bias + i_offset * _hidden_size;
    const OpDataType* b_f =  bias + f_offset * _hidden_size;
    const OpDataType* b_c =  bias + c_offset * _hidden_size;
    const OpDataType* b_o =  bias + o_offset * _hidden_size;
    const OpDataType* w_ci = weight_peephole + 0 * _hidden_size;
    const OpDataType* w_cf = weight_peephole + 1 * _hidden_size;
    const OpDataType* w_co = weight_peephole + 2 * _hidden_size;

    DLOG(INFO) << "seq_sum = " << seq_sum << ",emit length = " << emit_offset_vec.size();

    for (int word_id = 0; word_id < emit_length; word_id++) {
        int real_word_id = word_id;
        int last_word_id = word_id - 1;

        if (param.is_reverse && batch_size == 1) {
            real_word_id = emit_length - word_id - 1;
            last_word_id = real_word_id + 1;
        }

        int emit_word_id_start = emit_offset_vec[real_word_id];
        int emit_word_id_end = emit_offset_vec[real_word_id + 1];
        int emit_word_length = emit_word_id_end - emit_word_id_start;
        const float* hin;

        if (word_id == 0) {
            hin = h_init;
        } else {
            hin = inner_h_out + emit_offset_vec[last_word_id] * _hidden_size;
        }

//        DLOG(INFO) << "word_id = " << word_id << ",emit_start = " << emit_word_id_start << ",emit_end=" <<emit_word_id_end;
        float* hout = nullptr;
        hout = emit_offset_vec[real_word_id] * _hidden_size + inner_h_out;

        //wh
        _gemm_wh(emit_word_length, 4 * _hidden_size, _hidden_size, 1.0, hin, 1.f,
                 weight_h,
                 temp_wx+emit_word_id_start*4*_hidden_size, _ctx->get_compute_stream());


        CHECK_LE(_hidden_size, 1024) << "now not support hidden size > 1024 for paddle formula";
        int frame_per_block = _hidden_size <= 1024 ? _hidden_size : 1024;

        if (param.gate_activity == Active_sigmoid_fluid && param.cell_activity == Active_tanh_fluid
                && param.candidate_activity == Active_tanh_fluid) {
            if (param.with_peephole) {
                cal_lstm_kernel_batch_with_peephole << < emit_word_length, frame_per_block, 0
                                                    , _ctx->get_compute_stream() >> >
                                                    (temp_wx, b_i,b_f,b_c,b_o, w_ci,w_cf,w_co, inner_cell, _hidden_size, emit_word_id_start, hout);
            } else {
                cal_lstm_kernel_batch_without_peephole << < emit_word_length, frame_per_block, 0
                                                       , _ctx->get_compute_stream() >> >
                                                       (temp_wx, b_i,b_f,b_c,b_o, inner_cell, _hidden_size, emit_word_id_start, hout);
            }
        } else {
            if (param.with_peephole) {
                cal_lstm_kernel_batch_with_peephole_anyactivate << < emit_word_length, frame_per_block, 0
                        , _ctx->get_compute_stream() >> >
                          (temp_wx, b_i, b_f, b_c, b_o, w_ci, w_cf, w_co, inner_cell, _hidden_size, emit_word_id_start, param.gate_activity,
                                  param.cell_activity, param.candidate_activity, hout);
            } else{
                cal_lstm_kernel_batch_without_peephole_anyactivate << < emit_word_length, frame_per_block, 0
                        , _ctx->get_compute_stream() >> >
                          (temp_wx, b_i, b_f, b_c, b_o, inner_cell, _hidden_size, emit_word_id_start, param.gate_activity,
                                  param.cell_activity, param.candidate_activity, hout);
            }
        }
    }

    if (transform) {
        _seq_util.sorted_seq_2_seq(_temp_out.data(), outputs[0]->mutable_data(), _hidden_size,
                                   _ctx->get_compute_stream());
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;

};

template<>
SaberStatus
SaberLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch_once(
    const std::vector < DataTensor_in* >& inputs,
    std::vector < DataTensor_out* >& outputs,
    LstmParam < OpTensor >& param) {

    return SaberSuccess;
};

template<>
SaberStatus
SaberLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(
    const std::vector < DataTensor_in* >& inputs,
    std::vector < DataTensor_out* >& outputs,
    LstmParam < OpTensor >& param) {
    CHECK_EQ(inputs.size(),1)<<"only support input size = 1";
    CHECK_EQ(outputs.size(),1)<<"only support outputs size = 1";
    CHECK_EQ(param.init_hidden()==nullptr, true )<<"only support param.init_hidden() == nullptr";
    CHECK_EQ(param.num_layers,1)<<"only support param.num_layers==1";
    return dispatch_batch(inputs, outputs, param);

}

template class SaberLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
}

