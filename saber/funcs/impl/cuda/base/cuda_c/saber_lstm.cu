#include "saber/funcs/impl/cuda/saber_lstm.h"
#include "saber/core/tensor_op.h"
#include "cuda_inline_activation.h"
namespace anakin {

namespace saber {

template <typename Dtype>
__global__ void cal_lstm_kernel_batch_with_peephole_anyactivate(
        const Dtype* w_x,  const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o,
        const Dtype* w_ci, const Dtype* w_cf, const Dtype* w_co, Dtype* cell,const int hidden_size,
        const int aligned_hidden_size,const int batch_size,const int word_start_id,
        const ActiveType gate_activity, const ActiveType cell_activity,const ActiveType candidate_activity,Dtype* output
        ) {

    const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
    const int batch_id = thread_id/aligned_hidden_size;
    const int tid=thread_id%aligned_hidden_size;
    if (tid < hidden_size && batch_id<batch_size) {
        Dtype(*gat_act)(const Dtype)=Activate_inner<Dtype>(gate_activity);
        Dtype(*cell_act)(const Dtype)=Activate_inner<Dtype>(cell_activity);
        Dtype(*candi_act)(const Dtype)=Activate_inner<Dtype>(candidate_activity);

        const int emit_wx_offset = (word_start_id + batch_id) * hidden_size * 4;
        const Dtype* w_x_i = w_x  + emit_wx_offset;
        const Dtype* w_x_f = w_x_i + hidden_size ;
        const Dtype* w_x_c = w_x_f + hidden_size;
        const Dtype* w_x_o = w_x_c + hidden_size;


        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;

        const Dtype c_1 = gate_c_p[tid];
        const Dtype gate_i = gat_act(w_x_i[tid] + b_i[tid] + w_ci[tid] * c_1);
        const Dtype gate_f = gat_act(w_x_f[tid] + b_f[tid] + w_cf[tid] * c_1);

        const Dtype gate_c_s = cell_act(w_x_c[tid]  + b_c[tid]);
        const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const Dtype gate_o = gat_act(w_x_o[tid] + b_o[tid] + gate_c * w_co[tid]);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * candi_act(gate_c);
    }
}

template <typename Dtype>
__global__ void cal_lstm_kernel_batch_without_peephole_anyactivate(
        const Dtype* w_x,const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o, Dtype* cell,
        const int hidden_size, const int aligned_hidden_size,const int batch_size,const int word_start_id, const ActiveType gate_activity,const ActiveType cell_activity,const ActiveType candidate_activity,
        Dtype* output) {

    const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
    const int batch_id = thread_id/aligned_hidden_size;
    const int tid=thread_id%aligned_hidden_size;
    if (tid < hidden_size && batch_id<batch_size) {
        Dtype(*gat_act)(const Dtype)=Activate_inner<Dtype>(gate_activity);
        Dtype(*cell_act)(const Dtype)=Activate_inner<Dtype>(cell_activity);
        Dtype(*candi_act)(const Dtype)=Activate_inner<Dtype>(candidate_activity);

        const int emit_wx_offset = (word_start_id + batch_id) * hidden_size * 4;
        const Dtype* w_x_i = w_x + emit_wx_offset;
        const Dtype* w_x_f = w_x_i + hidden_size ;
        const Dtype* w_x_c = w_x_f + hidden_size;
        const Dtype* w_x_o = w_x_c + hidden_size;


        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;

        const Dtype c_1 = gate_c_p[tid];
        const Dtype gate_i = gat_act(w_x_i[tid]  + b_i[tid]);
        const Dtype gate_f = gat_act(w_x_f[tid]  + b_f[tid]);

        const Dtype gate_c_s = cell_act(w_x_c[tid]  + b_c[tid]);
        const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const Dtype gate_o = gat_act(w_x_o[tid]  + b_o[tid]);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * candi_act(gate_c);
//        printf("tid = %d, f = %f, i = %f, o = %f, hout = %f, w_x_i = %f, c_i = %f,c_out = %f, batch_id = %d\n",tid,gate_f,gate_i,gate_o,gate_h_p[tid],w_x_i[tid],c_1,gate_c,batch_id);
    }
}


template <typename Dtype>
__global__ void cal_lstm_kernel_batch_with_peephole(
        const Dtype* w_x,  const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o,
        const Dtype* w_ci, const Dtype* w_cf, const Dtype* w_co, Dtype* cell,const int hidden_size,
        const int aligned_hidden_size,const int batch_size, const int word_start_id,
        Dtype* output) {


    const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
    const int batch_id = thread_id/aligned_hidden_size;
    const int tid=thread_id%aligned_hidden_size;
    if (tid < hidden_size && batch_id<batch_size) {

        const int emit_wx_offset = (word_start_id + batch_id) * hidden_size * 4;
        const Dtype* w_x_i = w_x + emit_wx_offset;
        const Dtype* w_x_f = w_x_i + hidden_size ;
        const Dtype* w_x_c = w_x_f + hidden_size;
        const Dtype* w_x_o = w_x_c + hidden_size;

        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;

        const Dtype c_1 = gate_c_p[tid];
        const Dtype gate_i = Sigmoid(w_x_i[tid] + b_i[tid] + w_ci[tid] * c_1);
        const Dtype gate_f = Sigmoid(w_x_f[tid] + b_f[tid] + w_cf[tid] * c_1);

        const Dtype gate_c_s = Tanh(w_x_c[tid]  + b_c[tid]);
        const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const Dtype gate_o = Sigmoid(w_x_o[tid] + b_o[tid] + gate_c * w_co[tid]);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * Tanh(gate_c);
        //        printf("tid = %d, f = %f, i = %f, o = %f, hout = %f, w_x_i = %f, c_i = %f,c_out = %f, batch_id = %d\n",tid,gate_f,gate_i,gate_o,gate_h_p[tid],w_x_i[tid],c_1,gate_c,batch_id);
    }
}


template <typename Dtype>
__global__ void cal_lstm_kernel_batch_without_peephole(
        const Dtype* w_x,const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o, Dtype* cell,
        const int hidden_size, const int aligned_hidden_size,const int batch_size,const int word_start_id, Dtype* output) {

    const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
    const int batch_id = thread_id/aligned_hidden_size;
    const int tid=thread_id%aligned_hidden_size;
    if (tid < hidden_size && batch_id<batch_size) {

        const int emit_wx_offset = (word_start_id + batch_id) * hidden_size * 4;
        const Dtype* w_x_i = w_x + emit_wx_offset;
        const Dtype* w_x_f = w_x_i + hidden_size ;
        const Dtype* w_x_c = w_x_f + hidden_size;
        const Dtype* w_x_o = w_x_c + hidden_size;

        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;

        const Dtype c_1 = gate_c_p[tid];
        const Dtype gate_i = Sigmoid_fluid(w_x_i[tid]  + b_i[tid]);
        const Dtype gate_f = Sigmoid_fluid(w_x_f[tid]  + b_f[tid]);

        const Dtype gate_c_s = Tanh_fluid(w_x_c[tid]  + b_c[tid]);
        const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const Dtype gate_o = Sigmoid_fluid(w_x_o[tid]  + b_o[tid]);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * Tanh_fluid(gate_c);

    }
}

template<>
SaberStatus
SaberLstm<NV, AK_FLOAT>::dispatch_batch(
    const std::vector < Tensor<NV>* >& inputs,
    std::vector < Tensor<NV>* >& outputs,
    LstmParam < NV >& param) {
    Tensor<NV>* x = inputs[0];
    std::vector<int> offset_vec = x->get_seq_offset()[x->get_seq_offset().size()-1];

    int seq_sum = x->num();
    int batch_size = offset_vec.size() - 1;
    const OpDataType* x_data = (const OpDataType*)x->data();

    const OpDataType *weight_h = (const OpDataType *)(param.weight()->data())+4*_hidden_size*_word_size;
    const OpDataType *weight_w = (const OpDataType *)param.weight()->data();
    const OpDataType *bias = (const OpDataType *)param.bias()->data();
    const OpDataType *weight_peephole = (const OpDataType *)(param.bias()->data())+4*_hidden_size;
    const OpDataType* h_init = nullptr;
    const OpDataType* inner_x = (const OpDataType *)inputs[0]->data();
    OpDataType* inner_h_out = (OpDataType *)outputs[0]->mutable_data();
    OpDataType* inner_cell = nullptr;

    _gemm_wx = saber_find_fast_sass_gemm(false, false, seq_sum, 4 * _hidden_size,_word_size);
    _gemm_wh = saber_find_fast_sass_gemm(false, false, batch_size, 4 * _hidden_size, _hidden_size);

    utils::try_expand_tensor(_temp_map_dev,seq_sum);
    bool transform = _seq_util.get_sorted_map(offset_vec, this->_ctx->get_compute_stream());
    std::vector<int> emit_offset_vec=_seq_util.get_emit_offset_vec();
    int emit_length = emit_offset_vec.size()-1;

    if (inputs.size() > 1) {
        h_init = (const OpDataType *)inputs[1]->data();
        utils::try_expand_tensor(_init_hidden,batch_size * _hidden_size);
        h_init = (const OpDataType *)_init_hidden.data();
    } else if (param.init_hidden() != nullptr) {
        h_init = (const OpDataType *)param.init_hidden()->data();
        //FIXME:is it correct?
    } else {
        if (_temp_zero.valid_size() < batch_size * _hidden_size) {
            utils::try_expand_tensor(_temp_zero,batch_size * _hidden_size);
            CUDA_CHECK(cudaMemsetAsync(_temp_zero.mutable_data(), 0,
                                       sizeof(OpDataType)*batch_size * _hidden_size,
                                       _ctx->get_compute_stream()));
        }
        h_init = (const OpDataType *)_temp_zero.data();
    }

    utils::try_expand_tensor(_temp_wx,seq_sum * 4 * _hidden_size);
    utils::try_expand_tensor(_temp_wh,batch_size * 4 * _hidden_size);
    utils::try_expand_tensor(_temp_out,seq_sum * _hidden_size * param.num_direction);
    utils::try_expand_tensor(_temp_cell,batch_size * _hidden_size);

    if (transform) {
        utils::try_expand_tensor(_temp_x,seq_sum * _word_size);
        _seq_util.seq_2_sorted_seq(x_data, (OpDataType *)_temp_x.mutable_data(), _word_size, _ctx->get_compute_stream());

        inner_h_out = (OpDataType *)_temp_out.mutable_data();
        inner_x = (OpDataType *)_temp_x.mutable_data();

        if (inputs.size() > 1 || param.init_hidden() != nullptr) {
            CHECK(false) << "not support inner_h_init != nullptr";
        }
    }


    inner_cell = (OpDataType *)_temp_cell.mutable_data();
    CUDA_CHECK(cudaMemsetAsync(inner_cell, 0, sizeof(OpDataType)*batch_size * _hidden_size,
                               _ctx->get_compute_stream()));

    OpDataType* temp_wh = (OpDataType *)_temp_wh.mutable_data();
    OpDataType* temp_wx = (OpDataType *)_temp_wx.mutable_data();

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
    const OpDataType* w_ci = nullptr;
    const OpDataType* w_cf =nullptr;
    const OpDataType* w_co =nullptr;
    if(param.with_peephole){
        w_ci = weight_peephole + 0 * _hidden_size;
        w_cf = weight_peephole + 1 * _hidden_size;
        w_co = weight_peephole + 2 * _hidden_size;
    }


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
        const OpDataType* hin;

        if (word_id == 0) {
            hin = h_init;
        } else {
            hin = inner_h_out + emit_offset_vec[last_word_id] * _hidden_size;
        }

//        DLOG(INFO) << "word_id = " << word_id << ",emit_start = " << emit_word_id_start << ",emit_end=" <<emit_word_id_end;
        OpDataType* hout = nullptr;
        hout = emit_offset_vec[real_word_id] * _hidden_size + inner_h_out;


        //wh
        _gemm_wh(emit_word_length, 4 * _hidden_size, _hidden_size, 1.0, hin, 1.f,
                 weight_h,
                 temp_wx+emit_word_id_start*4*_hidden_size, _ctx->get_compute_stream());



        const int block_dim=512;
        const int grid_dim=round_up(emit_word_length*_aligned_hidden_size,block_dim);


        if (param.gate_activity == Active_sigmoid && param.cell_activity == Active_tanh
                && param.candidate_activity == Active_tanh) {
            if (param.with_peephole) {

                cal_lstm_kernel_batch_with_peephole << <grid_dim, block_dim , 0
                                                    , _ctx->get_compute_stream() >> >
                                                    (temp_wx, b_i,b_f,b_c,b_o, w_ci,w_cf,w_co, inner_cell, _hidden_size,_aligned_hidden_size,emit_word_length, emit_word_id_start, hout);
            } else {
                cal_lstm_kernel_batch_without_peephole << < grid_dim, block_dim , 0
                                                       , _ctx->get_compute_stream() >> >
                                                       (temp_wx, b_i,b_f,b_c,b_o, inner_cell, _hidden_size, _aligned_hidden_size,emit_word_length,emit_word_id_start, hout);
            }
        } else {
            if (param.with_peephole) {
                cal_lstm_kernel_batch_with_peephole_anyactivate << < grid_dim, block_dim , 0
                        , _ctx->get_compute_stream() >> >
                          (temp_wx, b_i, b_f, b_c, b_o, w_ci, w_cf, w_co, inner_cell, _hidden_size, _aligned_hidden_size,emit_word_length,emit_word_id_start, param.gate_activity,
                                  param.cell_activity, param.candidate_activity, hout);
            } else{
                cal_lstm_kernel_batch_without_peephole_anyactivate << < grid_dim, block_dim , 0
                        , _ctx->get_compute_stream() >> >
                          (temp_wx, b_i, b_f, b_c, b_o, inner_cell, _hidden_size,_aligned_hidden_size,emit_word_length, emit_word_id_start, param.gate_activity,
                                  param.cell_activity, param.candidate_activity, hout);
            }
        }
    }

    if (transform) {
        _seq_util.sorted_seq_2_seq((const OpDataType *)_temp_out.data(), (OpDataType *)outputs[0]->mutable_data(), _hidden_size,
                                   _ctx->get_compute_stream());
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;

};
//TODO:complate dispatch_once
template<>
SaberStatus
SaberLstm<NV, AK_FLOAT>::dispatch_once(
    const std::vector < Tensor<NV>* >& inputs,
    std::vector < Tensor<NV>* >& outputs,
    LstmParam < NV >& param) {

    return SaberSuccess;
};

template<>
SaberStatus
SaberLstm<NV, AK_FLOAT>::dispatch(
    const std::vector < Tensor<NV>* >& inputs,
    std::vector < Tensor<NV>* >& outputs,
    LstmParam < NV >& param) {
    CHECK_EQ(inputs.size(),1)<<"only support input size = 1";
    CHECK_EQ(outputs.size(),1)<<"only support outputs size = 1";
    CHECK_EQ(param.init_hidden()==nullptr, true )<<"only support param.init_hidden() == nullptr";
    CHECK_EQ(param.num_layers,1)<<"only support param.num_layers==1";

    return dispatch_batch(inputs, outputs, param);

}
DEFINE_OP_TEMPLATE(SaberLstm, LstmParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberLstm, LstmParam, NV, AK_INT8);
}
}

