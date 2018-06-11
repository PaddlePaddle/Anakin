/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H

#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "mkl_packed_weight.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/impl_lstm.h"
//#include "saber/funcs/impl/x86/kernel/activation_functions.h"

namespace anakin {
namespace saber {

template <class T>
struct LstmMetaValue {
  T *gate_value;
  T *prev_state_value;
  T *state_value;
  T *state_active_value;
  T *output_value;
  const T *check_ig;
  const T *check_fg;
  const T *check_og;
};

template <typename T>
void sigmoid(size_t len, T *x, T *y) {
#pragma omp parallel for if (len > 1)
    for (size_t i = 0; i < len; i++) {
        y[i] = 1. / (1. + exp(-x[i]));
    }
}

template <typename T>
void relu(size_t len, T *x, T *y) {
#pragma omp parallel for if (len > 1)
    for (size_t i = 0; i < len; i++) {
        y[i] = x[i] < 0 ? 0 : x[i];
    }
}

template <typename T>
void tanh(size_t len, T *x, T *y) {
#pragma omp parallel for if (len > 1)
    for (size_t i = 0; i < len; i++) {
        T e_2x = exp(2 * x[i]);
        y[i] = (e_2x - 1) / (e_2x + 1);
    }
}

template <typename T>
void stanh(size_t len, T *x, T *y) {
#pragma omp parallel for if (len > 1)
    for (size_t i = 0; i < len; i++) {
        T e_x = exp(4. * x[i] / 3.);
        y[i] = 1.7159 * (e_x - 1) / (e_x + 1);
    }
}

template <typename T>
void identity(size_t len, T *x, T *y) {
#pragma omp parallel for if (len > 1)
    for (size_t i = 0; i < len; i++) {
        y[i] = x[i];
    }
}

template <typename T>
struct Active {
    typedef void (*Act)(size_t, T*, T*);
};

static Active<float>::Act kActFloat[] = {
        nullptr,
        &sigmoid<float>,
        &relu<float>,
        &tanh<float>,
        nullptr,
        nullptr,
        &identity<float>,
        nullptr,
        nullptr,
        &stanh<float>
};

inline void activation(size_t len, float *src, float *dst, int index) {
    auto *func = kActFloat[index];
    if (!func) {
        LOG(ERROR) << "activation not implemented!";
    }
    func(len, src, dst);
}

template <class T>
class lstm {
 public:
  void operator()(T *value_in, T *value_ig, T *value_fg, T *value_og,
                             T *prev_state, T *state, T *state_atv, T *output,
                             T *checkI, T *checkF, T *checkO,
                             ActiveType active_node,
                             ActiveType active_gate,
                             ActiveType active_state,
                             size_t size = 1) {
    activation(size, value_in, value_in, active_node);
    T tmp = *value_ig + (*prev_state) * (*checkI);
    activation(size, &tmp, value_ig, active_gate);
    tmp = *value_fg + (*prev_state) * (*checkF);
    activation(size, &tmp, value_fg, active_gate);
    *state = (*value_in) * (*value_ig) + (*prev_state) * (*value_fg);
    tmp = *value_og + (*state) * (*checkO);
    activation(size, &tmp, value_og, active_gate);
    activation(size, state, state_atv, active_state);
    *output = (*value_og) * (*state_atv);
  }
};

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberLstm<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>: public ImplBase<
        Tensor<X86, inDtype, LayOutType_in>,
        Tensor<X86, outDtype, LayOutType_out>,
        Tensor<X86, OpDtype, LayOutType_op>,
        LstmParam<Tensor<X86, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    SaberLstm() :
    packed_w_x_(nullptr), packed_w_h_(nullptr),
    batch_h0_(nullptr), batch_c0_(nullptr) {
    }

    ~SaberLstm() {
        if (packed_w_x_) {
            delete packed_w_x_;
            packed_w_x_ = nullptr;
        }
        if (packed_w_h_) {
            delete packed_w_h_;
            packed_w_h_ = nullptr;
        }
        if (batch_h0_) {
            delete batch_h0_;
            batch_h0_ = nullptr;
        }
        if (batch_c0_) {
            delete batch_c0_;
            batch_c0_ = nullptr;
        }
        delete cell_out;
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             LstmParam<OpTensor> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               LstmParam<OpTensor> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 LstmParam<OpTensor> &param) override;

    virtual SaberStatus init_conf(
                                  const std::vector<DataTensor_in*>& inputs,
                                  std::vector<DataTensor_out*>& outputs,
                                  LstmParam<OpTensor>& param);

private:
    mkl_packed_weight<OpDtype, LayOutType_op> * packed_w_x_;
    mkl_packed_weight<OpDtype, LayOutType_op> * packed_w_h_;
    DataTensor_out _inner_cell_workspace;

    OpTensor *batch_h0_;
    OpTensor *batch_c0_;
    DataTensor_out *cell_out;
    virtual SaberStatus check_conf(const std::vector<DataTensor_in*>& inputs,
                                   std::vector<DataTensor_out*>& outputs,
                                   LstmParam<OpTensor>& param);
    virtual void compute(LstmMetaValue<DataType_in> value,
                      int frame_size, int batch_size,
                      const ActiveType &gate_act,
                      const ActiveType &cell_act,
                      const ActiveType &cand_act);
    virtual void cpu_lstm_forward(LstmMetaValue<DataType_in> value, int frame_size,
                      ActiveType active_node, ActiveType active_gate,
                      ActiveType active_state);
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
