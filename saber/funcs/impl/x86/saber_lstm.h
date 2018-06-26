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
#include "sequence2batch.h"

#include "saber/funcs/impl/impl_lstm.h"

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
            avx2_available_(false), max_thread_num_(1),
            packed_w_x_(nullptr), packed_w_h_(nullptr),
            batch_h0_(nullptr), batch_c0_(nullptr), check_ig_(nullptr),
            check_fg_(nullptr), check_og_(nullptr),
            xx_(nullptr), batch_xx_(nullptr), batch_hidden_(nullptr),
            batch_cell_(nullptr), batch_cell_act_(nullptr), aligned_hidden_size_(0) {
    }

    ~SaberLstm() {
        safe_free(&packed_w_x_);
        safe_free(&packed_w_h_);
        safe_free(&batch_h0_);
        safe_free(&batch_c0_);
        safe_free(&check_ig_);
        safe_free(&check_fg_);
        safe_free(&check_og_);
        safe_free(&xx_);
        safe_free(&batch_xx_);
        safe_free(&batch_hidden_);
        safe_free(&batch_cell_);
        safe_free(&batch_cell_act_);
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
    inline void safe_free(MatrixInfo<DataType_op> **ptr) {
        if (*ptr) {
            delete (*ptr);
            (*ptr) = nullptr;
        }
    }

    inline void safe_free(DataTensor_in **ptr) {
        if (*ptr) {
            delete (*ptr);
            (*ptr) = nullptr;
        }
    }

    inline void safe_free(mkl_packed_weight<OpDtype, LayOutType_op> **ptr) {
        if (*ptr) {
            delete (*ptr);
            (*ptr) = nullptr;
        }
    }

    inline DataTensor_in* request_buf_for_input(DataTensor_in *input, Shape required_shape) {
        if (input) {
            int len = 1;
            if (required_shape.size() == 0) {
                len = 0;
            }
            for (int i = 0; i < required_shape.size(); i++) {
                len *= required_shape[i];
            }
            if (input->size() < len) {
                input->re_alloc(required_shape);
            }
        } else {
            input = new DataTensor_in(required_shape);
        }
        return input;
    }

    bool avx2_available_;
    int max_thread_num_;

    mkl_packed_weight<OpDtype, LayOutType_op> *packed_w_x_;
    mkl_packed_weight<OpDtype, LayOutType_op> *packed_w_h_;
    OpTensor *batch_h0_;
    OpTensor *batch_c0_;
    OpTensor *check_ig_;
    OpTensor *check_fg_;
    OpTensor *check_og_;
    // buf for storing data after calculating x * [Wix, Wfx, Wcx, Wox]
    DataTensor_in *xx_;
    // buf for storing data after xx calculating seq to batch
    DataTensor_in *batch_xx_;

    // buf for storing batch tmp data
    DataTensor_out *batch_hidden_;
    DataTensor_out *batch_cell_;
    DataTensor_out *batch_cell_act_;
    /*aligned with 256bit(8 float)*/
    size_t aligned_hidden_size_;

    virtual SaberStatus check_conf(const std::vector<DataTensor_in*>& inputs,
                                   std::vector<DataTensor_out*>& outputs,
                                   LstmParam<OpTensor>& param);

    virtual void compute(LstmMetaValue<DataType_in> value,
                         int hidden_size, int batch_size,
                         const ActiveType &gate_act,
                         const ActiveType &cell_act,
                         const ActiveType &cand_act);
    virtual void compute_with_avx(LstmMetaValue<DataType_in> value,
                                  int hidden_size, int batch_size,
                                  const ActiveType &gate_act,
                                  const ActiveType &cell_act,
                                  const ActiveType &cand_act);
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
