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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_VENDER_LSTM_H

#include "saber/saber_types.h"
#include "saber/saber_funcs_param.h"
#include "mkl_packed_weight.h"
#include "sequence2batch.h"

#include "saber/funcs/impl/impl_lstm.h"

namespace anakin {
namespace saber {

template <class T>
struct LstmMetaValue {
    T* gate_value;
    T* prev_state_value;
    T* state_value;
    T* state_active_value;
    T* output_value;
    const T* check_ig;
    const T* check_fg;
    const T* check_og;
};

template <DataType OpDtype>
class VenderLstm<X86, OpDtype>: public ImplBase <
        X86, OpDtype,LstmParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    VenderLstm() :
        avx2_available_(false), max_thread_num_(1),
        packed_w_x_(nullptr), packed_w_h_(nullptr),
        batch_h0_(nullptr), batch_c0_(nullptr), check_ig_(nullptr),
        check_fg_(nullptr), check_og_(nullptr),
        xx_(nullptr), batch_xx_(nullptr), batch_hidden_(nullptr),
        batch_cell_(nullptr), batch_cell_act_(nullptr), aligned_hidden_size_(0) {
        //        LOG(INFO)<<"vender construct";
    }

    ~VenderLstm() {
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

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             LstmParam<X86>& param,
                             Context<X86>& ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               LstmParam<X86>& param,
                               Context<X86>& ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 LstmParam<X86>& param) override;

    virtual SaberStatus init_conf(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        LstmParam<X86>& param);

private:
    inline void safe_free(MatrixInfo<OpDataType>** ptr) {
        if (*ptr) {
            delete (*ptr);
            (*ptr) = nullptr;
        }
    }

    inline void safe_free(Tensor<X86>** ptr) {
        if (*ptr) {
            delete (*ptr);
            (*ptr) = nullptr;
        }
    }

    inline void safe_free(mkl_packed_weight<OpDataType, NCHW>** ptr) {
        if (*ptr) {
            delete (*ptr);
            (*ptr) = nullptr;
        }
    }

    inline Tensor<X86>* request_buf_for_input(Tensor<X86>* input, Shape required_shape) {
        if (input) {
            int len = 1;

            if (required_shape.size() == 0) {
                len = 0;
            }

            for (int i = 0; i < required_shape.size(); i++) {
                len *= required_shape[i];
            }

            if (input->size() < len) {
                input->re_alloc(required_shape,input->get_dtype());
            }
        } else {
            input = new Tensor<X86>(required_shape);
        }

        return input;
    }

    bool avx2_available_;
    int max_thread_num_;

    mkl_packed_weight<OpDataType, NCHW>* packed_w_x_;
    mkl_packed_weight<OpDataType, NCHW>* packed_w_h_;
    Tensor<X86>* batch_h0_;
    Tensor<X86>* batch_c0_;
    Tensor<X86>* check_ig_;
    Tensor<X86>* check_fg_;
    Tensor<X86>* check_og_;
    // buf for storing data after calculating x * [Wix, Wfx, Wcx, Wox]
    Tensor<X86>* xx_;
    // buf for storing data after xx calculating seq to batch
    Tensor<X86>* batch_xx_;

    // buf for storing batch tmp data
    Tensor<X86>* batch_hidden_;
    Tensor<X86>* batch_cell_;
    Tensor<X86>* batch_cell_act_;
    /*aligned with 256bit(8 float)*/
    int aligned_hidden_size_;

    virtual SaberStatus check_conf(const std::vector<Tensor<X86>*>& inputs,
                                   std::vector<Tensor<X86>*>& outputs,
                                   LstmParam<X86>& param);

    virtual void compute(LstmMetaValue<OpDataType> value,
                         int hidden_size, int batch_size,
                         const ActiveType& gate_act,
                         const ActiveType& cell_act,
                         const ActiveType& cand_act);
    virtual void compute_with_avx(LstmMetaValue<OpDataType> value,
                                  int hidden_size, int batch_size,
                                  const ActiveType& gate_act,
                                  const ActiveType& cell_act,
                                  const ActiveType& cand_act);
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
