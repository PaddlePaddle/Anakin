/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PAD_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PAD_H

#include "saber/funcs/impl/impl_pad.h"
#include "saber/core/data_traits.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPad<NV, OpDtype>:\
    public ImplBase<
            NV, OpDtype,
            PadParam<NV>> {

public:
    typedef typename DataTrait<NV, OpDtype> :: Dtype dtype;

    SaberPad() {}

    ~SaberPad() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             PadParam<NV> &param,
                             Context<NV> &ctx) {

        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               PadParam<NV> &param,
                               Context<NV> &ctx) {
        CHECK_EQ(2, param.pad_c.size());
        CHECK_EQ(2, param.pad_h.size());
        CHECK_EQ(2, param.pad_w.size());
        Shape out_stride = outputs[0]->get_stride();
        Shape in_stride = inputs[0]->get_stride();
        int in_n_index = inputs[0]->num_index();
        int in_c_index = inputs[0]->channel_index();
        int in_h_index = inputs[0]->height_index();
        int in_w_index = inputs[0]->width_index();
        int out_n_index = outputs[0]->num_index();
        int out_c_index = outputs[0]->channel_index();
        int out_h_index = outputs[0]->height_index();
        int out_w_index = outputs[0]->width_index();
        _out_n_stride = out_stride[out_n_index];
        _out_c_stride = out_stride[out_c_index];
        _out_h_stride = out_stride[out_h_index];
        _out_w_stride = out_stride[out_w_index];
        _in_n_stride = in_stride[in_n_index];
        _in_c_stride = in_stride[in_c_index];
        _in_h_stride = in_stride[in_h_index];
        _in_w_stride = in_stride[in_w_index];
        _img_offset = _out_c_stride * param.pad_c[0]\
            + _out_h_stride * param.pad_h[0] \
            + _out_w_stride * param.pad_w[0];

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs, \
        std::vector<Tensor<NV>*>& outputs, PadParam<NV> &param);
private:
    int _img_offset;
    int _in_n_stride;
    int _in_c_stride;
    int _in_h_stride;
    int _in_w_stride;
    int _out_n_stride;
    int _out_c_stride;
    int _out_h_stride;
    int _out_w_stride;
};

template class SaberPad<NV, AK_FLOAT>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PAD_H
