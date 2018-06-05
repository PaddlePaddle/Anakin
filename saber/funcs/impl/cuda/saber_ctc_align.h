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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CTC_ALIGN_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CTC_ALIGN_H
   
#include "saber/funcs/impl/impl_ctc_align.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberCtcAlign<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        CtcAlignParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberCtcAlign()
    {}

    ~SaberCtcAlign() {

    }

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        CtcAlignParam<OpTensor>& param, 
                        Context<NV> &ctx) {
        this->_ctx = ctx;
        Shape offset_shape = {inputs[0]->num(), 1, 1, 1};
        _in_offset.re_alloc(offset_shape);
        _out_offset.re_alloc(offset_shape);
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        CtcAlignParam<OpTensor>& param, 
                        Context<NV>& ctx) {
        Shape offset_shape = {inputs[0]->get_seq_offset().size(), 1, 1, 1};
        _in_offset.reshape(offset_shape);
        _out_offset.reshape(offset_shape);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        CtcAlignParam<OpTensor>& param);

private:
    Tensor<NV, AK_INT32, LayOutType_in> _in_offset;
    Tensor<NV, AK_INT32, LayOutType_out> _out_offset;
};

template class SaberCtcAlign<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}

}

#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
