/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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
#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SLICE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SLICE_H

#include "saber/funcs/impl/impl_slice.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberSlice<ARM, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        SliceParam<Tensor<ARM, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberSlice() {
        _slice_num = 4;
        _slice_size = 0;
    }
    ~SaberSlice() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             SliceParam<OpTensor> &param, Context<ARM> &ctx) override {
        // get context
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               SliceParam<OpTensor> &param, Context<ARM> &ctx) override {
        this->_ctx = ctx;
        _slice_num = inputs[0]->count_valid(0, param.axis);
        _slice_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
       return SaberSuccess;
    }

    virtual SaberStatus dispatch(std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 SliceParam<OpTensor> &param) override {
        int offset_slice_axis = 0;
        const InDataType* din = inputs[0]->data();
        const int in_slice_axis = inputs[0]->valid_shape[param.axis];
        for (int i = 0; i < outputs.size(); ++i) {
            OutDataType* dout = outputs[i]->mutable_data();
            const int out_slice_axis = outputs[i]->valid_shape[param.axis];
            for (int n = 0; n < _slice_num; ++n) {
                const int out_offset = n * out_slice_axis * _slice_size;
                const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
                memcpy((void*)(dout + out_offset), (void*)(din + in_offset), \
                sizeof(OutDataType) * out_slice_axis * _slice_size);
            }
            offset_slice_axis += out_slice_axis;
        }
        return SaberSuccess;
    }

private:
    int _slice_num;
    int _slice_size;
};

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_POOLING_H
