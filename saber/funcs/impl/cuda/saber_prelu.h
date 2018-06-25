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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PRELU_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PRELU_H

#include "saber/funcs/impl/impl_prelu.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
class SaberPrelu<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
            Tensor<NV, inDtype, LayOutType_in>,
            Tensor<NV, outDtype, LayOutType_out>,
            Tensor<NV, OpDtype, LayOutType_op>,
            PreluParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberPrelu() = default;
    ~SaberPrelu() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             PreluParam<OpTensor> &param,
                             Context<NV> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               PreluParam<OpTensor> &param,
                               Context<NV> &ctx) {
        // compute inner and outer size
        int channel_index = inputs[0]->channel_index();
        _dims = inputs[0]->dims();
        _size = inputs[0]->valid_size();
        _channels = inputs[0]->channel();
        _inner_size = inputs[0]->count_valid(channel_index + 1, _dims);
        _outer_size = inputs[0]->count_valid(0, channel_index);
        if (!param.channel_shared) {
                    CHECK_EQ(_channels, param.slope->valid_size()) << \
                "slope data size must = channels";
        }
        _is_continue_buf = outputs[0]->is_continue_mem() && inputs[0]->is_continue_mem();
        if (!_is_continue_buf) {
            Shape sh_input_real_stride = inputs[0]->get_stride();
            Shape sh_output_real_stride = outputs[0]->get_stride();

            //! re_alloc device memory
            Shape sh{1, 1, 1, _dims};
            _valid_shape.reshape(sh);
            _input_stride.reshape(sh);
            _output_stride.reshape(sh);

            CUDA_CHECK(cudaMemcpy(_valid_shape.mutable_data(), inputs[0]->valid_shape().data(), \
                sizeof(int) * _dims, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(_input_stride.mutable_data(), sh_input_real_stride.data(), \
                sizeof(int) * _dims, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(_output_stride.mutable_data(), sh_output_real_stride.data(), \
                sizeof(int) * _dims, cudaMemcpyHostToDevice));
        }
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 PreluParam<OpTensor> &param);

private:
    int _size;
    int _inner_size;
    int _outer_size;
    int _channels;
    int _dims;
    Tensor<NV, AK_INT32, LayOutType_in> _input_stride;
    Tensor<NV, AK_INT32, LayOutType_out> _output_stride;
    Tensor<NV, AK_INT32, LayOutType_op> _valid_shape;
    bool _is_continue_buf{true};
};


template class SaberPrelu<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
/*t
emplate class SaberPrelu<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NHWC, NHWC, NHWC>;
template class SaberPrelu<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, HW, HW, HW>;
template class SaberPrelu<NV, AK_INT8, AK_INT8, AK_INT8, NCHW, NCHW, NCHW>;
template class SaberPrelu<NV, AK_INT8, AK_INT8, AK_INT8, NHWC, NHWC, NHWC>;
template class SaberPrelu<NV, AK_INT8, AK_INT8, AK_INT8, HW, HW, HW>;
*/
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PRELU_H