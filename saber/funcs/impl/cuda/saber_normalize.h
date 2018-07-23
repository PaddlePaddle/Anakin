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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_NORMALIZE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_NORMALIZE_H

#include "saber/funcs/impl/impl_normalize.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
    class SaberNormalize<NV, OpDtype>:
    public ImplBase<
            NV,OpDtype,
            NormalizeParam<NV> > {

public:
    typedef Tensor<NV> DataTensor_in;
    typedef Tensor<NV> DataTensor_out;
    typedef Tensor<NV> OpTensor;

    //typedef typename DataTensor_in::Dtype InDataType;
    //typedef typename DataTensor_out::Dtype OutDataType;
    //typedef typename OpTensor::Dtype OpDataType;
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberNormalize() = default;
    ~SaberNormalize() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             NormalizeParam<NV> &param,
                             Context<NV> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               NormalizeParam<NV> &param,
                               Context<NV> &ctx) {
        // compute norm size
        int channel_index = inputs[0]->channel_index();
        _dims = inputs[0]->dims();
        _size = inputs[0]->valid_size();
        _channels = inputs[0]->channel();
        _batchs = inputs[0]->num();

        //! check the scale size
        if (param.has_scale) {
            if (!param.channel_shared) {
                        CHECK_EQ(_channels, param.scale->valid_size()) << \
                    "scale data size must = channels";
            }
        }

        //! size of data to compute square root sum (eg. H * W for channel, C * H * W for batch)
        if (param.across_spatial) {
            _norm_size = _batchs;
        } else {
            _norm_size = _channels * _batchs;
        }
        _channel_stride = inputs[0]->count_valid(channel_index + 1, _dims);
        _compute_size = _size / _norm_size;
        Shape sh_norm({1, 1, 1, _norm_size});
        _norm_reduce.reshape(sh_norm);

        _is_continue_buf = outputs[0]->is_continue_mem() && inputs[0]->is_continue_mem();
        if (!_is_continue_buf) {
            Shape sh_input_real_stride = inputs[0]->get_stride();
            Shape sh_output_real_stride = outputs[0]->get_stride();

            //! re_alloc device memory
            Shape sh({1, 1, 1, _dims});
            _valid_shape.reshape(sh);
            _input_stride.reshape(sh);
            _output_stride.reshape(sh);

            CUDA_CHECK(cudaMemcpy(_valid_shape.mutable_data(), \
                inputs[0]->valid_shape().data(), sizeof(int) * _dims, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(_input_stride.mutable_data(), \
                sh_input_real_stride.data(), sizeof(int) * _dims, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(_output_stride.mutable_data(), \
                sh_output_real_stride.data(), sizeof(int) * _dims, cudaMemcpyHostToDevice));
        }
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 NormalizeParam<NV> &param);


private:
    //Tensor<NV, OpDtype, NCHW> _norm_reduce;
    Tensor<NV> _norm_reduce;
    int _size;
    int _norm_size;
    int _compute_size;
    int _batchs;
    int _channels;
    int _dims;
    int _channel_stride;
    //todo:
    Tensor<NV> _input_stride;
    Tensor<NV> _output_stride;
    Tensor<NV> _valid_shape;

    bool _is_continue_buf{true};
};
//template class SaberNormalize<NV, AK_FLOAT>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_NORMALIZE_H
