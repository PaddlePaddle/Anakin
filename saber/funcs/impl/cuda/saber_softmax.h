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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SOFTMAX_H

#include "saber/funcs/impl/impl_softmax.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberSoftmax<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        SoftmaxParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef TargetWrapper<NV> API;
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberSoftmax() = default;

    ~SaberSoftmax() {}

    /**
     * \brief initial all cudnn resources here
     * @param inputs
     * @param outputs
     * @param param
     * @param ctx
     */
    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            SoftmaxParam<OpTensor>& param, Context<NV>& ctx) {

        //! get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            SoftmaxParam<OpTensor>& param, Context<NV>& ctx) {
        //! compute size
        Shape shape_in = inputs[0]->valid_shape();
        Shape shape_out = outputs[0]->valid_shape();
        CHECK_EQ(shape_in == shape_out, true) << "valid shapes must be the same";
        _outer_num = inputs[0]->count_valid(0, param.axis);
        _inner_num = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
        _axis_size = shape_in[param.axis];

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, API::get_device_id());
        size_t sharedmem_size = deviceProp.sharedMemPerBlock;
        _max_dimsize = sharedmem_size / sizeof(InDataType) / CUDA_NUM_THREADS;
        //LOG(INFO) << "shared memory size: " << sharedmem_size / 1024 << "Kb";
        Shape sh_tmp{1, 1, 1, _outer_num * _inner_num};
        if (_axis_size > _max_dimsize){
            //! re_alloc device memory
            _max_data.reshape(sh_tmp);
            _sum_data.reshape(sh_tmp);
        }

        //! CHECK whether the input or output tensor is with continuous buffer or not
        _is_continue_buf = outputs[0]->is_continue_mem() && inputs[0]->is_continue_mem();
        _dims = shape_in.size();
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
                          SoftmaxParam<OpTensor>& param);

private:

    //! get maximum size to select which softmax kernel to call
    //! _max_dimsize is compute from shared memory size
    bool _is_continue_buf{true};
    int _max_dimsize;
    int _inner_num;
    int _outer_num;
    int _axis_size;
    int _dims;
    Tensor<NV, AK_INT32, NCHW> _input_stride;
    Tensor<NV, AK_INT32, NCHW> _output_stride;
    Tensor<NV, AK_INT32, NCHW> _valid_shape;

    Tensor<NV, inDtype, NCHW> _max_data;
    Tensor<NV, inDtype, NCHW> _sum_data;
};
template class SaberSoftmax<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SOFTMAX_H
