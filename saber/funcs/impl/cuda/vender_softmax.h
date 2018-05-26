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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_SOFTMAX_H

#include "saber/funcs/impl/impl_define.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/saber_types.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderSoftmax<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        SoftmaxParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderSoftmax(){
        _handle = nullptr;
        _input_desc = nullptr;
        _output_desc = nullptr;
        _setup = false;
    }

    ~VenderSoftmax() {

        if (_setup){
            CUDNN_CHECK(cudnnDestroy(_handle));
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_desc));
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_desc));
        }
    }

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

        // ---- init cudnn resources ----

        this->_ctx = ctx;
        // ---- get cuda resources ----

        cudaStream_t cuda_stream = this->_ctx.get_compute_stream();

        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

        // ---- create cudnn Descs ----
        cudnn::createTensorDesc<InDataType>(&_input_desc);
        cudnn::createTensorDesc<OutDataType>(&_output_desc);

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            SoftmaxParam<OpTensor>& param, Context<NV> &ctx) {

        if (!inputs[0]->is_continue_mem() || !outputs[0]->is_continue_mem()) {
            //! unsupported type for cudnn
            return SaberInvalidValue;
        }

        //CHECK_EQ(inputs[0]->shape() == inputs[0]->valid_shape(), true) << \
                "cudnn softmax does not support tensor with roi";
        //CHECK_EQ(outputs[0]->shape() == outputs[0]->valid_shape(), true) << \
                "cudnn softmax does not support tensor with roi";

        Shape shape_in = inputs[0]->shape();
        if (!(ctx == this->_ctx)) {
            if (_handle != NULL) {
                CUDNN_CHECK(cudnnDestroy(_handle));
            }
            this->_ctx = ctx;
            cudaStream_t cuda_stream;
            cuda_stream = ctx.get_compute_stream();
            CUDNN_CHECK(cudnnCreate(&_handle));
            CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));
        }

        int outer_num = inputs[0]->count(0, param.axis);
        int inner_num = inputs[0]->count(param.axis + 1, inputs[0]->dims());

        int N = outer_num;
        int K = inputs[0]->shape()[param.axis];
        int H = inner_num;
        int W = 1;

        const int stride_w = 1;
        const int stride_h = W * stride_w;
        const int stride_c = H * stride_h;
        const int stride_n = K * stride_c;
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(_input_desc, \
            cudnn::cudnnOpWrapper<InDataType>::type, \
            N, K, H, W, stride_n, stride_c, stride_h, stride_w));
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(_output_desc, \
            cudnn::cudnnOpWrapper<InDataType>::type, \
            N, K, H, W, stride_n, stride_c, stride_h, stride_w));

        _setup = true;
        return SaberSuccess;
    }

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          SoftmaxParam<OpTensor> &param){
        cudaStream_t stream = this->_ctx.get_compute_stream();
        const InDataType* input_data = inputs[0]->data();
        InDataType * output_data = outputs[0]->mutable_data();
        CUDNN_CHECK(cudnnSoftmaxForward(_handle, CUDNN_SOFTMAX_ACCURATE, \
            CUDNN_SOFTMAX_MODE_CHANNEL, cudnn::cudnnTypeWrapper<InDataType>::kOne(), _input_desc, input_data, \
            cudnn::cudnnTypeWrapper<InDataType>::kZero(), _output_desc, output_data));
        //outputs[0]->record_event(stream);
        return SaberSuccess;
    }

private:
    bool _setup{false};
    cudnnHandle_t             _handle;
    cudnnTensorDescriptor_t _input_desc;
    cudnnTensorDescriptor_t _output_desc;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_SOFTMAX_H
