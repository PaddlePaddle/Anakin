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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DEFORMABLE_CONV_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DEFORMABLE_CONV_H

#include "saber/funcs/impl/impl_define.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberDeformableConv2D<NV, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, \
        Tensor<NV, outDtype, LayOutType_out>, \
        Tensor<NV, OpDtype, LayOutType_op>, \
        DeformableConvParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberDeformableConv2D()
        : _handle(NULL)
        , _conv_out_spatial_dim(0)
        , _kernel_dim(0)
        , _bottom_dim(0)
        , _offset_dim(0)
        , _col_offset(0)
        , _output_offset(0)
        , _kernel_offset(0)
    {}

    ~SaberDeformableConv2D() {
        if (_handle != NULL) {
            CUBLAS_CHECK(cublasDestroy(_handle));
        }
    }

    /**
     * [Create description] Init all cudnn resource here
     * @AuthorHTL
     * @DateTime  2018-02-01T16:13:06+0800
     * @param     inputs                    [description]
     * @param     outputs                   [description]
     * @param     conv_param                [conv parameters]
     */
    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            DeformableConvParam<OpTensor>& param, Context<NV>& ctx) {

        // ---- init cudnn resources ----
        this->_ctx = ctx;
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, this->_ctx.get_compute_stream()));

        _kernel_dim = param.weight()->channel()
                      * param.weight()->height()
                      * param.weight()->width();

        _bottom_dim = inputs[0]->channel()
                      * inputs[0]->height()
                      * inputs[0]->width();

        _offset_dim = inputs[1]->channel()
                      * inputs[1]->height()
                      * inputs[1]->width();

        Shape deform_col_buffer_shape = {1, _kernel_dim, outputs[0]->height(), outputs[0]->width()};
        _deform_col_buffer.re_alloc(deform_col_buffer_shape);

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            DeformableConvParam<OpTensor>& param, Context<NV>& ctx) {

        if (!(ctx == this->_ctx)) {
            this->_ctx = ctx;
            if (_handle != NULL) {
                CUBLAS_CHECK(cublasDestroy(_handle));
            }
            CUBLAS_CHECK(cublasCreate(&_handle));
            CUBLAS_CHECK(cublasSetStream(_handle, this->_ctx.get_compute_stream()));
        }

        int in_channel = inputs[0]->channel();
        int conv_out_channel = outputs[0]->channel();
        _conv_out_spatial_dim = outputs[0]->height() * outputs[0]->width();

        _kernel_dim = param.weight()->channel()
                      * param.weight()->height()
                      * param.weight()->width();

        _bottom_dim = inputs[0]->channel()
                      * inputs[0]->height()
                      * inputs[0]->width();

        _offset_dim = inputs[1]->channel()
                      * inputs[1]->height()
                      * inputs[1]->width();

        _col_offset = _kernel_dim * _conv_out_spatial_dim;
        _output_offset = conv_out_channel * _conv_out_spatial_dim;
        _kernel_offset = _kernel_dim * conv_out_channel;

        if ((outputs[0]->height() != _deform_col_buffer.height())
                || (outputs[0]->width() != _deform_col_buffer.width())) {

            Shape deform_col_buffer_shape = {1, _kernel_dim, outputs[0]->height(), outputs[0]->width()};
            _deform_col_buffer.reshape(deform_col_buffer_shape);
        }
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            DeformableConvParam<OpTensor>& param);

private:
    DataTensor_in _deform_col_buffer;
    cublasHandle_t _handle;

    int _conv_out_spatial_dim;
    int _kernel_dim;
    int _bottom_dim;
    int _offset_dim;
    int _col_offset;
    int _output_offset;
    int _kernel_offset;
};


}

}
#endif //ANAKIN_SABER_FUNCS_CUDNN_CONV2D_H
