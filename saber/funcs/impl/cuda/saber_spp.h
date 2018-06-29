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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SPP_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SPP_H
#include "saber/funcs/pooling.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_spp.h"

namespace anakin{

namespace saber{
#if 0
template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
class SaberSpp<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
            Tensor<NV, inDtype, LayOutType_in>,
            Tensor<NV, outDtype, LayOutType_out>,
            Tensor<NV, OpDtype, LayOutType_op>,
            SPPParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;
    typedef Pooling<NV, OpDtype, inDtype, outDtype, 
        LayOutType_op, LayOutType_in, LayOutType_out> Pooling_t;

    SaberSpp()
    {}

    ~SaberSpp() {
        for (auto pool : _pooling) {
            delete pool;
            pool = nullptr;
        }
        for (auto out : _pooling_output) {
            delete out;
            out = nullptr;
        }

    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             SPPParam<OpTensor> &param,
                             Context<NV> &ctx)  {
        this->_ctx = &ctx;
        _pooling.clear();
        _pooling_output.clear();
        _pooling_param.clear();
        _pooling_output.resize(param.pyramid_height);
        _pooling.resize(param.pyramid_height);
        _pooling_param.resize(param.pyramid_height);
        int out_w_index = outputs[0]->width_index();
        int out_h_index = outputs[0]->height_index();
        for (int i = 0; i < param.pyramid_height; i++) {
             int num_bins = pow(2, i);
             int window_h = std::ceil(inputs[0]->height() / static_cast<double>(num_bins));
             int window_w = std::ceil(inputs[0]->width() / static_cast<double>(num_bins));
             int pad_h = (window_h * num_bins - inputs[0]->height() + 1) / 2;
             int pad_w = (window_w * num_bins - inputs[0]->width() + 1) / 2;
             PoolingParam<OpTensor> pool_param(window_h, window_w, pad_h, pad_w
            , window_h, window_w, param.pool_type);

             Shape valid_shape = outputs[0]->valid_shape();
             valid_shape[out_w_index] = pow(2, i);
             valid_shape[out_h_index] = pow(2, i);
             _pooling[i] = new Pooling_t();
             _pooling_output[i] = new DataTensor_out(valid_shape);
             std::vector<DataTensor_out*> pool_outputs = {_pooling_output[i]};
             _pooling[i]->compute_output_shape(inputs, pool_outputs, pool_param);
             _pooling[i]->init(inputs, pool_outputs, pool_param, SPECIFY, VENDER_IMPL, ctx);
             _pooling_param[i] = pool_param;
             
        }
        
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             SPPParam<OpTensor> &param,
                             Context<NV> &ctx)  {
        return SaberSuccess;
    }

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 SPPParam<OpTensor> &param);

private:
    std::vector<Pooling_t*> _pooling;
    std::vector<PoolingParam<OpTensor>> _pooling_param;
    std::vector<DataTensor_out*> _pooling_output;
};
template class SaberSpp<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
#endif
}

}

#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
