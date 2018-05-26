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

#ifndef ANAKIN_SABER_FUNCS_CUDA_SABER_BOX_CODER_H
#define ANAKIN_SABER_FUNCS_CUDA_SABER_BOX_CODER_H
   
#include "saber/funcs/impl/impl_define.h"


namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberBoxCoder<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        BoxCoderParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberBoxCoder() = default;
    ~SaberBoxCoder() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        BoxCoderParam<OpTensor>& param, 
                        Context<NV> &ctx) {
        // get context
        this->_ctx = ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        BoxCoderParam<OpTensor>& param, 
                        Context<NV>& ctx) {

        //! inputs[0]: prior boxes, dims = 2 {2, boxes * 4(xmin, ymin, xmax, ymax)}, with variance
        //! inputs[1]: loc boxes, dims = 2 {N, boxes, }
        Shape sh_bbox = inputs[0]->valid_shape();
        Shape sh_conf = inputs[1]->valid_shape();

        CHECK_EQ(sh_conf[2], sh_bbox[1]) << \
            "Number of bboxes must match the number of scores per class.";

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        ArgmaxParam<OpTensor>& param) {
        LOG(FATAL) << "not impl yet";
        return SaberUnImplError;
    }


};
template class SaberBoxCoder<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_BOX_CODER_H