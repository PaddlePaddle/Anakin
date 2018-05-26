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

#ifndef ANAKIN_SABER_FUNCS_CUDA_SABER_MULTICLASS_NMS_H
#define ANAKIN_SABER_FUNCS_CUDA_SABER_MULTICLASS_NMS_H

#include "saber/funcs/impl/impl_define.h"
#include "saber/core/tensor.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberMultiClassNMS<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        MultiClassNMSParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberMultiClassNMS() = default;
    ~SaberMultiClassNMS() {
        if (_bbox_cpu_data) {
            fast_free(_bbox_cpu_data);
        }
        if (_conf_cpu_data) {
            fast_free(_conf_cpu_data);
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            MultiClassNMSParam<OpTensor>& param, Context<NV>& ctx) {
        // get context
        this->_ctx = ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            MultiClassNMSParam<OpTensor>& param, Context<NV> &ctx) {

        //! inputs[0]: bbox map, dims = 3 {N, boxes, 4(xmin, ymin, xmax, ymax)}
        //! inputs[1]: score map, dims = 3 {N, classes, boxes}
        Shape sh_bbox = inputs[0]->valid_shape();
        Shape sh_conf = inputs[1]->valid_shape();

        //! layout must be 3 dims, the priors(number of boxes) is in the second dim
        _num_priors = sh_bbox[1];

        CHECK_EQ(sh_conf[2], sh_bbox[1]) << \
            "Number of bboxes must match the number of scores per class.";

        _conf_cpu_data = (InDataType*)fast_malloc(sizeof(InDataType) * sh_conf.count());
        _bbox_cpu_data = (InDataType*)fast_malloc(sizeof(InDataType) * sh_bbox.count());

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          MultiClassNMSParam<OpTensor>& param);


private:
    int _num_priors;
    InDataType* _bbox_cpu_data{nullptr};
    InDataType* _conf_cpu_data{nullptr};
};
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_MULTICLASS_NMS_H
