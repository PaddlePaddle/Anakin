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

#ifndef ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H
#define ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H

#include "saber/funcs/impl/impl_detection_output.h"
#include "saber/core/tensor.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberDetectionOutput<ARM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        DetectionOutputParam<Tensor<ARM, OpDtype, LayOutType_op> > >
{
public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberDetectionOutput() = default;
    ~SaberDetectionOutput() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                      std::vector<DataTensor_out*>& outputs,
                      DetectionOutputParam<OpTensor> &param, Context<ARM> &ctx){
        // get context
        this->_ctx = ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                        std::vector<DataTensor_out*>& outputs,
                        DetectionOutputParam<OpTensor> &param, Context<ARM> &ctx){

        //! inputs[0]: location map, dims = 4 {N, boxes * 4, 1, 1}
        //! inputs[1]: confidence map, dims = 4 {N, boxes * classes, 1, 1}
        //! inputs[2]: prior boxes, dims = 4 {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}
        Shape sh_loc = inputs[0]->valid_shape();
        Shape sh_conf = inputs[1]->valid_shape();
        Shape sh_box = inputs[2]->valid_shape();
        //! shape {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}, boxes = size / 2 / 4
        //! layout must be 4 dims, the priors is in the last dim
        _num_priors = sh_box[3] / 4;
        int num = inputs[0]->num();
        if (param.class_num == 0) {
            _num_classes = inputs[1]->valid_size() / (num * _num_priors);
        } else {
            _num_classes = param.class_num;
        }
        if (param.share_location) {
            _num_loc_classes = 1;
        } else {
            _num_loc_classes = _num_classes;
            _bbox_permute.reshape(sh_loc);
        }

        _bbox_preds.reshape(sh_loc);
        _conf_permute.reshape(sh_conf);

        CHECK_EQ(_num_priors * _num_loc_classes * 4, sh_loc[1]) << \
		    "Number of priors must match number of location predictions.";
        CHECK_EQ(_num_priors * _num_classes, sh_conf[1]) << \
		    "Number of priors must match number of confidence predictions.";

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          DetectionOutputParam<OpTensor> &param);


private:
    int _num_classes;
    int _num_loc_classes;
    int _num_priors;
    OpTensor _bbox_preds;
    OpTensor _bbox_permute;
    OpTensor _conf_permute;
};

} //namespace saber

} //namespace anakin

#endif

#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H
