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

#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
class SaberDetectionOutput {
public:
    SaberDetectionOutput() = default;
    ~SaberDetectionOutput() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     DetectionOutputParam<Tensor<Dtype>> &param) {
        //! output tensor's dims = 2
        Shape shape_out;
        shape_out.resize(2);
        //CHECK_EQ(shape_out.dims(), 4) << "only support 4d layout";
        shape_out[0] = param.keep_top_k;
        shape_out[1] = 7;

        return outputs[0]->set_shape(shape_out);
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                      std::vector<Tensor<Dtype>*>& outputs,
                      DetectionOutputParam<Tensor<Dtype>> &param, Context &ctx){
        return create(inputs, outputs, param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,
                        std::vector<Tensor<Dtype>*>& outputs,
                        DetectionOutputParam<Tensor<Dtype>> &param, Context &ctx){

        //! inputs[0]: location map, dims = 2 {N, boxes * 4}
        //! inputs[1]: confidence map, dims = 2 {N, boxes * classes}
        //! inputs[2]: prior boxes, dims = 3 {1, 2, boxes * 4(xmin, ymin, xmax, ymax)}
        Shape sh_loc = inputs[0]->valid_shape();
        Shape sh_conf = inputs[1]->valid_shape();
        Shape sh_box = inputs[2]->valid_shape();
        //! shape {1, 2, boxes * 4(xmin, ymin, xmax, ymax)}, boxes = size / 2 / 4
        //! the priors is in the last dim

        CHECK_EQ(this->_shapes_in[0][0], this->_shapes_in[1][0]) << "input tensor must have same num";

        _num_priors = sh_box[2] / 4;

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

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs,
                          std::vector<Tensor<Dtype>*>& outputs,
                          DetectionOutputParam<Tensor<Dtype>> &param);


private:
    int _num_classes;
    int _num_loc_classes;
    int _num_priors;
    Tensor<Dtype> _bbox_preds;
    Tensor<Dtype> _bbox_permute;
    Tensor<Dtype> _conf_permute;
};

} //namepace lite

} //namespace saber

} //namespace anakin

#endif

#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H
