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

#ifndef ANAKIN_SABER_FUNCS_X86_SABER_DETECTION_OUTPUT_H
#define ANAKIN_SABER_FUNCS_X86_SABER_DETECTION_OUTPUT_H


#include "saber/funcs/impl/impl_detection_output.h"
#include "saber/core/data_traits.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberDetectionOutput<X86, OpDtype> : \
    public ImplBase<
        X86, OpDtype,
        DetectionOutputParam<X86> > 
{
public:

    SaberDetectionOutput() = default;
    ~SaberDetectionOutput() {}

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                            std::vector<Tensor<X86> *>& outputs,
                            DetectionOutputParam<X86>& param, Context<X86>& ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                            std::vector<Tensor<X86> *>& outputs,
                            DetectionOutputParam<X86>& param, Context<X86> &ctx) {

        _shared_loc = param.share_location;
        Shape sh_loc = inputs[0]->valid_shape();
        Shape sh_conf = inputs[1]->valid_shape();
        Shape sh_box;

        //fixme, only support{xmin, ymin, xmax, ymax} style box
        if (_shared_loc) {
            //! for one stage detector
            //! inputs[0]: location map, {N, boxes * 4}
            //! inputs[1]: confidence map, ssd: {N, classes, boxes}, yolov3: {N, boxes, classes}
            //! optional, ssd has 3 inputs, the last inputs is priorbox
            //! inputs[2]: prior boxes, dims = 4 {1, 2, boxes * 4(xmin, ymin, xmax, ymax)}
            CHECK_GE(inputs.size(), 2) << "detection_output op must has 2 inputs at least";
            bool is_ssd = inputs.size() > 2;
            if (is_ssd) {
                sh_box = inputs[2]->valid_shape();
            }
            //! boxes = sh_loc / 4
            _num_priors = sh_loc.count() / 4;
            if (param.class_num <= 0) {
                _num_classes = sh_conf.count() / _num_priors;
            } else {
                _num_classes = param.class_num;
            }
            _num_loc_classes = 1;
            if (is_ssd) {
                _bbox_preds.reshape(sh_loc);
                _conf_permute.reshape(sh_conf);
            }

        } else {
            //! for two stage detector
            //! inputs[0]: tensor with offset, location, {M, C, 4}
            //! inputs[1]: tensor with offset, confidence, {M, C}
            CHECK_EQ(sh_loc[0], sh_conf[0]) << "boxes number must be the same";
            _num_priors = sh_loc[0];
            if (param.class_num <= 0) {
                _num_classes = sh_conf.count() / _num_priors;
            } else {
                _num_classes = param.class_num;
            }
            _num_loc_classes = _num_classes;
            _bbox_permute.reshape(sh_loc);
            _conf_permute.reshape(sh_conf);
        }

        CHECK_EQ(_num_priors * _num_loc_classes * 4, sh_loc.count()) << \
            "Number of boxes must match number of location predictions.";
        CHECK_EQ(_num_priors * _num_classes, sh_conf.count()) << \
            "Number of boxes must match number of confidence predictions.";

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                          std::vector<Tensor<X86> *>& outputs,
                          DetectionOutputParam<X86>& param);


private:
    bool _shared_loc{true};
    int _num_classes;
    int _num_loc_classes;
    int _num_priors;
    Tensor<X86> _bbox_preds;
    Tensor<X86> _bbox_permute;
    Tensor<X86> _conf_permute;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H
