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

#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_DETECTION_OUTPUT_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_DETECTION_OUTPUT_H

#include "saber/lite/funcs/op_base.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

//template <ARMType ttype, DataType dtype>
class SaberDetectionOutput : public OpBase {
public:
    SaberDetectionOutput(){}
    SaberDetectionOutput(const ParamBase* param);
//    SaberDetectionOutput(bool share_loc,
//                         bool variance_encode,
//                         int class_num,
//                         int background_id,
//                         int keep_topk,
//                         CodeType type,
//                         float conf_thresh,
//                         int nms_topk,
//                         float nms_thresh = 0.3f,
//                         float nms_eta = 1.f);
    ~SaberDetectionOutput() {}

//    SaberStatus load_param(bool share_loc,
//                           bool variance_encode,
//                           int class_num,
//                           int background_id,
//                           int keep_topk,
//                           CodeType type,
//                           float conf_thresh,
//                           int nms_topk,
//                           float nms_thresh = 0.3f,
//                           float nms_eta = 1.f);

    virtual SaberStatus load_param(const ParamBase* param) override;

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
                      std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                          std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;


private:
    const DetectionOutputParam* _param;
//    bool _share_loacation{true};
//    bool _variance_encode_in_target{false};
    int _class_num;
//    int _background_id{0};
//    int _keep_top_k{-1};
//    CodeType _type{CENTER_SIZE};
//    float _conf_thresh;
//    int _nms_top_k;
//    float _nms_thresh{0.3f};
//    float _nms_eta{1.f};
    int _num_loc_classes;
    int _num_priors;
    Tensor<CPU, AK_FLOAT> _bbox_preds;
    Tensor<CPU, AK_FLOAT> _bbox_permute;
    Tensor<CPU, AK_FLOAT> _conf_permute;
};

} //namepace lite

} //namespace saber

} //namespace anakin

#endif

#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_DETECTION_OUTPUT_H
