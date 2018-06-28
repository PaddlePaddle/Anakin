#include "saber/funcs/impl/arm/saber_detection_output.h"
#include "saber/funcs/impl/detection_helper.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

void permute_conf(const float* conf_data, const int num,
                  const int num_priors, const int num_classes,
                  float* conf_preds) {
    for (int i = 0; i < num; ++i) {
        const float* batch_conf = conf_data + i * num_classes * num_priors;
        float* batch_data_permute = conf_preds + i * num_classes * num_priors;
        for (int p = 0; p < num_priors; ++p) {
            int start_idx = p * num_classes;
            for (int c = 0; c < num_classes; ++c) {
                batch_data_permute[c * num_priors + p] = batch_conf[start_idx + c];
            }
        }
    }
}


template <>
SaberStatus SaberDetectionOutput<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, \
NCHW, NCHW, NCHW>::dispatch(
        const std::vector<DataTensor_in *>& inputs,
        std::vector<DataTensor_out *>& outputs,
        DetectionOutputParam<OpTensor> &param) {

    DataTensor_in* t_loc = inputs[0];
    DataTensor_in* t_conf = inputs[1];
    DataTensor_in* t_prior = inputs[2];

    const int num = t_loc->num();

    const float* loc_data = t_loc->data();
    const float* prior_data = t_prior->data();
    const float* conf_data = t_conf->data();

    float* bbox_data = _bbox_preds.mutable_data();

    if (!param.share_location) {
        return SaberUnImplError;
    }

    //! Decode predictions.
    //! Retrieve all decoded location predictions.
    decode_bboxes(num, loc_data, prior_data, param.type, param.variance_encode_in_target, \
        _num_priors, param.share_location, _num_loc_classes, \
        param.background_id, bbox_data);

    //! Retrieve all confidences, permute to classes * boxes_size
    float* conf_permute_data = _conf_permute.mutable_data();
    permute_conf(conf_data, num, _num_priors, _num_classes, conf_permute_data);

    std::vector<float> result;

    nms_detect(bbox_data, conf_permute_data, result, num, this->_num_classes, _num_priors, param.background_id, \
        param.keep_top_k, param.nms_top_k, param.conf_thresh, param.nms_thresh, param.nms_eta, param.share_location);

    if(result.size() == 0) {
        result.resize(7);
        for (int i = 0; i < 7; ++i) {
            result[i] = -1.f;
        }
        outputs[0]->reshape({1, 1, 1, 7});
    } else {
        outputs[0]->reshape({1, 1, result.size() / 7, 7});
    }

    memcpy(outputs[0]->mutable_data(), result.data(), \
                result.size() * sizeof(float));

    return SaberSuccess;
}

} //namespace anakin

} //namespace anakin

#endif
