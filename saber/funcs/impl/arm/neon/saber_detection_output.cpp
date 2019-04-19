#include "saber/funcs/impl/arm/saber_detection_output.h"
#include "saber/funcs/impl/detection_helper.h"

namespace anakin{
namespace saber {

template <typename dtype>
void permute_data(const int nthreads, const dtype* data, const int num_classes,
                  const int num_data, const int num_dim, dtype* new_data) {
    for (int index = 0; index < nthreads; ++index) {
        const int i = index % num_dim;
        const int c = (index / num_dim) % num_classes;
        const int d = (index / num_dim / num_classes) % num_data;
        const int n = index / num_dim / num_classes / num_data;
        const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
        new_data[new_index] = data[index];
    }
}

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

template<>
SaberStatus SaberDetectionOutput<ARM, AK_FLOAT>::create(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        DetectionOutputParam<ARM> &param,
        Context<ARM> &ctx){

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

template <>
SaberStatus SaberDetectionOutput<ARM, AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        DetectionOutputParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif

    Tensor<ARM>* t_loc = inputs[0];
    Tensor<ARM>* t_conf = inputs[1];
    Tensor<ARM>* t_prior;
    std::vector<int> priors;
    CHECK_EQ(t_loc->get_dtype(), AK_FLOAT) << "input data type must be float";
    CHECK_EQ(t_conf->get_dtype(), AK_FLOAT) << "input data type must be float";

    const float* bbox_data_cpu = nullptr;
    const float* conf_data_cpu = nullptr;
    const int num = t_loc->num();

    if (_shared_loc) {
        //! for one stage
        for (int i = 0; i < num; ++i) {
            priors.push_back(_num_priors / num);
        }

        bool is_ssd = inputs.size() > 2;

        if (is_ssd) {
            t_prior = inputs[2];
            int num_priors = _num_priors / num;
            const float* loc_data = static_cast<const float*>(t_loc->data());
            const float* prior_data = static_cast<const float*>(t_prior->data());

            // Decode predictions.
            const float* bbox_data = static_cast<float*>(_bbox_preds.mutable_data());

            //! Decode predictions.
            //! Retrieve all decoded location predictions.
            decode_bboxes(num, loc_data, prior_data, param.type, param.variance_encode_in_target, \
                num_priors, param.share_location, _num_loc_classes, param.background_id, bbox_data);
            //! Retrieve all confidences, permute to classes * boxes_size
            const float* conf_data = static_cast<const float*>(t_conf->data());
            float* conf_permute_data = static_cast<float*>(_conf_permute.mutable_data());
            permute_conf(conf_data, num, num_priors, _num_classes, conf_permute_data);
            conf_data_cpu = conf_permute_data;
            bbox_data_cpu = bbox_data;
        } else { //! multiclass_nms
            bbox_data_cpu = static_cast<const float*>(t_loc->data());
            conf_data_cpu = static_cast<const float*>(t_conf->data());
        }
    } else {
        //! for two stage
        //! sizeof seq offset is N + 1
        auto conf_permute = static_cast<float*>(_conf_permute.mutable_data());
        auto bbox_permute = static_cast<float*>(_bbox_permute.mutable_data());
        auto conf_ori = static_cast<const float*>(t_conf->data());
        auto bbox_ori = static_cast<const float*>(t_loc->data());
        //! for two stage
        //! sizeof seq offset is N + 1
        auto offset = t_loc->get_seq_offset()[0];
        for (int i = 0; i < offset.size() - 1; ++i) {
            int num_priors = offset[i + 1] - offset[i];
            priors.push_back(num_priors);
            const float* conf_ori_batch = conf_ori + this->_num_classes * offset[i];
            const float* bbox_ori_batch = bbox_ori + this->_num_classes * 4 * offset[i];
            float* conf_permute_batch = conf_permute + this->_num_classes * offset[i];
            float* bbox_permute_batch = bbox_permute + this->_num_classes * 4 * offset[i];
            //! permute conf and bbox
            //! input bbox layout is [M, C, 4], multi-batch view: [{priors0, C, 4}, {priors1, C, 4}, ...]
            //! permute bbox data to [{C, priors0, 4}, {C, priors1, 4}, ...]
            //! input conf layout is [M, C], multi-batch view: [{priors0, C}, {priors1, C}, ...]
            //! permute conf data to [{C, priors0}, {C, priors1}, ...]
            permute_data<float>(num_priors * this->_num_classes, conf_ori_batch,
                                this->_num_classes, num_priors, 1, conf_permute_batch);
            permute_data<float>(num_priors * this->_num_classes * 4, bbox_ori_batch,
                                this->_num_classes, num_priors, 4, bbox_permute_batch);
        }
        bbox_data_cpu = bbox_permute;
        conf_data_cpu = conf_permute;
    }

    std::vector<float> result;
    nms_detect(bbox_data_cpu, conf_data_cpu, result, priors, _num_classes, param.background_id, \
        param.keep_top_k, param.nms_top_k, param.conf_thresh, param.nms_thresh, param.nms_eta, _shared_loc);

    if (result.size() == 0) {
        result.resize(7);
        for (int i = 0; i < 7; ++i) {
            result[i] = -1.f;
        }
        outputs[0]->reshape(Shape({1, 1, 1, 7}));
    } else {
        outputs[0]->reshape(Shape({1, 1, result.size() / 7, 7}));
    }

    memcpy(static_cast<float*>(outputs[0]->mutable_data()), result.data(), \
                result.size() * sizeof(float));
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "DetectionOutput : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("DetectionOutput", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
}
} // namespace anakin
