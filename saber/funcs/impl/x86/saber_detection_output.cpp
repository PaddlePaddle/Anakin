#include "saber/funcs/impl/x86/saber_detection_output.h"
#include "saber/funcs/impl/detection_helper.h"
namespace anakin{

namespace saber{
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

template <DataType OpDtype>
SaberStatus SaberDetectionOutput<X86, OpDtype>::dispatch(const std::vector<Tensor<X86> *>& inputs,
    std::vector<Tensor<X86> *>& outputs,
    DetectionOutputParam<X86>& param) {

    Tensor<X86>* t_loc = inputs[0];
    Tensor<X86>* t_conf = inputs[1];
    Tensor<X86>* t_prior;
    std::vector<int> priors;
    CHECK_EQ(t_loc->get_dtype(), AK_FLOAT) << "input data type must be float";
    CHECK_EQ(t_conf->get_dtype(), AK_FLOAT) << "input data type must be float";

    const float* bbox_data_cpu = nullptr;
    const float* conf_data_cpu = nullptr;

    if (_shared_loc) {
        //! for one stage
        const int num = t_loc->num();
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
            float* bbox_data = static_cast<float*>(_bbox_preds.mutable_data());
            const int loc_count = _bbox_preds.valid_size();
            decode_bboxes<float>(loc_count, loc_data, prior_data, param.type, \
                param.variance_encode_in_target, num_priors, param.share_location, \
                _num_loc_classes, param.background_id, bbox_data);
            // Retrieve all decoded location predictions.
            if (!param.share_location) {
                float* bbox_permute_data = static_cast<float*>(_bbox_permute.mutable_data());
                permute_data<float>(loc_count, bbox_data, _num_loc_classes, num_priors,
                                    4, bbox_permute_data);
            }
            // Retrieve all confidences.
            float* conf_permute_data = static_cast<float*>(_conf_permute.mutable_data());
            permute_data<float>(t_conf->valid_size(), static_cast<float*>(t_conf->data()), \
                 this->_num_classes, num_priors, 1, conf_permute_data);

            bbox_data_cpu = bbox_data;
            conf_data_cpu = conf_permute_data;
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
    nms_detect(bbox_data_cpu, conf_data_cpu, result, priors, this->_num_classes, param.background_id, \
        param.keep_top_k, param.nms_top_k, param.conf_thresh, param.nms_thresh, param.nms_eta, _shared_loc);

    if (result.size() == 0) {
        result.resize(7);
        for (int i = 0; i < 7; ++i) {
            result[i] = (float)-1;
        }
        outputs[0]->reshape(Shape({1, 1, 1, 7}));
    } else {
        outputs[0]->reshape(Shape({1, 1, static_cast<int>(result.size() / 7), 7}));
    }

    memcpy(outputs[0]->mutable_data(), result.data(), \
                result.size() * sizeof(float));

    return SaberSuccess;
}

template class SaberDetectionOutput<X86, AK_FLOAT>;
} //namespace anakin

} //namespace anakin
