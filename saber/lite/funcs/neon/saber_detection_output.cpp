#include "saber/lite/funcs/saber_detection_output.h"
#include "saber/lite/funcs/detection_lite.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{

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


SaberDetectionOutput::SaberDetectionOutput(const ParamBase *param) {
    _param = (const DetectionOutputParam*)param;
    this->_flag_param = true;
}

SaberDetectionOutput::~SaberDetectionOutput() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberDetectionOutput::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const DetectionOutputParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberDetectionOutput::load_param(std::istream &stream, const float *weights) {
    int class_num;
    float conf_thresh;
    int nms_topk;
    int bg_id;
    int keep_topk;
    int cd_type;
    float nms_thresh;
    float nms_eta;
    int share_loc;
    int encode_in_tar;
    stream >> class_num >> conf_thresh >> nms_topk >> bg_id >> keep_topk >> \
        cd_type >> nms_thresh >> nms_eta >> share_loc >> encode_in_tar;
    CodeType type = static_cast<CodeType>(cd_type);
    _param = new DetectionOutputParam(class_num, conf_thresh, nms_topk, \
        bg_id, keep_topk, type, nms_thresh, nms_eta, share_loc>0, encode_in_tar>0);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#if 0
SaberStatus SaberDetectionOutput::load_param(FILE *fp, const float *weights) {
    int class_num;
    float conf_thresh;
    int nms_topk;
    int bg_id;
    int keep_topk;
    int cd_type;
    float nms_thresh;
    float nms_eta;
    int share_loc;
    int encode_in_tar;

    fscanf(fp, "%d %f %d %d %d %d %f %f %d %d\n", &class_num, &conf_thresh, \
        &nms_topk, &bg_id, &keep_topk, &cd_type, &nms_thresh, &nms_eta, \
        &share_loc, &encode_in_tar);
    CodeType type = static_cast<CodeType>(cd_type);
    _param = new DetectionOutputParam(class_num, conf_thresh, nms_topk, \
        bg_id, keep_topk, type, nms_thresh, nms_eta, share_loc>0, encode_in_tar>0);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#endif
SaberStatus SaberDetectionOutput::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                       std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load detection_output param first\n");
        return SaberNotInitialized;
    }
    //! output tensor's dims = 2
    Shape shape_out;
    shape_out.resize(4);
    //CHECK_EQ(shape_out.dims(), 4) << "only support 4d layout";
    shape_out[0] = 1;
    shape_out[1] = 1;
    shape_out[2] = inputs[0]->num() * _param->_keep_top_k;
    shape_out[3] = 7;

    return outputs[0]->set_shape(shape_out);
}

SaberStatus SaberDetectionOutput::init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
                      std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) {

    if (!this->_flag_param) {
        printf("load detection_output param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;

    //! inputs[0]: location map, dims = 4 {N, boxes * 4, 1, 1}
    //! inputs[1]: confidence map, dims = 4 {N, boxes * classes, 1, 1}
    //! inputs[2]: prior boxes, dims = 4 {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}
    Shape sh_loc = inputs[0]->valid_shape();
    Shape sh_conf = inputs[1]->valid_shape();
    Shape sh_box = inputs[2]->valid_shape();
    //! shape {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}, boxes = size / 2 / 4
    //! layout must be 4 dims, the priors is in the last dim
    _num_priors = inputs[2]->valid_size() / 8;
    int num = inputs[0]->num();
    if (this->_param->_class_num == 0) {
        _class_num = inputs[1]->valid_size() / (num * _num_priors);
    } else {
        _class_num = this->_param->_class_num;
    }
    if (this->_param->_share_location) {
        _num_loc_classes = 1;
    } else {
        _num_loc_classes = _class_num;
        _bbox_permute.reshape(sh_loc);
    }

    _bbox_preds.reshape(sh_loc);
    _conf_permute.reshape(sh_conf);

    LCHECK_EQ(_num_priors * _num_loc_classes * 4, sh_loc[1], "Number of priors must match number of location predictions.");
    LCHECK_EQ(_num_priors * _class_num, sh_conf[1], "Number of priors must match number of confidence predictions.");

    this->_flag_init = true;
    return SaberSuccess;
}


//template <>
SaberStatus SaberDetectionOutput::dispatch(
        const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
        std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) {

    if (!this->_flag_init) {
        printf("init detection_output first\n");
        return SaberNotInitialized;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    Tensor<CPU, AK_FLOAT>* t_loc = inputs[0];
    Tensor<CPU, AK_FLOAT>* t_conf = inputs[1];
    Tensor<CPU, AK_FLOAT>* t_prior = inputs[2];

    const int num = t_loc->valid_shape()[0];

    const float* loc_data = t_loc->data();
    const float* prior_data = t_prior->data();
    const float* conf_data = t_conf->data();

    float* bbox_data = _bbox_preds.mutable_data();

    if (!this->_param->_share_location) {
        return SaberUnImplError;
    }

    //! Decode predictions.
    //! Retrieve all decoded location predictions.
    decode_bboxes(num, loc_data, prior_data, this->_param->_code_type, this->_param->_variance_encode_in_target, \
        _num_priors, this->_param->_share_location, _num_loc_classes, \
        this->_param->_background_id, bbox_data);

    //! Retrieve all confidences, permute to classes * boxes_size
    float* conf_permute_data = _conf_permute.mutable_data();
    permute_conf(conf_data, num, _num_priors, _class_num, conf_permute_data);

    std::vector<float> result;

    nms_detect(bbox_data, conf_permute_data, result, num, _class_num, _num_priors, this->_param->_background_id, \
        this->_param->_keep_top_k, this->_param->_nms_top_k, this->_param->_conf_thresh, this->_param->_nms_thresh, \
        this->_param->_nms_eta, this->_param->_share_location);

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
#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("detection_output time: %f\n", ts);
    OpTimer::add_timer("detection_optput", ts);
    OpTimer::add_timer("total", ts);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberDetectionOutput);
} //namespace lite

} //namespace saber

} //namespace anakin
