
#include "saber/funcs/impl/x86/saber_sproposal.h"
#include "mkl.h"
#include <limits>
#include <cmath>

namespace anakin {
namespace saber {

struct abox{
    float batch_ind;
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    bool operator < (const abox&tmp) const {
        return score < tmp.score;
    }
};

template<>
std::vector<float> SaberSProposal<X86, AK_FLOAT>::mkanchor(float w, float h, float x_ctr, float y_ctr){
    std::vector<float> tmp;
    tmp.push_back(x_ctr - 0.5 * (w - 1));
    tmp.push_back(y_ctr - 0.5 * (h - 1));
    tmp.push_back(x_ctr + 0.5 * (w - 1));
    tmp.push_back(y_ctr + 0.5 * (h - 1));
    return tmp;
}

template<>
std::vector<float> SaberSProposal<X86, AK_FLOAT>::whctrs(std::vector<float> anchor){
    std::vector<float> result;
    result.push_back(anchor[2] - anchor[0] + 1); //w
    result.push_back(anchor[3] - anchor[1] + 1); //h
    result.push_back((anchor[2] + anchor[0]) / 2); //ctrx
    result.push_back((anchor[3] + anchor[1]) / 2); //ctry
    return result;
}

template<>
std::vector<std::vector<float> > SaberSProposal<X86, AK_FLOAT>::scale_enum(std::vector<float> anchor){
    std::vector<std::vector<float> > result;
    std::vector<float> reform_anchor = whctrs(anchor);
    float x_ctr = reform_anchor[2];
    float y_ctr = reform_anchor[3];
    float w = reform_anchor[0];
    float h = reform_anchor[1];
    for (int i = 0; i < _anchor_scales.size(); ++i) {
        float ws = w * _anchor_scales[i];
        float hs = h * _anchor_scales[i];
        std::vector<float> tmp = mkanchor(ws, hs, x_ctr, y_ctr);
        result.push_back(tmp);
    }
    return result;
}

template<>
std::vector<std::vector<float> > SaberSProposal<X86, AK_FLOAT>::ratio_enum(std::vector<float> anchor){
    std::vector<std::vector<float> > result;
    std::vector<float> reform_anchor = whctrs(anchor);
    float x_ctr = reform_anchor[2];
    float y_ctr = reform_anchor[3];
    float size = reform_anchor[0] * reform_anchor[1];
    for (int i = 0; i < _ratios.size(); ++i) {
        float size_ratios = size / _ratios[i];
        float ws = round(std::sqrt(size_ratios));
        float hs = round(ws * _ratios[i]);
        std::vector<float> tmp = mkanchor(ws, hs, x_ctr, y_ctr);
        result.push_back(tmp);
    }
    return result;
}

template<>
void SaberSProposal<X86, AK_FLOAT>::generate_anchors(){
    //generate base anchor
    std::vector<float> base_anchor;
    base_anchor.push_back(0);
    base_anchor.push_back(0);
    base_anchor.push_back(_base_size - 1);
    base_anchor.push_back(_base_size - 1);
    //enum ratio anchors
    std::vector<std::vector<float> >ratio_anchors = ratio_enum(base_anchor);
    for (int i = 0; i < ratio_anchors.size(); ++i) {
        std::vector<std::vector<float> > tmp = scale_enum(ratio_anchors[i]);
        _gen_anchors.insert(_gen_anchors.end(), tmp.begin(), tmp.end());
    }
}

void nms(std::vector<abox> &input_boxes, float nms_thresh) {
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < input_boxes.size(); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < input_boxes.size(); ++i) {
        for (int j = i + 1; j < input_boxes.size();) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_thresh) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

template<>
SaberStatus SaberSProposal<X86, AK_FLOAT>::create(
        const std::vector<Tensor < X86> *>& inputs,
        std::vector<Tensor < X86> *>& outputs,
        SProposalParam <X86> &param,
        Context<X86> &ctx) {

    _map_width = inputs[1]->width(); //feat_width
    _map_height = inputs[1]->height(); //feat_height
    int length = std::max(_map_width, _map_height);
    int step = _map_width * _map_height;
    Shape local_anchors_shape({1, _anchors_nums * 4, _map_height, _map_width}, Layout_NCHW);
    Shape map_m_shape({length}, Layout_W);
    Shape step_shape({step}, Layout_W);
    _local_anchors.reshape(local_anchors_shape);
    _map_m_tensor.reshape(map_m_shape);
    _shift_x_tensor.reshape(step_shape);
    _shift_y_tensor.reshape(step_shape);
    return SaberSuccess;
}

template<>
SaberStatus SaberSProposal<X86, AK_FLOAT>::init(
        const std::vector<Tensor < X86> *>& inputs,
        std::vector<Tensor < X86> *>& outputs,
        SProposalParam <X86> &param,
        Context<X86> &ctx) {

    this->_ctx = &ctx;
    _anchor_scales.clear();
    _ratios.clear();
    _feat_stride = param.feat_stride;
    _base_size = param.basesize;
    _min_size = param.boxminsize;
    _pre_nms_topN = param.pre_nms_topn;
    _post_nms_topN = param.post_nms_topn;
    _nms_thresh = param.nms_thresh;
    int scales_num = param.scale.size();
    for (int i = 0; i < scales_num; ++i) {
        _anchor_scales.push_back(param.scale[i]);
    }
    int ratios_num = param.ratio.size();
    for (int i = 0; i < ratios_num; ++i) {
        _ratios.push_back(param.ratio[i]);
    }

    generate_anchors();

    _anchors_nums = _gen_anchors.size();
    Shape anchors_shape({_anchors_nums * 4}, Layout_W);
    _anchors_tensor.re_alloc(anchors_shape, AK_FLOAT);
    _anchors = (int*)_anchors_tensor.mutable_data();

    for (int i = 0; i<_gen_anchors.size(); ++i) {
        for (int j = 0; j < _gen_anchors[i].size(); ++j) {
            _anchors[i * 4 + j] = _gen_anchors[i][j];
        }
    }
    _map_width = inputs[1]->width(); //feat_width
    _map_height = inputs[1]->height(); //feat_height
    int length = std::max(_map_width, _map_height);
    int step = _map_width * _map_height;
    Shape local_anchors_shape({1, _anchors_nums * 4, _map_height, _map_width}, Layout_NCHW);
    Shape map_m_shape({length}, Layout_W);
    Shape step_shape({step}, Layout_W);
    _local_anchors.re_alloc(local_anchors_shape, AK_FLOAT);
    _map_m_tensor.re_alloc(map_m_shape, AK_FLOAT);
    _shift_x_tensor.re_alloc(step_shape, AK_FLOAT);
    _shift_y_tensor.re_alloc(step_shape, AK_FLOAT);
    return create(inputs, outputs, param, ctx);
}

template<>
SaberStatus SaberSProposal<X86, AK_FLOAT>::dispatch(
    const std::vector<Tensor < X86> *>& inputs,
    std::vector<Tensor < X86> *>& outputs,
    SProposalParam <X86> &param) {

    _map_width = inputs[1]->width(); //feat_width
    _map_height = inputs[1]->height(); //feat_height
    //int channel = inputs[1]->channel();

    //get boxs_delta,向右。
    auto m_box_ = inputs[1];
    //get sores 向右，前面_anchors_nums个位bg的得分，后面_anchors_nums为fg得分，我们需要的是后面的。
    auto m_score_ = inputs[0];
    //get im_info
    const float* img_info = (const float*)inputs[2]->data();
    int img_info_h = inputs[2]->height();
    int img_info_w = inputs[2]->width();
    _src_height = img_info[0];
    _src_width = img_info[1 * img_info_h * img_info_w];
    _src_scale = img_info[2 * img_info_h * img_info_w];

    //gen local anchors 向右
    int length = std::max(_map_width, _map_height);
    int step = _map_width * _map_height;
    int *_map_m = (int*)_map_m_tensor.mutable_data();
    for (int i = 0; i < length; ++i) {
        _map_m[i] = i * _feat_stride;
    }
    float *_shift_x = (float*)_shift_x_tensor.mutable_data();
    float *_shift_y = (float*)_shift_y_tensor.mutable_data();
    for (int i = 0; i < _map_height; ++i) {
        for (int j = 0; j < _map_width; ++j) {
            _shift_x[i * _map_width + j] = _map_m[j];
            _shift_y[i * _map_width + j] = _map_m[i];
        }
    }

    float *local_anchors_ptr = (float*)_local_anchors.mutable_data();
    for (int i = 0; i < _anchors_nums; ++i) {
        for (int j = 0; j < step; ++j) {
            (local_anchors_ptr + (i * 4 + 0) * step)[j] = float(_anchors[i * 4 + 0]);
        }
        for (int j = 0; j < step; ++j) {
            (local_anchors_ptr + (i * 4 + 1) * step)[j] = float(_anchors[i * 4 + 1]);
        }
        for (int j = 0; j < step; ++j) {
            (local_anchors_ptr + (i * 4 + 2) * step)[j] = float(_anchors[i * 4 + 2]);
        }
        for (int j = 0; j < step; ++j) {
            (local_anchors_ptr + (i * 4 + 3) * step)[j] = float(_anchors[i * 4 + 3]);
        }
        cblas_saxpy(step, float(1), _shift_x, 1, local_anchors_ptr + (i * 4 + 0) * step, 1);
        cblas_saxpy(step, float(1), _shift_x, 1, local_anchors_ptr + (i * 4 + 2) * step, 1);
        cblas_saxpy(step, float(1), _shift_y, 1, local_anchors_ptr + (i * 4 + 1) * step, 1);
        cblas_saxpy(step, float(1), _shift_y, 1, local_anchors_ptr + (i * 4 + 3) * step, 1);
    }

    //Convert anchors into proposals via bbox transformations

    int channel = m_box_->channel();
    int height = m_box_->height();
    int width = m_box_->width();
    int m_box_step = height * width;
    float* m_box_ptr = (float*)m_box_->mutable_data(); // bbox_deltas

    for (int i = 0; i < channel / 4; ++i) {

//        // [xmin, ymin, xmax, ymax] -> [width, height, ctr_x, ctr_y]
        cblas_saxpy(2 * m_box_step, float(-1),
                local_anchors_ptr + (i * 4 + 0) * m_box_step, 1,
                local_anchors_ptr + (i * 4 + 2) * m_box_step, 1);
        for (int i = 0; i < 2 * m_box_step; ++i) {
            (local_anchors_ptr + (i * 4 + 2) * m_box_step)[i] += float(1);
        }
        cblas_saxpy(2 * m_box_step, float(0.5),
                local_anchors_ptr + (i * 4 + 2) * m_box_step, 1,
                local_anchors_ptr + (i * 4 + 0) * m_box_step, 1);

        // add offset: ctr_x = ctr_x + tx * width_delta, ctr_y = ctr_y + ty * height_delta
        vsMul(2 * m_box_step,
                local_anchors_ptr + (i * 4 + 2) * m_box_step,
                m_box_ptr + (i * 4 + 0) * m_box_step,
                m_box_ptr + (i * 4 + 0) * m_box_step);

        vsAdd(2 * m_box_step,
                local_anchors_ptr + (i * 4 + 0) * m_box_step,
                m_box_ptr + (i * 4 + 0) * m_box_step,
                m_box_ptr + (i * 4 + 0) * m_box_step);

        // add offset: width = width * exp(width_delta), height = height * exp(height_delta)
        vsExp(2 * m_box_step,
                m_box_ptr + (i * 4 + 2) * m_box_step,
                m_box_ptr + (i * 4 + 2) * m_box_step);

        vsMul(2 * m_box_step,
                local_anchors_ptr + (i * 4 + 2) * m_box_step,
                m_box_ptr + (i * 4 + 2) * m_box_step,
                m_box_ptr + (i * 4 + 2) * m_box_step);
//
//        // do not reverse the quantities
//        // leaving [width, height, ctr_x, ctr_y] ->  [xmin, ymin, xmax, ymax] undone.
    }

    std::vector<abox> aboxes;

    int map_width = m_box_->width();
    int map_height = m_box_->height();
    int map_channel = m_box_->channel();
    const float *box = (const float*)m_box_->data(); // bbox_deltas
    const float *score = (const float*)m_score_->data(); // scores

    int offset_step = 4 * map_height * map_width;
    int one_step = map_height * map_width;
    int offset_w, offset_h, offset_x, offset_y, offset_s;

    for (int h = 0; h < map_height; ++h) {
        for (int w = 0; w < map_width; ++w) {
            offset_x = h * map_width + w;
            offset_y = offset_x + one_step;
            offset_w = offset_y + one_step;
            offset_h = offset_w + one_step;
            offset_s = one_step * _anchors_nums + h * map_width + w;
            for (int c = 0; c < map_channel / 4; ++c) {
                float width = box[offset_w], height = box[offset_h];
                abox tmp;
                tmp.batch_ind = 0;
                tmp.x1 = box[offset_x] - 0.5 * width;
                tmp.y1 = box[offset_y] - 0.5 * height;
                tmp.x2 = box[offset_x] + 0.5 * width;
                tmp.y2 = box[offset_y] + 0.5 * height;
                tmp.x1 = std::min(std::max(tmp.x1, 0.f), _src_width - 1.f);
                tmp.y1 = std::min(std::max(tmp.y1, 0.f), _src_height - 1.f);
                tmp.x2 = std::min(std::max(tmp.x2, 0.f), _src_width - 1.f);
                tmp.y2 = std::min(std::max(tmp.y2, 0.f), _src_height - 1.f);
                tmp.score = score[offset_s];
                aboxes.push_back(tmp);
                offset_x += offset_step;
                offset_y += offset_step;
                offset_w += offset_step;
                offset_h += offset_step;
                offset_s += one_step;
            }
        }
    }

    std::sort(aboxes.rbegin(), aboxes.rend()); //降序

    if (_pre_nms_topN > 0 && _pre_nms_topN < aboxes.size()) {
        int tmp = std::min((size_t)_pre_nms_topN, aboxes.size());
        aboxes.erase(aboxes.begin() + tmp, aboxes.end());
    }

    nms(aboxes,_nms_thresh);

    if (_post_nms_topN > 0) {
        int tmp = std::min((size_t)_post_nms_topN, aboxes.size());
        aboxes.erase(aboxes.begin() + tmp, aboxes.end());
    }
    Shape output_shape({1, aboxes.size(), 5, 1}, Layout_NCHW);
    outputs[0]->reshape(output_shape);
    float *top0 = (float*)outputs[0]->mutable_data();
    int output_offset = outputs[0]->height() * outputs[0]->width();
    for (int i = 0; i < aboxes.size(); ++i) {
        //caffe_copy(aboxes.size() * 5, (float*)aboxes.data(), top0);
        top0[0] = aboxes[i].batch_ind;
        top0[1] = aboxes[i].x1;
        top0[2] = aboxes[i].y1;
        top0[3] = aboxes[i].x2;
        top0[4] = aboxes[i].y2;
//        top0 += outputs[0]->offset(0, 1);
        top0 += output_offset;
    }

    return SaberSuccess;
}

template class SaberSProposal<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSProposal, SProposalParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSProposal, SProposalParam, X86, AK_INT8);

} //namespace saber.
} //namespace anakin.
