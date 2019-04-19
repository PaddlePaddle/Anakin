#include "saber/funcs/impl/arm/saber_sproposal.h"
#include "saber/funcs/impl/arm/neon/impl/neon_mathfun.h"

#include <cmath>

namespace anakin{

namespace saber{

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

template <>
std::vector<float> SaberSProposal<ARM, AK_FLOAT>::mkanchor(float w, float h, float x_ctr, float y_ctr){
    std::vector<float> tmp;
    tmp.push_back(x_ctr - 0.5 * (w - 1));
    tmp.push_back(y_ctr - 0.5 * (h - 1));
    tmp.push_back(x_ctr + 0.5 * (w - 1));
    tmp.push_back(y_ctr + 0.5 * (h - 1));
    return tmp;
}

template <>
std::vector<float> SaberSProposal<ARM, AK_FLOAT>::whctrs(std::vector<float> anchor){
    std::vector<float> result;
    result.push_back(anchor[2] - anchor[0] + 1); //w
    result.push_back(anchor[3] - anchor[1] + 1); //h
    result.push_back((anchor[2] + anchor[0]) / 2); //ctrx
    result.push_back((anchor[3] + anchor[1]) / 2); //ctry
    return result;
}

template <>
std::vector<std::vector<float> > SaberSProposal<ARM, AK_FLOAT>::scale_enum(std::vector<float> anchor){
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

template <>
std::vector<std::vector<float> > SaberSProposal<ARM, AK_FLOAT>::ratio_enum(std::vector<float> anchor){
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

template <>
void SaberSProposal<ARM, AK_FLOAT>::generate_anchors(){
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

template <>
SaberStatus SaberSProposal<ARM, AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SProposalParam<ARM>& param, Context<ARM> &ctx){
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
    float* _anchors = (float*)_anchors_tensor.mutable_data();

    for (int i = 0; i<_gen_anchors.size(); ++i) {
        for (int j = 0; j < _gen_anchors[i].size(); ++j) {
            _anchors[i * 4 + j] = _gen_anchors[i][j];
        }
    }
    _map_width = inputs[1]->width(); //feat_width
    _map_height = inputs[1]->height(); //feat_height
    int length = std::max(_map_width, _map_height);
    int step = _map_width * _map_height;
    Shape local_anchors_shape ({1, _anchors_nums * 4, _map_height, _map_width}, Layout_NCHW);
    Shape map_m_shape({length}, Layout_W);
    Shape step_shape({step}, Layout_W);
    _local_anchors.re_alloc(local_anchors_shape, AK_FLOAT);
    _map_m_tensor.re_alloc(map_m_shape, AK_FLOAT);
    _shift_x_tensor.re_alloc(step_shape, AK_FLOAT);
    _shift_y_tensor.re_alloc(step_shape, AK_FLOAT);
}

template <>
SaberStatus SaberSProposal<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        SProposalParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
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
    int size = 4 * step;
    float* anchors_ptr = (float*)_anchors_tensor.data();
    for (int i = 0; i < _anchors_nums; ++i) {
        float* din_ptr = anchors_ptr;
        float* x_ptr = _shift_x;
        float* y_ptr = _shift_y;
        float* dout_ptr0 = local_anchors_ptr;
        float* dout_ptr1 = dout_ptr0 + step;
        float* dout_ptr2 = dout_ptr1 + step;
        float* dout_ptr3 = dout_ptr2 + step;
        int cnt = step;
#ifdef __aarch64__
            asm volatile(
                "ld1 {v0.4s}, [%[din_ptr]]   \n" /*vld1q_f32(din_ptr0)*/
                "1:                             \n"
                "ldr     s4, [%[x_ptr]], #4   \n" /* load in, 1 float */
                "ldr     s5, [%[y_ptr]], #4   \n" /* load in, 1 float */
                "fadd   s0, s0, s4   \n" /*  add + x */
                "fadd   s1, s1, s5   \n" /*  add + y */
                "fadd   s2, s2, s4   \n" /*  add + x */
                "fadd   s3, s3, s5   \n" /*  add+ y */
                "subs %w[cnt], %w[cnt], #1 \n"
                "str s0, [%[dout_ptr0]], #4       \n"
                "str s1, [%[dout_ptr1]], #4       \n"
                "str s2, [%[dout_ptr2]], #4       \n"
                "str s3, [%[dout_ptr3]], #4       \n"
                "bne 1b \n"
                :[cnt] "+r" (cnt), [din_ptr] "+r" (din_ptr), [x_ptr] "+r" (x_ptr), [y_ptr] "+r" (y_ptr), \
                 [dout_ptr0] "+r" (dout_ptr0), [dout_ptr1] "+r" (dout_ptr1), [dout_ptr2] "+r" (dout_ptr2), \
                 [dout_ptr3] "+r" (dout_ptr3)
                :
                :"v0", "v1", "v2", "cc", "memory"
            );
#else
            asm volatile(
                "vld1.32  {d0-d1}, [%[din_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "1:                             \n"
                "vldm     %[x_ptr]!, {s4}         @ load 1 float\n" /* load in, 1 float */
                "vldm     %[y_ptr]!, {s5}         @ load 1 float\n" /* load in, 1 float */
                "vadd.f32   s0, s0, s4   \n" /*  add + x */
                "vadd.f32   s1, s1, s5   \n" /*  add + y */
                "vadd.f32  s2, s2, s4   \n" /*  add + x */
                "vadd.f32  s3, s3, s5   \n" /*  add+ y */
                "subs %[cnt], #1 \n"
                "vst1.32 {d0[0]}, [%[dout_ptr0]]!      @ save result\n"
                "vst1.32 {d0[1]}, [%[dout_ptr1]]!      @ save result\n"
                "vst1.32 {d1[0]}, [%[dout_ptr2]]!      @ save result\n"
                "vst1.32 {d1[1]}, [%[dout_ptr3]]!      @ save result\n"
                "bne 1b                 \n"
                :[cnt] "+r" (cnt), [din_ptr] "+r" (din_ptr), [x_ptr] "+r" (x_ptr), [y_ptr] "+r" (y_ptr), \
                 [dout_ptr0] "+r" (dout_ptr0), [dout_ptr1] "+r" (dout_ptr1), [dout_ptr2] "+r" (dout_ptr2), \
                 [dout_ptr3] "+r" (dout_ptr3)
                :
                :"q0", "q1", "q2", "cc", "memory"
            );
#endif
        anchors_ptr += 4;
        // _shift_x += step;
        // _shift_y += step;
        local_anchors_ptr += size;
    }

    //Convert anchors into proposals via bbox transformations
    int channel = m_box_->channel();
    int height = m_box_->height();
    int width = m_box_->width();
    int m_box_step = height * width;
    float* m_box_ptr = (float*)m_box_->mutable_data(); // bbox_deltas

    size = 4 * m_box_step;
    int len = 2 * m_box_step;
    int cnt = len / 8;
    int remain = len % 8;
    float32x4_t vone = vdupq_n_f32(1.0f);
    float32x4_t vhalf = vdupq_n_f32(0.5f);
    for (int i = 0; i < channel / 4; ++i) {
        float* x_ptr = &local_anchors_ptr[i * size];
        float* y_ptr = x_ptr + len;
        int loop_cnt  = cnt;
#ifdef __aarch64__
            asm volatile(
                "ld1 {v0.4s}, [%[x_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v1.4s}, [%[y_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "1:                             \n"
                "ld1 {v2.4s}, [%[x_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v3.4s}, [%[y_ptr]]        \n" /*vld1q_f32(din_ptr0)*/
                "fsub  v1.4s, v1.4s, v0.4s   \n" /*  sub y - x */
                "fsub  v3.4s, v3.4s, v2.4s   \n" /*  sub y - x */
                "sub %[y_ptr], %[y_ptr], #16   \n"
                "fadd  v1.4s, v1.4s, %[vone].4s \n"
                "fadd  v3.4s, v3.4s, %[vone].4s \n"
                "ld1 {v0.4s}, [%[x_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "subs %[cnt], %[cnt], #1      \n"
                "st1 {v1.4s}, [%[y_ptr]], #16 \n"
                "st1 {v3.4s}, [%[y_ptr]], #16 \n"
                "ld1 {v1.4s}, [%[y_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "bne 1b \n"
                :[cnt] "+r" (loop_cnt), [x_ptr] "+r" (x_ptr), [y_ptr] "+r" (y_ptr)
                :[vone] "w" (vone)
                :"v0", "v1", "v2", "v3", "cc", "memory"
            );
#else
            asm volatile(
                "vld1.32  {d0-d1}, [%[x_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vld1.32  {d2-d3}, [%[y_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "1:                             \n"
                "vld1.32  {d4-d5}, [%[x_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vld1.32  {d6-d7}, [%[y_ptr]]    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vsub.f32   q1, q1, q0   \n" /*  sub y - x */
                "vsub.f32   q3, q3, q2   \n" /*  sub y - x */
                "sub %[y_ptr], #16   \n"
                "vadd.f32 q1, q1, %q[vone] @ add +1 \n"
                "vadd.f32 q3, q3, %q[vone] @ add +1 \n"
                "vld1.32  {d0-d1}, [%[x_ptr]]!    @ load din r0\n"
                "subs %[cnt], #1 \n"
                "vst1.32 {d2-d3}, [%[y_ptr]]!      @ save result\n"
                "vst1.32 {d6-d7}, [%[y_ptr]]!      @ save result\n"
                "vld1.32  {d2-d3}, [%[y_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "bne 1b                 \n"
                :[cnt] "+r" (loop_cnt), [x_ptr] "+r" (x_ptr), [y_ptr] "+r" (y_ptr)
                :[vone] "w" (vone)
                :"q0", "q1", "q2", "q3", "cc", "memory"
            );
#endif
        x_ptr -= 4;
        y_ptr -= 4;
        for (int j = 0; j < remain; j++){
            y_ptr[j] = y_ptr[j] - x_ptr[j] + 1.0;
        }

        y_ptr = &local_anchors_ptr[i * size];
        x_ptr = x_ptr + len;
        loop_cnt  = cnt;
#ifdef __aarch64__
            asm volatile(
                "ld1 {v0.4s}, [%[x_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v1.4s}, [%[y_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "1:                             \n"
                "ld1 {v2.4s}, [%[x_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v3.4s}, [%[y_ptr]]        \n" /*vld1q_f32(din_ptr0)*/
                "fmla  v1.4s, v0.4s, %[vhalf].4s   \n" /*  mla y = y + 0.5x */
                "fsub  v3.4s, v2.4s, %[vhalf].4s   \n" /*  mla y = y + 0.5x */
                "sub %[y_ptr], %[y_ptr], #16   \n"
                "ld1 {v0.4s}, [%[x_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "subs %[cnt], %[cnt], #1      \n"
                "st1 {v1.4s}, [%[y_ptr]], #16 \n"
                "st1 {v3.4s}, [%[y_ptr]], #16 \n"
                "ld1 {v1.4s}, [%[y_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "bne 1b \n"
                :[cnt] "+r" (loop_cnt), [x_ptr] "+r" (x_ptr), [y_ptr] "+r" (y_ptr)
                :[vhalf] "w" (vhalf)
                :"v0", "v1", "v2", "v3", "cc", "memory"
            );
#else
            asm volatile(
                "vld1.32  {d0-d1}, [%[x_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vld1.32  {d2-d3}, [%[y_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "1:                             \n"
                "vld1.32  {d4-d5}, [%[x_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vld1.32  {d6-d7}, [%[y_ptr]]    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vmla.f32   q1, q0, %q[vhalf]   \n" /*  mla y = y + 0.5x */
                "vmla.f32   q3, q2, %q[vhalf]   \n" /*  mla y = y + 0.5x */
                "sub %[y_ptr], #16   \n"
                "vld1.32  {d0-d1}, [%[x_ptr]]!    @ load din r0\n"
                "subs %[cnt], #1 \n"
                "vst1.32 {d2-d3}, [%[y_ptr]]!      @ save result\n"
                "vst1.32 {d6-d7}, [%[y_ptr]]!      @ save result\n"
                "vld1.32  {d2-d3}, [%[y_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "bne 1b                 \n"
                :[cnt] "+r" (loop_cnt), [x_ptr] "+r" (x_ptr), [y_ptr] "+r" (y_ptr)
                :[vhalf] "w" (vhalf)
                :"q0", "q1", "q2", "q3", "cc", "memory"
            );
#endif
        x_ptr -= 4;
        y_ptr -= 4;
        for (int j = 0; j < remain; j++){
            y_ptr[j] = y_ptr[j] + 0.5 * x_ptr[j];
        }

        y_ptr = &m_box_ptr[i * size];
        float* x_ptr0 = &local_anchors_ptr[i * size];
        float* x_ptr1 = x_ptr0 + len;
        loop_cnt  = cnt;
        //y_ptr = y_ptr * x_ptr1 + x_ptr0
#ifdef __aarch64__
            asm volatile(
                "ld1 {v0.4s}, [%[x_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v1.4s}, [%[y_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v2.4s}, [%[x_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "1:                             \n"
                "ld1 {v3.4s}, [%[x_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v4.4s}, [%[y_ptr]]         \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v5.4s}, [%[x_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "fmla  v2.4s, v0.4s, v1.4s   \n" /*  mla x0 = x0 + y * x1 */
                "fsub  v5.4s, v3.4s, v4.4s   \n" /*  mla x0 = x0 + y * x1 */
                "sub %[y_ptr], %[y_ptr], #16   \n"
                "ld1 {v0.4s}, [%[x_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "ld1 {v2.4s}, [%[x_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "subs %[cnt], %[cnt], #1      \n"
                "st1 {v2.4s}, [%[y_ptr]], #16 \n"
                "st1 {v5.4s}, [%[y_ptr]], #16 \n"
                "ld1 {v1.4s}, [%[y_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
                "bne 1b \n"
                :[cnt] "+r" (loop_cnt), [x_ptr0] "+r" (x_ptr0), [y_ptr] "+r" (y_ptr), \
                 [x_ptr1] "+r" (x_ptr1)
                :
                :"v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory"
            );
#else
            asm volatile(
                "vld1.32  {d0-d1}, [%[x_ptr1]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vld1.32  {d2-d3}, [%[y_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vld1.32  {d4-d5}, [%[x_ptr0]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "1:                             \n"
                "vld1.32  {d6-d7}, [%[x_ptr1]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vld1.32  {d8-d9}, [%[y_ptr]]    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vld1.32  {d10-d11}, [%[x_ptr0]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "vmla.f32   q2, q0, q1   \n" /*  mla x0 = x0 + y * x1 */
                "vmla.f32   q5, q4, q3   \n" /*  mla x0 = x0 + y * x1 */
                "sub %[y_ptr], #16   \n"
                "vld1.32  {d0-d1}, [%[x_ptr1]]!    @ load din r0\n"
                "vld1.32  {d4-d5}, [%[x_ptr0]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "subs %[cnt], #1 \n"
                "vst1.32 {d4-d5}, [%[y_ptr]]!      @ save result\n"
                "vst1.32 {d10-d11}, [%[y_ptr]]!      @ save result\n"
                "vld1.32  {d2-d3}, [%[y_ptr]]!    @ load din r0\n" /*vld1q_f32(din_ptr0)*/
                "bne 1b                 \n"
                :[cnt] "+r" (loop_cnt), [x_ptr0] "+r" (x_ptr0), [y_ptr] "+r" (y_ptr), \
                 [x_ptr1] "+r" (x_ptr1)
                :
                :"q0", "q1", "q2", "q3", "q4", "q5", "cc", "memory"
            );
#endif
        x_ptr0 -= 4;
        x_ptr1 -= 4;
        y_ptr -= 4;
        for (int j = 0; j < remain; j++){
            y_ptr[j] = y_ptr[j] * x_ptr1[j] + x_ptr0[j];
        }
        y_ptr = &m_box_ptr[i * size + len];
        x_ptr = &local_anchors_ptr[i * size + len];

        //y = exp(y) * x
        for (int j = 0; j < cnt; j++){
            float32x4_t din0 = vld1q_f32(y_ptr);
            float32x4_t din1 = vld1q_f32(y_ptr + 4);
            float32x4_t x0 = vld1q_f32(x_ptr);
            float32x4_t x1 = vld1q_f32(x_ptr + 4);
            float32x4_t sum0 = exp_ps(din0);
            float32x4_t sum1 = exp_ps(din1);
            float32x4_t res0 = vmulq_f32(sum0, x0);
            float32x4_t res1 = vmulq_f32(sum1, x1);
            x_ptr += 8;
            vst1q_f32(y_ptr, res0);
            vst1q_f32(y_ptr + 4, res1);
            y_ptr += 8;
        }
        for (int j = 0; j < remain; j++){
            y_ptr[j] = exp(y_ptr[j]) * x_ptr[j];
        }
          // do not reverse the quantities
        // leaving [width, height, ctr_x, ctr_y] ->  [xmin, ymin, xmax, ymax] undone.
    }

    std::vector<abox> aboxes;
    int map_width = m_box_->width();
    int map_height = m_box_->height();
    int map_channel = m_box_->channel();

    const float *box = (const float*)m_box_->data(); // bbox_deltas
    const float *score = (const float*)m_score_->data(); // scores
    int offset_step = 4 * map_height * map_width;
    int one_step = map_height * map_width;
    int offset_w = 0;
    int offset_h = 0;
    int offset_x = 0;
    int offset_y = 0;
    int offset_s = 0;

    for (int h = 0; h < map_height; ++h) {
        for (int w = 0; w < map_width; ++w) {
            offset_x = h * map_width + w;
            offset_y = offset_x + one_step;
            offset_w = offset_y + one_step;
            offset_h = offset_w + one_step;
            offset_s = one_step * _anchors_nums + h * map_width + w;
            for (int c = 0; c < map_channel / 4; ++c) {
                float width = box[offset_w];
                float height = box[offset_h];
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
        top0 += output_offset;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Sproposal : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("Sproposal", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberSProposal, SProposalParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSProposal, SProposalParam, ARM, AK_INT8);

} //namespace anakin

} //namespace anakin
