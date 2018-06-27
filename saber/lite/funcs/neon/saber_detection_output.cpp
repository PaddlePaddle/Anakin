#include "saber/lite/funcs/saber_detection_output.h"
#ifdef USE_ARM_PLACE
#include <map>
#include "saber/lite/funcs/neon/impl/neon_mathfun.h"
#include <cmath>
#include <algorithm>
namespace anakin{

namespace saber{

namespace lite{

void decode_bbox_corner_variance_kernel(const int batch_num, \
        const float* loc_data, const float* prior_data, const float* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, float* bbox_data) {

    LCHECK_EQ(share_location, true, "decode boxes without share_location is unimplemented");

    int cnt = num_priors / 4;
    int len_batch = num_priors * 4;

    for (int n = 0; n < batch_num; ++n) {

        const float* ptr_loc_batch = loc_data + n * len_batch;
        float* ptr_bbox_batch = bbox_data + n * len_batch;
#pragma omp parallel for
        for (int i = 0; i < cnt; ++i) {
            int idx = i * 16;
            const float* ptr_loc = ptr_loc_batch + idx;
            const float* ptr_prior = prior_data + idx;
            float* ptr_bbox = ptr_bbox_batch + idx;

            float32x4_t vloc1 = vld1q_f32(ptr_loc);
            float32x4_t vloc2 = vld1q_f32(ptr_loc + 4);
            float32x4_t vloc3 = vld1q_f32(ptr_loc + 8);
            float32x4_t vloc4 = vld1q_f32(ptr_loc + 12);

            float32x4_t vprior1 = vld1q_f32(ptr_prior);
            float32x4_t vprior2 = vld1q_f32(ptr_prior + 4);
            float32x4_t vprior3 = vld1q_f32(ptr_prior + 8);
            float32x4_t vprior4 = vld1q_f32(ptr_prior + 12);

            vst1q_f32(ptr_bbox, vaddq_f32(vloc1, vprior1));
            vst1q_f32(ptr_bbox + 4, vaddq_f32(vloc2, vprior2));
            vst1q_f32(ptr_bbox + 8, vaddq_f32(vloc3, vprior3));
            vst1q_f32(ptr_bbox + 12, vaddq_f32(vloc4, vprior4));
        }
#pragma omp parallel for
        for (int i = cnt * 4; i < num_priors; i++) {
            int idx = i * 4;
            float32x4_t vloc = vld1q_f32(ptr_loc_batch + idx);
            float32x4_t vprior = vld1q_f32(prior_data + idx);
            vst1q_f32(ptr_bbox_batch + idx , vaddq_f32(vloc, vprior));
        }
    }
}

void decode_bbox_corner_no_variance_kernel(const int batch_num, \
        const float* loc_data, const float* prior_data, const float* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, float* bbox_data) {

    LCHECK_EQ(share_location, true, "decode boxes without share_location is unimplemented");

    int cnt = num_priors / 4;
    int len_batch = num_priors * 4;

    for (int n = 0; n < batch_num; ++n) {

        const float *ptr_loc_batch = loc_data + n * len_batch;
        float *ptr_bbox_batch = bbox_data + n * len_batch;

#pragma omp parallel for
        for (int i = 0; i < cnt; ++i) {
            int idx = i * 16;
            const float* ptr_loc = ptr_loc_batch + idx;
            const float* ptr_prior = prior_data + idx;
            const float* ptr_var = variance + idx;
            float* ptr_bbox = ptr_bbox_batch + idx;

            float32x4_t vloc1 = vld1q_f32(ptr_loc);
            float32x4_t vprior1 = vld1q_f32(ptr_prior);
            float32x4_t vvar1 = vld1q_f32(ptr_var);
            float32x4_t vout1 = vmulq_f32(vloc1, vvar1);

            float32x4_t vloc2 = vld1q_f32(ptr_loc + 4);
            float32x4_t vprior2 = vld1q_f32(ptr_prior + 4);
            float32x4_t vvar2 = vld1q_f32(ptr_var + 4);
            float32x4_t vout2 = vmulq_f32(vloc2, vvar2);

            float32x4_t vloc3 = vld1q_f32(ptr_loc + 8);
            float32x4_t vprior3 = vld1q_f32(ptr_prior + 8);
            float32x4_t vvar3 = vld1q_f32(ptr_var + 8);
            float32x4_t vout3 = vmulq_f32(vloc3, vvar3);

            float32x4_t vloc4 = vld1q_f32(ptr_loc + 12);
            float32x4_t vprior4 = vld1q_f32(ptr_prior + 12);
            float32x4_t vvar4 = vld1q_f32(ptr_var + 12);
            float32x4_t vout4 = vmulq_f32(vloc4, vvar4);

            vst1q_f32(ptr_bbox, vaddq_f32(vout1, vprior1));
            vst1q_f32(ptr_bbox + 4, vaddq_f32(vout2, vprior2));
            vst1q_f32(ptr_bbox + 8, vaddq_f32(vout3, vprior3));
            vst1q_f32(ptr_bbox + 12, vaddq_f32(vout4, vprior4));
        }

        for (int i = cnt * 4; i < num_priors; i++) {
            int idx = i * 4;
            float32x4_t vloc = vld1q_f32(ptr_loc_batch + idx);
            float32x4_t vprior = vld1q_f32(prior_data + idx);
            float32x4_t vvar = vld1q_f32(variance + idx);
            float32x4_t vout = vmulq_f32(vloc, vvar);
            vst1q_f32(ptr_bbox_batch + idx, vaddq_f32(vout, vprior));
        }
    }
}

void decode_bbox_center_variance_kernel(const int batch_num, \
        const float* loc_data, const float* prior_data, const float* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, float* bbox_data) {

    LCHECK_EQ(share_location, true, "decode boxes without share_location is unimplemented");

    int cnt = num_priors / 4;
    //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
    //! vloc   0: xmin, 1: ymin, 2: xmax, 3: ymax
    //! vvar
    float32x4_t vhalf = vdupq_n_f32(0.5f);

    int len_batch = num_priors * 4;

    for (int n = 0; n < batch_num; ++n) {

        const float *ptr_loc_batch = loc_data + n * len_batch;
        float *ptr_bbox_batch = bbox_data + n * len_batch;

#pragma omp parallel for
        for (int i = 0; i < cnt; ++i) {
            int idx = i * 16;
            const float* ptr_loc = ptr_loc_batch + idx;
            const float* ptr_prior = prior_data + idx;
            float* ptr_bbox = ptr_bbox_batch + idx;

            float32x4x4_t vprior = vld4q_f32(ptr_prior);
            float32x4x4_t vloc = vld4q_f32(ptr_loc);
            float32x4_t vprior_width = vsubq_f32(vprior.val[2], vprior.val[0]);
            float32x4_t vprior_height = vsubq_f32(vprior.val[3], vprior.val[1]);
            float32x4_t vprior_cx = vmulq_f32(vaddq_f32(vprior.val[0], vprior.val[2]), vhalf);
            float32x4_t vprior_cy = vmulq_f32(vaddq_f32(vprior.val[1], vprior.val[3]), vhalf);

            float32x4_t vdec_bbx_cx = vaddq_f32(vmulq_f32(vloc.val[0], vprior_width), vprior_cx);
            float32x4_t vdec_bbx_cy = vaddq_f32(vmulq_f32(vloc.val[1], vprior_height), vprior_cy);
            float32x4_t vdec_bbx_w = exp_ps(vloc.val[2]);
            float32x4_t vdec_bbx_h = exp_ps(vloc.val[3]);
            vprior_width = vmulq_f32(vprior_width, vhalf);
            vprior_height = vmulq_f32(vprior_height, vhalf);
            vdec_bbx_w = vmulq_f32(vdec_bbx_w, vprior_width);
            vdec_bbx_h = vmulq_f32(vdec_bbx_h, vprior_height);

            vloc.val[0] = vsubq_f32(vdec_bbx_cx, vdec_bbx_w);
            vloc.val[1] = vsubq_f32(vdec_bbx_cy, vdec_bbx_h);
            vloc.val[2] = vaddq_f32(vdec_bbx_cx, vdec_bbx_w);
            vloc.val[3] = vaddq_f32(vdec_bbx_cy, vdec_bbx_h);

            vst4q_f32(ptr_bbox, vloc);
        }
#pragma omp parallel for
        for (int i = cnt * 4; i < num_priors; i++) {
            int idx = i * 4;
            float p_xmin = prior_data[idx];
            float p_ymin = prior_data[idx + 1];
            float p_xmax = prior_data[idx + 2];
            float p_ymax = prior_data[idx + 3];
            float prior_width = p_xmax - p_xmin;
            float prior_height = p_ymax - p_ymin;
            float prior_center_x = (p_xmin + p_xmax) / 2.f;
            float prior_center_y = (p_ymin + p_ymax) / 2.f;

            float xmin = ptr_loc_batch[idx];
            float ymin = ptr_loc_batch[idx + 1];
            float xmax = ptr_loc_batch[idx + 2];
            float ymax = ptr_loc_batch[idx + 3];

            //! variance is encoded in target, we simply need to retore the offset predictions.
            float decode_bbox_center_x = xmin * prior_width + prior_center_x;
            float decode_bbox_center_y = ymin * prior_height + prior_center_y;
            float decode_bbox_width = expf(xmax) * prior_width;
            float decode_bbox_height = expf(ymax) * prior_height;

            ptr_bbox_batch[idx] = decode_bbox_center_x - decode_bbox_width / 2.f;
            ptr_bbox_batch[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
            ptr_bbox_batch[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
            ptr_bbox_batch[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
        }
    }
}

void decode_bbox_center_no_variance_kernel(const int batch_num, \
        const float* loc_data, const float* prior_data, const float* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, float* bbox_data) {

    LCHECK_EQ(share_location, true, "decode boxes without share_location is unimplemented");

    int cnt = num_priors / 4;
    //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
    //! vloc   0: xmin, 1: ymin, 2: xmax, 3: ymax
    //! vvar
    float32x4_t vhalf = vdupq_n_f32(0.5f);

    int len_batch = num_priors * 4;

    for (int n = 0; n < batch_num; ++n) {

        const float *ptr_loc_batch = loc_data + n * len_batch;
        float *ptr_bbox_batch = bbox_data + n * len_batch;

#pragma omp parallel for
        for (int i = 0; i < cnt; ++i) {

            int idx = i * 16;

            const float* ptr_loc = ptr_loc_batch + idx;
            const float* ptr_prior = prior_data + idx;
            const float* ptr_var = variance + idx;
            float* ptr_bbox = ptr_bbox_batch + idx;

            float32x4x4_t vprior = vld4q_f32(ptr_prior);
            float32x4x4_t vloc = vld4q_f32(ptr_loc);
            float32x4x4_t vvar = vld4q_f32(ptr_var);
            float32x4_t vprior_width = vsubq_f32(vprior.val[2], vprior.val[0]);
            float32x4_t vprior_height = vsubq_f32(vprior.val[3], vprior.val[1]);
            float32x4_t vprior_cx = vmulq_f32(vaddq_f32(vprior.val[0], vprior.val[2]), vhalf);
            float32x4_t vprior_cy = vmulq_f32(vaddq_f32(vprior.val[1], vprior.val[3]), vhalf);

            vloc.val[0] = vmulq_f32(vloc.val[0], vvar.val[0]);
            vloc.val[1] = vmulq_f32(vloc.val[1], vvar.val[1]);
            vloc.val[2] = vmulq_f32(vloc.val[2], vvar.val[2]);
            vloc.val[3] = vmulq_f32(vloc.val[3], vvar.val[3]);

            float32x4_t vdec_bbx_cx = vaddq_f32(vmulq_f32(vloc.val[0], vprior_width), vprior_cx);
            float32x4_t vdec_bbx_cy = vaddq_f32(vmulq_f32(vloc.val[1], vprior_height), vprior_cy);
            float32x4_t vdec_bbx_w = exp_ps(vloc.val[2]);
            float32x4_t vdec_bbx_h = exp_ps(vloc.val[3]);
            vprior_width = vmulq_f32(vprior_width, vhalf);
            vprior_height = vmulq_f32(vprior_height, vhalf);
            vdec_bbx_w = vmulq_f32(vdec_bbx_w, vprior_width);
            vdec_bbx_h = vmulq_f32(vdec_bbx_h, vprior_height);

            vloc.val[0] = vsubq_f32(vdec_bbx_cx, vdec_bbx_w);
            vloc.val[1] = vsubq_f32(vdec_bbx_cy, vdec_bbx_h);
            vloc.val[2] = vaddq_f32(vdec_bbx_cx, vdec_bbx_w);
            vloc.val[3] = vaddq_f32(vdec_bbx_cy, vdec_bbx_h);

            vst4q_f32(ptr_bbox, vloc);
        }

#pragma omp parallel for
        for (int i = cnt * 4; i < num_priors; i++) {
            int idx = i * 4;
            float p_xmin = prior_data[idx];
            float p_ymin = prior_data[idx + 1];
            float p_xmax = prior_data[idx + 2];
            float p_ymax = prior_data[idx + 3];
            float prior_width = p_xmax - p_xmin;
            float prior_height = p_ymax - p_ymin;
            float prior_center_x = (p_xmin + p_xmax) / 2.f;
            float prior_center_y = (p_ymin + p_ymax) / 2.f;

            float xmin = ptr_loc_batch[idx];
            float ymin = ptr_loc_batch[idx + 1];
            float xmax = ptr_loc_batch[idx + 2];
            float ymax = ptr_loc_batch[idx + 3];

            //! variance is encoded in target, we simply need to retore the offset predictions.
            float decode_bbox_center_x = variance[idx] * xmin * prior_width + prior_center_x;
            float decode_bbox_center_y = variance[idx + 1] * ymin * prior_height + prior_center_y;
            float decode_bbox_width = expf(variance[idx + 2] * xmax) * prior_width;
            float decode_bbox_height = expf(variance[idx + 3] * ymax) * prior_height;

            ptr_bbox_batch[idx] = decode_bbox_center_x - decode_bbox_width / 2.f;
            ptr_bbox_batch[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
            ptr_bbox_batch[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
            ptr_bbox_batch[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
        }
    }
}

void decode_bbox_corner_size_variance_kernel(const int batch_num, \
        const float* loc_data, const float* prior_data, const float* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, float* bbox_data) {

    LCHECK_EQ(share_location, true, "decode boxes without share_location is unimplemented");

    int cnt = num_priors / 4;
    //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
    //! bbx

    int len_batch = num_priors * 4;

    for (int n = 0; n < batch_num; ++n) {

        const float *ptr_loc_batch = loc_data + n * len_batch;
        float *ptr_bbox_batch = bbox_data + n * len_batch;

#pragma omp parallel for
        for (int i = 0; i < cnt; ++i) {

            int idx = i * 16;

            const float* ptr_loc = ptr_loc_batch + idx;
            const float* ptr_prior = prior_data + idx;
            const float* ptr_var = variance + idx;
            float* ptr_bbox = ptr_bbox_batch + idx;

            float32x4x4_t vprior = vld4q_f32(ptr_prior);
            float32x4x4_t vloc = vld4q_f32(ptr_loc);

            float32x4_t vprior_width = vsubq_f32(vprior.val[2], vprior.val[0]);
            float32x4_t vprior_height = vsubq_f32(vprior.val[3], vprior.val[1]);

            float32x4x4_t vbbx;
            vbbx.val[0] = vmulq_f32(vloc.val[0], vprior_width);
            vbbx.val[1] = vmulq_f32(vloc.val[1], vprior_height);
            vbbx.val[2] = vmulq_f32(vloc.val[2], vprior_width);
            vbbx.val[3] = vmulq_f32(vloc.val[3], vprior_height);

            vbbx.val[0] = vaddq_f32(vprior.val[0], vbbx.val[0]);
            vbbx.val[1] = vaddq_f32(vprior.val[1], vbbx.val[1]);
            vbbx.val[2] = vaddq_f32(vprior.val[2], vbbx.val[2]);
            vbbx.val[3] = vaddq_f32(vprior.val[3], vbbx.val[3]);

            vst4q_f32(ptr_bbox, vbbx);
        }

#pragma omp parallel for
        for (int i = cnt * 4; i < num_priors; i++) {
            int idx = i * 4;
            float p_xmin = prior_data[idx];
            float p_ymin = prior_data[idx + 1];
            float p_xmax = prior_data[idx + 2];
            float p_ymax = prior_data[idx + 3];
            float prior_width = p_xmax - p_xmin;
            float prior_height = p_ymax - p_ymin;

            ptr_bbox_batch[idx] = p_xmin + ptr_loc_batch[idx] * prior_width;
            ptr_bbox_batch[idx + 1] = p_ymin + ptr_loc_batch[idx + 1] * prior_height;
            ptr_bbox_batch[idx + 2] = p_xmax + ptr_loc_batch[idx + 2] * prior_width;
            ptr_bbox_batch[idx + 3] = p_ymax + ptr_loc_batch[idx + 3] * prior_height;
        }

    }
}

void decode_bbox_corner_size_no_variance_kernel(const int batch_num, \
        const float* loc_data, const float* prior_data, const float* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, float* bbox_data) {

    LCHECK_EQ(share_location, true, "decode boxes without share_location is unimplemented");

    int cnt = num_priors / 4;
    //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
    //! bbx

    int len_batch = num_priors * 4;

    for (int n = 0; n < batch_num; ++n) {

        const float *ptr_loc_batch = loc_data + n * len_batch;
        float *ptr_bbox_batch = bbox_data + n * len_batch;

#pragma omp parallel for
        for (int i = 0; i < cnt; ++i) {
            int idx = i * 16;

            const float* ptr_loc = ptr_loc_batch + idx;
            const float* ptr_prior = prior_data + idx;
            const float* ptr_var = variance + idx;
            float* ptr_bbox = ptr_bbox_batch + idx;

            float32x4x4_t vprior = vld4q_f32(ptr_prior);
            float32x4x4_t vloc = vld4q_f32(ptr_loc);

            float32x4_t vprior_width = vsubq_f32(vprior.val[2], vprior.val[0]);
            float32x4_t vprior_height = vsubq_f32(vprior.val[3], vprior.val[1]);

            float32x4x4_t vbbx;
            vbbx.val[0] = vmulq_f32(vloc.val[0], vprior_width);
            vbbx.val[1] = vmulq_f32(vloc.val[1], vprior_height);
            vbbx.val[2] = vmulq_f32(vloc.val[2], vprior_width);
            vbbx.val[3] = vmulq_f32(vloc.val[3], vprior_height);

            vloc = vld4q_f32(ptr_var);
            vbbx.val[0] = vmulq_f32(vbbx.val[0], vloc.val[0]);
            vbbx.val[1] = vmulq_f32(vbbx.val[1], vloc.val[1]);
            vbbx.val[2] = vmulq_f32(vbbx.val[2], vloc.val[2]);
            vbbx.val[3] = vmulq_f32(vbbx.val[3], vloc.val[3]);

            vbbx.val[0] = vaddq_f32(vprior.val[0], vbbx.val[0]);
            vbbx.val[1] = vaddq_f32(vprior.val[1], vbbx.val[1]);
            vbbx.val[2] = vaddq_f32(vprior.val[2], vbbx.val[2]);
            vbbx.val[3] = vaddq_f32(vprior.val[3], vbbx.val[3]);

            vst4q_f32(ptr_bbox, vbbx);
        }
#pragma omp parallel for
        for (int i = cnt * 4; i < num_priors; i++) {
            int idx = i * 4;
            float p_xmin = prior_data[idx];
            float p_ymin = prior_data[idx + 1];
            float p_xmax = prior_data[idx + 2];
            float p_ymax = prior_data[idx + 3];
            float prior_width = p_xmax - p_xmin;
            float prior_height = p_ymax - p_ymin;

            ptr_bbox_batch[idx] =
                    p_xmin + ptr_loc_batch[idx] * variance[idx] * prior_width;
            ptr_bbox_batch[idx + 1] =
                    p_ymin + ptr_loc_batch[idx + 1] * variance[idx + 1] * prior_height;
            ptr_bbox_batch[idx + 2] =
                    p_xmax + ptr_loc_batch[idx + 2] * variance[idx + 2] * prior_width;
            ptr_bbox_batch[idx + 3] =
                    p_ymax + ptr_loc_batch[idx + 3] * variance[idx + 3] * prior_height;
        }
    }
}

void decode_bboxes(const int batch_num, const float* loc_data, const float* prior_data, \
                     const CodeType code_type, const bool variance_encoded_in_target,\
                     const int num_priors, const bool share_location, \
                     const int num_loc_classes, const int background_label_id, \
                     float* bbox_data) {
    const float* variance_data = prior_data + 4 * num_priors;
    if (code_type == CORNER) {
        if (variance_encoded_in_target) {
            decode_bbox_corner_variance_kernel(batch_num, \
                loc_data, prior_data, variance_data, \
                num_priors, share_location, num_loc_classes, \
                background_label_id, bbox_data);
        } else {
            decode_bbox_corner_no_variance_kernel(batch_num, \
                loc_data, prior_data, variance_data, \
                num_priors, share_location, num_loc_classes, \
                background_label_id, bbox_data);
        }
    } else if (code_type == CENTER_SIZE) {
        if (variance_encoded_in_target) {
            decode_bbox_center_variance_kernel(batch_num, \
                loc_data, prior_data, variance_data, \
                num_priors, share_location, num_loc_classes, \
                background_label_id, bbox_data);
        } else {
            decode_bbox_center_no_variance_kernel(batch_num, \
                loc_data, prior_data, variance_data, \
                num_priors, share_location, num_loc_classes, \
                background_label_id, bbox_data);
        }
    } else if (code_type == CORNER_SIZE) {
        if (variance_encoded_in_target) {
            decode_bbox_corner_size_variance_kernel(batch_num, \
                loc_data, prior_data, variance_data, \
                num_priors, share_location, num_loc_classes, \
                background_label_id, bbox_data);
        } else {
            decode_bbox_corner_size_no_variance_kernel(batch_num, \
                loc_data, prior_data, variance_data, \
                num_priors, share_location, num_loc_classes, \
                background_label_id, bbox_data);
        }
    }
}

template <typename dtype>
static bool sort_score_pair_descend(const std::pair<float, dtype>& pair1, \
                                    const std::pair<float, dtype>& pair2) {
    return pair1.first > pair2.first;
}

void get_max_score_index(const float* scores, int num, float threshold, \
                         int top_k, std::vector<std::pair<float, int> >* score_index_vec) {
    //! Generate index score pairs.
    for (int i = 0; i < num; ++i) {
        if (scores[i] > threshold) {
            score_index_vec->push_back(std::make_pair(scores[i], i));
        }
    }

    //! Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec->begin(), score_index_vec->end(), \
                     sort_score_pair_descend<int>);

    //! Keep top_k scores if needed.
    if (top_k > -1 && top_k < score_index_vec->size()) {
        score_index_vec->resize(top_k);
    }
}

float bbox_size(const float* bbox, bool normalized = true) {
    if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0.f;
    } else {
        const float width = bbox[2] - bbox[0];
        const float height = bbox[3] - bbox[1];

        if (normalized) {
            return width * height;
        } else {
            // If bbox is not within range [0, 1].
            return (width + 1) * (height + 1);
        }
    }
}

float jaccard_overlap(const float* bbox1, const float* bbox2) {
    if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
        bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
        return 0.f;
    } else {
        const float inter_xmin = std::max(bbox1[0], bbox2[0]);
        const float inter_ymin = std::max(bbox1[1], bbox2[1]);
        const float inter_xmax = std::min(bbox1[2], bbox2[2]);
        const float inter_ymax = std::min(bbox1[3], bbox2[3]);

        const float inter_width = inter_xmax - inter_xmin;
        const float inter_height = inter_ymax - inter_ymin;
        const float inter_size = inter_width * inter_height;

        const float bbox1_size = bbox_size(bbox1);
        const float bbox2_size = bbox_size(bbox2);

        return inter_size / (bbox1_size + bbox2_size - inter_size);
    }
}

void apply_nms_fast(const float* bboxes, const float* scores, int num,
                    float score_threshold, float nms_threshold,
                    float eta, int top_k, std::vector<int>* indices) {
    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int>> score_index_vec;
    get_max_score_index(scores, num, score_threshold, top_k, &score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices->clear();

    while (score_index_vec.size() != 0) {
        const int idx = score_index_vec.front().second;
        bool keep = true;

        for (int k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = jaccard_overlap(bboxes + idx * 4, bboxes + kept_idx * 4);
                keep = overlap <= adaptive_threshold;
            } else {
                break;
            }
        }

        if (keep) {
            indices->push_back(idx);
        }

        score_index_vec.erase(score_index_vec.begin());

        if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
        }
    }
}

void nms_detect(const float* bbox_cpu_data, const float* conf_cpu_data, std::vector<float>& result, \
                int batch_num, int class_num, int num_priors, int background_id, \
                int keep_topk, int nms_topk, float conf_thresh, float nms_thresh, \
                float nms_eta, bool share_location) {

    int num_kept = 0;
    std::vector<std::map<int, std::vector<int>>> all_indices;

    for (int i = 0; i < batch_num; ++i) {
        std::map<int, std::vector<int>> indices;
        int num_det = 0;
        const int conf_idx = i * class_num * num_priors;
        int bbox_idx;

        if (share_location) {
            bbox_idx = i * num_priors * 4;
        } else {
            bbox_idx = conf_idx * 4;
        }

        for (int c = 0; c < class_num; ++c) {
            if (c == background_id) {
                // Ignore background class.
                continue;
            }

            const float* cur_conf_data = conf_cpu_data + conf_idx + c * num_priors;
            const float* cur_bbox_data = bbox_cpu_data + bbox_idx;

            if (!share_location) {
                cur_bbox_data += c * num_priors * 4;
            }

            apply_nms_fast(cur_bbox_data, cur_conf_data, num_priors, \
                           conf_thresh, nms_thresh, nms_eta, nms_topk, &(indices[c]));
            num_det += indices[c].size();
        }

        if (keep_topk > -1 && num_det > keep_topk) {
            std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;

            for (auto it = indices.begin(); it != indices.end(); ++it) {
                int label = it->first;
                const std::vector<int>& label_indices = it->second;

                for (int j = 0; j < label_indices.size(); ++j) {
                    int idx = label_indices[j];
                    float score = conf_cpu_data[conf_idx + label * num_priors + idx];
                    score_index_pairs.push_back(std::make_pair(score, std::make_pair(label, idx)));
                }
            }

            // Keep top k results per image.
            std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                             sort_score_pair_descend<std::pair<int, int>>);
            score_index_pairs.resize(keep_topk);
            // Store the new indices.
            std::map<int, std::vector<int>> new_indices;

            for (int j = 0; j < score_index_pairs.size(); ++j) {
                int label = score_index_pairs[j].second.first;
                int idx = score_index_pairs[j].second.second;
                new_indices[label].push_back(idx);
            }

            all_indices.push_back(new_indices);
            num_kept += keep_topk;
        } else {
            all_indices.push_back(indices);
            num_kept += num_det;
        }
    }

    if (num_kept == 0) {
        result.clear();
        return;
    } else {
        result.resize(num_kept * 7);
    }

    int count = 0;

    for (int i = 0; i < batch_num; ++i) {
        const int conf_idx = i * class_num * num_priors;
        int bbox_idx;

        if (share_location) {
            bbox_idx = i * num_priors * 4;
        } else {
            bbox_idx = conf_idx * 4;
        }

        for (auto it = all_indices[i].begin(); it != all_indices[i].end(); ++it) {
            int label = it->first;
            std::vector<int>& indices = it->second;
            const float* cur_conf_data =
                    conf_cpu_data + conf_idx + label * num_priors;
            const float* cur_bbox_data = bbox_cpu_data + bbox_idx;

            if (!share_location) {
                cur_bbox_data += label * num_priors * 4;
            }

            for (int j = 0; j < indices.size(); ++j) {
                int idx = indices[j];
                result[count * 7] = i;
                result[count * 7 + 1] = label;
                result[count * 7 + 2] = cur_conf_data[idx];

                for (int k = 0; k < 4; ++k) {
                    result[count * 7 + 3 + k] = cur_bbox_data[idx * 4 + k];
                }

                ++count;
            }
        }
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

SaberDetectionOutput::SaberDetectionOutput(bool share_loc,
                                           bool variance_encode,
                                           int class_num,
                                           int background_id,
                                           int keep_topk,
                                           CodeType type,
                                           float conf_thresh,
                                           int nms_topk,
                                           float nms_thresh,
                                           float nms_eta) {
    LITE_CHECK(load_param(share_loc, variance_encode, class_num, background_id, \
        keep_topk, type, conf_thresh, nms_topk, nms_thresh, nms_eta));
}

SaberStatus SaberDetectionOutput::load_param(bool share_loc,
                                             bool variance_encode,
                                             int class_num,
                                             int background_id,
                                             int keep_topk,
                                             CodeType type,
                                             float conf_thresh,
                                             int nms_topk,
                                             float nms_thresh,
                                             float nms_eta) {
    _share_loacation = share_loc;
    _variance_encode_in_target = variance_encode;
    _class_num = class_num;
    _background_id = background_id;
    _keep_top_k = keep_topk;
    _type = type;
    _conf_thresh = conf_thresh;
    _nms_top_k = nms_topk;
    _nms_thresh = nms_thresh;
    _nms_eta = nms_eta;
}

SaberStatus SaberDetectionOutput::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                       std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    //! output tensor's dims = 2
    Shape shape_out;
    shape_out.resize(2);
    //CHECK_EQ(shape_out.dims(), 4) << "only support 4d layout";
    shape_out[0] = inputs[0]->num() * _keep_top_k;
    shape_out[1] = 7;

    return outputs[0]->set_shape(shape_out);
}

SaberStatus SaberDetectionOutput::init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
                      std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) {
    _ctx = ctx;

//! inputs[0]: location map, dims = 2 {N, boxes * 4}
//! inputs[1]: confidence map, dims = 2 {N, boxes * classes}
//! inputs[2]: prior boxes, dims = 3 {1, 2, boxes * 4(xmin, ymin, xmax, ymax)}
    int size_loc = inputs[0]->valid_size();
    int size_conf = inputs[1]->valid_size();
    int size_prior = inputs[2]->valid_size();

    Shape sh_loc = inputs[0]->valid_shape();
    Shape sh_conf = inputs[1]->valid_shape();
//! shape {1, 2, boxes * 4(xmin, ymin, xmax, ymax)}, boxes = size / 2 / 4
//! the priors is in the last dim

    int num = inputs[0]->num();
    _num_priors = size_prior / 8;

    if (_class_num == 0) {
        _class_num = size_conf / (num * _num_priors);
    }
    if (_share_loacation) {
        _num_loc_classes = 1;
    } else {
        _num_loc_classes = _class_num;
        _bbox_permute.reshape(sh_loc);
    }

    _bbox_preds.reshape(sh_loc);
    _conf_permute.reshape(sh_conf);

    return SaberSuccess;
}


//template <>
SaberStatus SaberDetectionOutput::dispatch(
        const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
        std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) {

    Tensor<CPU, AK_FLOAT>* t_loc = inputs[0];
    Tensor<CPU, AK_FLOAT>* t_conf = inputs[1];
    Tensor<CPU, AK_FLOAT>* t_prior = inputs[2];

    const int num = t_loc->num();

    const float* loc_data = t_loc->data();
    const float* prior_data = t_prior->data();
    const float* conf_data = t_conf->data();

    float* bbox_data = _bbox_preds.mutable_data();

    if (!_share_loacation) {
        return SaberUnImplError;
    }

    //! Decode predictions.
    //! Retrieve all decoded location predictions.
    decode_bboxes(num, loc_data, prior_data, _type, _variance_encode_in_target, \
        _num_priors, _share_loacation, _num_loc_classes, \
        _background_id, bbox_data);

    //! Retrieve all confidences, permute to classes * boxes_size
    float* conf_permute_data = _conf_permute.mutable_data();
    permute_conf(conf_data, num, _num_priors, _class_num, conf_permute_data);

    std::vector<float> result;

    nms_detect(bbox_data, conf_permute_data, result, num, _class_num, _num_priors, _background_id, \
        _keep_top_k, _nms_top_k, _conf_thresh, _nms_thresh, _nms_eta, _share_loacation);

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

} //namespace lite

} //namespace saber

} //namespace anakin

#endif
