
#include "saber/utils.h"
#include "saber/core/common.h"
#include "saber/core/tensor.h"

#include <vector>
#include "thrust/functional.h"
#include "thrust/sort.h"
namespace anakin {

namespace saber {
// caffe util_nms.cu
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const *const a, float const *const b) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    const int row_size =
            min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
            min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 5];
    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 5 + 0] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
        block_boxes[threadIdx.x * 5 + 1] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
        block_boxes[threadIdx.x * 5 + 2] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
        block_boxes[threadIdx.x * 5 + 3] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
        block_boxes[threadIdx.x * 5 + 4] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float *cur_box = dev_boxes + cur_box_idx * 5;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

const std::vector<bool> nms_voting0(const float *boxes_dev, unsigned long long *mask_dev,
                               int boxes_num, float nms_overlap_thresh,
                               const int max_candidates,
                               const int top_n) {

    if ((max_candidates > 0) && (boxes_num > max_candidates)) {
        boxes_num = max_candidates;
    }
//    float *boxes_dev = NULL;
//    unsigned long long *mask_dev = NULL;

    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
//    CUDA_CHECK(cudaMalloc(&mask_dev,
//                          boxes_num * col_blocks * sizeof(unsigned long long)));

    dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
                DIVUP(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);
    nms_kernel << < blocks, threads >> > (boxes_num,
            nms_overlap_thresh,
            boxes_dev,
            mask_dev);

    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);

    CUDA_CHECK(cudaMemcpy(&mask_host[0],
                          mask_dev,
                          sizeof(unsigned long long) * boxes_num * col_blocks,
                          cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
    std::vector<bool> mask(boxes_num, false);
    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            ++num_to_keep;
            mask[i] = true;
            unsigned long long *p = &mask_host[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }
            if ((top_n > 0) && (num_to_keep >= top_n)) {
                break;
            }
        }
    }

//    CUDA_CHECK(cudaFree(mask_dev));
    return mask;
}

template <typename Dtype>
__global__ void rpn_cmp_conf_bbox_kernel(
        const int threads, const int num_anchors,
        const int map_height, const int map_width,
        const Dtype input_height, const Dtype input_width,
        const Dtype heat_map_a, const Dtype heat_map_b,
        const Dtype allow_border, const Dtype allow_border_ratio,
        const Dtype min_size_w, const Dtype min_size_h,
        const bool min_size_mode_and_else_or, const Dtype thr_obj,
        const Dtype bsz01, const bool do_bbox_norm,
        const Dtype mean0, const Dtype mean1,
        const Dtype mean2, const Dtype mean3,
        const Dtype std0, const Dtype std1,
        const Dtype std2, const Dtype std3,
        const bool refine_out_of_map_bbox, const Dtype* anc_data,
        const Dtype* prob_data, const Dtype* tgt_data,
        Dtype* conf_data, Dtype* bbox_data) {
    int map_size = map_height * map_width;
    CUDA_KERNEL_LOOP(index, threads) {
        int w = index % map_width;
        int h = (index / map_width) % map_height;
        int a = index / map_size;
        int off = h * map_width + w;

        Dtype  score = prob_data[(num_anchors + a) * map_size + off];
        if (score < thr_obj) {
            conf_data[index] = 0.0;
            continue;
        }

        int ax4 = a * 4;
        Dtype anchor_ctr_x = anc_data[ax4];
        Dtype anchor_ctr_y = anc_data[ax4 + 1];
        Dtype anchor_width = anc_data[ax4 + 2];
        Dtype anchor_height = anc_data[ax4 + 3];

        Dtype input_ctr_x = w * heat_map_a + heat_map_b + anchor_ctr_x;
        Dtype input_ctr_y = h * heat_map_a + heat_map_b + anchor_ctr_y;

        if (allow_border >= Dtype(0.0)
            || allow_border_ratio >= Dtype(0.0)) {
            Dtype x1 = input_ctr_x - 0.5 * (anchor_width - bsz01);
            Dtype y1 = input_ctr_y - 0.5 * (anchor_height - bsz01);
            Dtype x2 = x1 + anchor_width - bsz01;
            Dtype y2 = y1 + anchor_height - bsz01;
            if (allow_border >= Dtype(0.0) && (
                    x1 < -allow_border || y1 < -allow_border
                    || x2 > input_width - 1 + allow_border ||
                    y2 > input_height - 1 + allow_border)) {
                conf_data[index] = 0.0;
                continue;
            } else if (allow_border_ratio >= Dtype(0.0)) {
                Dtype x11 = max(Dtype(0), x1);
                Dtype y11 = max(Dtype(0), y1);
                Dtype x22 = min(input_width - 1, x2);
                Dtype y22 = min(input_height - 1, y2);
                if ((y22 - y11 + bsz01) * (x22 - x11 + bsz01)
                    / ((y2 - y1 + bsz01) * (x2 - x1 + bsz01))
                    < (1.0 - allow_border_ratio)) {
                    conf_data[index] = 0.0;
                    continue;
                }
            }
        }

        Dtype tg0 = tgt_data[ax4 * map_size + off];
        Dtype tg1 = tgt_data[(ax4 + 1) * map_size + off];
        Dtype tg2 = tgt_data[(ax4 + 2) * map_size + off];
        Dtype tg3 = tgt_data[(ax4 + 3) * map_size + off];
        if (do_bbox_norm) {
            tg0 = tg0 * std0 + mean0;
            tg1 = tg1 * std1 + mean1;
            tg2 = tg2 * std2 + mean2;
            tg3 = tg3 * std3 + mean3;
        }
        Dtype tw = anchor_width * exp(tg2);
        Dtype th = anchor_height * exp(tg3);

        Dtype ctx = tg0 * anchor_width + input_ctr_x;
        Dtype cty = tg1 * anchor_height + input_ctr_y;
        Dtype ltx = ctx - 0.5 * (tw - bsz01);
        Dtype lty = cty - 0.5 * (th - bsz01);
        Dtype rbx = ltx + tw - bsz01;
        Dtype rby = lty + th - bsz01;

        if (refine_out_of_map_bbox) {
            ltx = min(max(ltx, Dtype(0.0)), input_width -1);
            lty = min(max(lty, Dtype(0.0)), input_height -1);
            rbx = min(max(rbx, Dtype(0.0)), input_width -1);
            rby = min(max(rby, Dtype(0.0)), input_height -1);
        }

        if (min_size_mode_and_else_or) {
            if ((rbx - ltx + bsz01) < min_size_w
                || (rby - lty + bsz01) < min_size_h) {
                conf_data[index] = 0.0;
                continue;
            }
        } else {
            if ((rbx - ltx + bsz01) < min_size_w
                && (rby - lty + bsz01) < min_size_h) {
                conf_data[index] = 0.0;
                continue;
            }
        }

        conf_data[index] = score;
        bbox_data[index * 4] = ltx;
        bbox_data[index * 4 + 1] = lty;
        bbox_data[index * 4 + 2] = rbx;
        bbox_data[index * 4 + 3] = rby;

    }
}

template <typename Dtype>
void rpn_cmp_conf_bbox_gpu(const int num_anchors,
                           const int map_height, const int map_width,
                           const Dtype input_height, const Dtype input_width,
                           const Dtype heat_map_a, const Dtype heat_map_b,
                           const Dtype allow_border, const Dtype allow_border_ratio,
                           const Dtype min_size_w, const Dtype min_size_h,
                           const bool min_size_mode_and_else_or, const Dtype thr_obj,
                           const Dtype bsz01, const bool do_bbox_norm,
                           const Dtype mean0, const Dtype mean1,
                           const Dtype mean2, const Dtype mean3,
                           const Dtype std0, const Dtype std1,
                           const Dtype std2, const Dtype std3,
                           const bool refine_out_of_map_bbox, const Dtype* anc_data,
                           const Dtype* prob_data, const Dtype* tgt_data,
                           Dtype* conf_data, Dtype* bbox_data, Context<NV> *ctx) {
#ifdef ENABLE_DEBUG
#undef CUDA_NUM_THREADS
#define CUDA_NUM_THREADS 256
#endif
    int threads = num_anchors * map_height * map_width;
    rpn_cmp_conf_bbox_kernel<Dtype><<<CUDA_GET_BLOCKS(threads),
            CUDA_NUM_THREADS, 0, ctx->get_compute_stream()>>>(threads, num_anchors,
                    map_height, map_width,
                    input_height, input_width,
                    heat_map_a, heat_map_b,
                    allow_border, allow_border_ratio,
                    min_size_w, min_size_h,
                    min_size_mode_and_else_or, thr_obj,
                    bsz01, do_bbox_norm,
                    mean0, mean1, mean2, mean3,
                    std0, std1, std2, std3,
                    refine_out_of_map_bbox, anc_data,
                    prob_data, tgt_data,
                    conf_data, bbox_data);
    CUDA_POST_KERNEL_CHECK;
}
template void rpn_cmp_conf_bbox_gpu(const int num_anchors,
                                    const int map_height, const int map_width,
                                    const float input_height, const float input_width,
                                    const float heat_map_a, const float heat_map_b,
                                    const float allow_border, const float allow_border_ratio,
                                    const float min_size_w, const float min_size_h,
                                    const bool min_size_mode_and_else_or, const float thr_obj,
                                    const float bsz01, const bool do_bbox_norm,
                                    const float mean0, const float mean1,
                                    const float mean2, const float mean3,
                                    const float std0, const float std1,
                                    const float std2, const float std3,
                                    const bool refine_out_of_map_bbox, const float* anc_data,
                                    const float* prob_data, const float* tgt_data,
                                    float* conf_data, float* bbox_data, Context<NV> *ctx);

// rcnn
template <typename Dtype>
__global__ void rcnn_cmp_conf_bbox_kernel(const int num_rois,
                                          const Dtype input_height, const Dtype input_width,
                                          const Dtype allow_border, const Dtype allow_border_ratio,
                                          const Dtype min_size_w, const Dtype min_size_h,
                                          const bool min_size_mode_and_else_or, const Dtype thr_obj,
                                          const Dtype bsz01, const bool do_bbox_norm,
                                          const Dtype mean0, const Dtype mean1,
                                          const Dtype mean2, const Dtype mean3,
                                          const Dtype std0, const Dtype std1,
                                          const Dtype std2, const Dtype std3,
                                          const bool refine_out_of_map_bbox, const bool regress_agnostic,
                                          const int num_class, const Dtype* thr_cls,
                                          const Dtype* rois_data, const Dtype* prob_data,
                                          const Dtype* tgt_data, Dtype* conf_data,
                                          Dtype* bbox_data) {
    int probs_dim = num_class + 1;
    int cords_dim = (regress_agnostic ? 2 : (num_class + 1)) * 4;
    CUDA_KERNEL_LOOP(index, num_rois) {
        const Dtype* probs = prob_data + index * probs_dim;
        const Dtype* cords = tgt_data + index * cords_dim;
        const Dtype* rois = rois_data + index * 5;

        if ((1.0 - probs[0]) < thr_obj) {
            conf_data[index] = 0.0;
            continue;
        }

        if (int(rois[0]) == -1) {
            conf_data[index] = 0.0;
            continue;
        }

        Dtype score_max = -10e6;
        int cls_max = -1;
        for (int c = 0; c < num_class; c++) {
            Dtype score_c = probs[c + 1] - thr_cls[c];
            if (score_c > score_max) {
                score_max = score_c;
                cls_max = c;
            }
        }
        if (score_max < 0) {
            conf_data[index] = 0.0;
            continue;
        }

        if (allow_border >= 0.0
            || allow_border_ratio >= 0.0) {
            Dtype x1 = rois[1];
            Dtype y1 = rois[2];
            Dtype x2 = rois[3];
            Dtype y2 = rois[4];
            if (allow_border >= 0.0 && (
                    x1 < -allow_border || y1 < -allow_border
                    || x2 > input_width - 1 + allow_border ||
                    y2 > input_height - 1 + allow_border )) {
                conf_data[index] = 0.0;
                continue;
            } else if (allow_border_ratio >= 0.0) {
                Dtype x11 = max(Dtype(0.0), x1);
                Dtype y11 = max(Dtype(0.0), y1);
                Dtype x22 = min(input_width - 1, x2);
                Dtype y22 = min(input_height - 1, y2);
                if ((y22 - y11 + bsz01) * (x22 - x11 + bsz01)
                    / ((y2 - y1 + bsz01) * (x2 - x1 +bsz01))
                    < (1.0 - allow_border_ratio)) {
                    conf_data[index] = 0.0;
                    continue;
                }
            }
        }

        Dtype rois_w = rois[3] - rois[1] + bsz01;
        Dtype rois_h = rois[4] - rois[2] + bsz01;
        Dtype rois_ctr_x = rois[1] + 0.5 * (rois_w - bsz01);
        Dtype rois_ctr_y = rois[2] + 0.5 * (rois_h - bsz01);

        int cdst = regress_agnostic ? 4 : ((cls_max + 1) * 4);
        Dtype tg0 = cords[cdst];
        Dtype tg1 = cords[cdst + 1];
        Dtype tg2 = cords[cdst + 2];
        Dtype tg3 = cords[cdst + 3];
        if (do_bbox_norm) {
            tg0 = tg0 * std0 + mean0;
            tg1 = tg1 * std1 + mean1;
            tg2 = tg2 * std2 + mean2;
            tg3 = tg3 * std3 + mean3;
        }
        Dtype tw = rois_w * exp(tg2);
        Dtype th = rois_h * exp(tg3);

        Dtype ctx = tg0 * rois_w + rois_ctr_x;
        Dtype cty = tg1 * rois_h + rois_ctr_y;
        Dtype ltx = ctx - 0.5 * (tw - bsz01);
        Dtype lty = cty - 0.5 * (th - bsz01);
        Dtype rbx = ltx + tw - bsz01;
        Dtype rby = lty + th - bsz01;

        if (refine_out_of_map_bbox) {
            ltx = min(max(ltx, Dtype(0.0)), input_width -1);
            lty = min(max(lty, Dtype(0.0)), input_height -1);
            rbx = min(max(rbx, Dtype(0.0)), input_width -1);
            rby = min(max(rby, Dtype(0.0)), input_height -1);
        }

        if (min_size_mode_and_else_or) {
            if ((rbx - ltx + bsz01) < min_size_w
                || (rby - lty + bsz01) < min_size_h) {
                conf_data[index] = 0.0;
                continue;
            }
        } else {
            if ((rbx - ltx + bsz01) < min_size_w
                && (rby - lty + bsz01) < min_size_h) {
                conf_data[index] = 0.0;
                continue;
            }
        }

        conf_data[index] = probs[cls_max + 1];
        bbox_data[index * 4] = ltx;
        bbox_data[index * 4 + 1] = lty;
        bbox_data[index * 4 + 2] = rbx;
        bbox_data[index * 4 + 3] = rby;
    }
}

template <typename Dtype>
void rcnn_cmp_conf_bbox_gpu(const int num_rois,
                            const Dtype input_height, const Dtype input_width,
                            const Dtype allow_border, const Dtype allow_border_ratio,
                            const Dtype min_size_w, const Dtype min_size_h,
                            const bool min_size_mode_and_else_or, const Dtype thr_obj,
                            const Dtype bsz01, const bool do_bbox_norm,
                            const Dtype mean0, const Dtype mean1,
                            const Dtype mean2, const Dtype mean3,
                            const Dtype std0, const Dtype std1,
                            const Dtype std2, const Dtype std3,
                            const bool refine_out_of_map_bbox, const bool regress_agnostic,
                            const int num_class, const Dtype* thr_cls,
                            const Dtype* rois_data, const Dtype* prob_data,
                            const Dtype* tgt_data, Dtype* conf_data,
                            Dtype* bbox_data, Context<NV> *ctx) {
    int threads = num_rois;
    rcnn_cmp_conf_bbox_kernel<Dtype><<<CUDA_GET_BLOCKS(threads),
            CUDA_NUM_THREADS, 0, ctx->get_compute_stream()>>>(num_rois,
                    input_height, input_width,
                    allow_border, allow_border_ratio,
                    min_size_w, min_size_h,
                    min_size_mode_and_else_or, thr_obj,
                    bsz01, do_bbox_norm,
                    mean0, mean1,
                    mean2, mean3,
                    std0, std1,
                    std2, std3,
                    refine_out_of_map_bbox, regress_agnostic,
                    num_class, thr_cls,
                    rois_data, prob_data,
                    tgt_data, conf_data,
                    bbox_data);
    CUDA_POST_KERNEL_CHECK;
}

template void rcnn_cmp_conf_bbox_gpu(const int num_rois,
        const float input_height, const float input_width,
        const float allow_border, const float allow_border_ratio,
        const float min_size_w, const float min_size_h,
        const bool min_size_mode_and_else_or, const float thr_obj,
        const float bsz01, const bool do_bbox_norm,
        const float mean0, const float mean1,
        const float mean2, const float mean3,
        const float std0, const float std1,
        const float std2, const float std3,
        const bool refine_out_of_map_bbox, const bool regress_agnostic,
        const int num_class, const float* thr_cls,
        const float* rois_data, const float* prob_data,
        const float* tgt_data, float* conf_data,
        float* bbox_data, Context<NV> *ctx);

// nms, copy and modify some cuda codes form yolo
template <typename Dtype>
__host__ __device__ Dtype bbox_size_gpu(const Dtype *bbox, const Dtype bsz01) {
    if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
        return Dtype(0.);
    } else {
        return (bbox[2] - bbox[0] + bsz01) * (bbox[3] - bbox[1] + bsz01);
    }
}

template <typename Dtype>
__host__ __device__ Dtype jaccard_overlap_gpu(const Dtype *bbox1,
                                              const Dtype *bbox2, const Dtype bsz01) {
    if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
        bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
        return Dtype(0.);
    } else {
        const Dtype inter_xmin = max(bbox1[0], bbox2[0]);
        const Dtype inter_ymin = max(bbox1[1], bbox2[1]);
        const Dtype inter_xmax = min(bbox1[2], bbox2[2]);
        const Dtype inter_ymax = min(bbox1[3], bbox2[3]);

        const Dtype inter_width = inter_xmax - inter_xmin + bsz01;
        const Dtype inter_height = inter_ymax - inter_ymin + bsz01;
        const Dtype inter_size = inter_width * inter_height;

        const Dtype bbox1_size = bbox_size_gpu(bbox1, bsz01);
        const Dtype bbox2_size = bbox_size_gpu(bbox2, bsz01);

        return inter_size / (bbox1_size + bbox2_size - inter_size);
    }
}

template <typename Dtype>
__global__ void compute_overlapped_by_idx_kernel(
        const int nthreads, const Dtype *bbox_data, const int bbox_step,
        const Dtype overlap_threshold, const int *idx, const int num_idx,
        const Dtype bsz01, bool *overlapped_data) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < (nthreads); index += blockDim.x * gridDim.x) {
        const int j = index % num_idx;
        const int i = index / num_idx;
        if (i == j) {
            // Ignore same bbox.
            return;
        }
        // Compute overlap between i-th bbox and j-th bbox.
        const int start_loc_i = idx[i] * bbox_step;
        const int start_loc_j = idx[j] * bbox_step;
        const Dtype overlap = jaccard_overlap_gpu(bbox_data + start_loc_i,
                                                  bbox_data + start_loc_j,
                                                  bsz01);
        overlapped_data[index] = overlap > overlap_threshold;
    }
}

//template <typename Dtype>
//void compute_overlapped_by_idx_gpu(
//        const int nthreads, const Dtype *bbox_data, const int bbox_step,
//        const Dtype overlap_threshold, const int *idx, const int num_idx,
//        const Dtype bsz01, bool *overlapped_data) {
//    // NOLINT_NEXT_LINE(whitespace/operators)
//    const int thread_size = 256;
//    int block_size = (nthreads + thread_size - 1) / thread_size;
//    compute_overlapped_by_idx_kernel << < block_size, thread_size >> > (
//            nthreads, bbox_data, bbox_step, overlap_threshold, idx, num_idx,
//                    bsz01, overlapped_data);
//}

template <typename Dtype>
void compute_overlapped_by_idx_gpu(
        const int nthreads, const Dtype *bbox_data, const int bbox_step,
        const Dtype overlap_threshold, const int *idx, const int num_idx,
        const Dtype bsz01, bool *overlapped_data, const cudaStream_t &stream) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    const int thread_size = 256;
    int block_size = (nthreads + thread_size - 1) / thread_size;
//    printf("thread_size = %d, block_size = %d\n", thread_size, block_size);
    compute_overlapped_by_idx_kernel << < block_size, thread_size, 0, stream >> > (
            nthreads, bbox_data, bbox_step, overlap_threshold, idx, num_idx,
                    bsz01, overlapped_data);
    cudaDeviceSynchronize();
}

// Do nms, modified by mingli.
void apply_nms(const bool *overlapped, const int num, const int top_k,
               const std::vector<int> &idxes, std::vector<int> *indices,
               const int nmsed_num = 0, const int nmsed_loc = 0) {
    std::vector<bool> mask(num, false);
    if (nmsed_num > 0) {
        int k_x_num_add_nmsed_num = nmsed_num;
        for (int k = 0; k < nmsed_num; k++) {
            int k_x_num_add_p = k_x_num_add_nmsed_num;
            for (int p = nmsed_num; p < num; p++) {
                if (overlapped[k_x_num_add_p++]) {
                    mask[p] = true;
                }
            }
            k_x_num_add_nmsed_num += num;
        }
    }
    int count = nmsed_num;
    int k_x_num = (nmsed_num -1) * num;
    for (int k = nmsed_num; k < num; k++) {
        k_x_num += num;
        if (mask[k]) {
            continue;
        } else {
            indices->push_back(idxes[nmsed_loc + k - nmsed_num]);
            if (++count >= top_k) {
                break;
            }
            int k_x_num_add_p = k_x_num + k + 1;
            for (int p = k + 1; p < num; p++) {
                if (overlapped[k_x_num_add_p++]) {
                    mask[p] = true;
                }
            }
        }
    }
}

template <typename Dtype, typename PGlue_nv>
void apply_nms_gpu(const Dtype *bbox_data, const Dtype *conf_data,
                   const int num_bboxes, const int bbox_step, const Dtype confidence_threshold,
                   const int max_candidate_n, const int top_k, const Dtype nms_threshold,
                   const Dtype bsz01, std::vector<int> *indices,
                   PGlue_nv *overlapped, PGlue_nv *idx_sm,
                   Context<NV> *ctx, std::vector<int> *idx_ptr,
                   const int conf_step, const int conf_idx,
                   const int nms_gpu_max_n_per_time) {
    indices->clear();
    std::vector<int> idx;
    std::vector<Dtype> confidences;
    if (idx_ptr == NULL) {
        if (conf_step == 1) {
            for (int i = 0; i < num_bboxes; ++i) {
                if (conf_data[i] > confidence_threshold) {
                    idx.push_back(i);
                    confidences.push_back(conf_data[i]);
                }
            }
        } else {
            int i_x_step_add_idx = conf_idx;
            for (int i = 0; i < num_bboxes; ++i) {
                if (conf_data[i_x_step_add_idx] > confidence_threshold) {
                    idx.push_back(i);
                    confidences.push_back(conf_data[i_x_step_add_idx]);
                }
                i_x_step_add_idx += conf_step;
            }
        }
    } else {
        if (conf_step == 1) {
            for (int k = 0; k < idx_ptr->size(); k++) {
                int i = (*idx_ptr)[k];
                if (conf_data[i] > confidence_threshold) {
                    idx.push_back(i);
                    confidences.push_back(conf_data[i]);
                }
            }
        } else {
            for (int k = 0; k < idx_ptr->size(); k++) {
                int i = (*idx_ptr)[k];
                int i_x_step_add_idx = i * conf_step + conf_idx;
                if (conf_data[i_x_step_add_idx] > confidence_threshold) {
                    idx.push_back(i);
                    confidences.push_back(conf_data[i_x_step_add_idx]);
                }
            }
        }
    }
    int num_remain = confidences.size();
    if (num_remain == 0) {
        return;
    }
    if (nms_threshold >= Dtype(1.0)) {
        for (int i = 0; i < idx.size(); i++) {
            indices->push_back(idx[i]);
        }
        return;
    }

    thrust::sort_by_key(&confidences[0], &confidences[0] + num_remain, &idx[0],
                        thrust::greater<Dtype>());
    if (max_candidate_n > -1 && max_candidate_n < num_remain) {
        num_remain = max_candidate_n;
    }

    int idx_loc = 0;
    int indices_size_pre = 0;
    while (idx_loc < num_remain && indices->size() < top_k) {
        int *idx_data = (int*)idx_sm->host_mutable_data(ctx);
        std::copy(indices->begin() + indices_size_pre,
                  indices->end(), idx_data + indices_size_pre);
        int idx_num_cur_time = min(int(nms_gpu_max_n_per_time - indices->size()),
                                   int(num_remain - idx_loc));
        std::copy(idx.begin() + idx_loc, idx.begin() + idx_loc + idx_num_cur_time,
                  idx_data + indices->size());
        int candidate_n_cur_time = indices->size() + idx_num_cur_time;
        int total_bboxes = candidate_n_cur_time * candidate_n_cur_time;
        bool *overlapped_data = (bool*)overlapped->device_mutable_data(ctx);
        compute_overlapped_by_idx_gpu(total_bboxes, bbox_data, bbox_step,
                                      nms_threshold, (const int*)idx_sm->device_data(ctx),
                                      candidate_n_cur_time, bsz01, overlapped_data, ctx->get_compute_stream());
        const bool *overlapped_results = (const bool*)overlapped->host_data(ctx);
        indices_size_pre = indices->size();
        apply_nms(overlapped_results, candidate_n_cur_time, top_k,
                  idx, indices, indices->size(), idx_loc);
        idx_loc += idx_num_cur_time;
    }
}
template void apply_nms_gpu(const float *bbox_data, const float *conf_data,
                            const int num_bboxes, const int bbox_step, const float confidence_threshold,
                            const int max_candidate_n, const int top_k, const float nms_threshold,
                            const float bsz01, std::vector<int> *indices,
                            PGlue<Tensor<NV>, Tensor<NVHX86> > *overlapped,
                            PGlue<Tensor<NV>, Tensor<NVHX86> > *idx_sm,
                            Context<NV> *ctx, std::vector<int> *idx_ptr,
                            const int conf_step, const int conf_idx, const int nms_gpu_max_n_per_time);

template <typename Dtype>
void GenGrdFt_cpu(unsigned int im_width,
                       unsigned int im_height, unsigned int blob_width,
                       unsigned int blob_height, Dtype std_height,
                       const std::vector<Dtype> & cam_params, Dtype* grd_ft,
                       Dtype read_width_scale, Dtype read_height_scale,
                       unsigned int read_height_offset, unsigned int valid_param_idx_st,
                       bool trans_cam_pitch_to_zero, bool normalize_grd_ft,
                       unsigned int normalize_grd_ft_dim) {

    CHECK_GT(im_width, 0);
    CHECK_GT(im_height, 0);
    CHECK_GE(blob_width, im_width);
    CHECK_GE(blob_height, im_height);
    CHECK_GT(read_width_scale, 0);
    CHECK_GT(read_height_scale, 0);
    CHECK_LE(valid_param_idx_st + 6, cam_params.size());

    Dtype cam_xpz = cam_params[valid_param_idx_st + 0];
    Dtype cam_xct = cam_params[valid_param_idx_st + 1];
    Dtype cam_ypz = cam_params[valid_param_idx_st + 2];
    Dtype cam_yct = cam_params[valid_param_idx_st + 3];
    Dtype cam_hgrd = cam_params[valid_param_idx_st + 4];
    Dtype cam_pitch = cam_params[valid_param_idx_st + 5];
    CHECK_GT(cam_xpz, 0);
    CHECK_GT(cam_ypz, 0);
    CHECK_GT(cam_hgrd, 0);

    Dtype min_py_grd = cam_yct + cam_ypz * tan(cam_pitch);
    Dtype min_r_grd = (min_py_grd - read_height_offset)
                      * read_height_scale;
    for (int r = 0; r < im_height; r++) {
        Dtype py_grd;
        Dtype z_grd, y_grd;
        Dtype z_std_h_upon_grd, y_std_h_upon_grd;
        Dtype py_std_h_upon_grd, r_std_h_upon_grd;
        if (r > min_r_grd) {
            py_grd = r / read_height_scale + read_height_offset;
            z_grd = cam_ypz * cam_hgrd
                    / (py_grd - cam_yct - cam_ypz * tan(cam_pitch));
            y_grd = cam_hgrd + z_grd * tan(cam_pitch);
            z_std_h_upon_grd = z_grd + std_height
                    * (trans_cam_pitch_to_zero?0.0:tan(cam_pitch));
            y_std_h_upon_grd = y_grd - std_height
                    * (trans_cam_pitch_to_zero?1.0:cos(cam_pitch));
            py_std_h_upon_grd = cam_ypz * y_std_h_upon_grd
                    / z_std_h_upon_grd + cam_yct;
            r_std_h_upon_grd = (py_std_h_upon_grd - read_height_offset)
                    * read_height_scale;
        }
        for (int c = 0; c < im_width; c++) {
            if (r <= min_r_grd) {
                grd_ft[r * blob_width + c] = Dtype(0.0);
            } else {
                Dtype px_grd = c / read_width_scale;
                Dtype x_grd = (px_grd - cam_xct) * z_grd / cam_xpz;
                Dtype x_std_h_upon_grd = x_grd;
                Dtype px_std_h_upon_grd = cam_xpz * x_std_h_upon_grd
                        / z_std_h_upon_grd + cam_xct;
                Dtype c_std_h_upon_grd = px_std_h_upon_grd
                        * read_width_scale;
                Dtype std_h_prj_scale =
                        sqrt((c_std_h_upon_grd - c) * (c_std_h_upon_grd - c)
                        + (r_std_h_upon_grd - r) * (r_std_h_upon_grd - r));
                if (!normalize_grd_ft) {
                    grd_ft[r * blob_width + c] = std_h_prj_scale;
                } else {
                    int norm_chl = std::min<int>(normalize_grd_ft_dim - 1,
                            std::max<int>(0, static_cast<int>(
                                    std::ceil(std::log(std_h_prj_scale) / std::log(2.0)))));
                    grd_ft[(norm_chl * blob_height + r) * blob_width + c] =
                            std_h_prj_scale / std::pow(2.0, norm_chl);
                }
            }
        }
    }
}

template void GenGrdFt_cpu(unsigned int im_width,
        unsigned int im_height, unsigned int blob_width,
        unsigned int blob_height, float std_height,
        const std::vector<float> & cam_params, float* grd_ft,
        float read_width_scale, float read_height_scale,
        unsigned int read_height_offset, unsigned int valid_param_idx_st,
        bool trans_cam_pitch_to_zero,bool normalize_grd_ft,
        unsigned int normalize_grd_ft_dim);

template <typename Dtype>
__global__ void GenGrdFt_kernel(unsigned int im_width, unsigned int blob_width,
        unsigned int blob_height, unsigned int n, Dtype std_height, Dtype cam_xpz,
        Dtype cam_xct, Dtype cam_ypz, Dtype cam_yct, Dtype cam_hgrd, Dtype cam_pitch,
        Dtype cam_tanh, Dtype cam_ypz_x_tanh, Dtype std_height_x_tanh, Dtype std_height_x_cos,
        Dtype cam_ypz_x_cam_hgrd, Dtype read_width_scale, Dtype read_height_scale,
        unsigned int read_height_offset, Dtype min_py_grd, Dtype min_r_grd,
        bool normalize_grd_ft, unsigned int normalize_grd_ft_dim, Dtype* grd_ft_gpu_data) {

    CUDA_KERNEL_LOOP(index, n) {
        int r = index / im_width;
        int c = index % im_width;
        if (r <= min_r_grd) {
            grd_ft_gpu_data[r * blob_width + c] = Dtype(0.0);
        } else {
            Dtype py_grd = r / read_height_scale + read_height_offset;
            Dtype z_grd = cam_ypz_x_cam_hgrd
                          / (py_grd - cam_yct - cam_ypz_x_tanh);
            Dtype y_grd = cam_hgrd + z_grd * cam_tanh;
            Dtype z_std_h_upon_grd = z_grd + std_height_x_tanh;
            Dtype y_std_h_upon_grd = y_grd - std_height_x_cos;
            Dtype py_std_h_upon_grd = cam_ypz * y_std_h_upon_grd
                    / z_std_h_upon_grd + cam_yct;
            Dtype r_std_h_upon_grd = (py_std_h_upon_grd - read_height_offset)
                    * read_height_scale;
            Dtype px_grd = c / read_width_scale;
            Dtype x_grd = (px_grd - cam_xct) * z_grd / cam_xpz;
            Dtype x_std_h_upon_grd = x_grd;
            Dtype px_std_h_upon_grd = cam_xpz * x_std_h_upon_grd
                    / z_std_h_upon_grd + cam_xct;
            Dtype c_std_h_upon_grd = px_std_h_upon_grd * read_width_scale;
            Dtype std_h_prj_scale =
                    sqrt((c_std_h_upon_grd - c) * (c_std_h_upon_grd - c)
                         + (r_std_h_upon_grd - r) * (r_std_h_upon_grd - r));

            if (!normalize_grd_ft) {
                grd_ft_gpu_data[r * blob_width + c] = std_h_prj_scale;
            } else {
                int norm_chl = min(normalize_grd_ft_dim - 1, max(0,
                        int(ceil(log(std_h_prj_scale) / log(2.0)))));
                grd_ft_gpu_data[(norm_chl * blob_height + r) * blob_width + c] =
                        std_h_prj_scale / pow(2.0, norm_chl);
            }
        }
    }
}

template <typename Dtype>
void GenGrdFt_gpu(unsigned int im_width,
                       unsigned int im_height, unsigned int blob_width,
                       unsigned int blob_height, Dtype std_height,
                       const std::vector<Dtype> & cam_params, Dtype* grd_ft,
                       Dtype read_width_scale, Dtype read_height_scale,
                       unsigned int read_height_offset, unsigned int valid_param_idx_st,
                       bool trans_cam_pitch_to_zero, bool normalize_grd_ft,
                       unsigned int normalize_grd_ft_dim) {
    CHECK_GT(im_width, 0);
    CHECK_GT(im_height, 0);
    CHECK_GE(blob_width, im_width);
    CHECK_GE(blob_height, im_height);
    CHECK_GT(read_width_scale, 0);
    CHECK_GT(read_height_scale, 0);
    CHECK_LE(valid_param_idx_st + 6, cam_params.size());

    Dtype cam_xpz = cam_params[valid_param_idx_st + 0];
    Dtype cam_xct = cam_params[valid_param_idx_st + 1];
    Dtype cam_ypz = cam_params[valid_param_idx_st + 2];
    Dtype cam_yct = cam_params[valid_param_idx_st + 3];
    Dtype cam_hgrd = cam_params[valid_param_idx_st + 4];
    Dtype cam_pitch = cam_params[valid_param_idx_st + 5];
    CHECK_GT(cam_xpz, 0);
    CHECK_GT(cam_ypz, 0);
    CHECK_GT(cam_hgrd, 0);

    Dtype cam_tanh = tanh(cam_pitch);
    Dtype cam_ypz_x_tanh = cam_ypz * cam_tanh;
    Dtype std_height_x_tanh =  std_height
            * (trans_cam_pitch_to_zero ? 0.0 : tanh(cam_pitch));
    Dtype std_height_x_cos = std_height
            * (trans_cam_pitch_to_zero ? 1.0 : cos(cam_pitch));
    Dtype cam_ypz_x_cam_hgrd = cam_ypz * cam_hgrd;

    Dtype min_py_grd = cam_yct + cam_ypz_x_tanh;
    Dtype min_r_grd = (min_py_grd - read_height_offset)
                      * read_height_scale;

    int count = im_height * im_width;
    GenGrdFt_kernel<Dtype><<<CUDA_GET_BLOCKS(count, CUDA_NUM_THREADS),
            CUDA_NUM_THREADS>>>(
            im_width, blob_width, blob_height, count,
            std_height, cam_xpz, cam_xct, cam_ypz,
            cam_yct, cam_hgrd, cam_pitch, cam_tanh,
            cam_ypz_x_tanh, std_height_x_tanh,
            std_height_x_cos, cam_ypz_x_cam_hgrd,
            read_width_scale, read_height_scale,
            read_height_offset, min_py_grd,
            min_r_grd, normalize_grd_ft,
            normalize_grd_ft_dim, grd_ft);
    CUDA_POST_KERNEL_CHECK;
}

template void GenGrdFt_gpu(unsigned int im_width,
        unsigned int im_height, unsigned int blob_width,
        unsigned int blob_height, float std_height,
        const std::vector<float> & cam_params, float* grd_ft,
        float read_width_scale, float read_height_scale,
        unsigned int read_height_offset, unsigned int valid_param_idx_st,
        bool trans_cam_pitch_to_zero, bool normalize_grd_ft,
        unsigned int normalize_grd_ft_dim);

}
}