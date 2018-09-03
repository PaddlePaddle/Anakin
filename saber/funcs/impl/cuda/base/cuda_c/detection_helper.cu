#include "saber/funcs/impl/detection_helper.h"
namespace anakin{

namespace saber{

template <typename dtype>
__global__ void decode_bbox_corner_variance_kernel(const int count, \
        const dtype* loc_data, const dtype* prior_data, const dtype* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, dtype* bbox_data) {
    CUDA_KERNEL_LOOP(index, count) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;
        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            return;
        }
        //! variance is encoded in target, we simply need to add the offset predictions.
        bbox_data[idx] = prior_data[idx_p] + loc_data[idx];
        bbox_data[idx + 1] = prior_data[idx_p + 1] + loc_data[idx + 1];
        bbox_data[idx + 2] = prior_data[idx_p + 2] + loc_data[idx + 2];
        bbox_data[idx + 3] = prior_data[idx_p + 3] + loc_data[idx + 3];
    }
}

template <typename dtype>
__global__ void decode_bbox_corner_no_variance_kernel(const int count, \
        const dtype* loc_data, const dtype* prior_data, const dtype* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, dtype* bbox_data) {
    CUDA_KERNEL_LOOP(index, count) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;
        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            return;
        }
        //! variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[idx] = prior_data[idx_p] + loc_data[idx] * variance[idx_p];
        bbox_data[idx + 1] = prior_data[idx_p + 1] + loc_data[idx + 1] * variance[idx_p + 1];
        bbox_data[idx + 2] = prior_data[idx_p + 2] + loc_data[idx + 2] * variance[idx_p + 2];
        bbox_data[idx + 3] = prior_data[idx_p + 3] + loc_data[idx + 3] * variance[idx_p + 3];
    }
}

template <typename dtype>
__global__ void decode_bbox_center_variance_kernel(const int count, \
        const dtype* loc_data, const dtype* prior_data, const dtype* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, dtype* bbox_data) {
    CUDA_KERNEL_LOOP(index, count) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;
        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            return;
        }
        const dtype p_xmin = prior_data[idx_p];
        const dtype p_ymin = prior_data[idx_p + 1];
        const dtype p_xmax = prior_data[idx_p + 2];
        const dtype p_ymax = prior_data[idx_p + 3];
        const dtype prior_width = p_xmax - p_xmin;
        const dtype prior_height = p_ymax - p_ymin;
        const dtype prior_center_x = (p_xmin + p_xmax) / 2.;
        const dtype prior_center_y = (p_ymin + p_ymax) / 2.;

        const dtype xmin = loc_data[idx];
        const dtype ymin = loc_data[idx + 1];
        const dtype xmax = loc_data[idx + 2];
        const dtype ymax = loc_data[idx + 3];

        //! variance is encoded in target, we simply need to retore the offset predictions.
        dtype decode_bbox_center_x = xmin * prior_width + prior_center_x;
        dtype decode_bbox_center_y = ymin * prior_height + prior_center_y;
        dtype decode_bbox_width = exp(xmax) * prior_width;
        dtype decode_bbox_height = exp(ymax) * prior_height;

        bbox_data[idx] = decode_bbox_center_x - decode_bbox_width / 2.f;
        bbox_data[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
        bbox_data[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
        bbox_data[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
    }
}

template <typename dtype>
__global__ void decode_bbox_center_no_variance_kernel(const int count, \
        const dtype* loc_data, const dtype* prior_data, const dtype* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, dtype* bbox_data) {
    CUDA_KERNEL_LOOP(index, count) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;
        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            return;
        }
        const dtype p_xmin = prior_data[idx_p];
        const dtype p_ymin = prior_data[idx_p + 1];
        const dtype p_xmax = prior_data[idx_p + 2];
        const dtype p_ymax = prior_data[idx_p + 3];
        const dtype prior_width = p_xmax - p_xmin;
        const dtype prior_height = p_ymax - p_ymin;
        const dtype prior_center_x = (p_xmin + p_xmax) / 2.;
        const dtype prior_center_y = (p_ymin + p_ymax) / 2.;

        const dtype xmin = loc_data[idx];
        const dtype ymin = loc_data[idx + 1];
        const dtype xmax = loc_data[idx + 2];
        const dtype ymax = loc_data[idx + 3];

        //! variance is encoded in bbox, we need to scale the offset accordingly.
        dtype decode_bbox_center_x =
                variance[idx_p] * xmin * prior_width + prior_center_x;
        dtype decode_bbox_center_y =
                variance[idx_p + 1] * ymin * prior_height + prior_center_y;
        dtype decode_bbox_width =
                exp(variance[idx_p + 2] * xmax) * prior_width;
        dtype decode_bbox_height =
                exp(variance[idx_p + 3] * ymax) * prior_height;

        bbox_data[idx] = decode_bbox_center_x - decode_bbox_width / 2.f;
        bbox_data[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
        bbox_data[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
        bbox_data[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
    }
}

template <typename dtype>
__global__ void decode_bbox_corner_size_variance_kernel(const int count, \
        const dtype* loc_data, const dtype* prior_data, const dtype* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, dtype* bbox_data) {
    CUDA_KERNEL_LOOP(index, count) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;
        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            return;
        }
        const dtype p_xmin = prior_data[idx_p];
        const dtype p_ymin = prior_data[idx_p + 1];
        const dtype p_xmax = prior_data[idx_p + 2];
        const dtype p_ymax = prior_data[idx_p + 3];
        const dtype prior_width = p_xmax - p_xmin;
        const dtype prior_height = p_ymax - p_ymin;
        //! variance is encoded in target, we simply need to add the offset predictions.
        bbox_data[idx] = p_xmin + loc_data[idx] * prior_width;
        bbox_data[idx + 1] = p_ymin + loc_data[idx + 1] * prior_height;
        bbox_data[idx + 2] = p_xmax + loc_data[idx + 2] * prior_width;
        bbox_data[idx + 3] = p_ymax + loc_data[idx + 3] * prior_height;
    }
}

template <typename dtype>
__global__ void decode_bbox_corner_size_no_variance_kernel(const int count, \
        const dtype* loc_data, const dtype* prior_data, const dtype* variance, \
        const int num_priors, const bool share_location, const int num_loc_classes, \
        const int background_label_id, dtype* bbox_data) {
    CUDA_KERNEL_LOOP(index, count) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;
        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            return;
        }
        const dtype p_xmin = prior_data[idx_p];
        const dtype p_ymin = prior_data[idx_p + 1];
        const dtype p_xmax = prior_data[idx_p + 2];
        const dtype p_ymax = prior_data[idx_p + 3];
        const dtype prior_width = p_xmax - p_xmin;
        const dtype prior_height = p_ymax - p_ymin;
        //! variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[idx] =
                p_xmin + loc_data[idx] * variance[idx_p] * prior_width;
        bbox_data[idx + 1] =
                p_ymin + loc_data[idx + 1] * variance[idx_p + 1] * prior_height;
        bbox_data[idx + 2] =
                p_xmax + loc_data[idx + 2] * variance[idx_p + 2] * prior_width;
        bbox_data[idx + 3] =
                p_ymax + loc_data[idx + 3] * variance[idx_p + 3] * prior_height;
    }
}

template <typename Dtype>
void decode_bboxes(const int nthreads,
                     const Dtype* loc_data, const Dtype* prior_data,
                     const CodeType code_type, const bool variance_encoded_in_target,
                     const int num_priors, const bool share_location,
                     const int num_loc_classes, const int background_label_id,
                     Dtype* bbox_data, cudaStream_t stream) {
    int count = nthreads / 4;
    const Dtype* variance_data = prior_data + 4 * num_priors;
    if (code_type == CORNER) {
        if (variance_encoded_in_target) {
            decode_bbox_corner_variance_kernel<Dtype>\
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>\
                (count, loc_data, prior_data, variance_data, num_priors, share_location, \
                    num_loc_classes, background_label_id, bbox_data);
        } else {
            decode_bbox_corner_no_variance_kernel<Dtype>\
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>\
                (count, loc_data, prior_data, variance_data, num_priors, share_location, \
                    num_loc_classes, background_label_id, bbox_data);
        }
    } else if (code_type == CENTER_SIZE) {
        if (variance_encoded_in_target) {
            decode_bbox_center_variance_kernel<Dtype>\
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>\
                (count, loc_data, prior_data, variance_data, num_priors, share_location, \
                    num_loc_classes, background_label_id, bbox_data);
        } else {
            decode_bbox_center_no_variance_kernel<Dtype>\
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>\
                (count, loc_data, prior_data, variance_data, num_priors, share_location, \
                    num_loc_classes, background_label_id, bbox_data);
        }
    } else if (code_type == CORNER_SIZE) {
        if (variance_encoded_in_target) {
            decode_bbox_corner_size_variance_kernel<Dtype>\
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>\
                (count, loc_data, prior_data, variance_data, num_priors, share_location, \
                    num_loc_classes, background_label_id, bbox_data);
        } else {
            decode_bbox_corner_size_no_variance_kernel<Dtype>\
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>\
                (count, loc_data, prior_data, variance_data, num_priors, share_location, \
                    num_loc_classes, background_label_id, bbox_data);
        }
    }
}

template void decode_bboxes<float>(const int nthreads,
                       const float* loc_data, const float* prior_data,
                       const CodeType code_type, const bool variance_encoded_in_target,
                       const int num_priors, const bool share_location,
                       const int num_loc_classes, const int background_label_id,
                       float* bbox_data, cudaStream_t stream);
} //namespace anakin

} //namespace anakin
