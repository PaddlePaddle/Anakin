/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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
__kernel void permute_data_kernel(
    const global float* data,
    const int data_size,
    const int num_classes,
    const int num_data,
    const int num_dim,
    global float* new_data) {

    int index           = get_global_id(0);

    if (index < data_size) {
        const int i         = index % num_dim;
        const int c         = (index / num_dim) % num_classes;
        const int d         = (index / num_dim / num_classes) % num_data;
        const int n         = index / num_dim / num_classes / num_data;
        const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
        new_data[new_index] = data[index];
    }
}

__kernel void decode_bbox_corner_variance_kernel(/*const int count, \*/
    const global float* loc_data,
    const global float* prior_data,
    const int num_priors,
    const int share_location,
    const int num_loc_classes,
    const int background_label_id,
    global float* bbox_data) {

    int index       = get_global_id(0);
    const int c     = index % num_loc_classes;
    const int idx_p = (index % num_priors) * 4;
    const int idx   = index * 4;

    if (!share_location && c == background_label_id) {
        //! Ignore background class if not share_location.
        return;
    }

    //! variance is encoded in target, we simply need to add the offset predictions.
    bbox_data[idx]     = prior_data[idx_p] + loc_data[idx];
    bbox_data[idx + 1] = prior_data[idx_p + 1] + loc_data[idx + 1];
    bbox_data[idx + 2] = prior_data[idx_p + 2] + loc_data[idx + 2];
    bbox_data[idx + 3] = prior_data[idx_p + 3] + loc_data[idx + 3];
}

__kernel void decode_bbox_corner_no_variance_kernel(/*const int count, \*/
    const global float* loc_data,
    const global float* prior_data,
    const int num_priors,
    const int share_location,
    const int num_loc_classes,
    const int background_label_id,
    global float* bbox_data) {

    int index       = get_global_id(0);
    const int c     = index % num_loc_classes;
    const int idx_p = (index % num_priors) * 4;
    const int idx   = index * 4;
    const global float* variance = prior_data + num_priors * 4;

    if (!share_location && c == background_label_id) {
        //! Ignore background class if not share_location.
        return;
    }

    //! variance is encoded in bbox, we need to scale the offset accordingly.
    bbox_data[idx]     = prior_data[idx_p] + loc_data[idx] * variance[idx_p];
    bbox_data[idx + 1] = prior_data[idx_p + 1] + loc_data[idx + 1] * variance[idx_p + 1];
    bbox_data[idx + 2] = prior_data[idx_p + 2] + loc_data[idx + 2] * variance[idx_p + 2];
    bbox_data[idx + 3] = prior_data[idx_p + 3] + loc_data[idx + 3] * variance[idx_p + 3];
}

__kernel void decode_bbox_center_variance_kernel(/*const int count, \*/
    const global float* loc_data,
    const global float* prior_data,
    const int num_priors,
    const int share_location,
    const int num_loc_classes,
    const int background_label_id,
    global float* bbox_data) {

    int index       = get_global_id(0);
    const int c     = index % num_loc_classes;
    const int idx_p = (index % num_priors) * 4;
    const int idx   = index * 4;
    const global float* variance = prior_data + num_priors * 4;

    if (!share_location && c == background_label_id) {
        //! Ignore background class if not share_location.
        return;
    }

    const float p_xmin         = prior_data[idx_p];
    const float p_ymin         = prior_data[idx_p + 1];
    const float p_xmax         = prior_data[idx_p + 2];
    const float p_ymax         = prior_data[idx_p + 3];
    const float prior_width    = p_xmax - p_xmin;
    const float prior_height   = p_ymax - p_ymin;
    const float prior_center_x = (p_xmin + p_xmax) / 2.;
    const float prior_center_y = (p_ymin + p_ymax) / 2.;

    const float xmin = loc_data[idx];
    const float ymin = loc_data[idx + 1];
    const float xmax = loc_data[idx + 2];
    const float ymax = loc_data[idx + 3];

    //! variance is encoded in target, we simply need to retore the offset predictions.
    float decode_bbox_center_x = xmin * prior_width + prior_center_x;
    float decode_bbox_center_y = ymin * prior_height + prior_center_y;
    float decode_bbox_width    = exp(xmax) * prior_width;
    float decode_bbox_height   = exp(ymax) * prior_height;

    bbox_data[idx]     = decode_bbox_center_x - decode_bbox_width / 2.f;
    bbox_data[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
    bbox_data[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
    bbox_data[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
}

__kernel void decode_bbox_center_no_variance_kernel(/*const int count, \*/
    const global float* loc_data,
    const global float* prior_data,
    const int num_priors,
    const int share_location,
    const int num_loc_classes,
    const int background_label_id,
    global float* bbox_data) {

    int index       = get_global_id(0);
    const int c     = index % num_loc_classes;
    const int idx_p = (index % num_priors) * 4;
    const int idx   = index * 4;
    const global float* variance = prior_data + num_priors * 4;

    if (!share_location && c == background_label_id) {
        //! Ignore background class if not share_location.
        return;
    }

    const float p_xmin         = prior_data[idx_p];
    const float p_ymin         = prior_data[idx_p + 1];
    const float p_xmax         = prior_data[idx_p + 2];
    const float p_ymax         = prior_data[idx_p + 3];
    const float prior_width    = p_xmax - p_xmin;
    const float prior_height   = p_ymax - p_ymin;
    const float prior_center_x = (p_xmin + p_xmax) / 2.;
    const float prior_center_y = (p_ymin + p_ymax) / 2.;

    const float xmin = loc_data[idx];
    const float ymin = loc_data[idx + 1];
    const float xmax = loc_data[idx + 2];
    const float ymax = loc_data[idx + 3];

    //! variance is encoded in bbox, we need to scale the offset accordingly.
    float decode_bbox_center_x = variance[idx_p] * xmin * prior_width + prior_center_x;
    float decode_bbox_center_y = variance[idx_p + 1] * ymin * prior_height + prior_center_y;
    float decode_bbox_width    = exp(variance[idx_p + 2] * xmax) * prior_width;
    float decode_bbox_height   = exp(variance[idx_p + 3] * ymax) * prior_height;

    bbox_data[idx]     = decode_bbox_center_x - decode_bbox_width / 2.f;
    bbox_data[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
    bbox_data[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
    bbox_data[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
}

__kernel void decode_bbox_corner_size_variance_kernel(/*const int count, \*/
    const global float* loc_data,
    const global float* prior_data,
    const int num_priors,
    const int share_location,
    const int num_loc_classes,
    const int background_label_id,
    global float* bbox_data) {

    int index       = get_global_id(0);
    const int c     = index % num_loc_classes;
    const int idx_p = (index % num_priors) * 4;
    const int idx   = index * 4;

    if (!share_location && c == background_label_id) {
        //! Ignore background class if not share_location.
        return;
    }

    const float p_xmin       = prior_data[idx_p];
    const float p_ymin       = prior_data[idx_p + 1];
    const float p_xmax       = prior_data[idx_p + 2];
    const float p_ymax       = prior_data[idx_p + 3];
    const float prior_width  = p_xmax - p_xmin;
    const float prior_height = p_ymax - p_ymin;
    //! variance is encoded in target, we simply need to add the offset predictions.
    bbox_data[idx]     = p_xmin + loc_data[idx] * prior_width;
    bbox_data[idx + 1] = p_ymin + loc_data[idx + 1] * prior_height;
    bbox_data[idx + 2] = p_xmax + loc_data[idx + 2] * prior_width;
    bbox_data[idx + 3] = p_ymax + loc_data[idx + 3] * prior_height;
}

__kernel void decode_bbox_corner_size_no_variance_kernel(/*const int count, \*/
    const global float* loc_data,
    const global float* prior_data,
    const int num_priors,
    const int share_location,
    const int num_loc_classes,
    const int background_label_id,
    global float* bbox_data) {

    int index       = get_global_id(0);
    const int c     = index % num_loc_classes;
    const int idx_p = (index % num_priors) * 4;
    const int idx   = index * 4;
    const global float* variance = prior_data + num_priors * 4;

    if (!share_location && c == background_label_id) {
        //! Ignore background class if not share_location.
        return;
    }

    const float p_xmin       = prior_data[idx_p];
    const float p_ymin       = prior_data[idx_p + 1];
    const float p_xmax       = prior_data[idx_p + 2];
    const float p_ymax       = prior_data[idx_p + 3];
    const float prior_width  = p_xmax - p_xmin;
    const float prior_height = p_ymax - p_ymin;
    //! variance is encoded in bbox, we need to scale the offset accordingly.
    bbox_data[idx]     = p_xmin + loc_data[idx] * variance[idx_p] * prior_width;
    bbox_data[idx + 1] = p_ymin + loc_data[idx + 1] * variance[idx_p + 1] * prior_height;
    bbox_data[idx + 2] = p_xmax + loc_data[idx + 2] * variance[idx_p + 2] * prior_width;
    bbox_data[idx + 3] = p_ymax + loc_data[idx + 3] * variance[idx_p + 3] * prior_height;
}
