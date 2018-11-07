/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_DETECTION_HELPER_H
#define ANAKIN_SABER_FUNCS_DETECTION_HELPER_H

#include "saber/core/common.h"
#include <map>
#include <vector>

namespace anakin{

namespace saber{

template <typename dtype>
dtype jaccard_overlap(const dtype* bbox1, const dtype* bbox2);

template <typename dtype>
void apply_nms_fast(const dtype* bboxes, const dtype* scores, int num,
                        float score_threshold, float nms_threshold,
                        float eta, int top_k, std::vector<int>* indices);

template <typename dtype>
void nms_detect(const dtype* bbox_cpu_data,
                const dtype* conf_cpu_data, std::vector<dtype>& result, \
                int batch_num, int class_num, int num_priors, int background_id, \
                int keep_topk, int nms_topk, float conf_thresh, float nms_thresh,
                float nms_eta, bool share_location);

#if defined USE_CUDA 
template <typename Dtype>
void decode_bboxes(const int nthreads, const Dtype* loc_data, const Dtype* prior_data, \
                   const CodeType code_type, const bool variance_encoded_in_target, \
                   const int num_priors, const bool share_location, \
                   const int num_loc_classes, const int background_label_id, \
                   Dtype* bbox_data, cudaStream_t stream);
#endif
#if defined USE_X86_PLACE 
template <typename Dtype>
void decode_bboxes(const int nthreads, const Dtype* loc_data, const Dtype* prior_data, \
                   const CodeType code_type, const bool variance_encoded_in_target, \
                   const int num_priors, const bool share_location, \
                   const int num_loc_classes, const int background_label_id, \
                   Dtype* bbox_data);
#endif

#ifdef USE_ARM_PLACE
void decode_bboxes(const int batch_num, const float* loc_data, const float* prior_data, \
                       const CodeType code_type, const bool variance_encoded_in_target, \
                       const int num_priors, const bool share_location, \
                       const int num_loc_classes, const int background_label_id, \
                       float* bbox_data);
#endif
} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_DETECTION_HELPER_H
