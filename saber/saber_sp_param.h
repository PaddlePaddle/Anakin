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

#ifndef ANAKIN_SABER_SP_PARAM_H
#define ANAKIN_SABER_SP_PARAM_H

#include "anakin_config.h"
#include <vector>
#include <string>
#include "saber/core/shape.h"
#include "saber/core/tensor.h"
#include "saber/saber_types.h"

namespace anakin {

namespace saber {

template <typename vectors>
inline
bool compare_vectors(vectors& a, const vectors& b) {
    if (a.size() != b.size()) {
        return false;
    }

    bool comp = true;

    for (int i = 0; i < a.size(); ++i) {
        comp &= comp && (a[i] == b[i]);
    }

    return comp;
}

template <typename vectors>
inline
void copy_vectors(vectors& out, const vectors& in) {
    out.resize(in.size());

    for (int i = 0; i < out.size(); ++i) {
        out[i] = in[i];
    }
}

typedef enum {
    ATRS_NormType_NONE = 0,
    ATRS_NormType_WIDTH = 1,
    ATRS_NormType_HEIGHT = 2,
    ATRS_NormType_WIDTH_LOG = 3,
    ATRS_NormType_HEIGHT_LOG = 4,
} ATRS_NormType;

typedef enum {
    DetectionOutputSSD_HEIGHT_AND_WIDTH = 0,
    DetectionOutputSSD_HEIGHT_OR_WIDTH = 1
} DetectionOutputSSD_MIN_SIZE_MODE;

typedef enum {
    ProposalImgScaleToCamCoords_NormType_HEIGHT = 0,
    ProposalImgScaleToCamCoords_NormType_HEIGHT_LOG = 1
} ProposalImgScaleToCamCoords_NormType;

typedef enum {
    ProposalImgScaleToCamCoords_OrienType_PI = 0,
    ProposalImgScaleToCamCoords_OrienType_PI2 = 1
} ProposalImgScaleToCamCoords_OrienType;

template <typename TargetType>
struct DFMBPSROIAlignParam {
    DFMBPSROIAlignParam()
        : heat_map_a(0)
        , output_dim(0)
    {}
    DFMBPSROIAlignParam(float heat_map_a_in, int output_dim_in,
                        float heat_map_b_in = 0, float pad_ratio_in = 0,
                        float trans_std_in = 0.1, int sample_per_part_in = 4,
                        int group_height_in = 7,  int group_width_in = 7,
                        int pooled_height_in = 7, int pooled_width_in = 7,
                        int part_height_in = 7, int part_width_in = 7)
        : heat_map_a(heat_map_a_in), output_dim(output_dim_in)
        , heat_map_b(heat_map_b_in), pad_ratio(pad_ratio_in)
        , trans_std(trans_std_in), sample_per_part(sample_per_part_in)
        , group_height(group_height_in), group_width(group_width_in)
        , pooled_height(pooled_height_in), pooled_width(pooled_width_in)
        , part_height(part_height_in), part_width(part_width_in)
    {}
    ~DFMBPSROIAlignParam() {}

    DFMBPSROIAlignParam(const DFMBPSROIAlignParam& right)
        : heat_map_a(right.heat_map_a)
        , output_dim(right.output_dim)
        , heat_map_b(right.heat_map_b)
        , pad_ratio(right.pad_ratio)
        , trans_std(right.trans_std)
        , sample_per_part(right.sample_per_part)
        , group_height(right.group_height)
        , group_width(right.group_width)
        , pooled_height(right.pooled_height)
        , pooled_width(right.pooled_width)
        , part_height(right.part_height)
        , part_width(right.part_width)
    {}

    DFMBPSROIAlignParam& operator=(const DFMBPSROIAlignParam& right) {
        heat_map_a = right.heat_map_a;
        output_dim = right.output_dim;
        heat_map_b = right.heat_map_b;
        pad_ratio = right.pad_ratio;
        trans_std = right.trans_std;
        sample_per_part = right.sample_per_part;
        group_height = right.group_height;
        group_width = right.group_width;
        pooled_height = right.pooled_height;
        pooled_width = right.pooled_width;
        part_height = right.part_height;
        part_width = right.part_width;
        return *this;
    }

    bool operator==(const DFMBPSROIAlignParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (heat_map_a == right.heat_map_a);
        comp_eq &= comp_eq && (output_dim == right.output_dim);
        comp_eq &= comp_eq && (heat_map_b == right.heat_map_b);
        comp_eq &= comp_eq && (pad_ratio == right.pad_ratio);
        comp_eq &= comp_eq && (trans_std == right.trans_std);
        comp_eq &= comp_eq && (sample_per_part == right.sample_per_part);
        comp_eq &= comp_eq && (group_height == right.group_height);
        comp_eq &= comp_eq && (group_width == right.group_width);
        comp_eq &= comp_eq && (pooled_height == right.pooled_height);
        comp_eq &= comp_eq && (pooled_width == right.pooled_width);
        comp_eq &= comp_eq && (part_height == right.part_height);
        comp_eq &= comp_eq && (part_width == right.part_width);
        return comp_eq;
    }

    float heat_map_a;
    int output_dim;

    float heat_map_b;
    float pad_ratio;
    float trans_std;
    int sample_per_part;
    int group_height;
    int group_width;
    int pooled_height;
    int pooled_width;
    int part_height;
    int part_width;
};

template <typename TargetType>
struct BBoxRegParam {
    BBoxRegParam()
        : bbox_mean()
        , bbox_std()
        , has_param(false)
    {}
    BBoxRegParam(std::vector<float> bbox_mean_in,
                 std::vector<float> bbox_std_in)
        : bbox_mean(bbox_mean_in)
        , bbox_std(bbox_std_in)
        , has_param(true)
    {}
    ~BBoxRegParam() {}
    BBoxRegParam(const BBoxRegParam& right)
        : bbox_mean(right.bbox_mean)
        , bbox_std(right.bbox_std)
        , has_param(right.has_param)
    {}
    BBoxRegParam& operator=(const BBoxRegParam& right) {
        copy_vectors(bbox_mean, right.bbox_mean);
        copy_vectors(bbox_std, right.bbox_std);
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const BBoxRegParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (compare_vectors(bbox_mean, right.bbox_mean));
        comp_eq &= comp_eq && (compare_vectors(bbox_std, right.bbox_std));
        return comp_eq;
    }
    std::vector<float> bbox_mean;
    std::vector<float> bbox_std;
    bool has_param;
};

template <typename TargetType>
struct RegParam {
    RegParam()
        : mean()
        , std()
        , has_param(false)
    {}
    RegParam(std::vector<float> mean_in,
             std::vector<float> std_in)
        : mean(mean_in)
        , std(std_in)
        , has_param(true)
    {}
    ~RegParam() {}
    RegParam(const RegParam& right)
        : mean(right.mean)
        , std(right.std)
        , has_param(right.has_param)
    {}
    RegParam& operator=(const RegParam& right) {
        copy_vectors(mean, right.mean);
        copy_vectors(std, right.std);
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const RegParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (compare_vectors(mean, right.mean));
        comp_eq &= comp_eq && (compare_vectors(std, right.std));
        return comp_eq;
    }
    std::vector<float> mean;
    std::vector<float> std;
    bool has_param;
};

template <typename TargetType>
struct NMSSSDParam {
    NMSSSDParam()
        : overlap_ratio()
        , top_n()
        , max_candidate_n()
        , use_soft_nms()
        , voting()
        , vote_iou()
        , need_nms(true)
        , add_score(false)
        , nms_among_classes(false)
        , has_param(false)
        , nms_gpu_max_n_per_time(-1)
    {}
    NMSSSDParam(std::vector<float> overlap_ratio_in,
                std::vector<int> top_n_in,
                std::vector<int> max_candidate_n_in,
                std::vector<bool> use_soft_nms_in,
                std::vector<bool> voting_in,
                std::vector<float> vote_iou_in,
                bool need_nms_in = true,
                bool add_score_in = false,
                bool nms_among_classes_in = false,
                int nms_gpu_max_n_per_time_in = -1)
        : overlap_ratio(overlap_ratio_in)
        , top_n(top_n_in)
        , max_candidate_n(max_candidate_n_in)
        , use_soft_nms(use_soft_nms_in)
        , voting(voting_in)
        , vote_iou(vote_iou_in)
        , need_nms(need_nms_in)
        , add_score(add_score_in)
        , nms_among_classes(nms_among_classes_in)
        , nms_gpu_max_n_per_time(nms_gpu_max_n_per_time_in)
        , has_param(true)
    {}
    ~NMSSSDParam() {}
    NMSSSDParam(const NMSSSDParam& right)
        : overlap_ratio(right.overlap_ratio)
        , top_n(right.top_n)
        , max_candidate_n(right.max_candidate_n)
        , use_soft_nms(right.use_soft_nms)
        , voting(right.voting)
        , vote_iou(right.vote_iou)
        , need_nms(right.need_nms)
        , add_score(right.add_score)
        , nms_among_classes(right.nms_among_classes)
        , nms_gpu_max_n_per_time(right.nms_gpu_max_n_per_time)
        , has_param(right.has_param)
    {}
    NMSSSDParam& operator=(const NMSSSDParam& right) {
        copy_vectors(overlap_ratio, right.overlap_ratio);
        copy_vectors(top_n, right.top_n);
        copy_vectors(max_candidate_n, right.max_candidate_n);
        copy_vectors(use_soft_nms, right.use_soft_nms);
        copy_vectors(voting, right.voting);
        copy_vectors(vote_iou, right.vote_iou);
        need_nms = right.need_nms;
        add_score = right.add_score;
        nms_among_classes = right.nms_among_classes;
        nms_gpu_max_n_per_time = right.nms_gpu_max_n_per_time;
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const NMSSSDParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (compare_vectors(overlap_ratio, right.overlap_ratio));
        comp_eq &= comp_eq && (compare_vectors(top_n, right.top_n));
        comp_eq &= comp_eq && (compare_vectors(max_candidate_n, right.max_candidate_n));
        comp_eq &= comp_eq && (compare_vectors(use_soft_nms, right.use_soft_nms));
        comp_eq &= comp_eq && (compare_vectors(voting, right.voting));
        comp_eq &= comp_eq && (compare_vectors(vote_iou, right.vote_iou));
        comp_eq &= comp_eq && (need_nms == right.need_nms);
        comp_eq &= comp_eq && (add_score == right.add_score);
        comp_eq &= comp_eq && (nms_among_classes == right.nms_among_classes);
        comp_eq &= comp_eq && (nms_gpu_max_n_per_time == right.nms_gpu_max_n_per_time);
        return comp_eq;
    }
    std::vector<float> overlap_ratio;
    std::vector<int> top_n;
    std::vector<int> max_candidate_n;
    std::vector<bool> use_soft_nms;
    std::vector<bool> voting;
    std::vector<float> vote_iou;
    int nms_gpu_max_n_per_time;
    bool need_nms = true;
    bool add_score = false;
    bool nms_among_classes = false;
    bool has_param;
};

template <typename TargetType>
struct GenerateAnchorParam {
    GenerateAnchorParam()
        : ratios()
        , scales()
        , anchor_width()
        , anchor_height()
        , anchor_x1()
        , anchor_y1()
        , anchor_x2()
        , anchor_y2()
        , base_size(16)
        , zero_anchor_center(true)
        , has_param(false)
    {}
    GenerateAnchorParam(std::vector<float> ratios_in,
                        std::vector<float> scales_in,
                        std::vector<float> anchor_width_in,
                        std::vector<float> anchor_height_in,
                        std::vector<float> anchor_x1_in,
                        std::vector<float> anchor_y1_in,
                        std::vector<float> anchor_x2_in,
                        std::vector<float> anchor_y2_in,
                        float base_size_in = 16,
                        bool zero_anchor_center_in = true)
        : ratios(ratios_in)
        , scales(scales_in)
        , anchor_width(anchor_width_in)
        , anchor_height(anchor_height_in)
        , anchor_x1(anchor_x1_in)
        , anchor_y1(anchor_y1_in)
        , anchor_x2(anchor_x2_in)
        , anchor_y2(anchor_y2_in)
        , base_size(base_size_in)
        , zero_anchor_center(zero_anchor_center_in)
        , has_param(true)
    {}
    ~GenerateAnchorParam() {}
    GenerateAnchorParam(const GenerateAnchorParam& right)
        : ratios(right.ratios)
        , scales(right.scales)
        , anchor_width(right.anchor_width)
        , anchor_height(right.anchor_height)
        , anchor_x1(right.anchor_x1)
        , anchor_y1(right.anchor_y1)
        , anchor_x2(right.anchor_x2)
        , anchor_y2(right.anchor_y2)
        , base_size(right.base_size)
        , zero_anchor_center(right.zero_anchor_center)
        , has_param(right.has_param)
    {}
    GenerateAnchorParam& operator=(const GenerateAnchorParam& right) {
        copy_vectors(ratios, right.ratios);
        copy_vectors(scales, right.scales);
        copy_vectors(anchor_width, right.anchor_width);
        copy_vectors(anchor_height, right.anchor_height);
        copy_vectors(anchor_x1, right.anchor_x1);
        copy_vectors(anchor_y1, right.anchor_y1);
        copy_vectors(anchor_x2, right.anchor_x2);
        copy_vectors(anchor_y2, right.anchor_y2);
        base_size = right.base_size;
        zero_anchor_center = right.zero_anchor_center;
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const GenerateAnchorParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (compare_vectors(ratios, right.ratios));
        comp_eq &= comp_eq && (compare_vectors(scales, right.scales));
        comp_eq &= comp_eq && (compare_vectors(anchor_width, right.anchor_width));
        comp_eq &= comp_eq && (compare_vectors(anchor_height, right.anchor_height));
        comp_eq &= comp_eq && (compare_vectors(anchor_x1, right.anchor_x1));
        comp_eq &= comp_eq && (compare_vectors(anchor_y1, right.anchor_y1));
        comp_eq &= comp_eq && (compare_vectors(anchor_x2, right.anchor_x2));
        comp_eq &= comp_eq && (compare_vectors(anchor_y2, right.anchor_y2));
        comp_eq &= comp_eq && (base_size == right.base_size);
        comp_eq &= comp_eq && (zero_anchor_center == right.zero_anchor_center);
        return comp_eq;
    }
    std::vector<float> ratios;
    std::vector<float> scales;
    std::vector<float> anchor_width;
    std::vector<float> anchor_height;
    std::vector<float> anchor_x1;
    std::vector<float> anchor_y1;
    std::vector<float> anchor_x2;
    std::vector<float> anchor_y2;
    float base_size;
    bool zero_anchor_center;
    bool has_param;
};

template <typename TargetType>
struct KPTSParam {
    KPTSParam()
        : kpts_exist_bottom_idx(0)
        , kpts_reg_bottom_idx(0)
        , kpts_st_for_each_class()
        , kpts_ed_for_each_class()
        , kpts_reg_as_classify(false)
        , kpts_classify_width(0)
        , kpts_classify_height(0)
        , kpts_reg_norm_idx_st(0)
        , kpts_classify_pad_ratio(0.f)
        , has_param(false)
    {}
    KPTSParam(int kpts_exist_bottom_idx_in,
              int kpts_reg_bottom_idx_in,
              std::vector<int> kpts_st_for_each_class_in,
              std::vector<int> kpts_ed_for_each_class_in,
              bool kpts_reg_as_classify_in = false,
              int kpts_classify_width_in = 0,
              int kpts_classify_height_in = 0,
              int kpts_reg_norm_idx_st_in = 0,
              float kpts_classify_pad_ratio_in = 0.0f)
        : kpts_exist_bottom_idx(kpts_exist_bottom_idx_in)
        , kpts_reg_bottom_idx(kpts_reg_bottom_idx_in)
        , kpts_st_for_each_class(kpts_st_for_each_class_in)
        , kpts_ed_for_each_class(kpts_ed_for_each_class_in)
        , kpts_reg_as_classify(kpts_reg_as_classify_in)
        , kpts_classify_width(kpts_classify_width_in)
        , kpts_classify_height(kpts_classify_height_in)
        , kpts_reg_norm_idx_st(kpts_reg_norm_idx_st_in)
        , kpts_classify_pad_ratio(kpts_classify_pad_ratio_in)
        , has_param(true)
    {}
    ~KPTSParam() {}
    KPTSParam(const KPTSParam& right)
        : kpts_exist_bottom_idx(right.kpts_exist_bottom_idx)
        , kpts_reg_bottom_idx(right.kpts_reg_bottom_idx)
        , kpts_st_for_each_class(right.kpts_st_for_each_class)
        , kpts_ed_for_each_class(right.kpts_ed_for_each_class)
        , kpts_reg_as_classify(right.kpts_reg_as_classify)
        , kpts_classify_width(right.kpts_classify_width)
        , kpts_classify_height(right.kpts_classify_height)
        , kpts_reg_norm_idx_st(right.kpts_reg_norm_idx_st)
        , kpts_classify_pad_ratio(right.kpts_classify_pad_ratio)
        , has_param(right.has_param)
    {}
    KPTSParam& operator=(const KPTSParam& right) {
        kpts_exist_bottom_idx = right.kpts_exist_bottom_idx;
        kpts_reg_bottom_idx = right.kpts_reg_norm_idx_st;
        copy_vectors(kpts_st_for_each_class, right.kpts_st_for_each_class);
        copy_vectors(kpts_ed_for_each_class, right.kpts_ed_for_each_class);
        kpts_reg_as_classify = right.kpts_reg_as_classify;
        kpts_classify_width = right.kpts_classify_width;
        kpts_classify_height = right.kpts_classify_height;
        kpts_reg_norm_idx_st = right.kpts_reg_norm_idx_st;
        kpts_classify_pad_ratio = right.kpts_classify_pad_ratio;
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const KPTSParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (kpts_exist_bottom_idx == right.kpts_exist_bottom_idx);
        comp_eq &= comp_eq && (kpts_reg_bottom_idx == right.kpts_reg_bottom_idx);
        comp_eq &= comp_eq && (compare_vectors(kpts_st_for_each_class,
                                               right.kpts_st_for_each_class));
        comp_eq &= comp_eq && (compare_vectors(kpts_ed_for_each_class,
                                               right.kpts_ed_for_each_class));
        comp_eq &= comp_eq && (kpts_reg_as_classify == right.kpts_reg_as_classify);
        comp_eq &= comp_eq && (kpts_classify_width == right.kpts_classify_width);
        comp_eq &= comp_eq && (kpts_classify_height == right.kpts_classify_height);
        comp_eq &= comp_eq && (kpts_reg_norm_idx_st == right.kpts_reg_norm_idx_st);
        comp_eq &= comp_eq && (kpts_classify_pad_ratio == right.kpts_classify_pad_ratio);
        return comp_eq;
    }
    int kpts_exist_bottom_idx;
    int kpts_reg_bottom_idx;
    std::vector<int> kpts_st_for_each_class;
    std::vector<int> kpts_ed_for_each_class;
    bool kpts_reg_as_classify = false;
    int kpts_classify_width = 0;
    int kpts_classify_height = 0;
    int kpts_reg_norm_idx_st = 0;
    float kpts_classify_pad_ratio = 0.f;
    bool has_param;
};

template <typename TargetType>
struct ATRSParam {
    ATRSParam()
        : atrs_reg_bottom_idx(-1)
        , atrs_norm_type()
        , atrs_reg_norm_idx_st(0)
        , has_param(false)
    {}
    ATRSParam(int atrs_reg_bottom_idx_in,
              std::vector<ATRS_NormType> atrs_norm_type_in,
              int atrs_reg_norm_idx_st_in = 0)
        : atrs_reg_bottom_idx(atrs_reg_bottom_idx_in)
        , atrs_reg_norm_idx_st(atrs_reg_norm_idx_st_in)
        , atrs_norm_type(atrs_norm_type_in)
        , has_param(true)
    {}
    ~ATRSParam() {}
    ATRSParam(const ATRSParam& right)
        : atrs_reg_bottom_idx(right.atrs_reg_bottom_idx)
        , atrs_reg_norm_idx_st(right.atrs_reg_norm_idx_st)
        , atrs_norm_type(right.atrs_norm_type)
        , has_param(right.has_param)
    {}
    ATRSParam& operator=(const ATRSParam& right) {
        atrs_reg_bottom_idx = right.atrs_reg_bottom_idx;
        atrs_reg_norm_idx_st = right.atrs_reg_norm_idx_st;
        copy_vectors(atrs_norm_type, right.atrs_norm_type);
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const ATRSParam& right) {
        bool comp_eq = true;
        atrs_reg_bottom_idx = right.atrs_reg_bottom_idx;
        atrs_reg_norm_idx_st = right.atrs_reg_norm_idx_st;
        comp_eq &= comp_eq && (compare_vectors(atrs_norm_type,
                                               right.atrs_norm_type));
        return comp_eq;
    }
    int atrs_reg_bottom_idx;
    int atrs_reg_norm_idx_st;
    std::vector<ATRS_NormType> atrs_norm_type;
    bool has_param;
};

template <typename TargetType>
struct FTRSParam {
    FTRSParam()
        : ftrs_bottom_idx(-1)
        , has_param(false)
    {}
    FTRSParam(int ftrs_bottom_idx_in)
        : ftrs_bottom_idx(ftrs_bottom_idx_in)
        , has_param(true)
    {}
    FTRSParam(const FTRSParam& right)
        : ftrs_bottom_idx(right.ftrs_bottom_idx)
        , has_param(right.has_param)
    {}
    ~FTRSParam() {}
    FTRSParam& operator=(const FTRSParam& right) {
        ftrs_bottom_idx = right.ftrs_bottom_idx;
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const FTRSParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (ftrs_bottom_idx == right.ftrs_bottom_idx);
        return comp_eq;
    }
    int ftrs_bottom_idx;
    bool has_param;
};

template <typename TargetType>
struct SPMPParam {
    SPMPParam()
        : spmp_bottom_idx(-1)
        , spmp_class_aware()
        , spmp_label_width()
        , spmp_label_height()
        , spmp_pad_ratio()
        , has_param(false)
    {}
    SPMPParam(int spmp_bottom_idx_in,
              std::vector<bool> spmp_class_aware_in,
              std::vector<int> spmp_label_width_in,
              std::vector<int> spmp_label_height_in,
              std::vector<float> spmp_pad_ratio_in)

        : spmp_bottom_idx(spmp_bottom_idx_in)
        , spmp_class_aware(spmp_class_aware_in)
        , spmp_label_width(spmp_label_width_in)
        , spmp_label_height(spmp_label_height_in)
        , spmp_pad_ratio(spmp_pad_ratio_in)
        , has_param(true)
    {}
    ~SPMPParam() {}
    SPMPParam(const SPMPParam& right)
        : spmp_bottom_idx(right.spmp_bottom_idx)
        , spmp_class_aware(right.spmp_class_aware)
        , spmp_label_width(right.spmp_label_width)
        , spmp_label_height(right.spmp_label_height)
        , spmp_pad_ratio(right.spmp_pad_ratio)
        , has_param(right.has_param)
    {}
    SPMPParam& operator=(const SPMPParam& right) {
        spmp_bottom_idx = right.spmp_bottom_idx;
        copy_vectors(spmp_class_aware, right.spmp_class_aware);
        copy_vectors(spmp_label_width, right.spmp_label_width);
        copy_vectors(spmp_label_height, right.spmp_label_height);
        copy_vectors(spmp_pad_ratio, right.spmp_pad_ratio);
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const SPMPParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (spmp_bottom_idx == right.spmp_bottom_idx);
        comp_eq &= comp_eq && (compare_vectors(spmp_class_aware, right.spmp_class_aware));
        comp_eq &= comp_eq && (compare_vectors(spmp_label_width, right.spmp_label_width));
        comp_eq &= comp_eq && (compare_vectors(spmp_label_height, right.spmp_label_height));
        comp_eq &= comp_eq && (compare_vectors(spmp_pad_ratio, right.spmp_pad_ratio));
        return comp_eq;
    }
    int spmp_bottom_idx;
    std::vector<bool> spmp_class_aware;
    std::vector<int> spmp_label_width;
    std::vector<int> spmp_label_height;
    std::vector<float> spmp_pad_ratio;
    bool has_param;
};

template <typename TargetType>
struct Cam3dParam {
    Cam3dParam()
        : cam3d_bottom_idx(-1)
        , has_param(false)
    {}
    Cam3dParam(int cam3d_bottom_idx_in)
        : cam3d_bottom_idx(cam3d_bottom_idx_in)
        , has_param(true)
    {}
    Cam3dParam(const Cam3dParam& right)
        : cam3d_bottom_idx(right.cam3d_bottom_idx)
        , has_param(right.has_param)
    {}
    ~Cam3dParam() {}
    Cam3dParam& operator=(const Cam3dParam& right) {
        cam3d_bottom_idx = right.cam3d_bottom_idx;
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const Cam3dParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (cam3d_bottom_idx == right.cam3d_bottom_idx);
        return comp_eq;
    }
    int cam3d_bottom_idx;
    bool has_param;
};

template <typename TargetType>
struct DetectionOutputSSDParam {
    DetectionOutputSSDParam()
        : threshold()
        , class_indexes()
        , heat_map_a()
        , heat_map_b()
        , proposal_min_sqrt_area()
        , proposal_max_sqrt_area()
        , channel_per_scale(5)
        , class_name_list("")
        , num_class(1)
        , refine_out_of_map_bbox(false)
        , threshold_objectness(0.f)
        , bg_as_one_of_softmax(false)
        , use_target_type_rcnn(true)
        , im_width(0)
        , im_height(0)
        , rpn_proposal_output_score(false)
        , regress_agnostic(true)
        , allow_border(-1.0f)
        , allow_border_ratio(-1.0f)
        , bbox_size_add_one(true)
        , read_width_scale(1.0f)
        , read_height_scale(1.0f)
        , read_height_offset(0)
        , min_size_h(2.0f)
        , min_size_w(2.0f)
        , min_size_mode(DetectionOutputSSD_HEIGHT_AND_WIDTH)
        , gen_anchor_param()
        , nms_param()
        , kpts_param()
        , atrs_param()
        , ftrs_param()
        , spmp_param()
        , cam3d_param()
        , has_param(false)
    {}

    DetectionOutputSSDParam(std::vector<float> threshold_in,
                            std::vector<int> class_indexes_in,
                            std::vector<float> heat_map_a_in,
                            std::vector<float> heat_map_b_in,
                            std::vector<float> proposal_min_sqrt_area_in,
                            std::vector<float> proposal_max_sqrt_area_in,
                            bool refine_out_of_map_bbox_in = false,
                            int channel_per_scale_in = 5,
                            std::string class_name_list_in = "",
                            int num_class_in = 1,
                            float threshold_objectness_in = 0.f,
                            bool bg_as_one_of_softmax_in = false,
                            bool use_target_type_rcnn_in = true,
                            float im_width_in = 0,
                            float im_height_in = 0,
                            bool rpn_proposal_output_score_in = false,
                            bool regress_agnostic_in = true,
                            float allow_border_in = -1.0,
                            float allow_border_ratio_in = -1.0,
                            bool bbox_size_add_one_in = true,
                            float read_width_scale_in = 1.0f,
                            float read_height_scale_in = 1.0f,
                            int read_height_offset_in = 0,
                            float min_size_h_in = 2.0f,
                            float min_size_w_in = 2.0f,
                            DetectionOutputSSD_MIN_SIZE_MODE min_size_mode_in =
                                DetectionOutputSSD_HEIGHT_AND_WIDTH,
                            GenerateAnchorParam<TargetType> gen_anchor_param_in = GenerateAnchorParam<TargetType>(),
                            NMSSSDParam<TargetType> nms_param_in = NMSSSDParam<TargetType>(),
                            KPTSParam<TargetType> kpts_param_in = KPTSParam<TargetType>(),
                            ATRSParam<TargetType> atrs_param_in = ATRSParam<TargetType>(),
                            FTRSParam<TargetType> ftrs_param_in = FTRSParam<TargetType>(),
                            SPMPParam<TargetType> spmp_param_in = SPMPParam<TargetType>(),
                            Cam3dParam<TargetType> cam3d_param_in = Cam3dParam<TargetType>())

        : threshold(threshold_in)
        , class_indexes(class_indexes_in)
        , heat_map_a(heat_map_a_in)
        , heat_map_b(heat_map_b_in)
        , proposal_min_sqrt_area(proposal_min_sqrt_area_in)
        , proposal_max_sqrt_area(proposal_max_sqrt_area_in)
        , channel_per_scale(channel_per_scale_in)
        , class_name_list(class_name_list_in)
        , num_class(num_class_in)
        , refine_out_of_map_bbox(refine_out_of_map_bbox_in)
        , threshold_objectness(threshold_objectness_in)
        , bg_as_one_of_softmax(bg_as_one_of_softmax_in)
        , use_target_type_rcnn(use_target_type_rcnn_in)
        , im_width(im_width_in)
        , im_height(im_height_in)
        , rpn_proposal_output_score(rpn_proposal_output_score_in)
        , regress_agnostic(regress_agnostic_in)
        , allow_border(allow_border_in)
        , allow_border_ratio(allow_border_ratio_in)
        , bbox_size_add_one(bbox_size_add_one_in)
        , read_width_scale(read_width_scale_in)
        , read_height_scale(read_height_scale_in)
        , read_height_offset(read_height_offset_in)
        , min_size_h(min_size_h_in)
        , min_size_w(min_size_w_in)
        , min_size_mode(min_size_mode_in)
        , gen_anchor_param(gen_anchor_param_in)
        , nms_param(nms_param_in)
        , kpts_param(kpts_param_in)
        , atrs_param(atrs_param_in)
        , ftrs_param(ftrs_param_in)
        , spmp_param(spmp_param_in)
        , cam3d_param(cam3d_param_in)
        , has_param(true)
    {}

    DetectionOutputSSDParam(const DetectionOutputSSDParam& right)
        : threshold(right.threshold)
        , class_indexes(right.class_indexes)
        , heat_map_a(right.heat_map_a)
        , heat_map_b(right.heat_map_b)
        , proposal_min_sqrt_area(right.proposal_min_sqrt_area)
        , proposal_max_sqrt_area(right.proposal_max_sqrt_area)
        , channel_per_scale(right.channel_per_scale)
        , class_name_list(right.class_name_list)
        , num_class(right.num_class)
        , refine_out_of_map_bbox(right.refine_out_of_map_bbox)
        , threshold_objectness(right.threshold_objectness)
        , bg_as_one_of_softmax(right.bg_as_one_of_softmax)
        , use_target_type_rcnn(right.use_target_type_rcnn)
        , im_width(right.im_width)
        , im_height(right.im_height)
        , rpn_proposal_output_score(right.rpn_proposal_output_score)
        , regress_agnostic(right.regress_agnostic)
        , allow_border(right.allow_border)
        , allow_border_ratio(right.allow_border_ratio)
        , bbox_size_add_one(right.bbox_size_add_one)
        , read_width_scale(right.read_width_scale)
        , read_height_scale(right.read_height_scale)
        , read_height_offset(right.read_height_offset)
        , min_size_h(right.min_size_h)
        , min_size_w(right.min_size_w)
        , min_size_mode(right.min_size_mode)
        , gen_anchor_param(right.gen_anchor_param)
        , nms_param(right.nms_param)
        , kpts_param(right.kpts_param)
        , atrs_param(right.atrs_param)
        , ftrs_param(right.ftrs_param)
        , spmp_param(right.spmp_param)
        , cam3d_param(right.cam3d_param)
        , has_param(right.has_param)
    {}

    ~DetectionOutputSSDParam() {}

    DetectionOutputSSDParam& operator=(
        const DetectionOutputSSDParam& right) {
        copy_vectors(threshold, right.threshold);
        copy_vectors(class_indexes, right.class_indexes);
        copy_vectors(heat_map_a, right.heat_map_a);
        copy_vectors(heat_map_b, right.heat_map_b);
        copy_vectors(proposal_min_sqrt_area, right.proposal_min_sqrt_area);
        copy_vectors(proposal_max_sqrt_area, right.proposal_max_sqrt_area);
        channel_per_scale = right.channel_per_scale;
        class_name_list = right.class_name_list;
        num_class = right.num_class;
        refine_out_of_map_bbox = right.refine_out_of_map_bbox;
        threshold_objectness = right.threshold_objectness;
        bg_as_one_of_softmax = right.bg_as_one_of_softmax;
        use_target_type_rcnn = right.use_target_type_rcnn;
        im_width = right.im_width;
        im_height = right.im_height;
        rpn_proposal_output_score = right.rpn_proposal_output_score;
        regress_agnostic = right.regress_agnostic;
        allow_border = right.allow_border;
        allow_border_ratio = right.allow_border_ratio;
        bbox_size_add_one = right.bbox_size_add_one;
        read_width_scale = right.read_width_scale;
        read_height_scale = right.read_height_scale;
        read_height_offset = right.read_height_offset;
        min_size_h = right.min_size_h;
        min_size_w = right.min_size_w;
        min_size_mode = right.min_size_mode;
        gen_anchor_param = right.gen_anchor_param;
        nms_param = right.nms_param;
        kpts_param = right.kpts_param;
        atrs_param = right.atrs_param;
        ftrs_param = right.ftrs_param;
        spmp_param = right.spmp_param;
        cam3d_param = right.cam3d_param;
        has_param = right.has_param;
        return *this;
    }

    bool operator==(const DetectionOutputSSDParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (compare_vectors(threshold, right.threshold));
        comp_eq &= comp_eq && (compare_vectors(class_indexes, right.class_indexes));
        comp_eq &= comp_eq && (compare_vectors(heat_map_a, right.heat_map_a));
        comp_eq &= comp_eq && (compare_vectors(heat_map_b, right.heat_map_b));
        comp_eq &= comp_eq && (compare_vectors(proposal_min_sqrt_area, right.proposal_min_sqrt_area));
        comp_eq &= comp_eq && (compare_vectors(proposal_max_sqrt_area, right.proposal_max_sqrt_area));
        comp_eq &= comp_eq && (channel_per_scale == right.channel_per_scale);
        comp_eq &= comp_eq && (class_name_list == right.class_name_list);
        comp_eq &= comp_eq && (num_class == right.num_class);
        comp_eq &= comp_eq && (refine_out_of_map_bbox == right.refine_out_of_map_bbox);
        comp_eq &= comp_eq && (threshold_objectness == right.threshold_objectness);
        comp_eq &= comp_eq && (bg_as_one_of_softmax == right.bg_as_one_of_softmax);
        comp_eq &= comp_eq && (use_target_type_rcnn == right.use_target_type_rcnn);
        comp_eq &= comp_eq && (im_width == right.im_width);
        comp_eq &= comp_eq && (im_height == right.im_height);
        comp_eq &= comp_eq && (rpn_proposal_output_score == right.rpn_proposal_output_score);
        comp_eq &= comp_eq && (regress_agnostic == right.regress_agnostic);
        comp_eq &= comp_eq && (allow_border == right.allow_border);
        comp_eq &= comp_eq && (allow_border_ratio == right.allow_border_ratio);
        comp_eq &= comp_eq && (bbox_size_add_one == right.bbox_size_add_one);
        comp_eq &= comp_eq && (read_width_scale == right.read_width_scale);
        comp_eq &= comp_eq && (read_height_scale == right.read_height_scale);
        comp_eq &= comp_eq && (read_height_offset == right.read_height_offset);
        comp_eq &= comp_eq && (min_size_h == right.min_size_h);
        comp_eq &= comp_eq && (min_size_w == right.min_size_w);
        comp_eq &= comp_eq && (min_size_mode == right.min_size_mode);
        comp_eq &= comp_eq && (gen_anchor_param == right.gen_anchor_param);
        comp_eq &= comp_eq && (nms_param == right.nms_param);
        comp_eq &= comp_eq && (kpts_param == right.kpts_param);
        comp_eq &= comp_eq && (atrs_param == right.atrs_param);
        comp_eq &= comp_eq && (ftrs_param == right.ftrs_param);
        comp_eq &= comp_eq && (spmp_param == right.spmp_param);
        comp_eq &= comp_eq && (cam3d_param == right.cam3d_param);
        return comp_eq;
    }

    std::vector<float> threshold;
    std::vector<int> class_indexes;
    std::vector<float> heat_map_a;
    std::vector<float> heat_map_b;
    std::vector<float> proposal_min_sqrt_area;
    std::vector<float> proposal_max_sqrt_area;

    int channel_per_scale;
    std::string class_name_list;
    int num_class;
    bool refine_out_of_map_bbox;
    float threshold_objectness;
    bool bg_as_one_of_softmax;
    bool use_target_type_rcnn;
    float im_width;
    float im_height;
    bool rpn_proposal_output_score;
    bool regress_agnostic;
    float allow_border;
    float allow_border_ratio;
    bool bbox_size_add_one;
    float read_width_scale;
    float read_height_scale;
    int read_height_offset;
    float min_size_h;
    float min_size_w;

    DetectionOutputSSD_MIN_SIZE_MODE min_size_mode;
    GenerateAnchorParam<TargetType> gen_anchor_param;
    NMSSSDParam<TargetType> nms_param;
    KPTSParam<TargetType> kpts_param;
    ATRSParam<TargetType> atrs_param;
    FTRSParam<TargetType> ftrs_param;
    SPMPParam<TargetType> spmp_param;
    Cam3dParam<TargetType> cam3d_param;
    bool has_param;

};

template <typename TargetType>
struct ProposalParam {
    ProposalParam()
        : bbox_reg_param()
        , detection_output_ssd_param()
        , reg_param()
    {}

    ProposalParam(BBoxRegParam<TargetType> bbox_reg_param_in,
                  DetectionOutputSSDParam<TargetType> detection_output_ssd_param_in,
                  RegParam<TargetType> reg_param_in = RegParam<TargetType>())
        : bbox_reg_param(bbox_reg_param_in)
        , detection_output_ssd_param(detection_output_ssd_param_in)
        , reg_param(reg_param_in)
    {}

    ProposalParam(const ProposalParam& right)
        : bbox_reg_param(right.bbox_reg_param)
        , detection_output_ssd_param(right.detection_output_ssd_param)
        , reg_param(right.reg_param)
    {}

    ~ProposalParam() {}

    ProposalParam& operator=(const ProposalParam& right) {
        bbox_reg_param = right.bbox_reg_param;
        detection_output_ssd_param = right.detection_output_ssd_param;
        reg_param = right.reg_param;
        return *this;
    }
    bool operator==(const ProposalParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (bbox_reg_param == right.bbox_reg_param);
        comp_eq &= comp_eq && (detection_output_ssd_param == right.detection_output_ssd_param);
        comp_eq &= comp_eq && (reg_param == right.reg_param);
        return comp_eq;
    }

    BBoxRegParam<TargetType> bbox_reg_param;
    DetectionOutputSSDParam<TargetType> detection_output_ssd_param;
    RegParam<TargetType> reg_param;
};

template <typename TargetType>
struct ProposalImgScaleToCamCoordsParam {
    ProposalImgScaleToCamCoordsParam()
        : num_class(-1)
        , sub_class_num_class()
        , sub_class_bottom_idx()
        , cls_ids_zero_size3d_w()
        , cls_ids_zero_size3d_l()
        , cls_ids_zero_orien3d()
        , ctr_2d_means()
        , ctr_2d_stds()
        , prj_h_means()
        , prj_h_stds()
        , real_h_means()
        , real_h_stds()
        , real_w_means()
        , real_w_stds()
        , real_l_means()
        , real_l_stds()
        , sin_means()
        , sin_stds()
        , cos_means()
        , cos_stds()
        , real_h_means_as_whole()
        , real_h_stds_as_whole()
        , prj_h_norm_type(ProposalImgScaleToCamCoords_NormType_HEIGHT_LOG)
        , has_size3d_and_orien3d(false)
        , orien_type(ProposalImgScaleToCamCoords_OrienType_PI2)
        , cmp_pts_corner_3d(false)
        , cmp_pts_corner_2d(false)
        , cam_info_idx_st_in_im_info(0)
        , im_width_scale(1.0f)
        , im_height_scale(1.0f)
        , cords_offset_x(0.0f)
        , cords_offset_y(0.0f)
        , bbox_size_add_one(true)
        , rotate_coords_by_pitch(false)
        , with_trunc_ratio(false)
        , regress_ph_rh_as_whole(false)
        /*, refine_coords_by_bbox(false)
        , refine_min_dist(40.0f)
        , refine_dist_for_height_ratio_one(80.0f)
        , max_3d2d_height_ratio_for_min_dist(1.2f)*/

    {}

    ProposalImgScaleToCamCoordsParam(int num_class_in,
            std::vector<int> sub_class_num_class_in,
            std::vector<int> sub_class_bottom_idx_in,
            std::vector<int> cls_ids_zero_size3d_w_in,
            std::vector<int> cls_ids_zero_size3d_l_in,
            std::vector<int> cls_ids_zero_orien3d_in,
            std::vector<float> ctr_2d_means_in,
            std::vector<float> ctr_2d_stds_in,
            std::vector<float> prj_h_means_in,
            std::vector<float> prj_h_stds_in,
            std::vector<float> real_h_means_in,
            std::vector<float> real_h_stds_in,
            std::vector<float> real_w_means_in,
            std::vector<float> real_w_stds_in,
            std::vector<float> real_l_means_in,
            std::vector<float> real_l_stds_in,
            std::vector<float> sin_means_in,
            std::vector<float> sin_stds_in,
            std::vector<float> cos_means_in,
            std::vector<float> cos_stds_in,
            std::vector<float> real_h_means_as_whole_in,
            std::vector<float> real_h_std_as_whole_in,
            ProposalImgScaleToCamCoords_NormType prj_h_norm_type_in =
                    ProposalImgScaleToCamCoords_NormType_HEIGHT_LOG,
            bool has_size3d_and_orien3d_in = false,
            ProposalImgScaleToCamCoords_OrienType orien_type_in =
                    ProposalImgScaleToCamCoords_OrienType_PI2,
            bool cmp_pts_corner_3d_in = false,
            bool cmp_pts_corner_2d_in = false,
            int cam_info_idx_st_in_im_info_in = 0,
            float im_width_scale_in = 1.0f,
            float im_height_scale_in = 1.0f,
            float cords_offset_x_in = 0.0f,
            float cords_offset_y_in = 0.0f,
            bool bbox_size_add_one_in = true,
            bool rotate_coords_by_pitch_in = false,
            bool with_trunc_ratio_in = false,
            bool regress_ph_rh_as_whole_in = false
            /*bool refine_coords_by_bbox_in = false,
            float refine_min_dist_in = 40.0f,
            float refine_dist_for_height_ratio_one_in = 80.0f,
            float max_3d2d_height_ratio_for_min_dist_in = 1.2f,*/ )

        : num_class(num_class_in)
        , sub_class_num_class(sub_class_num_class_in)
        , sub_class_bottom_idx(sub_class_bottom_idx_in)
        , cls_ids_zero_size3d_w(cls_ids_zero_size3d_w_in)
        , cls_ids_zero_size3d_l(cls_ids_zero_size3d_l_in)
        , cls_ids_zero_orien3d(cls_ids_zero_orien3d_in)
        , ctr_2d_means(ctr_2d_means_in)
        , ctr_2d_stds(ctr_2d_stds_in)
        , prj_h_means(prj_h_means_in)
        , prj_h_stds(prj_h_stds_in)
        , real_h_means(real_h_means_in)
        , real_h_stds(real_h_stds_in)
        , real_w_means(real_w_means_in)
        , real_w_stds(real_w_stds_in)
        , real_l_means(real_l_means_in)
        , real_l_stds(real_l_stds_in)
        , sin_means(sin_means_in)
        , sin_stds(sin_stds_in)
        , cos_means(cos_means_in)
        , cos_stds(cos_stds_in)
        , real_h_means_as_whole(real_h_means_as_whole_in)
        , real_h_stds_as_whole(real_h_std_as_whole_in)
        , prj_h_norm_type(prj_h_norm_type_in)
        , has_size3d_and_orien3d(has_size3d_and_orien3d_in)
        , orien_type(orien_type_in)
        , cmp_pts_corner_3d(cmp_pts_corner_3d_in)
        , cmp_pts_corner_2d(cmp_pts_corner_2d_in)
        , cam_info_idx_st_in_im_info(cam_info_idx_st_in_im_info_in)
        , im_width_scale(im_width_scale_in)
        , im_height_scale(im_height_scale_in)
        , cords_offset_x(cords_offset_x_in)
        , cords_offset_y(cords_offset_y_in)
        , bbox_size_add_one(bbox_size_add_one_in)
        , rotate_coords_by_pitch(rotate_coords_by_pitch_in)
        , with_trunc_ratio(with_trunc_ratio_in)
        , regress_ph_rh_as_whole(regress_ph_rh_as_whole_in)
        /*, refine_coords_by_bbox(refine_coords_by_bbox_in)
        , refine_min_dist(refine_min_dist_in)
        , refine_dist_for_height_ratio_one(refine_dist_for_height_ratio_one_in)
        , max_3d2d_height_ratio_for_min_dist(max_3d2d_height_ratio_for_min_dist_in) */

    {}
    ProposalImgScaleToCamCoordsParam(const ProposalImgScaleToCamCoordsParam& right)
        : num_class(right.num_class)
        , sub_class_num_class(right.sub_class_num_class)
        , sub_class_bottom_idx(right.sub_class_bottom_idx)
        , cls_ids_zero_size3d_w(right.cls_ids_zero_size3d_w)
        , cls_ids_zero_size3d_l(right.cls_ids_zero_size3d_l)
        , cls_ids_zero_orien3d(right.cls_ids_zero_orien3d)
        , ctr_2d_means(right.ctr_2d_means)
        , ctr_2d_stds(right.ctr_2d_stds)
        , prj_h_means(right.prj_h_means)
        , prj_h_stds(right.prj_h_stds)
        , real_h_means(right.real_h_means)
        , real_h_stds(right.real_h_stds)
        , real_w_means(right.real_w_means)
        , real_w_stds(right.real_w_stds)
        , real_l_means(right.real_l_means)
        , real_l_stds(right.real_l_stds)
        , sin_means(right.sin_means)
        , sin_stds(right.sin_stds)
        , cos_means(right.cos_means)
        , cos_stds(right.cos_stds)
        , real_h_means_as_whole(right.real_h_means_as_whole)
        , real_h_stds_as_whole(right.real_h_std_as_whole)
        , prj_h_norm_type(right.prj_h_norm_type)
        , has_size3d_and_orien3d(right.has_size3d_and_orien3d)
        , orien_type(right.orien_type)
        , cmp_pts_corner_3d(right.cmp_pts_corner_3d)
        , cmp_pts_corner_2d(right.cmp_pts_corner_2d)
        , cam_info_idx_st_in_im_info(right.cam_info_idx_st_in_im_info)
        , im_width_scale(right.im_width_scale)
        , im_height_scale(right.im_height_scale)
        , cords_offset_x(right.cords_offset_x)
        , cords_offset_y(right.cords_offset_y)
        , bbox_size_add_one(right.bbox_size_add_one)
        , rotate_coords_by_pitch(right.rotate_coords_by_pitch)
        , with_trunc_ratio(right.with_trunc_ratio)
        , regress_ph_rh_as_whole(right.regress_ph_rh_as_whole)
        /*, refine_coords_by_bbox(right.refine_coords_by_bbox)
        , refine_min_dist(right.refine_min_dist)
        , refine_dist_for_height_ratio_one(right.refine_dist_for_height_ratio_one)
        , max_3d2d_height_ratio_for_min_dist(right.max_3d2d_height_ratio_for_min_dist) */
    {}

    ~ProposalImgScaleToCamCoordsParam() {}

    ProposalImgScaleToCamCoordsParam& operator=(const ProposalImgScaleToCamCoordsParam& right) {
        num_class = right.num_class;
        copy_vectors(sub_class_num_class, right.sub_class_num_class);
        copy_vectors(sub_class_bottom_idx, right.sub_class_bottom_idx);
        copy_vectors(cls_ids_zero_size3d_w, right.cls_ids_zero_size3d_w);
        copy_vectors(cls_ids_zero_size3d_l, right.cls_ids_zero_size3d_l);
        copy_vectors(cls_ids_zero_orien3d, right.cls_ids_zero_orien3d);
        copy_vectors(ctr_2d_means, right.ctr_2d_means);
        copy_vectors(ctr_2d_stds, right.ctr_2d_stds);
        copy_vectors(prj_h_means, right.prj_h_means);
        copy_vectors(prj_h_stds, right.prj_h_stds);
        copy_vectors(real_h_means, right.real_h_means);
        copy_vectors(real_h_stds, right.real_h_stds);
        copy_vectors(real_w_means, right.real_w_means);
        copy_vectors(real_w_stds, right.real_w_stds);
        copy_vectors(real_l_means, right.real_l_means);
        copy_vectors(real_l_stds, right.real_l_stds);
        copy_vectors(sin_means, right.sin_means);
        copy_vectors(sin_stds, right.sin_stds);
        copy_vectors(cos_means, right.cos_means);
        copy_vectors(cos_stds, right.cos_stds);
        copy_vectors(real_h_means_as_whole, right.real_h_means_as_whole);
        copy_vectors(real_h_stds_as_whole, right.real_h_stds_as_whole);
        prj_h_norm_type = right.prj_h_norm_type;
        has_size3d_and_orien3d = right.has_size3d_and_orien3d;
        orien_type = right.orien_type;
        cmp_pts_corner_3d = right.cmp_pts_corner_3d;
        cmp_pts_corner_2d = right.cmp_pts_corner_2d;
        cam_info_idx_st_in_im_info = right.cam_info_idx_st_in_im_info;
        im_width_scale = right.im_width_scale;
        im_height_scale = right.im_height_scale;
        cords_offset_x = right.cords_offset_x;
        cords_offset_y = right.cords_offset_y;
        bbox_size_add_one = right.bbox_size_add_one;
        rotate_coords_by_pitch = right.rotate_coords_by_pitch;
        with_trunc_ratio = right.with_trunc_ratio;
        regress_ph_rh_as_whole = right.regress_ph_rh_as_whole;

//        refine_coords_by_bbox = right.refine_coords_by_bbox;
//        refine_min_dist = right.refine_min_dist;
//        refine_dist_for_height_ratio_one = right.refine_dist_for_height_ratio_one;
//        max_3d2d_height_ratio_for_min_dist = right.max_3d2d_height_ratio_for_min_dist;
        return *this;
    }
    bool operator==(const ProposalImgScaleToCamCoordsParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (num_class = right.num_class);
        comp_eq &= comp_eq && (compare_vectors(sub_class_num_class, right.sub_class_num_class));
        comp_eq &= comp_eq && (compare_vectors(sub_class_bottom_idx, right.sub_class_bottom_idx));
        comp_eq &= comp_eq && (compare_vectors(cls_ids_zero_size3d_w, right.cls_ids_zero_size3d_w));
        comp_eq &= comp_eq && (compare_vectors(cls_ids_zero_size3d_l, right.cls_ids_zero_size3d_l));
        comp_eq &= comp_eq && (compare_vectors(cls_ids_zero_orien3d, right.cls_ids_zero_orien3d));
        comp_eq &= comp_eq && (compare_vectors(ctr_2d_means, right.ctr_2d_means));
        comp_eq &= comp_eq && (compare_vectors(ctr_2d_stds, right.ctr_2d_stds));
        comp_eq &= comp_eq && (compare_vectors(prj_h_means, right.prj_h_means));
        comp_eq &= comp_eq && (compare_vectors(prj_h_stds, right.prj_h_stds));
        comp_eq &= comp_eq && (compare_vectors(real_h_means, right.real_h_means));
        comp_eq &= comp_eq && (compare_vectors(real_h_stds, right.real_h_stds));
        comp_eq &= comp_eq && (compare_vectors(real_w_means, right.real_w_means));
        comp_eq &= comp_eq && (compare_vectors(real_w_stds, right.real_w_stds));
        comp_eq &= comp_eq && (compare_vectors(real_l_means, right.real_l_means));
        comp_eq &= comp_eq && (compare_vectors(real_l_stds, right.real_l_stds));
        comp_eq &= comp_eq && (compare_vectors(sin_means, right.sin_means));
        comp_eq &= comp_eq && (compare_vectors(sin_stds, right.sin_stds));
        comp_eq &= comp_eq && (compare_vectors(cos_means, right.cos_means));
        comp_eq &= comp_eq && (compare_vectors(cos_stds, right.cos_stds));
        comp_eq &= comp_eq && (compare_vectors(real_h_means_as_whole, right.real_h_means_as_whole));
        comp_eq &= comp_eq && (compare_vectors(real_h_stds_as_whole, right.real_h_stds_as_whole));
        comp_eq &= comp_eq && (prj_h_norm_type == right.prj_h_norm_type);
        comp_eq &= comp_eq && (has_size3d_and_orien3d == right.has_size3d_and_orien3d);
        comp_eq &= comp_eq && (orien_type == right.orien_type);
        comp_eq &= comp_eq && (cmp_pts_corner_3d == right.cmp_pts_corner_3d);
        comp_eq &= comp_eq && (cmp_pts_corner_2d == right.cmp_pts_corner_2d);
        comp_eq &= comp_eq && (cam_info_idx_st_in_im_info == right.cam_info_idx_st_in_im_info);
        comp_eq &= comp_eq && (im_width_scale == right.im_width_scale);
        comp_eq &= comp_eq && (im_height_scale == right.im_height_scale);
        comp_eq &= comp_eq && (cords_offset_x == right.cords_offset_x);
        comp_eq &= comp_eq && (cords_offset_y == right.cords_offset_y);
        comp_eq &= comp_eq && (bbox_size_add_one == right.bbox_size_add_one);
        comp_eq &= comp_eq && (rotate_coords_by_pitch == right.rotate_coords_by_pitch);
        comp_eq &= comp_eq && (with_trunc_ratio == right.with_trunc_ratio);
        comp_eq &= comp_eq && (regress_ph_rh_as_whole == right.regress_ph_rh_as_whole);
//        comp_eq &= comp_eq && (refine_coords_by_bbox == right.refine_coords_by_bbox);
//        comp_eq &= comp_eq && (refine_min_dist == right.refine_min_dist);
//        comp_eq &= comp_eq && (refine_dist_for_height_ratio_one == right.refine_dist_for_height_ratio_one);
//        comp_eq &= comp_eq
//                   && (max_3d2d_height_ratio_for_min_dist == right.max_3d2d_height_ratio_for_min_dist);

        return comp_eq;
    }

    int num_class;
    std::vector<int> sub_class_num_class;
    std::vector<int> sub_class_bottom_idx;
    std::vector<int> cls_ids_zero_size3d_w;
    std::vector<int> cls_ids_zero_size3d_l;
    std::vector<int> cls_ids_zero_orien3d;
    std::vector<float> ctr_2d_means;
    std::vector<float> ctr_2d_stds;
    std::vector<float> prj_h_means;
    std::vector<float> prj_h_stds;
    std::vector<float> real_h_means;
    std::vector<float> real_h_stds;
    std::vector<float> real_w_means;
    std::vector<float> real_w_stds;
    std::vector<float> real_l_means;
    std::vector<float> real_l_stds;
    std::vector<float> sin_means;
    std::vector<float> sin_stds;
    std::vector<float> cos_means;
    std::vector<float> cos_stds;
    std::vector<float> real_h_means_as_whole;
    std::vector<float> real_h_stds_as_whole;

    ProposalImgScaleToCamCoords_NormType prj_h_norm_type;
    bool has_size3d_and_orien3d;
    ProposalImgScaleToCamCoords_OrienType orien_type;
    bool cmp_pts_corner_3d;
    bool cmp_pts_corner_2d;
    int cam_info_idx_st_in_im_info;
    float im_width_scale;
    float im_height_scale;
    float cords_offset_x;
    float cords_offset_y;
    bool bbox_size_add_one;
    bool rotate_coords_by_pitch;
    bool with_trunc_ratio;
    bool regress_ph_rh_as_whole;
//    bool refine_coords_by_bbox;
//    float refine_min_dist;
//    float refine_dist_for_height_ratio_one;
//    float max_3d2d_height_ratio_for_min_dist;

};

template <typename TargetType>
struct RoisAnchorFeatureParam {
    RoisAnchorFeatureParam()
        : has_param(false)
        , min_anchor_size(8)
        , num_anchor_scales(15)
        , anchor_scale_pow_base(1.4142f)
        , anchor_wph_ratios()
        , num_top_iou_anchor(10)
        , min_num_top_iou_anchor(3)
        , iou_thr(0.5f)
        , ft_ratio_h(false)
        , ft_ratio_w(false)
        , ft_log_ratio_h(false)
        , ft_log_ratio_w(false)
        , bbox_size_add_one(true)
    {}
    RoisAnchorFeatureParam(
            std::vector<float> anchor_wph_ratios_in,
            float min_anchor_size_in = 8.f,
            int num_anchor_scales_in = 15,
            float anchor_scale_pow_base_in = 1.4142f,
            int num_top_iou_anchor_in = 10,
            int min_num_top_iou_anchor_in = 3,
            float iou_thr_in = 0.5f,
            bool ft_ratio_h_in = false,
            bool ft_ratio_w_in = false,
            bool ft_log_ratio_h_in = false,
            bool ft_log_ratio_w_in = false,
            bool bbox_size_add_one_in = true)
        : anchor_wph_ratios(anchor_wph_ratios_in)
        , min_anchor_size(min_anchor_size_in)
        , num_anchor_scales(num_anchor_scales_in)
        , anchor_scale_pow_base(anchor_scale_pow_base_in)
        , num_top_iou_anchor(num_top_iou_anchor_in)
        , min_num_top_iou_anchor(min_num_top_iou_anchor_in)
        , iou_thr(iou_thr_in)
        , ft_ratio_h(ft_ratio_h_in)
        , ft_ratio_w(ft_ratio_w_in)
        , ft_log_ratio_h(ft_log_ratio_h_in)
        , ft_log_ratio_w(ft_log_ratio_w_in)
        , bbox_size_add_one(bbox_size_add_one_in)
        , has_param(true)
    {}
    RoisAnchorFeatureParam(const RoisAnchorFeatureParam& right)
            : anchor_wph_ratios(right.anchor_wph_ratios)
            , min_anchor_size(right.min_anchor_size)
            , num_anchor_scales(right.num_anchor_scales)
            , anchor_scale_pow_base(right.anchor_scale_pow_base)
            , num_top_iou_anchor(right.num_top_iou_anchor)
            , min_num_top_iou_anchor(right.min_num_top_iou_anchor)
            , iou_thr(right.iou_thr)
            , ft_ratio_h(right.ft_ratio_h)
            , ft_ratio_w(right.ft_ratio_w)
            , ft_log_ratio_h(right.ft_log_ratio_h)
            , ft_log_ratio_w(right.ft_log_ratio_w)
            , bbox_size_add_one(right.bbox_size_add_one)
            , has_param(right.has_param)
    {}
    ~RoisAnchorFeatureParam() {}
    RoisAnchorFeatureParam& operator=(const RoisAnchorFeatureParam& right) {
        copy_vectors(anchor_wph_ratios, right.anchor_wph_ratios);
        min_anchor_size = right.min_anchor_size;
        num_anchor_scales = right.num_anchor_scales;
        anchor_scale_pow_base = right.anchor_scale_pow_base;
        num_top_iou_anchor = right.num_top_iou_anchor;
        min_num_top_iou_anchor = right.min_num_top_iou_anchor;
        iou_thr = right.iou_thr;
        ft_ratio_h = right.ft_ratio_h;
        ft_ratio_w = right.ft_ratio_w;
        ft_log_ratio_h = right.ft_log_ratio_h;
        ft_log_ratio_w = right.ft_log_ratio_w;
        bbox_size_add_one = right.bbox_size_add_one;
        has_param = right.has_param;
        return *this;
    }
    bool operator==(const RoisAnchorFeatureParam& right) {
        bool comp_eq = true;
        comp_eq &= comp_eq && (compare_vectors(anchor_wph_ratios, right.anchor_wph_ratios));
        comp_eq &= comp_eq && (min_anchor_size == right.min_anchor_size);
        comp_eq &= comp_eq && (num_anchor_scales == right.num_anchor_scales);
        comp_eq &= comp_eq && (anchor_scale_pow_base == right.anchor_scale_pow_base);
        comp_eq &= comp_eq && (num_top_iou_anchor == right.num_top_iou_anchor);
        comp_eq &= comp_eq && (min_num_top_iou_anchor == right.min_num_top_iou_anchor);
        comp_eq &= comp_eq && (iou_thr == right.iou_thr);
        comp_eq &= comp_eq && (ft_ratio_h == right.ft_ratio_h);
        comp_eq &= comp_eq && (ft_ratio_w == right.ft_ratio_w);
        comp_eq &= comp_eq && (ft_log_ratio_h == right.ft_log_ratio_h);
        comp_eq &= comp_eq && (ft_log_ratio_w == right.ft_log_ratio_w);
        comp_eq &= comp_eq && (bbox_size_add_one == right.bbox_size_add_one);
        comp_eq &= comp_eq && (has_param == right.has_param);
        return comp_eq;
    }

    bool has_param{false};
    float min_anchor_size{8};
    int num_anchor_scales{15};
    float anchor_scale_pow_base{1.4142};
    std::vector<float> anchor_wph_ratios;
    int num_top_iou_anchor{10};
    int min_num_top_iou_anchor{3};
    float iou_thr{0.5};
    bool ft_ratio_h{false};
    bool ft_ratio_w{false};
    bool ft_log_ratio_h{false};
    bool ft_log_ratio_w{false};
    bool bbox_size_add_one{true};
};

}
}
#endif //ANAKIN_SABER_SP_PARAM_H
