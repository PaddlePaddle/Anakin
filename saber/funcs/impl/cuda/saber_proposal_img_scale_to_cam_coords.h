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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_H

#include "saber/funcs/impl/impl_proposal_img_scale_to_cam_coords.h"
#include "anakin_config.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include <vector>
#include <set>

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberProposalImgScaleToCamCoords<NV, OpDtype>:\
    public ImplBase<
        NV, OpDtype,
        ProposalImgScaleToCamCoordsParam<NV> > {

public:

    SaberProposalImgScaleToCamCoords()
            : _rois_boxes_data_host_tensor(NULL)
            , _im_info_data_host_tensor(NULL)
            , _cam2d_data_host_tensor(NULL)
            , _prj_h_pred_data_host_tensor(NULL)
            , _real_h_pred_data_host_tensor(NULL)
            , _size3d_h_pred_data_host_tensor(NULL)
            , _size3d_w_pred_data_host_tensor(NULL)
            , _size3d_l_pred_data_host_tensor(NULL)
            , _orien3d_sin_pred_data_host_tensor(NULL)
            , _orien3d_cos_pred_data_host_tensor(NULL)
            , _trunc_ratio_pred_data_host_tensor(NULL)
            , _img_info_data_host_tensor(NULL)
            , _cam_coords_data_host_tensor(NULL)
            , _has_inited(false)
    {}

    ~SaberProposalImgScaleToCamCoords() {
        if (_rois_boxes_data_host_tensor != NULL) {
            delete _rois_boxes_data_host_tensor;
        }

        if (_im_info_data_host_tensor != NULL) {
            delete _im_info_data_host_tensor;
        }

        if (_cam2d_data_host_tensor != NULL) {
            delete _cam2d_data_host_tensor;
        }

        if (_prj_h_pred_data_host_tensor != NULL) {
            delete _prj_h_pred_data_host_tensor;
        }

        if (_real_h_pred_data_host_tensor != NULL) {
            delete _real_h_pred_data_host_tensor;
        }

        if (_size3d_h_pred_data_host_tensor != NULL) {
            delete _size3d_h_pred_data_host_tensor;
        }

        if (_size3d_w_pred_data_host_tensor != NULL) {
            delete _size3d_w_pred_data_host_tensor;
        }

        if (_size3d_l_pred_data_host_tensor != NULL) {
            delete _size3d_l_pred_data_host_tensor;
        }

        if (_orien3d_sin_pred_data_host_tensor != NULL) {
            delete _orien3d_sin_pred_data_host_tensor;
        }

        if (_orien3d_cos_pred_data_host_tensor != NULL) {
            delete _orien3d_cos_pred_data_host_tensor;
        }

        if (_trunc_ratio_pred_data_host_tensor != NULL) {
            delete _trunc_ratio_pred_data_host_tensor;
        }

        if (_img_info_data_host_tensor != NULL) {
            delete _img_info_data_host_tensor;
        }

        if (_cam_coords_data_host_tensor != NULL) {
            delete _cam_coords_data_host_tensor;
        }

        for (int i = 0; i < _sub_class_datas_host_tensor_v.size(); ++i) {
            delete _sub_class_datas_host_tensor_v[i];
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             ProposalImgScaleToCamCoordsParam<NV> &param,
                             Context<NV> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               ProposalImgScaleToCamCoordsParam<NV> &param,
                               Context<NV> &ctx) override;

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 ProposalImgScaleToCamCoordsParam<NV> &param) override;
private:
    int num_class_;
    std::vector<int> sub_class_num_class_;
    std::vector<int> sub_class_bottom_idx_;
    std::vector<int> sub_class_num_class_pre_sum_;
    int total_sub_class_num_;
    ProposalImgScaleToCamCoords_NormType prj_h_norm_type_;
    bool has_size3d_and_orien3d_;
    // with trunc ratio
    bool with_trunc_ratio_;
    ProposalImgScaleToCamCoords_OrienType orien_type_;
    std::set<int> cls_ids_zero_size3d_w_;
    std::set<int> cls_ids_zero_size3d_l_;
    std::set<int> cls_ids_zero_orien3d_;
    bool cmp_pts_corner_3d_;
    bool cmp_pts_corner_2d_;
    int num_top_channels_;
    int size3d_h_bottom_idx_;
    int size3d_w_bottom_idx_;
    int size3d_l_bottom_idx_;
    int orien3d_sin_bottom_idx_;
    int orien3d_cos_bottom_idx_;
    int trunc_ratio_bottom_idx_;
    int cam_info_idx_st_in_im_info_;
    bool need_ctr_2d_norm_;
    std::vector<float > ctr_2d_means_;
    std::vector<float > ctr_2d_stds_;
    bool need_prj_h_norm_;
    std::vector<float > prj_h_means_;
    std::vector<float > prj_h_stds_;
    bool need_real_h_norm_;
    std::vector<float > real_h_means_;
    std::vector<float > real_h_stds_;
    bool need_real_w_norm_;
    std::vector<float > real_w_means_;
    std::vector<float > real_w_stds_;
    bool need_real_l_norm_;
    std::vector<float > real_l_means_;
    std::vector<float > real_l_stds_;
    bool need_sin_norm_;
    std::vector<float > sin_means_;
    std::vector<float > sin_stds_;
    bool need_cos_norm_;
    std::vector<float > cos_means_;
    std::vector<float > cos_stds_;
    bool has_scale_offset_info_;
    float im_width_scale_;
    float im_height_scale_;
    float cords_offset_x_;
    float cords_offset_y_;
    bool bbox_size_add_one_;
    // rotate coords by pitch
    bool rotate_coords_by_pitch_;

    // whether regress ph rh as whole
    bool regress_ph_rh_as_whole_;
    bool need_real_h_norm_dps_;
    std::vector<float> real_h_means_dps_;
    std::vector<float> real_h_stds_dps_;

    Tensor<NVHX86>* _rois_boxes_data_host_tensor;
    Tensor<NVHX86>* _im_info_data_host_tensor;
    Tensor<NVHX86>* _cam2d_data_host_tensor;
    Tensor<NVHX86>* _prj_h_pred_data_host_tensor;
    Tensor<NVHX86>* _real_h_pred_data_host_tensor;
    Tensor<NVHX86>* _size3d_h_pred_data_host_tensor;
    Tensor<NVHX86>* _size3d_w_pred_data_host_tensor;
    Tensor<NVHX86>* _size3d_l_pred_data_host_tensor;
    Tensor<NVHX86>* _orien3d_sin_pred_data_host_tensor;
    Tensor<NVHX86>* _orien3d_cos_pred_data_host_tensor;
    Tensor<NVHX86>* _trunc_ratio_pred_data_host_tensor;
    Tensor<NVHX86>* _img_info_data_host_tensor;
    std::vector<Tensor<NVHX86> *> _sub_class_datas_host_tensor_v;
    //output
    Tensor<NVHX86>* _cam_coords_data_host_tensor;
    bool _has_inited;
};

}

}


#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
