#include "saber_proposal_img_scale_to_cam_coords.h"
#include "saber/core/tensor_op.h"
#include "saber/utils.h"
namespace anakin {
namespace saber {

template <>
SaberStatus SaberProposalImgScaleToCamCoords<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ProposalImgScaleToCamCoordsParam<NV> &param,
        Context<NV> &ctx) {
    this->_ctx = &ctx;

    if (_has_inited) {
        return SaberSuccess;
    }

    _has_inited = true;
    ProposalImgScaleToCamCoordsParam<NV>& scale_to_dist_param = param;
    num_class_ = scale_to_dist_param.num_class;
    int max_bottom_idx = 4;
    std::copy(scale_to_dist_param.sub_class_num_class.begin(),
              scale_to_dist_param.sub_class_num_class.end(),
              std::back_inserter(sub_class_num_class_));
    std::copy(scale_to_dist_param.sub_class_bottom_idx.begin(),
              scale_to_dist_param.sub_class_bottom_idx.end(),
              std::back_inserter(sub_class_bottom_idx_));
    CHECK_EQ(sub_class_num_class_.size(), sub_class_bottom_idx_.size());
    CHECK_LE(sub_class_num_class_.size(), num_class_);

    for (int k = 0; k < sub_class_num_class_.size(); k++) {
        CHECK_GT(sub_class_num_class_[k], 0);

        if (sub_class_num_class_[k] > 1) {
            CHECK_GE(sub_class_bottom_idx_[k], 0);
        }

        max_bottom_idx = std::max(max_bottom_idx, sub_class_bottom_idx_[k]);
    }

    num_top_channels_ = 3;
    has_size3d_and_orien3d_ = scale_to_dist_param.has_size3d_and_orien3d;
    // with trunc ratio
    with_trunc_ratio_ = scale_to_dist_param.with_trunc_ratio;

    if (has_size3d_and_orien3d_) {
        size3d_h_bottom_idx_ = ++max_bottom_idx;
        size3d_w_bottom_idx_ = ++max_bottom_idx;
        size3d_l_bottom_idx_ = ++max_bottom_idx;
        orien3d_sin_bottom_idx_ = ++max_bottom_idx;
        orien3d_cos_bottom_idx_ = ++max_bottom_idx;

        if (with_trunc_ratio_) {
            trunc_ratio_bottom_idx_ = ++max_bottom_idx;
        }

        num_top_channels_ += 4;
        orien_type_ = scale_to_dist_param.orien_type;
        cls_ids_zero_size3d_w_.insert(
                scale_to_dist_param.cls_ids_zero_size3d_w.begin(),
                scale_to_dist_param.cls_ids_zero_size3d_w.end());
        cls_ids_zero_size3d_l_.insert(
                scale_to_dist_param.cls_ids_zero_size3d_l.begin(),
                scale_to_dist_param.cls_ids_zero_size3d_l.end());
        cls_ids_zero_orien3d_.insert(
                scale_to_dist_param.cls_ids_zero_orien3d.begin(),
                scale_to_dist_param.cls_ids_zero_orien3d.end());
    }

    cmp_pts_corner_3d_ = scale_to_dist_param.cmp_pts_corner_3d;

    if (cmp_pts_corner_3d_) {
        CHECK(has_size3d_and_orien3d_);
        num_top_channels_ += 24;
    }

    cmp_pts_corner_2d_ = scale_to_dist_param.cmp_pts_corner_2d;

    if (cmp_pts_corner_2d_) {
        CHECK(cmp_pts_corner_3d_);
        num_top_channels_ += 16;
    }

    CHECK_LT(max_bottom_idx, inputs.size());

    if (inputs.size() > (max_bottom_idx + 1)) {
        has_scale_offset_info_ = true;
    } else {
        has_scale_offset_info_ = false;
    }

    if (sub_class_num_class_.size() < num_class_) {
        sub_class_num_class_.resize(num_class_, 1);
        sub_class_bottom_idx_.resize(num_class_, -1);
    }

    sub_class_num_class_pre_sum_.push_back(0);

    for (int k = 0; k < sub_class_num_class_.size(); k++) {
        sub_class_num_class_pre_sum_.push_back(
                sub_class_num_class_pre_sum_.back() + sub_class_num_class_[k]);
    }

    total_sub_class_num_ = sub_class_num_class_pre_sum_.back();
    prj_h_norm_type_ = scale_to_dist_param.prj_h_norm_type;
    bbox_size_add_one_ = scale_to_dist_param.bbox_size_add_one;
    cam_info_idx_st_in_im_info_ = scale_to_dist_param.cam_info_idx_st_in_im_info;
    need_ctr_2d_norm_ = false;
    std::copy(scale_to_dist_param.ctr_2d_means.begin(),
              scale_to_dist_param.ctr_2d_means.end(),
              std::back_inserter(ctr_2d_means_));
    std::copy(scale_to_dist_param.ctr_2d_stds.begin(),
              scale_to_dist_param.ctr_2d_stds.end(),
              std::back_inserter(ctr_2d_stds_));
    CHECK_EQ(ctr_2d_means_.size(), ctr_2d_stds_.size());

    if (ctr_2d_means_.size() > 0) {
        need_ctr_2d_norm_ = true;
        CHECK_EQ(ctr_2d_means_.size(), total_sub_class_num_ * 2);
    }

    regress_ph_rh_as_whole_ = scale_to_dist_param.regress_ph_rh_as_whole;

    need_prj_h_norm_ = false;
    std::copy(scale_to_dist_param.prj_h_means.begin(),
              scale_to_dist_param.prj_h_means.end(),
              std::back_inserter(prj_h_means_));
    std::copy(scale_to_dist_param.prj_h_stds.begin(),
              scale_to_dist_param.prj_h_stds.end(),
              std::back_inserter(prj_h_stds_));
    CHECK_EQ(prj_h_means_.size(), prj_h_stds_.size());

    if (prj_h_means_.size() > 0) {
        need_prj_h_norm_ = true;
        if (!regress_ph_rh_as_whole_) {
            CHECK_EQ(prj_h_means_.size(), total_sub_class_num_);
        } else {
            CHECK_EQ(prj_h_means_.size(), 1);
        }
    }

    need_real_h_norm_ = false;
    std::copy(scale_to_dist_param.real_h_means.begin(),
              scale_to_dist_param.real_h_means.end(),
              std::back_inserter(real_h_means_));
    std::copy(scale_to_dist_param.real_h_stds.begin(),
              scale_to_dist_param.real_h_stds.end(),
              std::back_inserter(real_h_stds_));
    CHECK_EQ(real_h_means_.size(), real_h_stds_.size());

    if (real_h_means_.size() > 0) {
        need_real_h_norm_ = true;
        CHECK_EQ(real_h_means_.size(), total_sub_class_num_);
    }
    need_real_h_norm_dps_ = false;
    if (regress_ph_rh_as_whole_) {
        std::copy(scale_to_dist_param.real_h_means_as_whole.begin(),
                  scale_to_dist_param.real_h_means_as_whole.end(),
                  std::back_inserter(real_h_means_dps_));
        std::copy(scale_to_dist_param.real_h_stds_as_whole.begin(),
                  scale_to_dist_param.real_h_stds_as_whole.end(),
                  std::back_inserter(real_h_stds_dps_));
        CHECK_EQ(real_h_means_dps_.size(), real_h_stds_dps_.size());
        if (real_h_means_dps_.size() > 0) {
            need_real_h_norm_dps_ = true;
            CHECK_EQ(real_h_means_dps_.size(), 1);
        }
    } else {
        need_real_h_norm_dps_ = need_real_h_norm_;
        real_h_means_dps_ = real_h_means_;
        real_h_stds_dps_ = real_h_stds_;
    }
    need_real_w_norm_ = false;
    std::copy(scale_to_dist_param.real_w_means.begin(),
              scale_to_dist_param.real_w_means.end(),
              std::back_inserter(real_w_means_));
    std::copy(scale_to_dist_param.real_w_stds.begin(),
              scale_to_dist_param.real_w_stds.end(),
              std::back_inserter(real_w_stds_));
    CHECK_EQ(real_w_means_.size(), real_w_stds_.size());

    if (real_w_means_.size() > 0) {
        need_real_w_norm_ = true;
        CHECK_EQ(real_w_means_.size(), total_sub_class_num_);
    }

    need_real_l_norm_ = false;
    std::copy(scale_to_dist_param.real_l_means.begin(),
              scale_to_dist_param.real_l_means.end(),
              std::back_inserter(real_l_means_));
    std::copy(scale_to_dist_param.real_l_stds.begin(),
              scale_to_dist_param.real_l_stds.end(),
              std::back_inserter(real_l_stds_));
    CHECK_EQ(real_l_means_.size(), real_l_stds_.size());

    if (real_l_means_.size() > 0) {
        need_real_l_norm_ = true;
        CHECK_EQ(real_l_means_.size(), total_sub_class_num_);
    }

    need_sin_norm_ = false;
    std::copy(scale_to_dist_param.sin_means.begin(),
              scale_to_dist_param.sin_means.end(),
              std::back_inserter(sin_means_));
    std::copy(scale_to_dist_param.sin_stds.begin(),
              scale_to_dist_param.sin_stds.end(),
              std::back_inserter(sin_stds_));
    CHECK_EQ(sin_means_.size(), sin_stds_.size());

    if (sin_means_.size() > 0) {
        need_sin_norm_ = true;
        CHECK_EQ(sin_means_.size(), total_sub_class_num_);
    }

    need_cos_norm_ = false;
    std::copy(scale_to_dist_param.cos_means.begin(),
              scale_to_dist_param.cos_means.end(),
              std::back_inserter(cos_means_));
    std::copy(scale_to_dist_param.cos_stds.begin(),
              scale_to_dist_param.cos_stds.end(),
              std::back_inserter(cos_stds_));
    CHECK_EQ(cos_means_.size(), cos_stds_.size());

    if (cos_means_.size() > 0) {
        need_cos_norm_ = true;
        CHECK_EQ(cos_means_.size(), total_sub_class_num_);
    }

    im_width_scale_ = scale_to_dist_param.im_width_scale;
    im_height_scale_ = scale_to_dist_param.im_height_scale;
    cords_offset_x_ = scale_to_dist_param.cords_offset_x;
    cords_offset_y_ = scale_to_dist_param.cords_offset_y;
    // rotate coords by pitch
    rotate_coords_by_pitch_ = scale_to_dist_param.rotate_coords_by_pitch;

    //    const unsigned int shuffle_rng_seed = caffe_rng_rand();
    //    shuffle_rng_.reset(new Caffe::RNG(shuffle_rng_seed));
    // Reshape
    //cam_coords num_rois x [x3d, y3d, z3d]
    //    outputs[0]->Reshape(1, num_top_channels_, 1, 1);
    //[img_id, x1, y1, x2, y2, prb0, prb1, ..., prb_cls_num]
    CHECK_EQ(inputs[0]->count_valid(1, inputs[1]->dims()), 6 + num_class_);
    CHECK_GT(inputs[0]->num(), 0);
    //[im_info]
    CHECK_GE(inputs[1]->count(1, inputs[1]->dims()), cam_info_idx_st_in_im_info_ + 6);
    //[cam_ctr_pt]
    CHECK_EQ(inputs[2]->num(), inputs[0]->num());
    CHECK_GE(inputs[2]->count_valid(1, inputs[2]->dims()), total_sub_class_num_ * 2);
    //[prj_h_pred]
    CHECK_EQ(inputs[3]->num(), inputs[0]->num());
    if (!regress_ph_rh_as_whole_) {
        CHECK_EQ(inputs[3]->count_valid(1, inputs[3]->dims()), total_sub_class_num_);
    } else {
        CHECK_EQ(inputs[3]->count_valid(1, inputs[3]->dims()), 1);
    }
    //[real_h_pred]
    CHECK_EQ(inputs[4]->num(), inputs[0]->num());
    if (!regress_ph_rh_as_whole_) {
        CHECK_EQ(inputs[4]->count_valid(1, inputs[4]->dims()), total_sub_class_num_);
    } else {
        CHECK_EQ(inputs[4]->count_valid(1, inputs[4]->dims()), 1);
    }

    //[sub_class_bottom blobs]
    for (int k = 0; k < sub_class_bottom_idx_.size(); k++) {
        if (sub_class_num_class_[k] <= 1) {
            continue;
        }

        int sub_cls_top_idx = sub_class_bottom_idx_[k];
        CHECK_EQ(inputs[sub_cls_top_idx]->num(), inputs[0]->num());
        CHECK_EQ(inputs[sub_cls_top_idx]->count(1,
                inputs[sub_cls_top_idx]->dims()), sub_class_num_class_[k]);
    }

    if (has_size3d_and_orien3d_) {
        //[size3d_h]
        CHECK_EQ(inputs[size3d_h_bottom_idx_]->num(), inputs[0]->num());
        CHECK_EQ(inputs[size3d_h_bottom_idx_]->count(1,
                inputs[size3d_h_bottom_idx_]->dims()), total_sub_class_num_);
        //[size3d_w]
        CHECK_EQ(inputs[size3d_w_bottom_idx_]->num(), inputs[0]->num());
        CHECK_EQ(inputs[size3d_w_bottom_idx_]->count(1,
                inputs[size3d_w_bottom_idx_]->dims()), total_sub_class_num_);
        //[size3d_l]
        CHECK_EQ(inputs[size3d_l_bottom_idx_]->num(), inputs[0]->num());
        CHECK_EQ(inputs[size3d_l_bottom_idx_]->count(1,
                inputs[size3d_l_bottom_idx_]->dims()), total_sub_class_num_);
        //[orien3d_sin]
        CHECK_EQ(inputs[orien3d_sin_bottom_idx_]->num(), inputs[0]->num());
        CHECK_EQ(inputs[orien3d_sin_bottom_idx_]->count(1,
                inputs[orien3d_sin_bottom_idx_]->dims()), total_sub_class_num_);
        //[orien3d_cos]
        CHECK_EQ(inputs[orien3d_cos_bottom_idx_]->num(), inputs[0]->num());
        CHECK_EQ(inputs[orien3d_cos_bottom_idx_]->count(1,
                inputs[orien3d_cos_bottom_idx_]->dims()), total_sub_class_num_);

        //[trunc_ratio]
        if (with_trunc_ratio_) {
            CHECK_EQ(inputs[trunc_ratio_bottom_idx_]->num(), inputs[0]->num());
            CHECK_EQ(inputs[trunc_ratio_bottom_idx_]->count(1,
                    inputs[trunc_ratio_bottom_idx_]->dims()), total_sub_class_num_);
        }
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberProposalImgScaleToCamCoords<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ProposalImgScaleToCamCoordsParam<NV> &param,
        Context<NV> &ctx) {

    this->_ctx = &ctx;
    _rois_boxes_data_host_tensor = new Tensor<NVHX86>();
    _im_info_data_host_tensor = new Tensor<NVHX86>();
    _cam2d_data_host_tensor = new Tensor<NVHX86>();
    _prj_h_pred_data_host_tensor = new Tensor<NVHX86>();
    _real_h_pred_data_host_tensor = new Tensor<NVHX86>();
    _size3d_h_pred_data_host_tensor = new Tensor<NVHX86>();
    _size3d_w_pred_data_host_tensor = new Tensor<NVHX86>();
    _size3d_l_pred_data_host_tensor = new Tensor<NVHX86>();
    _orien3d_sin_pred_data_host_tensor = new Tensor<NVHX86>();
    _orien3d_cos_pred_data_host_tensor = new Tensor<NVHX86>();
    _trunc_ratio_pred_data_host_tensor = new Tensor<NVHX86>();
    _img_info_data_host_tensor = new Tensor<NVHX86>();
    _cam_coords_data_host_tensor = new Tensor<NVHX86>();
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberProposalImgScaleToCamCoords<NV, AK_FLOAT>::dispatch(const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ProposalImgScaleToCamCoordsParam<NV> &param) {

    const double PI=3.1415926;
    //RESHAPE PART
    _rois_boxes_data_host_tensor->reshape(inputs[0]->valid_shape());
    _im_info_data_host_tensor->reshape(inputs[1]->valid_shape());
    _cam2d_data_host_tensor->reshape(inputs[2]->valid_shape());
    _prj_h_pred_data_host_tensor->reshape(inputs[3]->valid_shape());
    _real_h_pred_data_host_tensor->reshape(inputs[4]->valid_shape());
    _img_info_data_host_tensor->reshape(inputs.back()->valid_shape());
    // resize tensor_vector.
    _sub_class_datas_host_tensor_v.resize(sub_class_bottom_idx_.size());

    for (int i = 0; i < _sub_class_datas_host_tensor_v.size(); ++i) {
        _sub_class_datas_host_tensor_v[i] = new Tensor<NVHX86>();
    }

    // reshape
    if (has_size3d_and_orien3d_) {
        _size3d_h_pred_data_host_tensor->reshape(inputs[size3d_h_bottom_idx_]->valid_shape());
        _size3d_w_pred_data_host_tensor->reshape(inputs[size3d_w_bottom_idx_]->valid_shape());
        _size3d_l_pred_data_host_tensor->reshape(inputs[size3d_l_bottom_idx_]->valid_shape());
        _orien3d_sin_pred_data_host_tensor->reshape(inputs[orien3d_sin_bottom_idx_]->valid_shape());
        _orien3d_cos_pred_data_host_tensor->reshape(inputs[orien3d_cos_bottom_idx_]->valid_shape());

        if (with_trunc_ratio_) {
            _trunc_ratio_pred_data_host_tensor->reshape(inputs[trunc_ratio_bottom_idx_]->valid_shape());
        }
    }

    // finished reshape
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    //inputs
    //[img_id, x1, y1, x2, y2, [score] ]
    int num_rois = inputs[0]->num();
    const int rois_dim = inputs[0]->count_valid(1, inputs[0]->dims());
    _rois_boxes_data_host_tensor->async_copy_from(*inputs[0], cuda_stream);
    inputs[0]->record_event(cuda_stream);
    inputs[0]->sync();
    _im_info_data_host_tensor->async_copy_from(*inputs[1], cuda_stream);
    inputs[1]->record_event(cuda_stream);
    inputs[1]->sync();
    _cam2d_data_host_tensor->async_copy_from(*inputs[2], cuda_stream);
    inputs[2]->record_event(cuda_stream);
    inputs[2]->sync();
    _prj_h_pred_data_host_tensor->async_copy_from(*inputs[3], cuda_stream);
    inputs[3]->record_event(cuda_stream);
    inputs[3]->sync();
    _real_h_pred_data_host_tensor->async_copy_from(*inputs[4], cuda_stream);
    inputs[4]->record_event(cuda_stream);
    inputs[4]->sync();
    const float* rois_boxes_data = (const float*)_rois_boxes_data_host_tensor->data();
    //num_img x [..., cam_xpz, cam_xct, cam_ypz, cam_yct, cam_hgrd, cam_pitch, ...]
    const int num_img = inputs[1]->num();
    const int im_info_dim = inputs[1]->count(1, inputs[1]->dims());
    const float* im_info_data = (const float*)_im_info_data_host_tensor->data();
    //[ctr_x_sub_c1, ctr_y_sub_c1, ctr_x_sub_c2, ctr_y_sub_c2, ...]
    const float* cam2d_data = (const float*)_cam2d_data_host_tensor->data();
    //[prj_h_sub_c1, prj_h_sub_c2, ...]
    const float* prj_h_pred_data = (const float*)_prj_h_pred_data_host_tensor->data();
    //[real_h_sub_c1, real_h_sub_c2, ...]
    const float* real_h_pred_data = (const float*)_real_h_pred_data_host_tensor->data();
    // =======================================================================
    //sub_class_bottom_blobs
    std::vector<const float* > sub_class_datas;

    for (int k = 0; k < sub_class_bottom_idx_.size(); k++) {
        if (sub_class_num_class_[k] <= 1) {
            sub_class_datas.push_back(NULL);
            continue;
        }

        int sub_cls_top_idx = sub_class_bottom_idx_[k];
        //        sub_class_datas.push_back(inputs[sub_cls_top_idx]->cpu_data());
        // These is very special, The host tensor is a vector...
        _sub_class_datas_host_tensor_v[k]->reshape(inputs[sub_cls_top_idx]->valid_shape());
        _sub_class_datas_host_tensor_v[k]->async_copy_from(*inputs[sub_cls_top_idx], cuda_stream);
        inputs[sub_cls_top_idx]->record_event(cuda_stream);
        inputs[sub_cls_top_idx]->sync();
        sub_class_datas.push_back((const float*)_sub_class_datas_host_tensor_v[k]->data());
    }
    //[size3d_h_sub_c1, size3d_h_sub_c2, ...]
    const float* size3d_h_pred_data = NULL;
    //[size3d_w_sub_c1, size3d_w_sub_c2, ...]
    const float* size3d_w_pred_data = NULL;
    //[size3d_l_sub_c1, size3d_l_sub_c2, ...]
    const float* size3d_l_pred_data = NULL;
    //[orien3d_sin_sub_c1, orien3d_sin_sub_c2, ...]
    const float* orien3d_sin_pred_data = NULL;
    //[orien3d_cos_sub_c1, orien3d_cos_sub_c2, ...]
    const float* orien3d_cos_pred_data = NULL;
    //[trunc_ratio_sub_c1, trunc_ratio_sub_c2, ...]
    const float* trunc_ratio_pred_data = NULL;

    if (has_size3d_and_orien3d_) {
        _size3d_h_pred_data_host_tensor->async_copy_from(*inputs[size3d_h_bottom_idx_], cuda_stream);
        inputs[size3d_h_bottom_idx_]->record_event(cuda_stream);
        inputs[size3d_h_bottom_idx_]->sync();
        _size3d_w_pred_data_host_tensor->async_copy_from(*inputs[size3d_w_bottom_idx_], cuda_stream);
        inputs[size3d_w_bottom_idx_]->record_event(cuda_stream);
        inputs[size3d_w_bottom_idx_]->sync();
        _size3d_l_pred_data_host_tensor->async_copy_from(*inputs[size3d_l_bottom_idx_], cuda_stream);
        inputs[size3d_l_bottom_idx_]->record_event(cuda_stream);
        inputs[size3d_l_bottom_idx_]->sync();
        _orien3d_sin_pred_data_host_tensor->async_copy_from(*inputs[orien3d_sin_bottom_idx_], cuda_stream);
        inputs[orien3d_sin_bottom_idx_]->record_event(cuda_stream);
        inputs[orien3d_sin_bottom_idx_]->sync();
        _orien3d_cos_pred_data_host_tensor->async_copy_from(*inputs[orien3d_cos_bottom_idx_], cuda_stream);
        inputs[orien3d_cos_bottom_idx_]->record_event(cuda_stream);
        inputs[orien3d_cos_bottom_idx_]->sync();

        if (with_trunc_ratio_) {
            _trunc_ratio_pred_data_host_tensor->async_copy_from(*inputs[trunc_ratio_bottom_idx_], cuda_stream);
            inputs[trunc_ratio_bottom_idx_]->record_event(cuda_stream);
            inputs[trunc_ratio_bottom_idx_]->sync();
        }

        size3d_h_pred_data = (const float*)_size3d_h_pred_data_host_tensor->data();
        size3d_w_pred_data = (const float*)_size3d_w_pred_data_host_tensor->data();
        size3d_l_pred_data = (const float*)_size3d_l_pred_data_host_tensor->data();
        orien3d_sin_pred_data = (const float*)_orien3d_sin_pred_data_host_tensor->data();
        orien3d_cos_pred_data = (const float*)_orien3d_cos_pred_data_host_tensor->data();

        if (with_trunc_ratio_) {
            trunc_ratio_pred_data = (const float*)_trunc_ratio_pred_data_host_tensor->data();
        }
    }

    //[im_width, im_height, im_width_scale, im_height_scale, cords_offset_x, cords_offset_y]
    float im_width_scale = im_width_scale_;
    float im_height_scale = im_height_scale_;
    float cords_offset_x = cords_offset_x_;
    float cords_offset_y = cords_offset_y_;
    if (has_scale_offset_info_) {
        CHECK_EQ(inputs.back()->num(), num_img);
        CHECK_EQ(inputs.back()->count(1, inputs.back()->dims()), 6);
    }

    float bsz01 = bbox_size_add_one_ ? float(1.0) : float(0.0);
    //prepare the outputs
    Shape output_shape({num_rois, num_top_channels_, 1, 1}, Layout_NCHW);
    outputs[0]->reshape(output_shape);
    // output must reshape in dispatch [zs]
    _cam_coords_data_host_tensor->reshape(output_shape);
    float* cam_coords_data = (float*)_cam_coords_data_host_tensor->mutable_data();
    memset(cam_coords_data, 0, sizeof(float) * _cam_coords_data_host_tensor->valid_size());
//    fill_tensor_const(*_cam_coords_data_host_tensor, 0);

    //    caffe_set(outputs[0]->count(), float(0), cam_coords_data);
    for (int i = 0; i < num_rois; i++) {
        int img_id = static_cast<int>(rois_boxes_data[i * rois_dim + 0]);
        CHECK_LT(img_id, num_img);
        //im_size_scale_offset_info
        if (has_scale_offset_info_ ) {
            //            const float * img_info_data = inputs.back()->cpu_data();
            _img_info_data_host_tensor->async_copy_from(*inputs.back(), cuda_stream);
            inputs.back()->record_event(cuda_stream);
            inputs.back()->sync();
            const float*  img_info_data = (const float*)_img_info_data_host_tensor->data();
            im_width_scale = img_info_data[img_id * 6 + 2];
            im_height_scale = img_info_data[img_id * 6 + 3];
            cords_offset_x = img_info_data[img_id * 6 + 4];
            cords_offset_y = img_info_data[img_id * 6 + 5];
            CHECK_GT(im_width_scale, 0);
            CHECK_GT(im_height_scale, 0);
            CHECK_LT(abs(im_width_scale - im_height_scale), 0.01);
        }
        //roi_info
        float ltx = rois_boxes_data[i * rois_dim + 1];
        float lty = rois_boxes_data[i * rois_dim + 2];
        float rbx = rois_boxes_data[i * rois_dim + 3];
        float rby = rois_boxes_data[i * rois_dim + 4];
        float rois_w = rbx - ltx + bsz01;
        float rois_h = rby - lty + bsz01;
        float rois_ctr_x = ltx + 0.5 * (rois_w - bsz01);
        float rois_ctr_y = lty + 0.5 * (rois_h - bsz01);
        CHECK_GT(rois_w, 0);
        CHECK_GT(rois_h, 0);
        ltx = ltx / im_width_scale + cords_offset_x;
        lty = lty / im_height_scale + cords_offset_y;
        rbx = rbx / im_width_scale + cords_offset_x;
        rby = rby / im_height_scale + cords_offset_y;
        rois_ctr_x = rois_ctr_x / im_width_scale + cords_offset_x;
        rois_ctr_y = rois_ctr_y / im_height_scale + cords_offset_y;
        rois_w /= im_width_scale;
        rois_h /= im_height_scale;
        //cam_info
        float cam_xpz =
                im_info_data[img_id * im_info_dim + cam_info_idx_st_in_im_info_ + 0];
        float cam_xct =
                im_info_data[img_id * im_info_dim + cam_info_idx_st_in_im_info_ + 1];
        float cam_ypz =
                im_info_data[img_id * im_info_dim + cam_info_idx_st_in_im_info_ + 2];
        float cam_yct =
                im_info_data[img_id * im_info_dim + cam_info_idx_st_in_im_info_ + 3];
        float cam_pitch =
                im_info_data[img_id * im_info_dim + cam_info_idx_st_in_im_info_ + 5];
        //total_sub_class_id
        int class_id = 0;
        for (int c = 1; c < num_class_; c++) {
            if (rois_boxes_data[i * rois_dim + 6 + c] >
                rois_boxes_data[i * rois_dim + 6 + class_id]) {
                class_id = c;
            }
        }
        int total_sub_class_id = sub_class_num_class_pre_sum_[class_id];
        int sub_class_num = sub_class_num_class_[class_id];
        if (sub_class_num > 1) {
            int sub_class_id = 0;
            for (int sc = 1; sc < sub_class_num; sc++) {
                if (sub_class_datas[class_id][i * sub_class_num + sc]
                    > sub_class_datas[class_id][i * sub_class_num + sub_class_id]) {
                    sub_class_id = sc;
                }
            }
            total_sub_class_id += sub_class_id;
        }
        CHECK_LT(total_sub_class_id, total_sub_class_num_);
        //cam2d_ctr
        float cam2d_x =
                cam2d_data[i * total_sub_class_num_ * 2 + total_sub_class_id * 2];
        float cam2d_y =
                cam2d_data[i * total_sub_class_num_ * 2 + total_sub_class_id * 2 + 1];
        if (need_ctr_2d_norm_) {
            cam2d_x *= ctr_2d_stds_[total_sub_class_id * 2];
            cam2d_x += ctr_2d_means_[total_sub_class_id * 2];
            cam2d_y *= ctr_2d_stds_[total_sub_class_id * 2 + 1];
            cam2d_y += ctr_2d_means_[total_sub_class_id * 2 + 1];
        }
        cam2d_x = cam2d_x * rois_w + rois_ctr_x;
        cam2d_y = cam2d_y * rois_h + rois_ctr_y;
        int ph_rh_regress_num = (regress_ph_rh_as_whole_ ? 1 : total_sub_class_num_);
        int ph_rh_regress_idx = (regress_ph_rh_as_whole_ ? 0 : total_sub_class_id);
        //prj_h_pred
        float prj_h_pred =
                prj_h_pred_data[i * ph_rh_regress_num + ph_rh_regress_idx];
        if (need_prj_h_norm_) {
            prj_h_pred = prj_h_pred *
                         prj_h_stds_[ph_rh_regress_idx] + prj_h_means_[ph_rh_regress_idx];
        }
        if (prj_h_norm_type_ == ProposalImgScaleToCamCoords_NormType_HEIGHT) {
            prj_h_pred *= rois_h;
        } else if (prj_h_norm_type_ == ProposalImgScaleToCamCoords_NormType_HEIGHT_LOG) {
            prj_h_pred = exp(prj_h_pred) * rois_h;
        } else {
            CHECK(false)<<"NOT IMPLEMENTED";
        }
        //real_h_pred
        float real_h_pred = real_h_pred_data[i * ph_rh_regress_num + ph_rh_regress_idx];
        if (need_real_h_norm_dps_) {
            real_h_pred = real_h_pred * real_h_stds_dps_[ph_rh_regress_idx]
                          + real_h_means_dps_[ph_rh_regress_idx];
        }
        //cmp distance_err
        float k1, k2, u, v;
        coef2dTo3d(cam_xpz, cam_xct, cam_ypz, cam_yct,
                   cam_pitch, cam2d_x, cam2d_y, k1, k2, u, v);
        float x, y, z;
        cord2dTo3d(k1, k2, u, v, prj_h_pred, real_h_pred, x, y, z);
        cam_coords_data[i * num_top_channels_ + 0] = x;
        cam_coords_data[i * num_top_channels_ + 1] = y;
        cam_coords_data[i * num_top_channels_ + 2] = z;
        //has_size3d_and_orien3d
        if (has_size3d_and_orien3d_) {
            //h,w,l
            float size3d_h_pred =
                    size3d_h_pred_data[i * total_sub_class_num_ + total_sub_class_id];
            if (need_real_h_norm_) {
                size3d_h_pred = size3d_h_pred *
                                real_h_stds_[total_sub_class_id] + real_h_means_[total_sub_class_id];
            }
            float size3d_w_pred =
                    size3d_w_pred_data[i * total_sub_class_num_ + total_sub_class_id];
            if (need_real_w_norm_) {
                size3d_w_pred = size3d_w_pred *
                                real_w_stds_[total_sub_class_id] + real_w_means_[total_sub_class_id];
            }
            float size3d_l_pred =
                    size3d_l_pred_data[i * total_sub_class_num_ + total_sub_class_id];
            if (need_real_l_norm_) {
                size3d_l_pred = size3d_l_pred *
                                real_l_stds_[total_sub_class_id] + real_l_means_[total_sub_class_id];
            }
            //orien
            float orien3d_sin_pred =
                    orien3d_sin_pred_data[i * total_sub_class_num_ + total_sub_class_id];
            if (need_sin_norm_) {
                orien3d_sin_pred = orien3d_sin_pred *
                                   sin_stds_[total_sub_class_id] + sin_means_[total_sub_class_id];
            }
            float orien3d_cos_pred =
                    orien3d_cos_pred_data[i * total_sub_class_num_ + total_sub_class_id];
            if (need_cos_norm_) {
                orien3d_cos_pred = orien3d_cos_pred *
                                   cos_stds_[total_sub_class_id] + cos_means_[total_sub_class_id];
            }
            float ctr_glb_yaw = atan2(x, z);
            if (ctr_glb_yaw < 0) {
                ctr_glb_yaw += 2 * PI;
            }
            float obj_local_yaw = atan2(orien3d_sin_pred, orien3d_cos_pred);
            if (obj_local_yaw < 0) {
                obj_local_yaw += 2 * PI;
            }
            float obj_glb_yaw = 0, obj_glb_yaw_kitti = 0;
            if (orien_type_ == ProposalImgScaleToCamCoords_OrienType_PI2) {
                obj_glb_yaw = ctr_glb_yaw + obj_local_yaw;
                if (obj_glb_yaw > 2 * PI) {
                    obj_glb_yaw -= 2 * PI;
                }
                obj_glb_yaw_kitti = obj_glb_yaw - PI / 2;
                if (obj_glb_yaw_kitti > PI) {
                    obj_glb_yaw_kitti -= 2 * PI;
                }
            } else if (orien_type_ == ProposalImgScaleToCamCoords_OrienType_PI) {
                float ctr_glb_yaw_mod_pi = ctr_glb_yaw;
                if (ctr_glb_yaw_mod_pi > PI) {
                    ctr_glb_yaw_mod_pi -= PI;
                }
                obj_local_yaw /= 2;
                obj_glb_yaw = ctr_glb_yaw_mod_pi + obj_local_yaw;
                if (obj_glb_yaw > PI) {
                    obj_glb_yaw -= PI;
                }
                obj_glb_yaw_kitti = obj_glb_yaw - PI / 2;
            } else {
                CHECK(false)<<"NOT IMPLEMENTED";
            }
            if (cls_ids_zero_size3d_w_.find(class_id) != cls_ids_zero_size3d_w_.end()) {
                size3d_w_pred = 0;
            }
            if (cls_ids_zero_size3d_l_.find(class_id) != cls_ids_zero_size3d_l_.end()) {
                size3d_l_pred = 0;
            }
            if (cls_ids_zero_orien3d_.find(class_id) != cls_ids_zero_orien3d_.end()) {
                obj_glb_yaw = 0;
                obj_glb_yaw_kitti = 0;
            }
            cam_coords_data[i * num_top_channels_ + 3] = size3d_h_pred;
            cam_coords_data[i * num_top_channels_ + 4] = size3d_w_pred;
            cam_coords_data[i * num_top_channels_ + 5] = size3d_l_pred;
            cam_coords_data[i * num_top_channels_ + 6] = obj_glb_yaw_kitti;
            // refine x, y, z by trunc ratio
            if (with_trunc_ratio_) {
                float trunc_ratio =
                        trunc_ratio_pred_data[i * total_sub_class_num_ + total_sub_class_id];
                trunc_ratio = std::max<float>(0.0, std::min<float>(1.0, trunc_ratio));
                if (trunc_ratio > 0.05) {
                    float trunc_glb_yaw = 0.0;
                    if (ctr_glb_yaw < PI) {
                                CHECK_LE(ctr_glb_yaw, 0.5 * PI + 1e-3);
                        if (obj_glb_yaw < ctr_glb_yaw) {
                            trunc_glb_yaw = obj_glb_yaw + PI;
                        } else if (obj_glb_yaw > (ctr_glb_yaw + PI)) {
                            trunc_glb_yaw = obj_glb_yaw - PI;
                        } else {
                            trunc_glb_yaw = obj_glb_yaw;
                        }
                    } else {
                                CHECK_GE(ctr_glb_yaw, 1.5 * PI - 1e-3);
                        if (obj_glb_yaw < (ctr_glb_yaw - PI)) {
                            trunc_glb_yaw = obj_glb_yaw + PI;
                        } else if (obj_glb_yaw > ctr_glb_yaw){
                            trunc_glb_yaw = obj_glb_yaw - PI;
                        } else {
                            trunc_glb_yaw = obj_glb_yaw;
                        }
                    }
                    z += 0.5 * size3d_l_pred * trunc_ratio * cos(trunc_glb_yaw);
                    x += 0.5 * size3d_l_pred * trunc_ratio * sin(trunc_glb_yaw);
                    cam_coords_data[i * num_top_channels_ + 0] = x;
                    cam_coords_data[i * num_top_channels_ + 1] = y;
                    cam_coords_data[i * num_top_channels_ + 2] = z;
                }
            }
            //pts_corner_3d
            if (cmp_pts_corner_3d_) {
                float h = size3d_h_pred;
                float w = size3d_w_pred;
                float l = size3d_l_pred;
                float o = obj_glb_yaw;
                float pts_3d[8][3] = {
                        {-w/2, h/2, l/2},
                        {w/2, h/2, l/2},
                        {w/2, h/2, -l/2},
                        {-w/2, h/2, -l/2},
                        {-w/2, -h/2, l/2},
                        {w/2, -h/2, l/2},
                        {w/2, -h/2, -l/2},
                        {-w/2, -h/2, -l/2}
                };
                for (int p = 0; p < 8; p++) {
                    // rotate by obj_glb_yaw
                    float tmp_z = pts_3d[p][2] * cos(o) - pts_3d[p][0] * sin(o);
                    float tmp_x = pts_3d[p][2] * sin(o) + pts_3d[p][0] * cos(o);
                    pts_3d[p][2] = tmp_z;
                    pts_3d[p][0] = tmp_x;

                    // rotate by cam_pitch
                    if (rotate_coords_by_pitch_) {
                        float o_pitch = -1.0 * cam_pitch;
                        tmp_z = pts_3d[p][2] * cos(o_pitch) - pts_3d[p][1] * sin(o_pitch);
                        float tmp_y = pts_3d[p][2] * sin(o_pitch) - pts_3d[p][1] * cos(o_pitch);
                        pts_3d[p][2] = tmp_z;
                        pts_3d[p][1] = tmp_y;
                    }

                    pts_3d[p][0] += x;
                    pts_3d[p][1] += y;
                    pts_3d[p][2] += z;
                    cam_coords_data[i * num_top_channels_ + 7 + p * 3 + 0] = pts_3d[p][0];
                    cam_coords_data[i * num_top_channels_ + 7 + p * 3 + 1] = pts_3d[p][1];
                    cam_coords_data[i * num_top_channels_ + 7 + p * 3 + 2] = pts_3d[p][2];
                }
                //pts_corner_2d
                if (cmp_pts_corner_2d_) {
                    for (int p = 0; p < 8; p++) {
                        if (pts_3d[p][2] < 1e-6) {
                            int bt_p = p % 4;
                            int cr_p = ((bt_p % 2 == 0) ? (bt_p + 3) : (bt_p + 1)) % 4;
                            if (p >= 4) {
                                cr_p += 4;
                            }
                            float r = (1e-6 - pts_3d[p][2]) /
                                      (pts_3d[cr_p][2] - pts_3d[p][2]);
                            pts_3d[p][0] += (pts_3d[cr_p][0] - pts_3d[p][0]) * r;
                            pts_3d[p][1] += (pts_3d[cr_p][1] - pts_3d[p][1]) * r;
                            pts_3d[p][2] = 1e-6;
                        }
                        cam_coords_data[i * num_top_channels_ + 31 + p * 2 + 0]
                                = cam_xpz * pts_3d[p][0] / pts_3d[p][2] + cam_xct;
                        cam_coords_data[i * num_top_channels_ + 31 + p * 2 + 1]
                                = cam_ypz * pts_3d[p][1] / pts_3d[p][2] + cam_yct;
                    }
                }
            }
        }
    }
    outputs[0]->async_copy_from(*_cam_coords_data_host_tensor, cuda_stream);
    return SaberSuccess;
}

template class SaberProposalImgScaleToCamCoords<NV, AK_FLOAT>;

DEFINE_OP_TEMPLATE(SaberProposalImgScaleToCamCoords,
        ProposalImgScaleToCamCoordsParam, NV, AK_HALF);

DEFINE_OP_TEMPLATE(SaberProposalImgScaleToCamCoords,
        ProposalImgScaleToCamCoordsParam, NV, AK_INT8);
}
}