#include "saber/funcs/impl/cuda/saber_rcnn_det_output_with_attr.h"
#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/saber_funcs_param.h"
#include <cfloat>
namespace anakin {
namespace saber {

template <>
SaberStatus SaberRCNNDetOutputWithAttr<NV, AK_FLOAT>::create(
        const std::vector<OpTensor*>& inputs,
        std::vector<OpTensor*>& outputs,
        ProposalParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    // ROIOutputSSD create
    ImplROIOutputSSD<NV, AK_FLOAT>::create(inputs, outputs, param, ctx);
    // max bottom idx
    int max_bottom_idx = 0;

    if (this->has_kpts_) {
        max_bottom_idx = std::max<int>(this->kpts_reg_bottom_idx_,
                std::max<int>(this->kpts_exist_bottom_idx_, max_bottom_idx));
    }

    if (this->has_atrs_) {
        max_bottom_idx = std::max<int>(this->atrs_reg_bottom_idx_, max_bottom_idx);
    }

    if (this->has_ftrs_) {
        max_bottom_idx = std::max<int>(this->ftrs_bottom_idx_, max_bottom_idx);
    }

    if (this->has_spmp_) {
        max_bottom_idx = std::max<int>(this->spmp_bottom_idx_, max_bottom_idx);
    }

    if (this->has_cam3d_) {
        max_bottom_idx = std::max<int>(this->cam3d_bottom_idx_, max_bottom_idx);
    }

    if (inputs.size() > (max_bottom_idx + 1)) {
        has_img_info_ = true;
    } else {
        has_img_info_ = false;
    }

    CHECK_GT(this->num_class_, 0);
    rois_dim_ = 5 + this->num_class_ + 1;

    if (this->has_spmp_) {
        CHECK(!this->refine_out_of_map_bbox_);
    }

    num_rois_ = inputs[0]->num();
    CHECK_EQ(rois_dim_, inputs[0]->channel());

    if (this->has_kpts_) {
        CHECK_EQ(num_rois_, inputs[this->kpts_reg_bottom_idx_]->num());

        if (!this->kpts_reg_as_classify_) {
            kpts_reg_dim_ = inputs[this->kpts_reg_bottom_idx_]->channel();
            CHECK_EQ(kpts_reg_dim_ % 2, 0);
            num_kpts_ = kpts_reg_dim_ / 2;

            if (this->kpts_do_norm_) {
                CHECK_GT(this->reg_means_.size(),
                    (this->kpts_reg_norm_idx_st_ + (num_kpts_ - 1) * 2 + 1));
            }
        } else {
            num_kpts_ =  inputs[this->kpts_reg_bottom_idx_]->channel();
            kpts_reg_dim_ = num_kpts_ *
                    this->kpts_classify_width_ * this->kpts_classify_height_;
            CHECK_EQ(this->kpts_classify_width_,
                    inputs[this->kpts_reg_bottom_idx_]->width());
            CHECK_EQ(this->kpts_classify_height_,
                    inputs[this->kpts_reg_bottom_idx_]->height());
        }

        if (this->kpts_exist_bottom_idx_ >= 0) {
            CHECK_EQ(num_rois_, inputs[this->kpts_exist_bottom_idx_]->num());
            kpts_cls_dim_ = inputs[this->kpts_exist_bottom_idx_]->channel();
            CHECK_EQ(kpts_cls_dim_, num_kpts_ * 2);
        }
    }

    if (this->has_atrs_) {
        CHECK_EQ(num_rois_, inputs[this->atrs_reg_bottom_idx_]->num());
        num_atrs_ = inputs[this->atrs_reg_bottom_idx_]->channel();

        if (this->atrs_do_norm_) {
            CHECK_GT(this->reg_means_.size(),
                    (this->atrs_reg_norm_idx_st_ + num_atrs_ - 1));
        }

        if (this->atrs_norm_type_.size() == 0) {
            this->atrs_norm_type_.resize(num_atrs_, ATRS_NormType_NONE);
        } else if (this->atrs_norm_type_.size() == 1) {
            this->atrs_norm_type_.resize(num_atrs_, this->atrs_norm_type_[0]);
        } else {
            CHECK_EQ(this->atrs_norm_type_.size(), num_atrs_);
        }
    }

    if (this->has_ftrs_) {
        CHECK_EQ(num_rois_, inputs[this->ftrs_bottom_idx_]->num());
        num_ftrs_ = inputs[this->ftrs_bottom_idx_]->channel();
    }

    if (this->has_spmp_) {
        CHECK_EQ(num_rois_, inputs[this->spmp_bottom_idx_]->num());
        CHECK_EQ(this->spmp_dim_sum_,
                inputs[this->spmp_bottom_idx_]->count(1,
                        inputs[this->spmp_bottom_idx_]->dims()));
    }

    if (this->has_cam3d_) {
//        CHECK_EQ(num_rois_, inputs[this->cam3d_bottom_idx_]->num());
        num_cam3d_ = inputs[this->cam3d_bottom_idx_]->count(1,
                inputs[this->cam3d_bottom_idx_]->dims());
    }

    return SaberSuccess;
}

template<>
SaberStatus SaberRCNNDetOutputWithAttr<NV, AK_FLOAT>::init(
        const std::vector<OpTensor*>& inputs,
        std::vector<OpTensor*>& outputs,
        ProposalParam<NV>& param, Context<NV>& ctx) {
    this->_ctx = &ctx;
    // new tensor first, These are the host tensors
    _img_info_data_host_tensor = new Tensor<NVHX86>();
    _rois_st_host_tensor = new Tensor<NVHX86>();
    _kpts_reg_st_host_tensor = new Tensor<NVHX86>();
    _kpts_exist_st_host_tensor = new Tensor<NVHX86>();
    _atrs_reg_st_host_tensor = new Tensor<NVHX86>();
    _ftrs_st_host_tensor = new Tensor<NVHX86>();
    _spmp_st_host_tensor = new Tensor<NVHX86>();
    _cam3d_st_host_tensor = new Tensor<NVHX86>();
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberRCNNDetOutputWithAttr<NV, AK_FLOAT>::dispatch(
    const std::vector<OpTensor*>& inputs,
    std::vector<OpTensor*>& outputs,
    ProposalParam<NV>& param) {
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    // CPU TENSOR RESHAPE PART
    // prepare to update to cpu
    _img_info_data_host_tensor->reshape(inputs.back()->valid_shape());
    _rois_st_host_tensor->reshape(inputs[0]->valid_shape());

    if (this->has_kpts_) {
        _kpts_reg_st_host_tensor->reshape(
            inputs[this->kpts_reg_bottom_idx_]->valid_shape());

        if (this->kpts_exist_bottom_idx_ >= 0) {
            _kpts_exist_st_host_tensor->reshape(
                inputs[this->kpts_exist_bottom_idx_]->valid_shape());
        }
    }

    if (this->has_atrs_) {
        _atrs_reg_st_host_tensor->reshape(
            inputs[this->atrs_reg_bottom_idx_]->valid_shape());
    }

    if (this->has_ftrs_) {
        _ftrs_st_host_tensor->reshape(
            inputs[this->ftrs_bottom_idx_]->valid_shape());
    }

    if (this->has_spmp_) {
        _spmp_st_host_tensor->reshape(
            inputs[this->spmp_bottom_idx_]->valid_shape());
    }

    if (this->has_cam3d_) {
        _cam3d_st_host_tensor->reshape(
            inputs[this->cam3d_bottom_idx_]->valid_shape());
    }

    //    LOG(INFO) << "finished reshape tons of tensors ";
    OpDataType im_height = this->im_height_, im_width = this->im_width_;
    bool is_input_paramid = false;
    std::vector<OpDataType> im_width_scale = std::vector<OpDataType>(1, this->read_width_scale_);
    std::vector<OpDataType> im_height_scale = std::vector<OpDataType>(1, this->read_height_scale_);
    std::vector<OpDataType> cords_offset_x = std::vector<OpDataType>(1, OpDataType(0));
    std::vector<OpDataType> cords_offset_y = std::vector<OpDataType>(1, this->read_height_offset_);
    OpDataType min_size_w_cur = this->min_size_w_;
    OpDataType min_size_h_cur = this->min_size_h_;

    if (has_img_info_) {
        if (inputs.back()->count(1, inputs.back()->dims()) == 6) {
            //            const OpDataType* img_info_data = inputs.back()->cpu_data();
            _img_info_data_host_tensor->async_copy_from(*inputs.back(), cuda_stream);
            inputs.back()->record_event(cuda_stream);
            inputs.back()->sync();
            const OpDataType* img_info_data = static_cast<const OpDataType*>
                                              (_img_info_data_host_tensor->data());
            im_width = img_info_data[0];
            im_height = img_info_data[1];
            im_width_scale.clear();
            im_height_scale.clear();
            cords_offset_x.clear();
            cords_offset_y.clear();

            for (int n = 0; n < inputs.back()->num(); n++) {
                im_width_scale.push_back(img_info_data[n * 6 + 2]);
                im_height_scale.push_back(img_info_data[n * 6 + 3]);
                CHECK_GT(im_width_scale[n], 0);
                CHECK_GT(im_height_scale[n], 0);
                cords_offset_x.push_back(img_info_data[n * 6 + 4]);
                cords_offset_y.push_back(img_info_data[n * 6 + 5]);
            }
        } else {
            //            CHECK_GT(inputs.back()->count(), 7);
            //            this->pyramid_image_data_param_.ReadFromSerialized(*(inputs.back()), 0);
            //            im_height = pyramid_image_data_param_.img_h_;
            //            im_width = pyramid_image_data_param_.img_w_;
            //            is_input_paramid = true;
        }
    }

    if (this->refine_out_of_map_bbox_) {
        CHECK_GT(im_width, 0);
        CHECK_GT(im_height, 0);
    }

    OpDataType bsz01 = this->bbox_size_add_one_ ? OpDataType(1.0) : OpDataType(0.0);
    //    const OpDataType* rois_st = inputs[0]->cpu_data();
    _rois_st_host_tensor->async_copy_from(*inputs[0], cuda_stream);
    inputs[0]->record_event(cuda_stream);
    inputs[0]->sync();
    const OpDataType* rois_st = static_cast<const OpDataType*>(_rois_st_host_tensor->data());
    const OpDataType* kpts_exist_st = NULL;
    const OpDataType* kpts_reg_st = NULL;

    if (this->has_kpts_) {
        //kpts_reg_st = inputs[this->kpts_reg_bottom_idx_]->cpu_data();
        //        _kpts_reg_st_host_tensor->reshape(inputs[this->kpts_reg_bottom_idx_]->shape());
        _kpts_reg_st_host_tensor->async_copy_from(*inputs[this->kpts_reg_bottom_idx_], cuda_stream);
        inputs[this->kpts_reg_bottom_idx_]->record_event(cuda_stream);
        inputs[this->kpts_reg_bottom_idx_]->sync();
        kpts_reg_st = static_cast<const OpDataType*>(_kpts_reg_st_host_tensor->data());

        if (this->kpts_exist_bottom_idx_ >= 0) {
            //kpts_exist_st = inputs[this->kpts_exist_bottom_idx_]->cpu_data();
            //            _kpts_exist_st_host_tensor->reshape(inputs[this->kpts_exist_bottom_idx_]->shape());
            _kpts_exist_st_host_tensor->async_copy_from(
                *inputs[this->kpts_exist_bottom_idx_], cuda_stream);
            inputs[this->kpts_exist_bottom_idx_]->record_event(cuda_stream);
            inputs[this->kpts_exist_bottom_idx_]->sync();
            kpts_exist_st = static_cast<const OpDataType*>(_kpts_exist_st_host_tensor->data());
        }
    }

    const OpDataType* atrs_reg_st = NULL;

    if (this->has_atrs_) {
        //        atrs_reg_st = inputs[this->atrs_reg_bottom_idx_]->cpu_data();
        //        _atrs_reg_st_host_tensor->reshape(inputs[this->atrs_reg_bottom_idx_]->shape());
        _atrs_reg_st_host_tensor->async_copy_from(*inputs[this->atrs_reg_bottom_idx_], cuda_stream);
        inputs[this->atrs_reg_bottom_idx_]->record_event(cuda_stream);
        inputs[this->atrs_reg_bottom_idx_]->sync();
        atrs_reg_st = static_cast<const OpDataType*>(_atrs_reg_st_host_tensor->data());
    }

    const OpDataType* ftrs_st = NULL;

    if (this->has_ftrs_) {
        //        ftrs_st = inputs[this->ftrs_bottom_idx_]->cpu_data();
        //        _ftrs_st_host_tensor->reshape(inputs[this->ftrs_bottom_idx_]->shape());
        _ftrs_st_host_tensor->async_copy_from(*inputs[this->ftrs_bottom_idx_], cuda_stream);
        inputs[this->ftrs_bottom_idx_]->record_event(cuda_stream);
        inputs[this->ftrs_bottom_idx_]->sync();
        ftrs_st = static_cast<const OpDataType*>(_ftrs_st_host_tensor->data());
    }

    const OpDataType* spmp_st = NULL;

    if (this->has_spmp_) {
        //        spmp_st = inputs[this->spmp_bottom_idx_]->cpu_data();
        //        _spmp_st_host_tensor->reshape(inputs[this->spmp_bottom_idx_]->shape());
        _spmp_st_host_tensor->async_copy_from(*inputs[this->spmp_bottom_idx_], cuda_stream);
        inputs[this->spmp_bottom_idx_]->record_event(cuda_stream);
        inputs[this->spmp_bottom_idx_]->sync();
        spmp_st = static_cast<const OpDataType*>(_spmp_st_host_tensor->data());
    }

    const OpDataType* cam3d_st = NULL;

    if (this->has_cam3d_) {
        //        cam3d_st = inputs[this->cam3d_bottom_idx_]->cpu_data();
        //        _cam3d_st_host_tensor->reshape(inputs[this->cam3d_bottom_idx_]->shape());
        _cam3d_st_host_tensor->async_copy_from(*inputs[this->cam3d_bottom_idx_], cuda_stream);
        inputs[this->cam3d_bottom_idx_]->record_event(cuda_stream);
        inputs[this->cam3d_bottom_idx_]->sync();
        cam3d_st = static_cast<const OpDataType*>(_cam3d_st_host_tensor->data());
    }

    for (int i = 0; i < num_rois_; ++i) {
        //[imid, x1, y1, x2, y2, prb0, prb1, ...]
        const OpDataType* rois = rois_st + i * rois_dim_;
        //[tgx0, tgy0, tgx1, tgy1, tgx2, tgy2, ...] or [num_kpts x kpts_w x kpts_h]
        const OpDataType* kpts_reg = (kpts_reg_st != NULL) ?
                                     (kpts_reg_st + i * kpts_reg_dim_) : NULL;
        //[prb00, prb10, prb20, ..., prb01, prb11, prb21, ...]
        const OpDataType* kpts_exist = (kpts_exist_st != NULL) ?
                                       (kpts_exist_st + i * kpts_cls_dim_) : NULL;
        //[tg0, tg1, tg2, ..., ]
        const OpDataType* atrs_reg = this->has_atrs_ ?
                                     (atrs_reg_st + i * num_atrs_) : NULL;
        //[ft0, ft1, ft2, ..., ]
        const OpDataType* ftrs = this->has_ftrs_ ?
                                 (ftrs_st + i * num_ftrs_) : NULL;
        //[ft0, ft1, ft2, ..., ]
        const OpDataType* spmp = this->has_spmp_ ?
                                 (spmp_st + i * this->spmp_dim_sum_) : NULL;
        //[cam3d0, cam3d1, cam3d2, ..., ]
        const OpDataType* cam3d = this->has_cam3d_ ?
                                  (cam3d_st + i * num_cam3d_) : NULL;

        // filter those width low probs
        if ((1.0 - rois[5]) < this->threshold_objectness_) {
            continue;
        }

        OpDataType score_max = -FLT_MAX;
        int cls_max = -1;

        for (int c = 0; c < this->num_class_; c++) {
            OpDataType score_c = rois[5 + c + 1] - this->threshold_[c];

            if (score_c > score_max) {
                score_max = score_c;
                cls_max = c;
            }
        }

        if (score_max < 0) {
            continue;
        }

        CHECK_GE(cls_max, 0);
        int imid = int(rois[0]);
        BBox<OpDataType> bbox;
        bbox.id = imid;
        bbox.score = rois[5 + cls_max + 1];
        OpDataType ltx = rois[1];
        OpDataType lty = rois[2];
        OpDataType rbx = rois[3];
        OpDataType rby = rois[4];
        // heat_map_a == 1, so it can be used for kpts regress
        OpDataType rois_w = rbx - ltx + bsz01;
        // heat_map_a == 1, so it can be used for kpts regress
        OpDataType rois_h = rby - lty + bsz01;
        OpDataType rois_ctr_x = ltx + 0.5f * (rois_w - bsz01);
        OpDataType rois_ctr_y = lty + 0.5f * (rois_h - bsz01);

        if (is_input_paramid) {
            //            RectBlockPacking<OpDataType>&  block_packer =
            //                this->pyramid_image_data_param_.rect_block_packer_;
            //            int block_id = this->pyramid_image_data_param_.GetBlockIdBy(imid);
            //            const OpDataType heat_map_a = 1;
            //            OpDataType cur_map_start_x = 0, cur_map_start_y = 0;
            //            block_packer.GetFeatureMapStartCoordsByBlockId(block_id,
            //                    heat_map_a, cur_map_start_y, cur_map_start_x);
            //            OpDataType center_buffered_img_w = cur_map_start_x + rois_ctr_x;
            //            OpDataType center_buffered_img_h = cur_map_start_y + rois_ctr_y;
            //            int roi_id = block_packer.GetRoiIdByBufferedImgCoords(
            //                    int(center_buffered_img_h), int(center_buffered_img_w));
            //            if (roi_id <= -1) {
            //                continue;
            //            }
            //            block_packer.GetInputImgCoords(roi_id, cur_map_start_y + lty,
            //                    cur_map_start_x + ltx, bbox.y1, bbox.x1);
            //            block_packer.GetInputImgCoords(roi_id, cur_map_start_y + rby,
            //                    cur_map_start_x + rbx, bbox.y2, bbox.x2);
        } else {
            bbox.x1 = ltx;
            bbox.y1 = lty;
            bbox.x2 = rbx;
            bbox.y2 = rby;
        }

        if (this->has_kpts_) {
            for (int k = 0; k < num_kpts_; k++) {
                OpDataType prb = (kpts_exist != NULL) ? kpts_exist[num_kpts_ + k] : 1.0f;
                OpDataType ptx = 0, pty = 0;

                if (!this->kpts_reg_as_classify_) {
                    OpDataType tgx = kpts_reg[k * 2];
                    OpDataType tgy = kpts_reg[k * 2 + 1];

                    if (this->kpts_do_norm_) {
                        tgx *= this->reg_stds_[this->kpts_reg_norm_idx_st_ + k * 2];
                        tgx += this->reg_means_[this->kpts_reg_norm_idx_st_ + k * 2];
                        tgy *= this->reg_stds_[this->kpts_reg_norm_idx_st_ + k * 2 + 1];
                        tgy += this->reg_means_[this->kpts_reg_norm_idx_st_ + k * 2 + 1];
                    }

                    ptx = tgx * rois_w + 0.5f * (bbox.x1 + bbox.x2);
                    pty = tgy * rois_h + 0.5f * (bbox.y1 + bbox.y2);
                } else {
                    int reg_idx_st = k *
                                     this->kpts_classify_width_ * this->kpts_classify_height_;
                    int reg_idx_ed = reg_idx_st +
                                     this->kpts_classify_width_ * this->kpts_classify_height_;
                    int max_reg_idx = -1;
                    OpDataType max_prb = -FLT_MAX;

                    for (int reg_idx = reg_idx_st; reg_idx < reg_idx_ed; reg_idx++) {
                        if (kpts_reg[reg_idx] > max_prb) {
                            max_reg_idx = reg_idx;
                            max_prb = kpts_reg[reg_idx];
                        }
                    }

                    CHECK_GT(max_reg_idx, -1);
                    OpDataType pad_w = rois_w * this->kpts_classify_pad_ratio_;
                    OpDataType pad_h = rois_h * this->kpts_classify_pad_ratio_;
                    ptx = bbox.x1 - pad_w + (rois_w + 2 * pad_w)
                          / this->kpts_classify_width_ *
                          (max_reg_idx % this->kpts_classify_width_ + 0.5f);
                    pty = bbox.y1 - pad_h + (rois_h + 2 * pad_h)
                          / this->kpts_classify_height_ *
                          (max_reg_idx / this->kpts_classify_width_ + 0.5f);
                }

                bbox.kpts.push_back(std::make_pair(ptx, pty));
                bbox.kpts_prbs.push_back(prb);
            }
        }

        if (this->has_atrs_) {
            for (int k = 0; k < num_atrs_; k++) {
                OpDataType tg = atrs_reg[k];

                if (this->atrs_do_norm_) {
                    tg *= this->reg_stds_[this->atrs_reg_norm_idx_st_ + k];
                    tg += this->reg_means_[this->atrs_reg_norm_idx_st_ + k];
                }

                if (this->atrs_norm_type_[k] == ATRS_NormType_WIDTH) {
                    tg *= rois_w;
                } else if (this->atrs_norm_type_[k] == ATRS_NormType_HEIGHT) {
                    tg *= rois_h;
                } else if (this->atrs_norm_type_[k] == ATRS_NormType_WIDTH_LOG) {
                    tg = expf(tg) * rois_w;
                } else if (this->atrs_norm_type_[k] == ATRS_NormType_HEIGHT_LOG) {
                    tg = expf(tg) * rois_h;
                }

                bbox.atrs.push_back(tg);
            }
        }

        if (this->has_ftrs_) {
            for (int k = 0; k < num_ftrs_; k++) {
                OpDataType ft = ftrs[k];
                bbox.ftrs.push_back(ft);
            }
        }

        if (this->has_spmp_) {
            for (int k = 0; k < this->num_spmp_; k++) {
                std::vector<OpDataType> sp;

                if (!this->spmp_class_aware_[k]) {
                    sp.push_back(this->spmp_label_width_[k]);
                    sp.push_back(this->spmp_label_height_[k]);
                    sp.push_back(this->spmp_pad_ratio_[k]);
                    int sp_st = this->spmp_dim_st_[k];
                    int sp_ed = sp_st +
                                this->spmp_label_width_[k] * this->spmp_label_height_[k];

                    for (int s = sp_st; s < sp_ed; s++) {
                        sp.push_back(spmp[s]);
                    }
                }

                bbox.spmp.push_back(sp);
            }
        }

        if (this->has_cam3d_) {
            if (num_cam3d_ >= 3) {
                bbox.cam3d.x = cam3d[0];
                bbox.cam3d.y = cam3d[1];
                bbox.cam3d.z = cam3d[2];
            }

            if (num_cam3d_ >= 6) {
                bbox.cam3d.h = cam3d[3];
                bbox.cam3d.w = cam3d[4];
                bbox.cam3d.l = cam3d[5];
            }

            if (num_cam3d_ >= 7) {
                bbox.cam3d.o = cam3d[6];
            }

            if (num_cam3d_ >= 31) {
                for (int k = 0; k < 8; k++) {
                    std::vector<OpDataType> xyz;
                    xyz.push_back(cam3d[7 + k * 3 + 0]);
                    xyz.push_back(cam3d[7 + k * 3 + 1]);
                    xyz.push_back(cam3d[7 + k * 3 + 2]);
                    bbox.cam3d.pts3d.push_back(xyz);
                }
            }

            if (num_cam3d_ >= 47) {
                for (int k = 0; k < 8; k++) {
                    std::vector<OpDataType> xy;
                    xy.push_back(cam3d[31 + k * 2 + 0]);
                    xy.push_back(cam3d[31 + k * 2 + 1]);
                    bbox.cam3d.pts2d.push_back(xy);
                }
            }
        }

        if (this->refine_out_of_map_bbox_) {
            bbox.x1 = std::min(std::max(bbox.x1, 0.f), im_width - 1.f);
            bbox.y1 = std::min(std::max(bbox.y1, 0.f), im_height - 1.f);
            bbox.x2 = std::min(std::max(bbox.x2, 0.f), im_width - 1.f);
            bbox.y2 = std::min(std::max(bbox.y2, 0.f), im_height - 1.f);
        }

        OpDataType bw = bbox.x2 - bbox.x1 + bsz01;
        OpDataType bh = bbox.y2 - bbox.y1 + bsz01;

        if (this->min_size_mode_
                == DetectionOutputSSD_HEIGHT_AND_WIDTH) {
            if (bw < min_size_w_cur || bh < min_size_h_cur) {
                continue;
            }
        } else if (this->min_size_mode_
                   == DetectionOutputSSD_HEIGHT_OR_WIDTH) {
            if (bw < min_size_w_cur && bh < min_size_h_cur) {
                continue;
            }
        } else {
            CHECK(false);
        }

        for (int c = 0; c < this->num_class_; ++c) {
            if (rois[5 + c + 1] < this->threshold_[c]) {
                continue;
            }

            bbox.score = rois[5 + c + 1];

            // deal width class aware spmps
            if (this->has_spmp_) {
                for (int k = 0; k < this->num_spmp_; k++) {
                    if (!this->spmp_class_aware_[k]) {
                        continue;
                    }

                    bbox.spmp[k].clear();
                    bbox.spmp[k].push_back(this->spmp_label_width_[k]);
                    bbox.spmp[k].push_back(this->spmp_label_height_[k]);
                    bbox.spmp[k].push_back(this->spmp_pad_ratio_[k]);
                    int sp_st = this->spmp_dim_st_[k] +
                                c * this->spmp_label_width_[k] * this->spmp_label_height_[k];
                    int sp_ed = sp_st +
                                this->spmp_label_width_[k] * this->spmp_label_height_[k];

                    for (int s = sp_st; s < sp_ed; s++) {
                        bbox.spmp[k].push_back(spmp[s]);
                    }
                }
            }

            this->all_candidate_bboxes_[c].push_back(bbox);
        }
    }

    //    if (!is_input_paramid || this->pyramid_image_data_param_.forward_iter_id_ ==
    //            (this->pyramid_image_data_param_.forward_times_for_cur_sample_ - 1))
    {
        for (int class_id = 0; class_id < this->num_class_; ++class_id) {
            std::vector<BBox<OpDataType> >& cur_box_list = this->all_candidate_bboxes_[class_id];
            std::vector<BBox<OpDataType> >& cur_outbox_list = this->output_bboxes_[class_id];

            if (this->nms_use_soft_nms_[class_id]) {
                this->is_candidate_bbox_selected_ = soft_nms_lm(cur_box_list,
                                                    this->nms_overlap_ratio_[class_id], this->nms_top_n_[class_id],
                                                    this->nms_max_candidate_n_[class_id], this->bbox_size_add_one_,
                                                    this->nms_voting_[class_id], this->nms_vote_iou_[class_id]);
            } else {
                this->is_candidate_bbox_selected_ = nms_lm(cur_box_list,
                                                    this->nms_overlap_ratio_[class_id], this->nms_top_n_[class_id],
                                                    false, this->nms_max_candidate_n_[class_id],
                                                    this->bbox_size_add_one_, this->nms_voting_[class_id],
                                                    this->nms_vote_iou_[class_id]);
            }

            cur_outbox_list.clear();

            for (int i = 0; i < this->is_candidate_bbox_selected_.size(); ++i) {
                if (this->is_candidate_bbox_selected_[i]) {
                    int id = im_width_scale.size() > 1 ? cur_box_list[i].id : 0;
                    CHECK_LT(id, im_width_scale.size());
                    cur_box_list[i].x1 = cur_box_list[i].x1
                                         / im_width_scale[id] + cords_offset_x[id];
                    cur_box_list[i].y1 = cur_box_list[i].y1
                                         / im_height_scale[id] + cords_offset_y[id];
                    cur_box_list[i].x2 = cur_box_list[i].x2
                                         / im_width_scale[id] + cords_offset_x[id];
                    cur_box_list[i].y2 = cur_box_list[i].y2
                                         / im_height_scale[id] + cords_offset_y[id];

                    for (int k = 0; k < cur_box_list[i].kpts.size(); k++) {
                        cur_box_list[i].kpts[k].first = cur_box_list[i].kpts[k].first
                                                        / im_width_scale[id] + cords_offset_x[id];
                        cur_box_list[i].kpts[k].second = cur_box_list[i].kpts[k].second
                                                         / im_height_scale[id] + cords_offset_y[id];

                        if ((k < this->kpts_st_for_each_class_[class_id]
                                || k >= this->kpts_ed_for_each_class_[class_id])) {
                            cur_box_list[i].kpts_prbs[k] = 0;
                        }
                    }

                    for (int k = 0; k < cur_box_list[i].atrs.size(); k++) {
                        if (this->atrs_norm_type_[k] == ATRS_NormType_WIDTH ||
                                this->atrs_norm_type_[k] == ATRS_NormType_WIDTH_LOG) {
                            cur_box_list[i].atrs[k] /= im_height_scale[id];
                        } else if (this->atrs_norm_type_[k] == ATRS_NormType_HEIGHT ||
                                   this->atrs_norm_type_[k] == ATRS_NormType_HEIGHT_LOG) {
                            cur_box_list[i].atrs[k] /= im_height_scale[id];
                        }
                    }

                    cur_outbox_list.push_back(cur_box_list[i]);
                }
            }

            cur_box_list.clear();
        }
    }
    return SaberSuccess;
}
template class SaberRCNNDetOutputWithAttr<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberRCNNDetOutputWithAttr, ProposalParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberRCNNDetOutputWithAttr, ProposalParam, NV, AK_INT8);

}
}