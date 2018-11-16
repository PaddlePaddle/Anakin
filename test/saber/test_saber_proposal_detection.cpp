#include "core/context.h"
#include "saber/funcs/rpn_proposal_ssd.h"
#include "saber/funcs/rcnn_proposal.h"
#include "saber/funcs/rcnn_det_output_with_attr.h"
#include "saber/funcs/proposal_img_scale_to_cam_coords.h"
#include "saber/funcs/rois_anchor_feature.h"
#include "test/saber/test_saber_func.h"
#include "tensor_op.h"
#include "saber/saber_funcs_param.h"
#include "saber_types.h"
#include <vector>

#ifdef NVIDIA_GPU

using namespace anakin::saber;

#define USE_DUMP_TENSOR 0

TEST(TestSaberFunc, test_rcnn_det_output_with_attr) {
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    typedef Tensor<NVHX86> TensorHf4;
    typedef Tensor<NV> TensorDf4;
    std::vector<float> rcnn_rois;
    std::vector<float> cam_coords;
    std::vector<float> im_info;
    TensorHf4 rcnn_rois_host;
    TensorDf4 rcnn_rois_dev;
    TensorHf4 cam_coords_host;
    TensorDf4 cam_coords_dev;
    TensorHf4 im_info_host;
    TensorDf4 im_info_dev;
#if USE_DUMP_TENSOR

    if (read_file(rcnn_rois, "./tensors/rcnn_rois.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(cam_coords, "./tensors/cam_coords.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(im_info, "./tensors/im_info.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

#endif
    Shape rcnn_rois_shape({9, 11, 1, 1});
    Shape cam_coords_shape({9, 47, 1, 1});
    Shape im_info_shape({1, 6, 1, 1});
    rcnn_rois_dev.re_alloc(rcnn_rois_shape);
    rcnn_rois_host.re_alloc(rcnn_rois_shape);
    cam_coords_dev.re_alloc(cam_coords_shape);
    cam_coords_host.re_alloc(cam_coords_shape);
    im_info_dev.re_alloc(im_info_shape);
    im_info_host.re_alloc(im_info_shape);
    fill_tensor_rand(rcnn_rois_host);
    fill_tensor_rand(cam_coords_host);
    static_cast<float*>(im_info_host.mutable_data())[0] = 1408;
    static_cast<float*>(im_info_host.mutable_data())[1] = 800;
    static_cast<float*>(im_info_host.mutable_data())[2] = 0.733;
    static_cast<float*>(im_info_host.mutable_data())[3] = 0.733;
    static_cast<float*>(im_info_host.mutable_data())[4] = 0;
    static_cast<float*>(im_info_host.mutable_data())[5] = 0;
#if USE_DUMP_TENSOR

//    for (int i = 0; i < rcnn_rois_host.size(); ++i) {
//        ((float*)rcnn_rois_host.mutable_data())[i] = rcnn_rois[i];
//    }
//
//    for (int i = 0; i < cam_coords_host.size(); ++i) {
//        ((float*)cam_coords_host.mutable_data())[i] = cam_coords[i];
//    }
//
//    for (int i = 0; i < im_info_host.size(); ++i) {
//        static_cast<float*>(im_info_host.mutable_data())[i] = im_info[i];
//    }

#endif
    rcnn_rois_dev.copy_from(rcnn_rois_host);
    cam_coords_dev.copy_from(cam_coords_host);
    im_info_dev.copy_from(im_info_host);
    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&rcnn_rois_dev);
    inputs.push_back(&cam_coords_dev);
    inputs.push_back(&im_info_dev);
    Context<NV> ctx1(0, 1, 1);
    BBoxRegParam<NV> bbox_reg_param;
    DetectionOutputSSDParam<NV> detection_param({}, {}, {}, {}, {}, {});
    detection_param.num_class = 5;
    detection_param.min_size_h = 18.335f;
    detection_param.min_size_w = 18.335f;
    detection_param.min_size_mode = DetectionOutputSSD_HEIGHT_OR_WIDTH;
    detection_param.read_height_offset = 0;
    detection_param.read_height_scale = 1.f;
    detection_param.read_width_scale = 1.f;
    detection_param.threshold_objectness = 0.f;
    Cam3dParam<NV> cam_3d_param(1);
    detection_param.refine_out_of_map_bbox = true;
    NMSSSDParam<NV> nms_param({0.5}, {1000}, {12000}, {false}, {false}, {0.6}, 600);
    detection_param.nms_param = nms_param;
    detection_param.cam3d_param = cam_3d_param;
    detection_param.has_param = true;
    ProposalParam<NV> rcnn_det_out_param(bbox_reg_param, detection_param);
    RCNNDetOutputWithAttr<NV, AK_FLOAT> rcnn_det_out;
    rcnn_det_out.compute_output_shape(inputs, outputs, rcnn_det_out_param);
    rcnn_det_out.init(inputs, outputs, rcnn_det_out_param, SPECIFY, SABER_IMPL, ctx1);
    LOG(INFO) << " about to operate!!!";
    rcnn_det_out(inputs, outputs, rcnn_det_out_param, ctx1);
    //    print_tensor_device(psroi_rcnn_rois_dev);
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
}

TEST(TestSaberFunc, test_rpn_proposal_ssd) {
    typedef Tensor<NVHX86> TensorHf4;
    typedef Tensor<NV> TensorDf4;
    std::vector<float> bottom_0;
    std::vector<float> bottom_1;
    std::vector<float> bottom_2;
    std::vector<float> top_0;
#if USE_DUMP_TENSOR

    if (read_file(bottom_0, "./tensors/RPNProposalSSD_bottom_0_1_32_33_55_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_1, "./tensors/RPNProposalSSD_bottom_1_1_64_33_55_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_2, "./tensors/RPNProposalSSD_bottom_2_1_6_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(top_0, "./tensors/RPNProposalSSD_top_0_226_5_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

#endif
    Shape b0_s({1, 32, 33, 55}, Layout_NCHW);
    Shape b1_s({1, 64, 33, 55}, Layout_NCHW);
    Shape b2_s({1, 6, 1, 1}, Layout_NCHW);
    Shape t0_s({226, 5, 1, 1}, Layout_NCHW);

    TensorDf4 bottom_0_d;
    TensorDf4 bottom_1_d;
    TensorDf4 bottom_2_d;
    TensorDf4 top_0_d;
    TensorHf4 bottom_0_h;
    TensorHf4 bottom_1_h;
    TensorHf4 bottom_2_h;
    TensorHf4 top_0_h;
    bottom_0_d.re_alloc(b0_s, AK_FLOAT);
    bottom_1_d.re_alloc(b1_s, AK_FLOAT);
    bottom_2_d.re_alloc(b2_s, AK_FLOAT);
    top_0_d.re_alloc(t0_s, AK_FLOAT);
    bottom_0_h.re_alloc(b0_s, AK_FLOAT);
    bottom_1_h.re_alloc(b1_s, AK_FLOAT);
    bottom_2_h.re_alloc(b2_s, AK_FLOAT);
    top_0_h.re_alloc(t0_s, AK_FLOAT);

    static_cast<float*>(bottom_2_h.mutable_data())[0] = 1408;
    static_cast<float*>(bottom_2_h.mutable_data())[1] = 800;
    static_cast<float*>(bottom_2_h.mutable_data())[2] = 0.733;
    static_cast<float*>(bottom_2_h.mutable_data())[3] = 0.733;
    static_cast<float*>(bottom_2_h.mutable_data())[4] = 0;
    static_cast<float*>(bottom_2_h.mutable_data())[5] = 0;
#if USE_DUMP_TENSOR

    for (int i = 0; i < bottom_0_h.size(); ++i) {
        ((float*)bottom_0_h.mutable_data())[i] = bottom_0[i];
    }

    for (int i = 0; i < bottom_1_h.size(); ++i) {
        ((float*)bottom_1_h.mutable_data())[i] = bottom_1[i];
    }

    for (int i = 0; i < bottom_2_h.size(); ++i) {
        ((float*)bottom_2_h.mutable_data())[i] = bottom_2[i];
    }

    for (int i = 0; i < top_0_h.size(); ++i) {
        ((float*)top_0_h.mutable_data())[i] = top_0[i];
    }

#endif

    bottom_0_d.copy_from(bottom_0_h);
    bottom_1_d.copy_from(bottom_1_h);
    bottom_2_d.copy_from(bottom_2_h);
    top_0_d.copy_from(top_0_h);
    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    TensorDf4 top_d;
    TensorHf4 top_h;

    inputs.push_back(&bottom_0_d);
    inputs.push_back(&bottom_1_d);
    inputs.push_back(&bottom_2_d);
    outputs.push_back(&top_d);

    Context<NV> ctx1(0, 1, 1);
    BBoxRegParam<NV> bbox_reg_param({0.005863, 0.004745, -0.092382f, -0.116431f},
    {0.148774, 0.125426, 0.431396, 0.346370});
    DetectionOutputSSDParam<NV> detection_param({}, {}, {16}, {}, {}, {});
    detection_param.min_size_h = 12.834500f;
    detection_param.min_size_w = 12.834500f;
    detection_param.min_size_mode = DetectionOutputSSD_HEIGHT_OR_WIDTH;
    detection_param.threshold_objectness = 0.2f;
    detection_param.refine_out_of_map_bbox = true;
    GenerateAnchorParam<NV> gen_anchor_p({}, {}, {
        22.627417, 32.000000, 45.254834, 45.254834,
        64.000000, 90.509668, 90.509668, 128.000000,
        181.019336, 181.019336, 256.000000, 362.038672,
        362.038672, 512.000000, 724.077344, 800.000000
    }, {
        45.254834, 32.000000, 22.627417, 90.509668,
        64.000000, 45.254834, 181.019336, 128.000000,
        90.509668, 362.038672, 256.000000, 181.019336,
        724.077344, 512.000000, 362.038672, 800.000000
    },
    {}, {}, {}, {});
    NMSSSDParam<NV> nms_param({0.7}, {300}, {3000}, {false}, {false}, {0.7});
    detection_param.gen_anchor_param = gen_anchor_p;
    detection_param.nms_param = nms_param;
    detection_param.has_param = true;
    ProposalParam<NV> rpn_proposal_ssd_param(bbox_reg_param, detection_param);
    RPNProposalSSD<NV, AK_FLOAT> rpn_proposal_ssd;
    rpn_proposal_ssd.compute_output_shape(inputs, outputs, rpn_proposal_ssd_param);
    top_d.re_alloc(top_d.shape());
    rpn_proposal_ssd.init(inputs, outputs, rpn_proposal_ssd_param, SPECIFY, SABER_IMPL, ctx1);
    rpn_proposal_ssd(inputs, outputs, rpn_proposal_ssd_param, ctx1);
    // print_tensor_device(psroi_rois_dev);
    cudaDeviceSynchronize();
    top_d.record_event(ctx1.get_compute_stream());
    top_d.sync();
    LOG(INFO) << " shape: " << top_d.shape()[0] <<
              ", " << top_d.shape()[1] <<
              ", " << top_d.shape()[2] <<
              ", " << top_d.shape()[3];
    top_h.re_alloc(top_d.valid_shape());
//    print_tensor(top_d);
#if USE_DUMP_TENSOR

//    for (int i = 0; i < rois_check.size(); ++i) {
//        rois_check.mutable_data()[i] = rois[i];
//    }

#endif
    top_h.copy_from(top_d);
#if USE_DUMP_TENSOR

//    for (int i = 0; i < rois_check.size(); ++i) {
//        if (fabs(rois_check.data()[i] - rois_host.data()[i]) > 0.001) {
//            LOG(FATAL) << "results error" << i;
//        }
//    }

    LOG(INFO) << "results passed!!!";
#endif
    CUDA_POST_KERNEL_CHECK;
}

TEST(TestSaberFunc, test_rcnn_proposal) {
    typedef Tensor<NVHX86> TensorHf4;
    typedef Tensor<NV> TensorDf4;
    std::vector<float> bottom_0;
    std::vector<float> bottom_1;
    std::vector<float> bottom_2;
    std::vector<float> bottom_3;
    std::vector<float> top_0;
    TensorHf4 bottom_0_h;
    TensorDf4 bottom_0_d;
    TensorHf4 bottom_1_h;
    TensorDf4 bottom_1_d;
    TensorHf4 bottom_2_h;
    TensorDf4 bottom_2_d;
    TensorHf4 bottom_3_h;
    TensorDf4 bottom_3_d;
    TensorHf4 top_0_h;
    TensorDf4 top_0_d;

#if USE_DUMP_TENSOR

    if (read_file(bottom_0, "./tensors/RCNNProposal_bottom_0_226_6_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_1, "./tensors/RCNNProposal_bottom_1_226_24_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_2, "./tensors/RCNNProposal_bottom_2_226_5_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_3, "./tensors/RCNNProposal_bottom_3_1_6_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(top_0, "./tensors/RCNNProposal_top_0_56_11_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

#endif
    Shape b0_s({226, 6, 1, 1}, Layout_NCHW);
    Shape b1_s({226, 24, 1, 1}, Layout_NCHW);
    Shape b2_s({226, 5, 1, 1}, Layout_NCHW);
    Shape b3_s({1, 6, 1, 1}, Layout_NCHW);
    Shape t0_s({56, 11, 1, 1}, Layout_NCHW);

    bottom_0_d.re_alloc(b0_s, AK_FLOAT);
    bottom_1_d.re_alloc(b1_s, AK_FLOAT);
    bottom_2_d.re_alloc(b2_s, AK_FLOAT);
    bottom_3_d.re_alloc(b3_s, AK_FLOAT);
    top_0_d.re_alloc(t0_s, AK_FLOAT);

    bottom_0_h.re_alloc(b0_s, AK_FLOAT);
    bottom_1_h.re_alloc(b1_s, AK_FLOAT);
    bottom_2_h.re_alloc(b2_s, AK_FLOAT);
    bottom_3_h.re_alloc(b3_s, AK_FLOAT);
    top_0_h.re_alloc(t0_s, AK_FLOAT);

    static_cast<float*>(bottom_3_h.mutable_data())[0] = 1408;
    static_cast<float*>(bottom_3_h.mutable_data())[1] = 800;
    static_cast<float*>(bottom_3_h.mutable_data())[2] = 0.733;
    static_cast<float*>(bottom_3_h.mutable_data())[3] = 0.733;
    static_cast<float*>(bottom_3_h.mutable_data())[4] = 0;
    static_cast<float*>(bottom_3_h.mutable_data())[5] = 0;

#if USE_DUMP_TENSOR

    for (int i = 0; i < bottom_0_h.size(); ++i) {
        ((float*)bottom_0_h.mutable_data())[i] = bottom_0[i];
    }
    for (int i = 0; i < bottom_1_h.size(); ++i) {
        ((float*)bottom_1_h.mutable_data())[i] = bottom_1[i];
    }
    for (int i = 0; i < bottom_2_h.size(); ++i) {
        ((float*)bottom_2_h.mutable_data())[i] = bottom_2[i];
    }
    for (int i = 0; i < bottom_3_h.size(); ++i) {
        ((float*)bottom_3_h.mutable_data())[i] = bottom_3[i];
    }
    for (int i = 0; i < top_0_h.size(); ++i) {
        ((float*)top_0_h.mutable_data())[i] = top_0[i];
    }
#endif
    bottom_0_d.copy_from(bottom_0_h);
    bottom_1_d.copy_from(bottom_1_h);
    bottom_2_d.copy_from(bottom_2_h);
    bottom_3_d.copy_from(bottom_3_h);
    top_0_d.copy_from(top_0_h);

    TensorDf4 top_d;
    TensorDf4 top_h;
    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;

    inputs.push_back(&bottom_0_d);
    inputs.push_back(&bottom_1_d);
    inputs.push_back(&bottom_2_d);
    inputs.push_back(&bottom_3_d);
    outputs.push_back(&top_d);

    Context<NV> ctx1(0, 1, 1);
    BBoxRegParam<NV> bbox_reg_param({0.f, 0.f, 0.f, 0.f},
    {0.1f, 0.1f, 0.2f, 0.2f});
    DetectionOutputSSDParam<NV> detection_param({0.f, 0.f, 0.f, 0.f, 0.f}, {}, {}, {}, {}, {});
    detection_param.min_size_h = 18.335f;
    detection_param.min_size_w = 18.335f;
    detection_param.min_size_mode = DetectionOutputSSD_HEIGHT_OR_WIDTH;
    detection_param.threshold_objectness = 0.f;
    detection_param.refine_out_of_map_bbox = true;
    detection_param.num_class = 5;
    detection_param.rpn_proposal_output_score = true;
    detection_param.regress_agnostic = false;
    NMSSSDParam<NV> nms_param({0.5}, {300}, {300}, {false}, {false}, {0.6});
    detection_param.nms_param = nms_param;
    detection_param.has_param = true;
    ProposalParam<NV> rcnn_proposal_param(bbox_reg_param, detection_param);
    RCNNProposal<NV, AK_FLOAT> rcnn_proposal;
    rcnn_proposal.compute_output_shape(inputs, outputs, rcnn_proposal_param);
    top_d.re_alloc(top_d.valid_shape());
    LOG(INFO) << " about to init!!!";
    rcnn_proposal.init(inputs, outputs, rcnn_proposal_param, SPECIFY, SABER_IMPL, ctx1);
    LOG(INFO) << " about to operate!!!";
    rcnn_proposal(inputs, outputs, rcnn_proposal_param, ctx1);
//    print_tensor_valid(top_d);
    cudaDeviceSynchronize();
    top_d.record_event(ctx1.get_compute_stream());
    top_d.sync();
    LOG(INFO) << " shape: " << top_d.shape()[0] <<
              ", " << top_d.shape()[1] <<
              ", " << top_d.shape()[2] <<
              ", " << top_d.shape()[3];
    top_h.re_alloc(top_d.valid_shape());
    top_h.copy_from(top_d);
#if USE_DUMP_TENSOR
#endif

}

TEST(TestSaberFunc, test_proposcal_to_cam_coords) {
    typedef Tensor<NVHX86> TensorHf4;
    typedef Tensor<NV> TensorDf4;

    std::vector<float> bottom_0;
    std::vector<float> bottom_1;
    std::vector<float> bottom_2;
    std::vector<float> bottom_3;
    std::vector<float> bottom_4;
    std::vector<float> bottom_5;
    std::vector<float> bottom_6;
    std::vector<float> bottom_7;
    std::vector<float> bottom_8;
    std::vector<float> bottom_9;
    std::vector<float> bottom_10;
    std::vector<float> bottom_11;
    std::vector<float> bottom_12;
    std::vector<float> top_0;

    TensorHf4 bottom_0_h;
    TensorDf4 bottom_0_d;
    TensorHf4 bottom_1_h;
    TensorDf4 bottom_1_d;
    TensorHf4 bottom_2_h;
    TensorDf4 bottom_2_d;
    TensorHf4 bottom_3_h;
    TensorDf4 bottom_3_d;
    TensorHf4 bottom_4_h;
    TensorDf4 bottom_4_d;
    TensorHf4 bottom_5_h;
    TensorDf4 bottom_5_d;
    TensorHf4 bottom_6_h;
    TensorDf4 bottom_6_d;
    TensorHf4 bottom_7_h;
    TensorDf4 bottom_7_d;
    TensorHf4 bottom_8_h;
    TensorDf4 bottom_8_d;
    TensorHf4 bottom_9_h;
    TensorDf4 bottom_9_d;
    TensorHf4 bottom_10_h;
    TensorDf4 bottom_10_d;
    TensorHf4 bottom_11_h;
    TensorDf4 bottom_11_d;
    TensorHf4 bottom_12_h;
    TensorDf4 bottom_12_d;
    TensorHf4 top_0_h;
    TensorDf4 top_0_d;

#if USE_DUMP_TENSOR

    if (read_file(bottom_0, "./tensors/ProposalImgScaleToCamCoords_bottom_0_56_11_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_1, "./tensors/ProposalImgScaleToCamCoords_bottom_1_1_6_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_2, "./tensors/ProposalImgScaleToCamCoords_bottom_2_56_16_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_3, "./tensors/ProposalImgScaleToCamCoords_bottom_3_56_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_4, "./tensors/ProposalImgScaleToCamCoords_bottom_4_56_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_5, "./tensors/ProposalImgScaleToCamCoords_bottom_5_56_4_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_6, "./tensors/ProposalImgScaleToCamCoords_bottom_6_56_8_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_7, "./tensors/ProposalImgScaleToCamCoords_bottom_7_56_8_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_8, "./tensors/ProposalImgScaleToCamCoords_bottom_8_56_8_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_9, "./tensors/ProposalImgScaleToCamCoords_bottom_9_56_8_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_10, "./tensors/ProposalImgScaleToCamCoords_bottom_10_56_8_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_11, "./tensors/ProposalImgScaleToCamCoords_bottom_11_56_8_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(bottom_12, "./tensors/ProposalImgScaleToCamCoords_bottom_12_1_6_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
    if (read_file(top_0, "./tensors/ProposalImgScaleToCamCoords_top_0_56_47_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
#endif
    Shape b0_s({56, 11, 1, 1}, Layout_NCHW);
    Shape b1_s({1, 6, 1, 1}, Layout_NCHW);
    Shape b2_s({56, 16, 1, 1}, Layout_NCHW);
    Shape b3_s({56, 1, 1, 1}, Layout_NCHW);
    Shape b4_s({56, 1, 1, 1}, Layout_NCHW);
    Shape b5_s({56, 4, 1, 1}, Layout_NCHW);
    Shape b6_s({56, 8, 1, 1}, Layout_NCHW);
    Shape b7_s({56, 8, 1, 1}, Layout_NCHW);
    Shape b8_s({56, 8, 1, 1}, Layout_NCHW);
    Shape b9_s({56, 8, 1, 1}, Layout_NCHW);
    Shape b10_s({56, 8, 1, 1}, Layout_NCHW);
    Shape b11_s({56, 8, 1, 1}, Layout_NCHW);
    Shape b12_s({1, 6, 1, 1}, Layout_NCHW);
    Shape t0_s({56, 8, 1, 1}, Layout_NCHW);

    bottom_0_d.re_alloc(b0_s, AK_FLOAT);
    bottom_1_d.re_alloc(b1_s, AK_FLOAT);
    bottom_2_d.re_alloc(b2_s, AK_FLOAT);
    bottom_3_d.re_alloc(b3_s, AK_FLOAT);
    bottom_4_d.re_alloc(b4_s, AK_FLOAT);
    bottom_5_d.re_alloc(b5_s, AK_FLOAT);
    bottom_6_d.re_alloc(b6_s, AK_FLOAT);
    bottom_7_d.re_alloc(b7_s, AK_FLOAT);
    bottom_8_d.re_alloc(b8_s, AK_FLOAT);
    bottom_9_d.re_alloc(b9_s, AK_FLOAT);
    bottom_10_d.re_alloc(b10_s, AK_FLOAT);
    bottom_11_d.re_alloc(b11_s, AK_FLOAT);
    bottom_12_d.re_alloc(b12_s, AK_FLOAT);
    top_0_d.re_alloc(t0_s, AK_FLOAT);

    bottom_0_h.re_alloc(b0_s, AK_FLOAT);
    bottom_1_h.re_alloc(b1_s, AK_FLOAT);
    bottom_2_h.re_alloc(b2_s, AK_FLOAT);
    bottom_3_h.re_alloc(b3_s, AK_FLOAT);
    bottom_4_h.re_alloc(b4_s, AK_FLOAT);
    bottom_5_h.re_alloc(b5_s, AK_FLOAT);
    bottom_6_h.re_alloc(b6_s, AK_FLOAT);
    bottom_7_h.re_alloc(b7_s, AK_FLOAT);
    bottom_8_h.re_alloc(b8_s, AK_FLOAT);
    bottom_9_h.re_alloc(b9_s, AK_FLOAT);
    bottom_10_h.re_alloc(b10_s, AK_FLOAT);
    bottom_11_h.re_alloc(b11_s, AK_FLOAT);
    bottom_12_h.re_alloc(b12_s, AK_FLOAT);
    top_0_h.re_alloc(t0_s, AK_FLOAT);

    ((float*)bottom_1_h.mutable_data())[0] = 2022.560059f;
    ((float*)bottom_1_h.mutable_data())[1] = 989.388977f;
    ((float*)bottom_1_h.mutable_data())[2] = 2014.050049f;
    ((float*)bottom_1_h.mutable_data())[3] = 570.614990f;
    ((float*)bottom_1_h.mutable_data())[4] = 1.489000f;
    ((float*)bottom_1_h.mutable_data())[5] = -0.020000f;
    static_cast<float*>(bottom_12_h.mutable_data())[0] = 1408;
    static_cast<float*>(bottom_12_h.mutable_data())[1] = 800;
    static_cast<float*>(bottom_12_h.mutable_data())[2] = 0.733;
    static_cast<float*>(bottom_12_h.mutable_data())[3] = 0.733;
    static_cast<float*>(bottom_12_h.mutable_data())[4] = 0;
    static_cast<float*>(bottom_12_h.mutable_data())[5] = 0;

#if USE_DUMP_TENSOR

    for (int i = 0; i < bottom_0.size(); ++i) {
        ((float*)bottom_0_h.mutable_data())[i] = bottom_0[i];
    }
    for (int i = 0; i < bottom_1.size(); ++i) {
        ((float*)bottom_1_h.mutable_data())[i] = bottom_1[i];
    }
    for (int i = 0; i < bottom_2.size(); ++i) {
        ((float*)bottom_2_h.mutable_data())[i] = bottom_2[i];
    }
    for (int i = 0; i < bottom_3.size(); ++i) {
        ((float*)bottom_3_h.mutable_data())[i] = bottom_3[i];
    }
    for (int i = 0; i < bottom_4.size(); ++i) {
        ((float*)bottom_4_h.mutable_data())[i] = bottom_4[i];
    }
    for (int i = 0; i < bottom_5.size(); ++i) {
        ((float*)bottom_5_h.mutable_data())[i] = bottom_5[i];
    }
    for (int i = 0; i < bottom_6.size(); ++i) {
        ((float*)bottom_6_h.mutable_data())[i] = bottom_6[i];
    }
    for (int i = 0; i < bottom_7.size(); ++i) {
        ((float*)bottom_7_h.mutable_data())[i] = bottom_7[i];
    }
    for (int i = 0; i < bottom_8.size(); ++i) {
        ((float*)bottom_8_h.mutable_data())[i] = bottom_8[i];
    }
    for (int i = 0; i < bottom_9.size(); ++i) {
        ((float*)bottom_9_h.mutable_data())[i] = bottom_9[i];
    }
    for (int i = 0; i < bottom_10.size(); ++i) {
        ((float*)bottom_10_h.mutable_data())[i] = bottom_10[i];
    }
    for (int i = 0; i < bottom_11.size(); ++i) {
        ((float*)bottom_11_h.mutable_data())[i] = bottom_11[i];
    }
    for (int i = 0; i < bottom_12.size(); ++i) {
        ((float*)bottom_12_h.mutable_data())[i] = bottom_12[i];
    }
    for (int i = 0; i < top_0.size(); ++i) {
        ((float*)top_0_h.mutable_data())[i] = top_0[i];
    }

#endif
    bottom_0_d.copy_from(bottom_0_h);
    bottom_1_d.copy_from(bottom_1_h);
    bottom_2_d.copy_from(bottom_2_h);
    bottom_3_d.copy_from(bottom_3_h);
    bottom_4_d.copy_from(bottom_4_h);
    bottom_5_d.copy_from(bottom_5_h);
    bottom_6_d.copy_from(bottom_6_h);
    bottom_7_d.copy_from(bottom_7_h);
    bottom_8_d.copy_from(bottom_8_h);
    bottom_9_d.copy_from(bottom_9_h);
    bottom_10_d.copy_from(bottom_10_h);
    bottom_11_d.copy_from(bottom_11_h);
    bottom_12_d.copy_from(bottom_12_h);

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    TensorDf4 top_d;
    TensorHf4 top_h;

    inputs.push_back(&bottom_0_d);
    inputs.push_back(&bottom_1_d);
    inputs.push_back(&bottom_2_d);
    inputs.push_back(&bottom_3_d);
    inputs.push_back(&bottom_4_d);
    inputs.push_back(&bottom_5_d);
    inputs.push_back(&bottom_6_d);
    inputs.push_back(&bottom_7_d);
    inputs.push_back(&bottom_8_d);
    inputs.push_back(&bottom_9_d);
    inputs.push_back(&bottom_10_d);
    inputs.push_back(&bottom_11_d);
    inputs.push_back(&bottom_12_d);
    outputs.push_back(&top_d);

    Context<NV> ctx1(0, 1, 1);
    ProposalImgScaleToCamCoordsParam<NV> proposal_img_param(
    5, {4}, {5}, {1, 2, 4}, {2, 4}, {2, 4}, {
        0.001113, -0.046562, -0.013609, -0.022395,
        0.007158, -0.034202, -0.025865, -0.011027,
        0.013731, -0.006640, 0.011485, 0.007306,
        0.006771, -0.003883, 0.005066, -0.013441
    }, {
        0.163892, 0.127315, 0.160005, 0.134646,
        0.163298, 0.131755, 0.178858, 0.132139,
        0.117660, 0.110846, 0.095226, 0.092817,
        0.135504, 0.115147, 0.075242, 0.134552
    }, {-0.136100}, {0.255000}, {
        1.420370, 2.492460, 1.831520, 2.968770,
        1.584690, 1.638900, 1.546240, 0.650876
    }, {
        0.181710, 0.568219, 0.258787, 0.334992,
        0.147350, 0.217205, 0.203272, 0.111005
    }, {
        1.615090, 1.963140, 1.678670, 2.700380,
        0.000000, 0.000000, 1.044710, 0.000000
    }, {
        0.203030, 0.448403, 0.256539, 0.416969,
        1.000000, 1.000000, 0.265105, 1.000000
    }, {
        4.058110, 5.989820, 4.398580, 11.843000,
        1.524510, 0.000000, 2.373930, 0.000000
    }, {
        0.521282, 1.915760, 0.528109, 1.933210,
        0.269335, 1.000000, 0.534754, 1.000000
    }, {
        0.119747, 0.122519, 0.101991, 0.137277,
        0.128069, 0.000000, 0.168376, 0.000000
    }, {
        0.500000, 0.500000, 0.500000, 0.500000,
        0.500000, 1.000000, 0.500000, 1.000000
    }, {
        0.630652, 0.600944, 0.669030, 0.605602,
        0.220681, 0.000000, 0.267483, 0.000000
    }, {
        0.500000, 0.500000, 0.500000, 0.500000,
        0.500000, 1.000000, 0.500000, 1.000000
    }, {1.789900}, {0.603600}
    );

    proposal_img_param.prj_h_norm_type = ProposalImgScaleToCamCoords_NormType_HEIGHT_LOG;
    proposal_img_param.cam_info_idx_st_in_im_info = 0;
    proposal_img_param.cords_offset_y = 0;
    proposal_img_param.im_width_scale = 1.f;
    proposal_img_param.im_height_scale =  1.f;
    proposal_img_param.has_size3d_and_orien3d = true;
    proposal_img_param.orien_type = ProposalImgScaleToCamCoords_OrienType_PI;
    proposal_img_param.cmp_pts_corner_3d = true;
    proposal_img_param.cmp_pts_corner_2d = true;
    proposal_img_param.with_trunc_ratio = true;
    proposal_img_param.regress_ph_rh_as_whole = true;
    ProposalImgScaleToCamCoords<NV, AK_FLOAT> proposal_img;
    proposal_img.compute_output_shape(inputs, outputs, proposal_img_param);
    top_d.re_alloc(outputs[0]->valid_shape());

    LOG(INFO) << " about to init!!!";
    proposal_img.init(inputs, outputs, proposal_img_param, SPECIFY, SABER_IMPL, ctx1);
    LOG(INFO) << " about to operate!!!";
    proposal_img(inputs, outputs, proposal_img_param, ctx1);
    LOG(INFO) << "finished operate";
    top_d.record_event(ctx1.get_compute_stream());
    top_d.sync();
    LOG(INFO) << " shape: " << top_d.shape()[0] <<
              ", " << top_d.shape()[1] <<
              ", " << top_d.shape()[2] <<
              ", " << top_d.shape()[3];
    top_h.re_alloc(top_d.valid_shape());
    top_h.copy_from(top_d);
//    print_tensor_valid(top_h);
#if USE_DUMP_TENSOR
    double max_r = 0.0;
    double max_d = 0.0;
    tensor_cmp_host((const float*)top_h.data(), (const float*)top_0_h.data(),
            top_0_h.valid_size(), max_r, max_d);

    if (max_r < 0.001) {
        LOG(INFO) << "results passed!!!";
    } else {
        LOG(INFO) << "results fail!!!";
    }
#endif
}
#endif

int main(int argc, const char** argv) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}