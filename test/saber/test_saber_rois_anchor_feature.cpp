#include "core/context.h"
#include "funcs/rois_anchor_feature.h"
#include "tensor_op.h"
#include "saber/saber_funcs_param.h"
#include "saber_types.h"
#include "test_saber_func.h"

using namespace anakin::saber;
#ifdef NVIDIA_GPU

TEST(TestSaberFunc, test_rois_anchor_feature_results) {
    Env<NV>::env_init();
#define USE_DUMP_TENSOR 0
    typedef Tensor<X86> TensorHf4;
    typedef Tensor<NV> TensorDf4;
    std::vector<float> b0;
    std::vector<float> t0;
#if USE_DUMP_TENSOR

    if (read_file(b0, "./tensors/RoisAnchorFeature_bottom_0_226_5_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(t0, "./tensors/RoisAnchorFeature_top_0_226_420_1_1_.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }
#endif

    Shape b0_s({226, 5, 1, 1}, Layout_NCHW);
    Shape t0_s({226, 420, 1, 1}, Layout_NCHW);
    TensorDf4 b0_d; // 0
    TensorHf4 b0_h;
    TensorDf4 t0_d;
    TensorHf4 t0_h;

    RoisAnchorFeatureParam<NV> rois_anchor_feature_param(
            {0.330000, 0.5, 0.67, 1.00, 1.5, 2.0, 3.0});
    rois_anchor_feature_param.ft_ratio_h = true;
    rois_anchor_feature_param.ft_ratio_w = true;
    rois_anchor_feature_param.ft_log_ratio_h = true;
    rois_anchor_feature_param.ft_log_ratio_w = true;
    b0_d.re_alloc(b0_s, AK_FLOAT);
    b0_h.re_alloc(b0_s, AK_FLOAT);
    t0_d.re_alloc(t0_s, AK_FLOAT);
    t0_h.re_alloc(t0_s, AK_FLOAT);
    fill_tensor_rand(b0_h);

#if USE_DUMP_TENSOR

    for (int i = 0; i < b0.size(); ++i) {
        static_cast<float*>(b0_h.mutable_data())[i] = b0[i];
    }

#endif
    b0_d.copy_from(b0_h);
    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&b0_d);
    outputs.push_back(&t0_d);
    Context<NV> ctx1(0, 1, 1);
    RoisAnchorFeature<NV, AK_FLOAT> rois_anchor_feature;
    rois_anchor_feature.compute_output_shape(inputs, outputs,  rois_anchor_feature_param);
            LOG(INFO) << " out shape" <<t0_d.valid_shape();
    t0_d.re_alloc(t0_d.valid_shape(),AK_FLOAT);
    t0_h.re_alloc(t0_d.valid_shape(),AK_FLOAT);

    rois_anchor_feature.init(inputs, outputs, rois_anchor_feature_param, SPECIFY, SABER_IMPL, ctx1);
    rois_anchor_feature(inputs, outputs, rois_anchor_feature_param, ctx1);
    //    print_tensor_device(t0_d);
    cudaDeviceSynchronize();
    t0_d.record_event(ctx1.get_compute_stream());
    t0_d.sync();
    t0_h.copy_from(t0_d);
//    print_tensor_valid(t0_d);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
#if USE_DUMP_TENSOR
    for (int i = 0; i < t0.size(); ++i) {
        if (fabs(t0[i] - static_cast<const float *>(t0_h.data())[i]) > 0.001) {
            LOG(FATAL) << "results error" << i;
        }
    }
    LOG(INFO) << "results passed!!!";
#endif
    CUDA_POST_KERNEL_CHECK;
}
#endif

int main(int argc, const char** argv) {

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}