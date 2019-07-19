#include "core/context.h"
#include "funcs/dfmb_psroi_align.h"
#include "tensor_op.h"
#include "saber/saber_funcs_param.h"
#include "saber_types.h"
#include "test_saber_func.h"
#include "debug.h"

using namespace anakin::saber;
#ifdef NVIDIA_GPU

TEST(TestSaberFunc, test_dfmb_psroi_align_results) {
    Env<NV>::env_init();
#define USE_DUMP_TENSOR 1
    typedef Tensor<X86> TensorHf4;
    typedef Tensor<NV> TensorDf4;
    std::vector<float> rois;
    std::vector<float> ft_add_left_right;
    std::vector<float> psroi_rois;
#if USE_DUMP_TENSOR

    if (read_file(ft_add_left_right,
                  "./dfmb_anakin/record+DFMBPSROIAlign+psroi_rois_3d+in+0+1_588_16_16_.txt", ' ', 1) != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(rois, "./record+DFMBPSROIAlign+psroi_rois_3d+in+1+29_5_1_1_.txt", ' ', 1) != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(psroi_rois, "record+DFMBPSROIAlign+psroi_rois_3d+out+0+29_12_7_7_.txt", ' ',
                  1) != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

#endif
    //    Shape ft_add_left_right_shape({1, 490, 16, 16});
    //    Shape rois_shape({17, 5, 1, 1});
    Shape ft_add_left_right_shape({1, 588, 16, 16});
    Shape rois_shape({29, 5, 1, 1});
    TensorDf4 ft_add_left_right_dev; // 0
    TensorHf4 ft_add_left_right_host;
    TensorDf4 rois_dev; // 1
    TensorHf4 rois_host;
    TensorDf4 psroi_rois_dev;
    TensorHf4 psroi_rois_host;
    TensorHf4 psroi_check;
    float heat_map_a = 16;
    int output_dim = 12;
    DFMBPSROIAlignParam<NV> dfmb_psroi_align_param(heat_map_a, output_dim);
    ft_add_left_right_dev.re_alloc(ft_add_left_right_shape, AK_FLOAT);
    ft_add_left_right_host.re_alloc(ft_add_left_right_shape, AK_FLOAT);
    rois_dev.re_alloc(rois_shape, AK_FLOAT);
    rois_host.re_alloc(rois_shape, AK_FLOAT);
    fill_tensor_rand(ft_add_left_right_host);
    fill_tensor_rand(rois_host);
#if USE_DUMP_TENSOR

    for (int i = 0; i < ft_add_left_right.size(); ++i) {
        static_cast<float*>(ft_add_left_right_host.mutable_data())[i] = ft_add_left_right[i];
    }

    for (int i = 0; i < rois.size(); ++i) {
        static_cast<float*>(rois_host.mutable_data())[i] = rois[i];
    }

#endif
    //    print_tensor(ft_add_left_right_host);
    ft_add_left_right_dev.copy_from(ft_add_left_right_host);
    rois_dev.copy_from(rois_host);
    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&ft_add_left_right_dev);
    inputs.push_back(&rois_dev);
    outputs.push_back(&psroi_rois_dev);
    Context<NV> ctx1(0, 1, 1);
    DFMBPSROIAlign<NV, AK_FLOAT> dfmb_psroi_align;
    dfmb_psroi_align.compute_output_shape(inputs, outputs,  dfmb_psroi_align_param);
    LOG(INFO) << " out shape" << psroi_rois_dev.valid_shape();
    psroi_rois_dev.re_alloc(psroi_rois_dev.valid_shape(), AK_FLOAT);
    psroi_rois_host.re_alloc(psroi_rois_dev.valid_shape(), AK_FLOAT);
    psroi_check.re_alloc(psroi_rois_dev.valid_shape(), AK_FLOAT);
#if USE_DUMP_TENSOR

    for (int i = 0; i < psroi_rois.size(); ++i) {
        static_cast<float*>(psroi_check.mutable_data())[i] = psroi_rois[i];
    }

#endif
    dfmb_psroi_align.init(inputs, outputs, dfmb_psroi_align_param, SPECIFY, SABER_IMPL, ctx1);
    dfmb_psroi_align(inputs, outputs, dfmb_psroi_align_param, ctx1);
    //    print_tensor_device(psroi_rois_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
    psroi_rois_dev.record_event(ctx1.get_compute_stream());
    psroi_rois_dev.sync();
    psroi_rois_host.copy_from(psroi_rois_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
    write_tensorfile(psroi_rois_host, "ak_result.txt");
#if USE_DUMP_TENSOR

    for (int i = 0; i < psroi_rois.size(); ++i) {
        if (fabs(static_cast<const float*>(psroi_check.data())[i] - static_cast<const float*>
                 (psroi_rois_host.data())[i]) > 0.001) {
            LOG(FATAL) << "results error" << i;
        }
    }

    LOG(INFO) << "results passed!!!";
#endif
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}
#endif

int main(int argc, const char** argv) {

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}