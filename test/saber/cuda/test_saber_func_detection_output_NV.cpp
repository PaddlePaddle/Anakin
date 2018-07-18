#include "core/context.h"
#include "funcs/detection_output.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"

using namespace anakin::saber;

#define USE_DUMP_TENSOR 0

TEST(TestSaberFuncNV, test_detection_output) {

    int iter = 10;

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    //! batch size = 16, boxes = 1917, loc = boxes * 4
    Shape sh_loc{16, 7668, 1, 1};
    //! batch size = 16, boxes = 1917, conf = boxes * 21
    Shape sh_conf{16, 40257, 1, 1};
    //! size = 2, boxes * 4
    Shape sh_prior{1, 1, 2, 7668};

    Shape sh_res_cmp{1, 1, 16, 7};

#if USE_DUMP_TENSOR
    std::vector<float> loc_data; //!first input tensor
    std::vector<float> conf_data;//! second input tensor
    std::vector<float> prior_data;//! third input tensor

    std::vector<float> result_data;//! output tensor to compare with

    if (read_file(loc_data, "../../test/saber/data/loc_data.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(conf_data, "../../test/saber/data/conf_data.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(prior_data, "../../test/saber/data/prior_data.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(result_data, "../../test/saber/data/detection_output.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    TensorDf4 tdloc(loc_data.data(), X86(), 0, sh_loc);
    TensorDf4 tdconf(conf_data.data(), X86(), 0, sh_conf);
    TensorDf4 tdprior(prior_data.data(), X86(), 0, sh_prior);
#else

    TensorDf4 tdloc(sh_loc);
    TensorDf4 tdconf(sh_conf);
    TensorDf4 tdprior(sh_prior);
    fill_tensor_device_rand(tdloc, 0.f, 1.f);
    fill_tensor_device_rand(tdconf, 0.f, .2f);
    fill_tensor_device_rand(tdprior, 0.f, 1.f);
#endif

    TensorDf4 tdout;

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;

    inputs.push_back(&tdloc);
    inputs.push_back(&tdconf);
    inputs.push_back(&tdprior);
    outputs.push_back(&tdout);

    DetectionOutputParam<TensorDf4> param;
    param.init(21, 0, 100, 100, 0.45f, 0.25f, true, false, 2);

    Context<NV> ctx_dev(0, 1, 1);

    DetectionOutput<NV, AK_FLOAT> det_dev;

    LOG(INFO) << "detection output compute output shape";
    SABER_CHECK(det_dev.compute_output_shape(inputs, outputs, param));
    Shape va_sh{1, 1, param.keep_top_k, 7};
    LOG(INFO) << "output shape pre alloc: " << va_sh[0] << ", " << va_sh[1] << \
              ", " << va_sh[2] << ", " << va_sh[3];
    CHECK_EQ(va_sh == outputs[0]->valid_shape(), true) << "compute shape error";

    LOG(INFO) << "detection output init";
    SABER_CHECK(det_dev.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx_dev));

    LOG(INFO) << "detection output compute";
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < iter; ++i) {
        SABER_CHECK(det_dev(inputs, outputs, param, ctx_dev));
        outputs[0]->record_event(ctx_dev.get_compute_stream());
        outputs[0]->sync();
        //cudaDeviceSynchronize();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    LOG(INFO) << "output size: " << outputs[0]->valid_shape()[0] << ", " << \
              outputs[0]->valid_shape()[1] << ", " << outputs[0]->valid_shape()[2] << \
              ", " << outputs[0]->valid_shape()[3];
    LOG(INFO) << "total time: " << ts << "avg time: " << ts / iter;

    print_tensor_device(*outputs[0]);
    cudaDeviceSynchronize();

#if USE_DUMP_TENSOR
    TensorHf4 thout(outputs[0]->valid_shape());
    thout.copy_from(*outputs[0]);

    CHECK_EQ(thout.size(), result_data.size()) << "detection compute error";

    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(result_data.data(), thout.data(), thout.size(), max_ratio, max_diff);
    CHECK_EQ(max_ratio < 1e-6f, true) << "detection compute error";
    LOG(INFO) << "detection output error: " << max_diff << ", max_ratio: " << max_ratio;
#else
    LOG(INFO) << "current unit test need read tensor from disk file";

#endif //USE_DUMP_TENSOR
}

int main(int argc, const char** argv) {
    Env<NV>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

