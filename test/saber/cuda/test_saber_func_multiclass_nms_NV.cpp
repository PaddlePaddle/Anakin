#include "core/context.h"
#include "funcs/multiclass_nms.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"

using namespace anakin::saber;
#define USE_DUMP_TENSOR 0

TEST(TestSaberFuncNV, test_multiclass_nms) {

    int iter = 10;

    typedef Tensor<X86, AK_FLOAT, NHW> TensorHf3;
    typedef Tensor<NV, AK_FLOAT, NHW> TensorDf3;

    typedef Tensor<X86, AK_FLOAT, NW> TensorHf2;
    typedef Tensor<NV, AK_FLOAT, NW> TensorDf2;

    //! batch size = 16, boxes = 1917, loc = boxes * 4
    Shape sh_loc{16, 1917, 4};
    //! batch size = 16, boxes = 1917, conf = boxes * 21
    Shape sh_conf{16, 21, 1917};

#if USE_DUMP_TENSOR
    std::vector<float> loc_data; //!first input tensor
    std::vector<float> conf_data;//! second input tensor

    std::vector<float> result_data;//! output tensor to compare with

    if (read_file(loc_data, "../../test/saber/data/bbox_data.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(conf_data, "../../test/saber/data/score_data.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(result_data, "../../test/saber/data/detection_output.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    LOG(INFO) << "read bbox data, size: " << loc_data.size();
    LOG(INFO) << "read score data, size: " << conf_data.size();

    TensorDf3 tdloc(loc_data.data(), X86(), 0, sh_loc);
    TensorDf3 tdconf(conf_data.data(), X86(), 0, sh_conf);

#else
    TensorDf3 tdloc(sh_loc);
    TensorDf3 tdconf(sh_conf);
    fill_tensor_device_rand(tdloc, 0.1f, 0.9f);
    fill_tensor_device_rand(tdconf, 0.1f, 0.3f);
#endif

    TensorDf2 tdout;

    std::vector<TensorDf3*> inputs;
    std::vector<TensorDf2*> outputs;

    inputs.push_back(&tdloc);
    inputs.push_back(&tdconf);
    outputs.push_back(&tdout);

    MultiClassNMSParam<TensorDf3> param;
    param.init(0, 100, 100, 0.45f, 0.25f);

    Context<NV> ctx_dev(0, 1, 1);

    MultiClassNMS<NV, AK_FLOAT> det_dev;

    LOG(INFO) << "detection output compute output shape";
    SABER_CHECK(det_dev.compute_output_shape(inputs, outputs, param));
    Shape va_sh{1, 7};
    LOG(INFO) << "output shape pre alloc: " << va_sh[0] << ", " << va_sh[1];
    CHECK_EQ(va_sh == outputs[0]->valid_shape(), true) << "compute shape error";

    LOG(INFO) << "detection output init";
    SABER_CHECK(det_dev.init(inputs, outputs, param, RUNTIME, SABER_IMPL, ctx_dev));

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
              outputs[0]->valid_shape()[1];
    LOG(INFO) << "total time: " << ts << "avg time: " << ts / iter;

    print_tensor_device(*outputs[0]);
    cudaDeviceSynchronize();

#if USE_DUMP_TENSOR
    TensorHf2 thout(outputs[0]->valid_shape());
    thout.copy_from(*outputs[0]);

    TensorHf2 thcmp(result_data.data(), X86(), 0, Shape{16, 7});
    print_tensor_host(thcmp);

    CHECK_EQ(thout.size(), result_data.size()) << "detection compute error";

    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(result_data.data(), thout.data(), thout.size(), max_ratio, max_diff);
    CHECK_EQ(max_ratio < 1e-5f, true) << "detection compute error";
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

