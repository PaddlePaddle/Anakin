#include "funcs/detection_output.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"

using namespace anakin::saber;

int cluster = 0;
int threads = 4;

#define USE_DUMP_TENSOR 1

TEST(TestSaberFuncTest, test_detection_output) {


    Context<ARM> ctx1;
    PowerMode mode = cluster == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
            LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
                LOG(INFO) << "number of threads: " << thread;
#endif
    }

    int iter = 100;

    typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;

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

    if (read_file(loc_data, "data/loc_data.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(conf_data, "data/conf_data.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(prior_data, "data/prior_data.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    if (read_file(result_data, "data/detection_output.txt") != 0) {
        LOG(FATAL) << "file not exist!!!";
    }

    TensorHf4 tdloc(loc_data.data(), ARM(), 0, sh_loc);
    TensorHf4 tdconf(conf_data.data(), ARM(), 0, sh_conf);
    TensorHf4 tdprior(prior_data.data(), ARM(), 0, sh_prior);
#else

    TensorHf4 tdloc(sh_loc);
    TensorHf4 tdconf(sh_conf);
    TensorHf4 tdprior(sh_prior);
    fill_tensor_host_rand(tdloc, 0.f, 1.f);
    fill_tensor_host_rand(tdconf, 0.f, .2f);
    fill_tensor_host_rand(tdprior, 0.f, 1.f);
#endif

    TensorHf4 tdout;

    std::vector<TensorHf4*> inputs;
    std::vector<TensorHf4*> outputs;

    inputs.push_back(&tdloc);
    inputs.push_back(&tdconf);
    inputs.push_back(&tdprior);
    outputs.push_back(&tdout);

    DetectionOutputParam<TensorHf4> param;
    param.init(21, 0, 100, 100, 0.45f, 0.25f, true, false, 2);

    DetectionOutput<ARM, AK_FLOAT> det_dev;

    LOG(INFO) << "detection output compute output shape";
    SABER_CHECK(det_dev.compute_output_shape(inputs, outputs, param));
    Shape va_sh{1, 1, param.keep_top_k, 7};
    LOG(INFO) << "output shape pre alloc: " << va_sh[0] << ", " << va_sh[1] << \
              ", " << va_sh[2] << ", " << va_sh[3];
    CHECK_EQ(va_sh == outputs[0]->valid_shape(), true) << "compute shape error";

    LOG(INFO) << "detection output init";
    SABER_CHECK(det_dev.init(inputs, outputs, param, RUNTIME, SABER_IMPL, ctx1));

    LOG(INFO) << "detection output compute";
    double to = 0;
    double tmin = 1000000;
    SaberTimer<ARM> t1;
    for (int i = 0; i < iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        SABER_CHECK(det_dev(inputs, outputs, param, ctx1));
        //outputs[0]->record_event(ctx1.get_compute_stream());
        //outputs[0]->sync();
        //cudaDeviceSynchronize();
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < tmin) {
            tmin = t1.get_average_ms();
        }
    }
    LOG(INFO) << "output size: " << outputs[0]->valid_shape()[0] << ", " << \
              outputs[0]->valid_shape()[1] << ", " << outputs[0]->valid_shape()[2] << \
              ", " << outputs[0]->valid_shape()[3];
    LOG(INFO) << "avg time: " << to / iter << ", min time: " << tmin;

    print_tensor_host(*outputs[0]);

#if USE_DUMP_TENSOR
    TensorHf4 thout(outputs[0]->valid_shape());
    thout.copy_from(*outputs[0]);

    CHECK_EQ(thout.size(), result_data.size()) << "detection compute error";

    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(result_data.data(), thout.data(), thout.size(), max_ratio, max_diff);
    LOG(INFO) << "detection output error: " << max_diff << ", max_ratio: " << max_ratio;
    CHECK_EQ(max_ratio < 1e-5f, true) << "detection compute error";
#else
    LOG(INFO) << "current unit test need read tensor from disk file";

#endif //USE_DUMP_TENSOR
}

int main(int argc, const char** argv) {
    Env<ARM>::env_init();

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

