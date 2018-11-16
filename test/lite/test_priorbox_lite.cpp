#include "saber/lite/funcs/saber_priorbox.h"
#include "test_lite.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 1;

const bool FLAG_RELU = false;

typedef Tensor<CPU> TensorHf4;

void test_arm_priorbox(std::vector<TensorHf4*>& tin, \
    int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;

    Context ctx1;
    PowerMode mode = SABER_POWER_HIGH;
    ctx1.set_run_mode(mode, 1);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    const int test_iter = 100;

    TensorHf4 tout_saber;
    std::vector<TensorHf4*> tvout_saber;
    tvout_saber.push_back(&tout_saber);

    LOG(INFO) << "create priorbox param";
    std::vector<float> min_size{60.f};
    std::vector<float> max_size;
    std::vector<float> aspect_ratio{2};
    std::vector<float> fixed_size{256.f};
    std::vector<float> density{1.0f};
    std::vector<float> fixed_ratio{1.0f};
    std::vector<float> variance{0.1f, 0.1f, 0.2f, 0.2f};
    bool flip = true;
    bool clip = false;
    float step_h = 0;
    float step_w = 0;
    int img_w = 0;
    int img_h = 0;
    float offset = 0.5;

    std::vector<PriorType> order;

    order.push_back(PRIOR_MIN);
    order.push_back(PRIOR_MAX);
    order.push_back(PRIOR_COM);

    SaberPriorBox priorbox_saber;

    //PriorBoxParam param(variance, flip, clip, img_w, img_h, step_w, step_h, offset, order, \
                                    min_size, max_size, aspect_ratio);
   PriorBoxParam param(variance, flip, clip, img_w, img_h, step_w, step_h, offset, order, \
                                    std::vector<float>(), std::vector<float>(), std::vector<float>(), \
                                    fixed_size, fixed_ratio, density);



    LOG(INFO) << "saber priorbox impl init";
    priorbox_saber.load_param(&param);

    priorbox_saber.compute_output_shape(tin, tvout_saber);
    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    Shape shape_out{1, 2, tin[0]->width() * tin[0]->height() * 4 * param._prior_num};

    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

 //   SABER_CHECK(priorbox_saber.init(tin, tvout_saber, param, SPECIFY, SABER_IMPL, ctx1));
     LOG(INFO) << "PriorBox initialization";
    priorbox_saber.init(tin, tvout_saber, ctx1);

    //! compute
    LOG(INFO) << "saber priorbox compute";
    to = 0;
    t1.clear();
    t1.start();

    for (int i = 0; i < test_iter; ++i) {
        priorbox_saber.dispatch(tin, tvout_saber);
    }

    t1.end();
    float ts = t1.get_average_ms();
    printf("total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    print_tensor(*tvout_saber[0]);

}


TEST(TestSaberLite, test_func_priorbox_arm) {

    int width = 300;
    int height = 300;
    int channel = 3;
    int num = 1;
    int w_fea = 19;
    int h_fea = 19;
    int c_fea = 512;

    LOG(INFO) << " input data size, num=" << num << ", channel=" << \
        channel << ", height=" << height << ", width=" << width;

    LOG(INFO) << " input feature tensor size, num=" << num << ", channel=" << \
        c_fea << ", height=" << h_fea << ", width=" << w_fea;
    //! create input output tensor
    Shape sh_fea{num, c_fea, h_fea, w_fea};
    Shape sh_data{num, channel, height, width};
    TensorHf4 tfea(sh_fea);
    TensorHf4 tdata(sh_data);

    std::vector<TensorHf4*> tin;

    tin.push_back(&tfea);
    tin.push_back(&tdata);

    test_arm_priorbox(tin, threads, cluster);
}

int main(int argc, const char** argv){

    Env::env_init();

    // initial logger
    //logger::init(argv[0]);

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

