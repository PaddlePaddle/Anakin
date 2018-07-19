#include "saber/lite/funcs/saber_concat.h"
#include "test_lite.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 4;

typedef Tensor<CPU, AK_FLOAT> TensorHf4;

TEST(TestSaberLite, test_func_concat_arm) {

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

    int concat_axis = 2; // height
    int w_in = 8;
    int sum_hin = 8;
    std::vector<int> h_in = {1, 2, 3, 2};
    int ch_in = 2;
    int num_in = 2;

    Shape shape_out(num_in, ch_in, sum_hin, w_in);

    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in[0] << ", width=" << w_in;
    LOG(INFO) << "input 4 tensor: h = " << h_in[0] << ", " << \
              h_in[1] << ", " << h_in[2] << ", " << h_in[3];
    LOG(INFO) << "concat axis= " << concat_axis;

    std::vector<TensorHf4*> vin;
    std::vector<TensorHf4*> vout;

    TensorHf4 th1, th2, th3, th4;
    Shape sh1(num_in, ch_in, h_in[0], w_in);
    Shape sh2(num_in, ch_in, h_in[1], w_in);
    Shape sh3(num_in, ch_in, h_in[2], w_in);
    Shape sh4(num_in, ch_in, h_in[3], w_in);
    th1.re_alloc(sh1);
    th2.re_alloc(sh2);
    th3.re_alloc(sh3);
    th4.re_alloc(sh4);

    for (int j = 0; j < th1.size(); ++j) {
        th1.mutable_data()[j] = j;
    }

    for (int j = 0; j < th2.size(); ++j) {
        th2.mutable_data()[j] = j;
    }

    for (int j = 0; j < th3.size(); ++j) {
        th3.mutable_data()[j] = j;
    }

    for (int j = 0; j < th4.size(); ++j) {
        th4.mutable_data()[j] = j;
    }

    vin.push_back(&th1);
    vin.push_back(&th2);
    vin.push_back(&th3);
    vin.push_back(&th4);

    TensorHf4 tdev_out;

    vout.push_back(&tdev_out);


    SaberConcat concat_lite;
    ConcatParam param(concat_axis);
    concat_lite.load_param(&param);

    LOG(INFO) << "concat compute output shape";
    concat_lite.compute_output_shape(vin, vout);
    LOG(INFO) << "output shape: " << tdev_out.valid_shape()[0] << ", " \
              << tdev_out.valid_shape()[1] << ", " << tdev_out.valid_shape()[2] \
              << ", " << tdev_out.valid_shape()[3];
    tdev_out.re_alloc(shape_out);

    LOG(INFO) << "concat initialization";
    concat_lite.init(vin, vout, ctx1);

    LOG(INFO) << "concat compute";
    t1.clear();
    t1.start();

    for (int i = 0; i < test_iter; ++i) {
        concat_lite.dispatch(vin, vout);
    }

    t1.end();
    float ts = t1.get_average_ms();
    printf("total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    print_tensor(*vout[0]);
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env::env_init(4);

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

