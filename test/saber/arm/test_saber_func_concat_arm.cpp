#include "core/context.h"
#include "funcs/concat.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

int cluster = 0;
int threads = 4;

typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;

TEST(TestSaberFuncTest, test_func_concat_arm) {

    const int test_iter = 100;

    int concat_axis = 2; // height
    int w_in = 8;
    int sum_hin = 8;
    std::vector<int> h_in = {1, 2, 3, 2};
    int ch_in = 2;
    int num_in = 2;

    Shape shape_out(num_in, ch_in, sum_hin, w_in);

    ConcatParam<TensorHf4> param(concat_axis);

    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in[0] << ", width=" << w_in;
    LOG(INFO) << "input 4 tensor: h = " << h_in[0] << ", " << \
              h_in[1] << ", " << h_in[2] << ", " << h_in[3];
    LOG(INFO) << "concat axis= " << param.axis;

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

    // start Reshape & doInfer
    Context<ARM> ctx1(0, 1, 1);

    Concat<ARM, AK_FLOAT> concat_arm;

    TensorHf4 tdev_out;

    vout.push_back(&tdev_out);

    LOG(INFO) << "concat compute output shape";
    concat_arm.compute_output_shape(vin, vout, param);
    LOG(INFO) << "output shape: " << tdev_out.valid_shape()[0] << ", " \
              << tdev_out.valid_shape()[1] << ", " << tdev_out.valid_shape()[2] \
              << ", " << tdev_out.valid_shape()[3];
    tdev_out.re_alloc(shape_out);

    LOG(INFO) << "concat initialization";
    concat_arm.init(vin, vout, param, SPECIFY, SABER_IMPL, ctx1);

    LOG(INFO) << "concat compute";
    SaberTimer<ARM> t1;
    t1.clear();
    t1.start(ctx1);

    for (int i = 0; i < test_iter; ++i) {
        concat_arm(vin, vout, param, ctx1);
        vout[0]->record_event(ctx1.get_compute_stream());
        vout[0]->sync();
    }

    t1.end(ctx1);
    float ts = t1.get_average_ms();
    printf("total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    print_tensor_host(*vout[0]);
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<ARM>::env_init(4);

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

