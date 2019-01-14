#include "test_lite.h"
#include "saber/lite/funcs/saber_softmax.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 4;
int num = 1;
int ch = 1971;
int h = 21;
int w = 1;
int axis = 2;
typedef Tensor<CPU> TensorHf4;

#define COMPARE_RESULT 1

void softmax_basic(TensorHf4& tin, int axis, TensorHf4& tout) {
    Shape shin = tin.valid_shape();
    Shape shtmp = shin;
    int axis_size = shin[axis];
    shtmp[axis] = 1;

    int cnt = shtmp.count();
    int inner_num = tin.count(axis + 1, tin.dims());
    int outer_num = tin.count(0, axis);

    //TensorHf4 tmax(shtmp);

    const float* din = static_cast<const float*>(tin.data());
    float* dout = static_cast<float*>(tout.mutable_data());
    //float* dtmp = tmax.mutable_data();

    for (int i = 0; i < cnt; ++i) {
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        float max_data = din[real_index];
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            max_data = din[real_index] > max_data? din[real_index] : max_data;
        }
        //printf("max data: %.2f\n", max_data);

        real_index = idx_outer * inner_num + idx_inner;
        //! sub, exp and sum
        dout[real_index] = expf(din[real_index] - max_data);
        float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            dout[real_index] = expf(din[real_index] - max_data);
            sum_data += dout[real_index];
        }

        //printf("sum exp data: %.2f\n", sum_data);

        float sum_inv = 1.f / sum_data;

        real_index = idx_outer * inner_num + idx_inner;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            dout[real_index] *= sum_inv;
            real_index += inner_num;
        }
    }
}

TEST(TestSaberLite, test_func_softmax_arm) {
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtine context";
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

    int test_iter = 1;

    int softmax_axis = axis; // channel
    int w_in = w;
    int h_in = h;
    int ch_in = ch;
    int num_in = num;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
        ch_in << ", height=" << h_in << ", width=" << w_in;

    LOG(INFO) << "softmax axis= " << softmax_axis;

    std::vector<TensorHf4*> vin;
    std::vector<TensorHf4*> vout;

    Tensor<CPU> thin(shape_in);
    float* din = static_cast<float*>(thin.mutable_data());
    for (int i = 0; i < thin.size(); ++i) {
        din[i] = i % 4;
    }
    TensorHf4 tout;
    TensorHf4 tout_basic(shape_out);
    vin.push_back(&thin);

#if COMPARE_RESULT
    softmax_basic(thin, softmax_axis, tout_basic);
    //print_tensor(tout_basic);
#endif

    SaberSoftmax softmax_lite;
    SoftmaxParam param(softmax_axis);
    softmax_lite.load_param(&param);

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    vout.push_back(&tout);
    softmax_lite.compute_output_shape(vin, vout);
    CHECK_EQ(shape_out == vout[0]->valid_shape(), true) << "compute shape error";

    LOG(INFO) << "re-alloc tensor buffer";
    vout[0]->re_alloc(vout[0]->valid_shape());

    LOG(INFO) << "softmax initialized to saber impl";
    softmax_lite.init(vin, vout, ctx1);

    SaberTimer t1;

    LOG(INFO) << "saber softmax compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        softmax_lite.dispatch(vin, vout);
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber softmax total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
    //print_tensor(*vout[0]);

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    //TensorHf4 tdiff(tout_basic.valid_shape());
    //tensor_diff(tout_basic, tout_saber, tdiff);
    //print_tensor_host(tdiff);
    tensor_cmp_host(tout_basic, tout, max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
}

int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    Env::env_init(4);

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if (argc >= 4) {
        axis = atoi(argv[3]);
    }
    if (argc >= 5 && argc <= 8) {
        num = atoi(argv[4]);
        ch = atoi(argv[5]);
        h = atoi(argv[6]);
        w = atoi(argv[7]);
    }

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

