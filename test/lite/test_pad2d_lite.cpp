#include "test_lite.h"
#include "saber/lite/funcs/saber_pad2d.h"
#include "saber/saber_types.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;
int test_iter = 1;

int num_in = 9;
int ch_in = 9;
int w_in = 9;
int h_in = 9;
int pad_top = 0;
int pad_bottom = 0;
int pad_left = 0;
int pad_right = 0;
int pad_mode = 0;
float pad_value = 0;
int cluster = 0;
int threads = 4;
typedef Tensor<CPU> TensorHf4;

#define COMPARE_RESULT 1


void basic_pad2d(const float* din, float* dout, int n, int c, int h, int w, int pad_top, int pad_bottom, \
                    int pad_left, int pad_right, PadMode pad_mode, float pad_value){
    int in_w = w - pad_left - pad_right;
    int in_h = h - pad_bottom - pad_top;
    int spatial_size_out = w * h;
    int spatial_size_in = in_w * in_h;
#pragma omp parallel for
    for (int i = 0; i < n * c; ++i) {
        const float* din_batch = din + i * spatial_size_in;
        float* dout_batch = dout + i * spatial_size_out;
        int in_y = 0;
        int in_x = 0;
        for (int y = 0; y < h; ++y){
            for (int x = 0; x < w; ++x){
                switch (pad_mode){
                    case PAD_CONSTANT:
                        in_y = y - pad_top;
                        in_x = x - pad_left;
                        dout_batch[y * w + x] = (in_x >= 0 && in_x < in_w) &&  (in_y >= 0 && in_y < in_h) ? \
                                                    din_batch[in_y * in_w + in_x] : pad_value;
                        break;
                    case PAD_EDGE:
                        in_x = std::min(std::max(pad_left, x), in_w + pad_left - 1) - pad_left;
                        in_y = std::min(std::max(pad_top, y), in_h + pad_top - 1) - pad_top;
                        dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
                        break;
                    case PAD_REFLECT:
                        in_y = y - pad_top;
                        in_x = x - pad_left;
                        in_y = std::max(in_y, -in_y);
                        in_y = std::min(in_y, 2 * in_h - in_y - 2);
                        in_x = std::max(in_x, -in_x);
                        in_x = std::min(in_x, 2 * in_w - in_x - 2);
                        dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
                        break;
                    default:
                        LOGE("ERROR: unknown pad mode: %d\n", pad_mode);
                }
            }
        }
    }
}

void test_pad2d(std::vector<TensorHf4*>& tin, std::vector<int> pad_w, \
    std::vector<int> pad_h, int pad_mode, float pad_value, int cluster_id, int threads) {

    Context ctx1;
    LOG(INFO) << "set runtime context";
    PowerMode mode = (PowerMode)cluster;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }
    TensorHf4 tout_basic;
    TensorHf4 tout_saber;

    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;

    tvout_saber.push_back(&tout_saber);
    tvout_basic.push_back(&tout_basic);

    int numin = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();

    LOG(INFO) << "pool2d param: ";
    LOG(INFO) << "img_num = " << numin;
    LOG(INFO) << "in_channels = " << chin;
    LOG(INFO) << "img_h = " << hin;
    LOG(INFO) << "img_w = " << win;
    LOG(INFO) << "pad_mode = " << pad_mode;
    LOG(INFO) << "pad_value = " << pad_value;
    LOG(INFO) << "pad_top = " << pad_h[0];
    LOG(INFO) << "pad_bottom = " << pad_h[1];
    LOG(INFO) << "pad_left = " << pad_w[0];
    LOG(INFO) << "pad_right = " << pad_w[1];

    Shape shape_out = tin[0]->valid_shape();
    for (int i = 0; i < 4; i++){
        shape_out[i] = tin[0]->valid_shape()[i];
    }
    shape_out[2] += pad_top + pad_bottom;
    shape_out[3] += pad_left + pad_right;
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    const float* din = (const float*)tin[0]->data();

#if COMPARE_RESULT
    LOG(INFO) << "run basic pad2d for precision comparation";
    tout_basic.re_alloc(shape_out);
    float* dout_basic = (float*)tout_basic.mutable_data();
    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        basic_pad2d(din, dout_basic, shape_out[0] , shape_out[1], shape_out[2], shape_out[3], pad_h[0], pad_h[1], \
                    pad_w[0], pad_w[1], (PadMode)pad_mode, pad_value);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "basic pad2d running time, ave: " << to / test_iter << ", min time: " << min_time;
#endif

    SaberPad2D pad2d_saber;
    Pad2DParam pad2d_param(pad_h, pad_w, pad_value, (PadMode)pad_mode);
    LOG(INFO) << "saber pad2d load param";
    pad2d_saber.load_param(&pad2d_param);

    LOG(INFO) << "saber pad2d compute output shape";
    pad2d_saber.compute_output_shape(tin, tvout_saber);
    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape_1: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
        << sh_out_saber[2] << ", " << sh_out_saber[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber pad2d impl init";
    CHECK_EQ(pad2d_saber.init(tin, tvout_saber, ctx1), SaberSuccess) << "init error";

    //! compute
    LOG(INFO) << "saber pad2d compute";
    double to1 = 0;
    SaberTimer t2;
    double min_time1 = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start();
        pad2d_saber.dispatch(tin, tvout_saber);
        t2.end();
        to1 += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time1 = t2.get_average_ms();
        }
    }
    LOG(INFO) << "saber pad2d running time, ave: " << to1 / test_iter << ", min time: " << min_time1;


#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
//    print_tensor(tout_basic);
//    print_tensor(tout_saber);
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";

#endif
}

#if 1
TEST(TestSaberLite, test_func_pad2d_lite) {

    int num = num_in;
    int chin = ch_in;
    int hin = h_in;
    int win = w_in;

    Shape shape_in(num, chin, hin, win);

    std::vector<TensorHf4*> tin;
    TensorHf4 tdin;
    tdin.re_alloc(shape_in, AK_FLOAT);

    tdin.set_dtype(AK_FLOAT);
    fill_tensor_rand(tdin, -1.f, 1.f);
    tin.push_back(&tdin);

    std::vector<int>pad_h;
    std::vector<int>pad_w;
    pad_h.push_back(pad_top);
    pad_h.push_back(pad_bottom);
    pad_w.push_back(pad_left);
    pad_w.push_back(pad_right);

    test_pad2d(tin, pad_w, pad_h, pad_mode, pad_value, cluster, threads);
}
#endif

int main(int argc, const char** argv){

    Env::env_init();

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if (argc >= 4){
        test_iter = atoi(argv[3]);
    }
    if (argc >= 5 ) {
        pad_mode = atoi(argv[4]);
    }
    if (argc >= 6){
        pad_value = atof(argv[5]);
    }
    if (argc >= 7) {
        if (argc < 14) {
            LOG(ERROR) << "usage: ./" << argv[0] << " cluster  threads  test_iter " << \
                "pad_mode pad_value num ch_in h_in w_in pad_top pad_bottom pad_left pad_right";
            return 0;
        }
        num_in = atoi(argv[6]);
        ch_in = atoi(argv[7]);
        h_in = atoi(argv[8]);
        w_in = atoi(argv[9]);
        pad_top = atoi(argv[10]);
        pad_bottom = atoi(argv[11]);
        pad_left = atoi(argv[12]);
        pad_right = atoi(argv[13]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}


