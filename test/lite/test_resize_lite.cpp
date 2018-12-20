#include "test_lite.h"
#include "saber/lite/funcs/saber_resize.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 4;
int w_in = 128;
int h_in = 128;
int num_in = 1;
int ch_in = 3;
float width_scale = 2.0f;
float height_scale = 2.0f;
int resize_type = 0;
int log_flag = 0;
typedef Tensor<CPU> TensorHf4;
#define COMPARE_RESULT 1

typedef void (*resize_basic_func)(const float* in_data, int count, int h_in, int w_in, \
            float* out_data, int h_out, int w_out, float width_scale, float height_scale);

void resize_basic_custom(const float* in_data, int count, int h_in, int w_in, \
            float* out_data, int h_out, int w_out, float width_scale, float height_scale) {

    int spatial_in = h_in * w_in;
    int spatial_out = h_out * w_out;
#pragma omp parallel for
    for (int i = 0; i < count; ++i){
        for (int s = 0; s < spatial_out; ++s){
            int x_out = s % w_out;
            int y_out = s / w_out;
            float x_in = x_out * width_scale;
            float y_in = y_out * height_scale;
            int x_in_start = (int)x_in;
            int y_in_start = (int)y_in;
            x_in -= x_in_start;
            y_in -= y_in_start;

            int x_in_end = x_in_start + 1;
            int y_in_end = y_in_start + 1;

            const float w00 = (1.0f - y_in) * (1.0f - x_in);
            const float w01 = x_in * (1.0 - y_in);
            const float w10 = y_in * (1.0 - x_in);
            const float w11 = x_in * y_in;

            int tl_index = y_in_start * w_in + x_in_start;
            int tr_index = y_in_start * w_in + x_in_end;
            int bl_index = y_in_end * w_in + x_in_start;
            int br_index = y_in_end * w_in + x_in_end;

            float tl = in_data[tl_index + i * spatial_in];
            float tr = (x_in_end >= w_in) ? 0 : in_data[tr_index + i * spatial_in];
            float bl = (y_in_end >= h_in) ? 0 : in_data[bl_index + i * spatial_in];
            float br = ((x_in_end >= w_in) || (y_in_end >= h_in)) ? 0 : in_data[br_index + i * spatial_in];
            out_data[s + i * spatial_out] = w00 * tl + w01 * tr + w10 * bl + w11 * br;
        }
    }
}

void resize_basic_corner_align(const float* in_data, int count, int h_in, int w_in, \
            float* out_data, int h_out, int w_out, float width_scale, float height_scale) {

    int spatial_in = h_in * w_in;
    int spatial_out = h_out * w_out;
    width_scale = (float)(w_in - 1) / (w_out - 1);
    height_scale = (float)(h_in - 1) / (h_out - 1);
#pragma omp parallel for
    for (int i = 0; i < count; ++i){
        for (int s = 0; s < spatial_out; ++s){
            int x_out = s % w_out;
            int y_out = s / w_out;
            float x_in = x_out * width_scale;
            float y_in = y_out * height_scale;
            int x_in_start = (int)x_in;
            int y_in_start = (int)y_in;
            x_in -= x_in_start;
            y_in -= y_in_start;

            int x_id = x_in_start < w_in - 1 ? 1 : 0;
            int y_id = y_in_start < h_in - 1 ? 1 : 0;
            int x_in_end = x_in_start + x_id;
            int y_in_end = y_in_start + y_id;

            const float w00 = (1.0f - y_in) * (1.0f - x_in);
            const float w01 = x_in * (1.0 - y_in);
            const float w10 = y_in * (1.0 - x_in);
            const float w11 = x_in * y_in;

            int tl_index = y_in_start * w_in + x_in_start;
            int tr_index = y_in_start * w_in + x_in_end;
            int bl_index = y_in_end * w_in + x_in_start;
            int br_index = y_in_end * w_in + x_in_end;

            float tl = in_data[tl_index + i * spatial_in];
            float tr = in_data[tr_index + i * spatial_in];
            float bl = in_data[bl_index + i * spatial_in];
            float br = in_data[br_index + i * spatial_in];
            out_data[s + i * spatial_out] = w00 * tl + w01 * tr + w10 * bl + w11 * br;
        }
    }
}

void resize_basic_corner_no_align(const float* in_data, int count, int h_in, int w_in, \
            float* out_data, int h_out, int w_out, float width_scale, float height_scale) {

    int spatial_in = h_in * w_in;
    int spatial_out = h_out * w_out;
    width_scale = (float)w_in / w_out;
    height_scale = (float)h_in / h_out;
#pragma omp parallel for
    for (int i = 0; i < count; ++i){
        for (int s = 0; s < spatial_out; ++s){
            int x_out = s % w_out;
            int y_out = s / w_out;
            float x_in = width_scale * (x_out + 0.5f) - 0.5f;
            float y_in = height_scale * (y_out + 0.5f) - 0.5f;
            x_in = x_in < 0 ? 0.f : x_in;
            y_in = y_in < 0 ? 0.f : y_in;
            int x_in_start = (int)x_in;
            int y_in_start = (int)y_in;
            x_in -= x_in_start;
            y_in -= y_in_start;

            int x_id = x_in_start < w_in - 1 ? 1 : 0;
            int y_id = y_in_start < h_in - 1 ? 1 : 0;
            int x_in_end = x_in_start + x_id;
            int y_in_end = y_in_start + y_id;

            const float w00 = (1.0f - y_in) * (1.0f - x_in);
            const float w01 = x_in * (1.0 - y_in);
            const float w10 = y_in * (1.0 - x_in);
            const float w11 = x_in * y_in;

            int tl_index = y_in_start * w_in + x_in_start;
            int tr_index = y_in_start * w_in + x_in_end;
            int bl_index = y_in_end * w_in + x_in_start;
            int br_index = y_in_end * w_in + x_in_end;

            float tl = in_data[tl_index + i * spatial_in];
            float tr = in_data[tr_index + i * spatial_in];
            float bl = in_data[bl_index + i * spatial_in];
            float br = in_data[br_index + i * spatial_in];
            out_data[s + i * spatial_out] = w00 * tl + w01 * tr + w10 * bl + w11 * br;
        }
    }
}
TEST(TestSaberLite, test_func_resize_arm) {
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

    int test_iter = 30;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out(num_in, ch_in, int(h_in * height_scale), int(w_in * width_scale));

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << ch_in << ", height=" << h_in << ", width=" << w_in;

    std::vector<TensorHf4*> vin;
    std::vector<TensorHf4*> vout;

    Tensor<CPU> thin(shape_in);

    //fill_tensor_rand(thin, -1.0, 1.0);
    float* in_data = (float*)thin.mutable_data();
    for (int i = 0; i < thin.valid_size(); ++i){
        in_data[i] = (i + 1) % 255;
    }
    TensorHf4 tout(shape_out);
    TensorHf4 tout_basic(shape_out);
    vin.push_back(&thin);

#if COMPARE_RESULT
    resize_basic_func impl;
    switch (resize_type){
        case 0:
            impl = resize_basic_corner_align;
            break;
        case 1:
            impl = resize_basic_corner_no_align;
            break;
        case 2:
            impl = resize_basic_custom;
            break;
        default:
            LOGE("Unknown resize type %d\n", resize_type);
    }
    SaberTimer timer;
    double to = 0;
    double min = 100000;
    printf("resize type: %d\n", resize_type);
    for (int i = 0; i < test_iter; ++i) {
        timer.clear();
        timer.start();
        impl((const float*)thin.data(),shape_out[0] * shape_out[1], shape_in[2], shape_in[3], \
            (float*)tout_basic.mutable_data(), shape_out[2], shape_out[3], 1.0f / width_scale, 1.0f / height_scale);
        timer.end();
        double tdiff = timer.get_average_ms();
        to += tdiff;
        if (tdiff < min) {
            min = tdiff;
        }
    }
#endif

    SaberResize resize_lite;
    ResizeParam param((ResizeType)resize_type, width_scale, height_scale);
    resize_lite.load_param(&param);

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    vout.push_back(&tout);
    resize_lite.compute_output_shape(vin, vout);
    CHECK_EQ(shape_out == vout[0]->valid_shape(), true) << "compute shape error";

    LOG(INFO) << "re-alloc tensor buffer";
    vout[0]->re_alloc(vout[0]->valid_shape());

    LOG(INFO) << "resize initialized to saber impl";
    resize_lite.init(vin, vout, ctx1);

    SaberTimer t1;

    LOG(INFO) << "saber resize compute";
    double sum = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        resize_lite.dispatch(vin, vout);
        t1.end();
        double tdiff = t1.get_average_ms();
        sum += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(tout_basic, tout, max_ratio, max_diff);
    CHECK_EQ(fabsf(max_ratio) < 1e-5f || max_diff < 1e-4, true) << "compute result error" \
     << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    printf("basic resize total time: %.4fms, avg time : %.4fms, min time: %.4fms\n", to, to / test_iter, min);
#endif
    printf("saber resize total time : %.4fms, avg time : %.4fms, min time: %.4fms\n", sum, sum / test_iter, min_time);
    // print_tensor(*vin[0]);
    // print_tensor(tout_basic);
    // print_tensor(*vout[0]);
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
    if (argc >= 4){
        num_in = atoi(argv[3]);
    }
    if (argc >= 5){
        ch_in = atoi(argv[4]);
    }
    if (argc >= 6){
        h_in = atoi(argv[5]);
    }
    if (argc >= 7){
        w_in = atoi(argv[6]);
    }
    if (argc >= 8){
        width_scale = atof(argv[7]);
    }
    if (argc >= 9){
        height_scale = atof(argv[8]);
    }
    if (argc >= 10){
        resize_type = atoi(argv[9]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

