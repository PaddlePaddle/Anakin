#include "test_lite.h"
#include "saber/lite/funcs/neon/impl/pooling_arm_impl.h"
#include "saber/lite/funcs/saber_pooling.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 4;
int test_iter = 10;
bool compare_result = false;
int num = 1;
int ch_in = 32;
int h_in = 112;
int w_in = 112;
int pool_case = 0;

typedef void (*POOL_FUNC)(const void*, void*, int, int, int, int, \
                          int, int, int, PoolingType, bool, int, int, int, int, int, int);
typedef Tensor<CPU> TensorH;

void pooling_basic_test(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          PoolingType type, bool global, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int pad_w, int pad_h) {
    //no need to pad input tensor, border is zero pad inside this function

    int size_channel_in = win * hin;
    int size_channel_out = wout * hout;

    signed char* data_out = static_cast<signed char*>(dout);
    const signed char* data_in = static_cast<const signed char*>(din);

    if (global) {
        switch (type) {
            case Pooling_max:
                for (int n = 0; n < num; ++n) {
                    signed char* data_out_batch = data_out + n * chout * size_channel_out;
                    const signed char* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                    for (int c = 0; c < chout; ++c) {
                        const signed char* data_in_channel = data_in_batch + c * size_channel_in;//in address
                        signed char max_val = std::numeric_limits<signed char>::min();
                        for (int i = 0; i < size_channel_in; ++i) {
                            if (max_val < data_in_channel[i]){
                                max_val = data_in_channel[i];
                            }
                            data_out_batch[c] = max_val;
                        }
                    }
                }
                break;

            case Pooling_average_include_padding:

            case Pooling_average_exclude_padding:
                for (int n = 0; n < num; ++n) {
                    signed char* data_out_batch = data_out + n * chout * size_channel_out;
                    const signed char* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                    for (int c = 0; c < chout; ++c) {
                        const signed char* data_in_channel = data_in_batch + c * size_channel_in;//in address
                        int sum = 0;
                        for (int i = 0; i < size_channel_in; ++i) {
                            sum += int(data_in_channel[i]);
                        }
                        data_out_batch[c] = (signed char)(sum / size_channel_in);
                    }
                }
                break;
            default:
                //printf("not support\n");
                LOGE("not support\n");
        }
        return;
    }

    switch (type) {
        case Pooling_max:
            for (int n = 0; n < num; ++n) {
                signed char* data_out_channel = data_out + n * chout * size_channel_out;
                const signed char* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < chout; q++) {

                    signed char* data_out_row = data_out_channel + q * size_channel_out;
                    const signed char* data_in_channel = data_in_batch + q * size_channel_in;

                    for (int i = 0; i < hout; i++) {
                        for (int j = 0; j < wout; j++) {
                            int hstart = i * stride_h - pad_h;
                            int wstart = j * stride_w - pad_w;
                            int hend = std::min(hstart + kernel_h, hin + pad_h);
                            int wend = std::min(wstart + kernel_w, win + pad_w);
                            hstart = std::max(hstart, 0);
                            wstart = std::max(wstart, 0);
                            hend = std::min(hend, hin);
                            wend = std::min(wend, win);

                            signed char max_val = std::numeric_limits<signed char>::min();
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    if (data_in_channel[h * win + w] > max_val){
                                        max_val = data_in_channel[h * win + w];
                                    }
                                }
                            }
                            data_out_row[j] = max_val;
                        }
                        data_out_row += wout;
                    }
                }
            }
            break;

        case Pooling_average_include_padding:
            for (int n = 0; n < num; ++n) {
                int pool_size = kernel_w * kernel_h;
                signed char* data_out_channel = data_out + n * chout * size_channel_out;
                const signed char* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < chout; q++) {

                    signed char* data_out_row = data_out_channel + q * size_channel_out;
                    const signed char* data_in_channel = data_in_batch + q * size_channel_in;
                    for (int i = 0; i < hout; i++) {
                        for (int j = 0; j < wout; j++) {
                            int hstart = i * stride_h - pad_h;
                            int wstart = j * stride_w - pad_w;
                            int hend = std::min(hstart + kernel_h, hin + pad_h);
                            int wend = std::min(wstart + kernel_w, win + pad_w);
                            hstart = std::max(hstart, 0);
                            wstart = std::max(wstart, 0);
                            hend = std::min(hend, hin);
                            wend = std::min(wend, win);

                            int sum = 0;
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    sum += int(data_in_channel[h * win + w]);
                                }
                            }
                            data_out_row[j] = (signed char)(sum / pool_size);
                        }
                        data_out_row += wout;
                    }
                }
            }
            break;
        case Pooling_average_exclude_padding:
            for (int n = 0; n < num; ++n) {
                signed char* data_out_channel = data_out + n * chout * size_channel_out;
                const signed char* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < chout; q++) {

                    signed char* data_out_row = data_out_channel + q * size_channel_out;
                    const signed char* data_in_channel = data_in_batch + q * size_channel_in;
                    for (int i = 0; i < hout; i++) {
                        for (int j = 0; j < wout; j++) {
                            int hstart = i * stride_h - pad_h;
                            int wstart = j * stride_w - pad_w;
                            int hend = std::min(hstart + kernel_h, hin + pad_h);
                            int wend = std::min(wstart + kernel_w, win + pad_w);
                            hstart = std::max(hstart, 0);
                            wstart = std::max(wstart, 0);
                            hend = std::min(hend, hin);
                            wend = std::min(wend, win);

                            int sum = 0;
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    sum += int(data_in_channel[h * win + w]);
                                }
                            }
                            int pool_size = (hend - hstart) * (wend - wstart);
                            data_out_row[j] = (signed char)(sum / pool_size);
                        }
                        data_out_row += wout;
                    }
                }
            }
            break;
        default:
            //printf("not support\n");
            LOGE("not support\n");
    }
}
void test_arm_pooling_int8(TensorH& tin, int threads, int cluster_id, int pool_case) {
    
#ifdef __aarch64__
    LOG(INFO) << "using arm64";
#else
    LOG(INFO) << "using armv7";
#endif
    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;
    SaberTimer t2;
    
    Context ctx1;
    PowerMode mode = cluster_id == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }
    
    TensorH tout_basic;
    TensorH tout_saber;
    
    
    int num = tin.num();
    int chin = tin.channel();
    int hin = tin.height();
    int win = tin.width();
    
    LOG(INFO) << "pooling param: ";
    LOG(INFO) << "img_num = " << num;
    LOG(INFO) << "in_channels = " << chin;
    LOG(INFO) << "img_h = " << hin;
    LOG(INFO) << "img_w = " << win;

    int kernel = 2;
    int stride = 2;
    int pad = 0;
    bool global = false;
    POOL_FUNC pool_func = nullptr;
    PoolingType type = Pooling_max;
 
    switch (pool_case){
        case 0:  //global
            global = true;
            pool_func = pooling_global_int8;
            type = Pooling_max;
            LOG(INFO) << "pool case: global pooling";
            break;
        case 1: //2x2s2 max
            kernel = 2;
            stride = 2;
            pad = 0;
            global = false;
            pool_func = pooling2x2s2_max_int8;
            type = Pooling_max;
            LOG(INFO) << "pool case: pooling2x2s2_max";
            break;
        case 2:  //3x3s1p1 max
            kernel = 3;
            stride = 1;
            pad = 1;
            global = false;
            pool_func = pooling3x3s1p1_max_int8;
            type = Pooling_max;
            LOG(INFO) << "pool case: pooling3x3s1p1_max";
            break;
        case 3: //3x3s2p1 max
            kernel = 3;
            stride = 2;
            pad = 1;
            global = false;
            pool_func = pooling3x3s2p1_max_int8;
            type = Pooling_max;
            LOG(INFO) << "pool case: pooling3x3s2p1_max";
            break;
        case 4: //3x3s2p0 max
            kernel = 3;
            stride = 2;
            pad = 0;
            global = false;
            pool_func = pooling3x3s2p0_max_int8;
            type = Pooling_max;
            LOG(INFO) << "pool case: pooling3x3s2p0_max";
            break;
        case 5: //2x2s2 ave
            kernel = 2;
            stride = 2;
            pad = 0;
            global = false;
            pool_func = pooling2x2s2_ave_int8;
            type = Pooling_average_exclude_padding;
            LOG(INFO) << "pool case: pooling2x2s2_ave";
            break;
        default:
            LOG(FATAL) << "kernel: " << kernel << ", stride: " << stride << ", pad: " \
                        << pad << ", no implement";
            break;
    }
    int wout = 1;
    int hout = 1;
    if (!global) {
        int hin = tin.height(); // P
        hout = static_cast<int>(std::max(0.f, ceilf(static_cast<float>(
                                                         hin + 2 * pad - kernel) / stride))) + 1;
        int win = tin.width(); // Q
        wout = static_cast<int>(std::max(0.f, ceilf(static_cast<float>(
                                                         win + 2 * pad - kernel) / stride))) + 1;
    }
    Shape shape_out(num, chin, hout, wout);
    if (compare_result) {
        tout_basic.re_alloc(shape_out, AK_INT8);
        LOG(INFO) << "basic pooling compute";
        to = 0;
        min_time = 1000000;
        for (int i = 0; i < test_iter; ++i) {
            t1.clear();
            t1.start();
            const void* in = (const void*)tin.data();
            void* out = (void*)tout_basic.mutable_data();
            
            pooling_basic_test(in, out, num, chin, hout, wout, chin, hin, win, type, global, kernel, \
                          kernel, stride, stride, pad, pad);
            
            t1.end();
            to += t1.get_average_ms();
            if (t1.get_average_ms() < min_time) {
                min_time = t1.get_average_ms();
            }
        }
        LOG(INFO) << "basic pooling running time, ave: " << to / test_iter << ", min time: " << min_time;
    }

    tout_saber.re_alloc(shape_out, AK_INT8);
    LOG(INFO) << "saber pooling compute";
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start();
        const void* in = (const void*)tin.data();
        void* out = (void*)tout_saber.mutable_data();
        //pooling_global_int8(in, out, num, chin, hout, wout, chin, hin, win, type, global, kernel, \
                          kernel, stride, stride, pad, pad);
        //pooling2x2s2_max_int8(in, out, num, chin, hout, wout, chin, hin, win, type, global, kernel, \
                          kernel, stride, stride, pad, pad);
        //pooling3x3s1p1_max_int8(in, out, num, chin, hout, wout, chin, hin, win, type, global, kernel, \
                          kernel, stride, stride, pad, pad);
        //pooling3x3s2p1_max_int8(in, out, num, chin, hout, wout, chin, hin, win, type, global, kernel, \
                          kernel, stride, stride, pad, pad);
        //pooling3x3s2p0_max_int8(in, out, num, chin, hout, wout, chin, hin, win, type, global, kernel, \
                          kernel, stride, stride, pad, pad);
        pool_func(in, out, num, chin, hout, wout, chin, hin, win, type, global, kernel, \
                          kernel, stride, stride, pad, pad);
        t2.end();
        to += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
        LOG(INFO) << "saber pooling running time, ave: " << to / test_iter << ", min time: " << min_time;
    }   
    
    if (compare_result) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        print_tensor(tin);
        print_tensor(tout_basic);
        print_tensor(tout_saber);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        CHECK_EQ(fabsf(max_ratio) < 1e-4f, true) << "compute result error";
    }
}

#if 1
TEST(TestSaberLite, test_func_pooling_global_arm) {
    
    Shape shape_in(num, ch_in, h_in, w_in);
    
    TensorH tdin;
    tdin.re_alloc(shape_in, AK_INT8);
    signed char* in = (signed char*)tdin.mutable_data();
    srand(time(NULL));
    for (int i = 0; i < tdin.size(); i++){
        *in = char(rand() % 256 - 128);
        in++;
    }
    
    test_arm_pooling_int8(tdin, threads, cluster, pool_case);
}
#endif



int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    Env::env_init();
    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if (argc >= 4) {
        test_iter = atoi(argv[3]);
    }
    if (argc >= 5) {
        compare_result = atoi(argv[4]) > 0;
    }
    if (argc >= 6) {
        if (argc < 10) {
            LOG(ERROR) << "usage: ./" << argv[0] << " cluster  threads  test_iter " << \
            " compare_result num ch_in h_in w_in";
            return 0;
        }
        num = atoi(argv[5]);
        ch_in = atoi(argv[6]);
        h_in = atoi(argv[7]);
        w_in = atoi(argv[8]);
        pool_case = atoi(argv[9]);
    }


    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

