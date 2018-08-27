#include "test_lite.h"
#include "saber/lite/funcs/saber_pooling.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;


int cluster = 0;
int threads = 4;
int test_iter = 10;

bool compare_result = false;
bool global_pool = false;

int num = 1;
int ch_in = 32;
int h_in = 112;
int w_in = 112;

int kernel = 2;
int pad = 0;
int stride = 2;

PoolingType type = Pooling_max;

typedef Tensor<CPU, AK_FLOAT> TensorHf4;

#define COMPARE_RESULT 1
void pooling_basic(const float* din, float* dout, \
                   int num, int chout, int hout, int wout, \
                   int chin, int hin, int win, \
                   PoolingType type, bool global, int kernel_w, int kernel_h, \
                   int stride_w, int stride_h, int pad_w, int pad_h) {
    //no need to pad input tensor, border is zero pad inside this function
    int size_channel_in = win * hin;
    int size_channel_out = wout * hout;
    
    float* data_out = dout;
    const float* data_in = din;
    
    if (global) {
        switch (type) {
            case Pooling_max:
                for (int n = 0; n < num; ++n) {
                    float* data_out_batch = data_out + n * chout * size_channel_out;
                    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                    for (int c = 0; c < chout; ++c) {
                        const float* data_in_channel = data_in_batch + c * size_channel_in;//in address
                        data_out_batch[c] = data_in_channel[0];
                        for (int i = 0; i < size_channel_in; ++i) {
                            data_out_batch[c] = data_out_batch[c] > data_in_channel[i] ? \
                            data_out_batch[c] : data_in_channel[i];
                        }
                    }
                }
                break;
                
            case Pooling_average_include_padding:
                
            case Pooling_average_exclude_padding:
                for (int n = 0; n < num; ++n) {
                    float* data_out_batch = data_out + n * chout * size_channel_out;
                    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                    for (int c = 0; c < chout; ++c) {
                        const float* data_in_channel = data_in_batch + c * size_channel_in;//in address
                        float sum = 0.f;
                        for (int i = 0; i < size_channel_in; ++i) {
                            sum += data_in_channel[i];
                        }
                        data_out_batch[c] = sum / size_channel_in;
                    }
                }
                break;
            default:
                printf("not support\n");
        }
        return;
    }
    
    switch (type) {
        case Pooling_max:
            for (int n = 0; n < num; ++n) {
                float* data_out_channel = data_out + n * chout * size_channel_out;
                const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < chout; q++) {
                    
                    float* data_out_row = data_out_channel + q * size_channel_out;
                    const float* data_in_channel = data_in_batch + q * size_channel_in;
                    
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
                            
                            data_out_row[j] = data_in_channel[hstart * win + wstart];
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    data_out_row[j] = data_out_row[j] > \
                                    data_in_channel[h * win + w] ? \
                                    data_out_row[j] : data_in_channel[h * win + w];
                                }
                            }
                        }
                        data_out_row += wout;
                    }
                }
            }
            break;
            
        case Pooling_average_include_padding:
            for (int n = 0; n < num; ++n) {
                int pool_size = kernel_w * kernel_h;//(hend - hstart) * (wend - wstart);//problem
                float* data_out_channel = data_out + n * chout * size_channel_out;
                const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < chout; q++) {
                    
                    float* data_out_row = data_out_channel + q * size_channel_out;
                    const float* data_in_channel = data_in_batch + q * size_channel_in;
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
                            
                            data_out_row[j] = data_in_channel[hstart * win + wstart];
                            float sum = 0.f;
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    sum += data_in_channel[h * win + w];
                                }
                            }
                            data_out_row[j] = sum / pool_size;
                        }
                        data_out_row += wout;
                    }
                }
            }
            break;
        case Pooling_average_exclude_padding:
            for (int n = 0; n < num; ++n) {
                float* data_out_channel = data_out + n * chout * size_channel_out;
                const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < chout; q++) {
                    
                    float* data_out_row = data_out_channel + q * size_channel_out;
                    const float* data_in_channel = data_in_batch + q * size_channel_in;
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
                            
                            data_out_row[j] = data_in_channel[hstart * win + wstart];
                            float sum = 0.f;
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    sum += data_in_channel[h * win + w];
                                }
                            }
                            int pool_size = (hend - hstart) * (wend - wstart);
                            data_out_row[j] = sum / pool_size;
                        }
                        data_out_row += wout;
                    }
                }
            }
            break;
        default:
            printf("not support\n");
    }
}
/*TEST(TestSaberLite, test_func_activation_arm) {
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

    int test_iter = 10;

    int w_in = 5;
    int h_in = 5;
    int ch_in = 5;
    int num_in = 5;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
        ch_in << ", height=" << h_in << ", width=" << w_in;

    std::vector<TensorHf4*> vin;
    std::vector<TensorHf4*> vout;

    Tensor<CPU, AK_FLOAT> thin(shape_in);
    fill_tensor_rand(thin, -1.f, 1.f);
    TensorHf4 tout;
    TensorHf4 tout_basic(shape_out);
    vin.push_back(&thin);

#if COMPARE_RESULT
    activation_basic(thin, active_type, tout_basic);
    //print_tensor_host(tout_basic);
#endif
    
    SaberActivation activation_lite;
    ActivationParam param(active_type, 0.f, 1.0f);
    activation_lite.load_param(&param);

    vout.push_back(&tout);
    activation_lite.compute_output_shape(vin, vout);
    CHECK_EQ(shape_out == vout[0]->valid_shape(), true) << "compute shape error";

    LOG(INFO) << "re-alloc tensor buffer";
    vout[0]->re_alloc(vout[0]->valid_shape());

    LOG(INFO) << "activation initialized to saber impl";
    activation_lite.init(vin, vout, ctx1);

    SaberTimer t1;

    LOG(INFO) << "saber activation compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        activation_lite.dispatch(vin, vout);
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber activation total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
    //print_tensor_host(*vout[0]);

#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
  
    tensor_cmp_host(tout_basic.data(), tout.data(), tout_basic.valid_size(), max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
}*/
void test_arm_pooling(std::vector<TensorHf4*>& tin, \
                      int kernel, int stride, int pad, \
                      PoolingType type, bool global, int threads, int cluster_id) {
    
    //int test_iter = 1000;
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
    
    TensorHf4 tout_basic;
    TensorHf4 tout_saber;
    
    TensorHf4* thin = tin[0];
    std::vector<TensorHf4*> vin;
    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;
    //vin.push_back(&thin);
    tvout_saber.push_back(&tout_saber);
    tvout_basic.push_back(&tout_basic);
    
    int num = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();
    
    LOG(INFO) << "pooling param: ";
    LOG(INFO) << " img_num = " << num;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
    LOG(INFO) << "kernel size = " << kernel;
    LOG(INFO) << "stride = " << stride;
    LOG(INFO) << "pad = " << pad;
    LOG(INFO) << "type = " << type;
    int wout = 1;
    int hout = 1;
    if(!global) {
        int hin = tin[0]->height(); // P
        hout = static_cast<int>(std::max(0.f,ceilf(static_cast<float>(
                                                         hin + 2 * pad - kernel) / stride))) + 1;
        int win = tin[0]->width(); // Q
        wout = static_cast<int>(std::max(0.f,ceilf(static_cast<float>(
                                                         win + 2 * pad - kernel) / stride))) + 1;
    }
    Shape shape_out{num, chin, hout, wout};
    PoolParam pooling_param(type,global,kernel,kernel, stride,stride,pad, pad);
    //LOG(INFO) << "input tensor";
    //print_tensor_host(*tin[0]);
    
    if (compare_result) {
        LOG(INFO) << "run basic pooling for precision comparation";
        tout_basic.re_alloc(shape_out);
        //pooling_basic(tout_basic, *thin, type,global, kernel, \
        kernel, stride, stride, pad, pad);
        //print_tensor_host(tout_basic);
        LOG(INFO) << "basic pooling compute";
        to = 0;
        min_time = 1000000;
        for (int i = 0; i < test_iter; ++i) {
            t1.clear();
            t1.start();
            const float* in=thin->data();
            float* out =tout_basic.mutable_data();
            
            pooling_basic(in,out, num, chin,hout,wout,chin,hin,win,type,global, kernel, \
                          kernel, stride, stride, pad, pad);
            
            //float* out1 =tout_saber.mutable_data();
            
           // pooling_basic(in,out1, num, chin,hout,wout,chin,hin,win,3,global, kernel, \
                          kernel, stride, stride, pad, pad);
            //tvout_basic[0]->record_event(ctx1.get_compute_stream());
            //tvout_basic[0]->sync();
            t1.end();
            to += t1.get_average_ms();
            if (t1.get_average_ms() < min_time) {
                min_time = t1.get_average_ms();
            }
        }
        LOG(INFO) << "basic pooling running time, ave: " << to / test_iter << ", min time: " << min_time;
        // print_tensor_host(tout_basic);
        
    }
    
    SaberPooling pooling_saber;
    pooling_saber.load_param(&pooling_param);
    pooling_saber.compute_output_shape(tin, tvout_saber);
    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape_1: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
    << sh_out_saber[2] << ", " << sh_out_saber[3];
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
    << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";
    
    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);
    
    LOG(INFO) << "saber pooling impl init";
    pooling_saber.init(tin, tvout_saber, ctx1);
    
    //print_tensor_host(*thin);
    
    //! compute
    LOG(INFO) << "saber pooling compute";
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start();
        //const float* in=thin->data();
        //float* out1 =tout_saber.mutable_data();
        
        //pooling_basic(in,out1, num, chin,hout,wout,chin,hin,win,3,global, kernel, \
                      kernel, stride, stride, pad, pad);
        pooling_saber.dispatch(tin,tvout_saber);
        //pooling3x3s2_max(tout_saber,*thin,type,global,kernel, \
        kernel, stride, stride, pad, pad);
        //tvout_saber[0]->record_event(ctx1.get_compute_stream());
        //tvout_saber[0]->sync();
        //pooling_basic()
        t2.end();
        to += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
    }
    LOG(INFO) << "saber pooling running time, ave: " << to / test_iter << ", min time: " << min_time;
    //print_tensor_host(tout_saber);
    
    if (compare_result) {
        double max_ratio = 0;
        double max_diff = 0;
        TensorHf4 tdiff(tout_basic.valid_shape());
        //tensor_diff(tout_basic, tout_saber, tdiff);
        //print_tensor_host(tdiff);
        for(int i =0;i<num;i++)
        {
            for(int j=0;j<ch_in;j++){
                printf("%.4f ",thin->data()[i*num+j]);
            }
            printf("\n");
            
        }
         printf("\n");
        for(int i =0;i<num;i++)
        {
            for(int j=0;j<ch_in;j++){
                printf("%.4f ",tout_saber.data()[i*num+j]);
            }
            printf("\n");
            
        }
        printf("\n");
        for(int i =0;i<num;i++)
        {
            for(int j=0;j<ch_in;j++){
                printf("%.4f ",tout_basic.data()[i*num+j]);
            }
            printf("\n");
            
        }
       
         printf("\n");
         printf("\n");
         printf("\n");
        for(int i =0;i<wout;i++)
        {
            
            for(int j=0;j<hout;j++){
                printf("%.4f ",tout_saber.data()[i*wout+j]);
            }
            printf("\n");
            
        }
         printf("\n");
        for(int i =0;i<wout;i++)
        {
            for(int j=0;j<hout;j++){
                printf("%.4f ",tout_basic.data()[i*wout+j]);
            }
            printf("\n");
            
        }
        
        tensor_cmp_host(tout_basic.data(), tout_saber.data(), tout_basic.valid_size(), max_ratio, max_diff);
        
        // LOG(INFO) << "tout_basic";
        // print_tensor_host(tout_basic);
        // LOG(INFO) << "tout_saber";
        // print_tensor_host(tout_saber);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
    }
}

#if 1
TEST(TestSaberLite, test_func_pooling_global_arm) {
    
    Shape shape_in(num, ch_in, h_in, w_in);
    
    TensorHf4 tdin;
    
    tdin.re_alloc(shape_in);
    float* in=tdin.mutable_data();
    for(int i=0;i<tdin.size();i++){
        *in=-1.0f-i;
        in++;
    }
    //fill_tensor_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, 1.f);
    
    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);
    
    test_arm_pooling(tin, kernel, stride, pad, type, global_pool, threads, cluster);
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
        global_pool = atoi(argv[5]) > 0;
    }
    if(argc >= 7) {
        if (argc < 14) {
            LOG(ERROR) << "usage: ./" << argv[0] << " cluster  threads  test_iter " << \
            " compare_result global_pool num ch_in h_in w_in kernel pad stride pool_type";
            return 0;
        }
        num = atoi(argv[6]);
        ch_in = atoi(argv[7]);
        h_in = atoi(argv[8]);
        w_in = atoi(argv[9]);
        kernel = atoi(argv[10]);
        pad = atoi(argv[11]);
        stride = atoi(argv[12]);
        type = (PoolingType)atoi(argv[13]);
    }

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

