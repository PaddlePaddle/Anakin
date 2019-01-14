#include "test_lite.h"
#include "saber/lite/funcs/neon/impl/conv_block_utils.h"
using namespace anakin::saber;
using namespace anakin::saber::lite;

typedef Tensor<CPU> TensorHf4;

int g_cluster = 0;
int g_threads = 1;
bool g_basic_test = false;
int g_test_iter = 100;
bool g_compared_result = true;
int g_ch_n = 4;
int g_hei_n = 1;
int g_num = 4;
int g_channel = 16;
int g_height = 112;
int g_width = 112;
int g_kernel_size = 9;

/*preprocessing weights
* input weights: [chout, chin/ group, 3, 3] --> outputs weights: [chout / n, chin/ group, 3, 3 * n]
*/
template <typename dtype>
void conv_trans_weights_numc_basic(const dtype* din, dtype* dout, int chout, int chin, int n, int kernel_size) {
    if (n <= 0){
        LOGE("ch_n and hei_n are more than zero\n");
        return;
    }
    int c_loop = chout / n;
    int chout_round = (chout + n - 1) / n;
    int win_stride = chin * kernel_size;
    int wout_stride = n * win_stride;
    int co = 0;
    for (; co < c_loop; ++co) {
        dtype* dout_c = dout + co * wout_stride;
        const dtype *din_array[n];
        din_array[0] = din + co * wout_stride;
        for (int i = 1; i < n; i++){
            din_array[i] = din_array[i - 1] + win_stride;
        }
        for (int ci = 0; ci < chin; ++ci) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int i = 0; i < n; i++){
                    *(dout_c++) = * (din_array[i]++);
                }
            }
        }
    }
    // pad final chout
    if (chout_round > c_loop) {
        dtype* dout_c = dout + c_loop * wout_stride;
        const dtype *din_array[n];
        din_array[0] = din + c_loop * wout_stride;
        for (int i = 1; i < n; i++){
            din_array[i] = din_array[i - 1] + win_stride;
        }
        //deal remain
        int cremain = chout_round * n - chout;
        for (int i = 1; i <= cremain; i++){
            din_array[n - i] = din_array[0];
        }
        for (int ci = 0; ci < chin; ++ci) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int i = 0; i < n; i++){
                    *(dout_c++) = * (din_array[i]++);
                }
            }
        }
    }
}

/*preprocessing inputs
* input din: [1, chin, he-hs, we - ws] --> outputs dout: [n, chin, 1, we - ws]
* n = he - hs
*/
template <typename dtype>
void prepack_input_nxw_basic(const dtype* din, dtype* dout, int n, int hs, int he, int ws, int we, \
    int channel, int width, int height, dtype* zero_ptr) {

    if (n <= 0){
        LOGE("hei_n is more than zero\n");
        return;
    }
    int w0 = ws < 0 ? 0 : ws;
    int w1 = we > width ? width : we;
    int h0 = hs < 0 ? 0: hs;
    int h1 = he > height ? height : he;

    int size_w = we - ws;
    int size_wc_len = size_w * channel;
    int size_c = width * height;

    int valid_w = w1 - w0;
    int valid_h = h1 - h0;
    size_t valid_w_byte = valid_w * sizeof(dtype);

    dtype *out_array[n];
    out_array[0] = dout;
    for (int i = 1; i < n; i++){
        out_array[i] = out_array[i - 1] + size_wc_len;
    }

    dtype* ptr_zero;
    memset(ptr_zero, 0, valid_w_byte);
    for (int c = 0; c < channel; ++c) {
        int j = 0;
        //valid height
        for (int i = hs; i < he; i++){
        	//get address
        	dtype *in_array = din + i * width;
            if (i < 0 || i >= height){
                in_array = ptr_zero;
            }
            for (int w = ws; w < w0; ++w) {
                *(out_array[j]++) = 0.f;
            }
            memcpy(out_array[j], in_array, valid_w_byte);
            out_array[j] += valid_w;
            for (int w = w1; w < we; ++w) {
                *(out_array[j]++) = 0.f;
            }
            j++;
        }
        //remain
        // for (int i = valid_h; i < n; i++){
        // 	for (int w = ws; w < we; w++){
        // 		*(out_array[i]++) = 0.f;
        // 	}
        // }
        din += size_c;
    }
}

/*wirte result in outputs
* input din: [n, c / n, h, w * n], output dout: [n, c, h, w]
*/
template <typename dtype>
void write_to_output_nxw_basic(const dtype* din, dtype* dout, int ch_n, int hei_n, int cs, int ce, int hs, int he,\
    int ws, int we, int channel, int height, int width, bool flag_relu, dtype* trash_ptr) {

    if (ch_n <= 0 || hei_n <= 0){
        LOGE("ch_n and hei_n are more than zero\n");
        return;
    }
    int size_c_out = width * height;

    dtype *dout_array[ch_n];
    dout_array[0] = dout + cs * size_c_out + hs * width + ws;
    for (int i = 1; i < ch_n; i++){
        dout_array[i] = dout_array[i - 1] + size_c_out;
    }

    const dtype* ptr_din = din;

    if (ce > channel) {
        int cremain = ce - channel;
        for (int i = cremain; i > 0; i--){
            dout_array[ch_n - i] = trash_ptr;
        }
    }

    int size_h = (he > height ? height : he) - hs;
    for (int i = 0; i < hei_n; i++){
        for (int j = 0; j < width; j++){
            int size_w = i * width;
            for (int c = 0; c < ch_n; c++){
                dtype *ptr = dout_array[c] + size_w;
                if (flag_relu){
                    *ptr = *ptr_din > 0 ? *ptr_din : 0;
                }else{
                    *ptr = *ptr_din;
                }
                ptr_din++;
            }
        }
    }
}

template <typename dtype>
void fill_packed_bias_nxmw_basic(const dtype* bias, dtype* dout, int ch_n, int hei_n, int wround){
    if (ch_n <= 0 || hei_n <= 0){
        LOGE("ch_n and hei_n are more than zero\n");
        return;
    }
    for(int i = 0; i < hei_n; i++){
        for (int j = 0; j < wround; j++){
            const dtype* bias_ptr = bias;
            for (int k = 0; k < ch_n; k++){
                *dout = * bias_ptr;
                dout++;
                bias_ptr++;
    		}
    	}
    }
}

SaberStatus test_arm_conv_block_utils(int n, int c, int h, int w, \
    int ch_n, int hei_n, int kernel_size, int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;

    Context ctx1;
    PowerMode mode = (PowerMode)cluster_id;
    ctx1.set_run_mode(mode, thread_num);
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

    TensorHf4 tout_basic_int;
    TensorHf4 tout_saber_int;

    Shape shin = {n, c, h, w};
    TensorHf4 thin;
    TensorHf4 thin32;

    thin.re_alloc(shin, AK_FLOAT);
    fill_tensor_rand(thin, -1.f, 1.f);
    // fill_tensor_const(thin, 1.f);

    thin32.re_alloc(shin, AK_INT32);
    fill_tensor_rand(thin32, -1.f, 1.f);

    LOG(INFO) << "conv block param: ";
    LOG(INFO) << " img_num = " << n;
    LOG(INFO) << " in_channels = " << c;
    LOG(INFO) << " img_h = " << h;
    LOG(INFO) << " img_w = " << w;
    LOG(INFO) << " ch_n = " << ch_n;
    LOG(INFO) << " hei_n = " << hei_n;
    LOG(INFO) << " kernel_size = " << kernel_size;

   //c1 -> cn
    int hout = h;

    int wout = w * ch_n;

    int chout = c / ch_n + c % ch_n;

    //cn->c1
    int hout_c = h;

    int wout_c = w / ch_n;

    int chout_c = c * ch_n;

    Shape shape_out{n, chout, hout, wout};
    LOG(INFO) << " chout = " << chout;
    LOG(INFO) << " hout = " << hout;
    LOG(INFO) << " wout = " << wout;

    const float* din = static_cast<const float*>(thin.data());
    const int* din_int32 = static_cast<const int*>(thin32.data());

    //! compute
    LOG(INFO) << "saber conv block compute";
    to = 0;
    tout_saber.re_alloc(shape_out, AK_FLOAT);
    fill_tensor_const(tout_saber, 0.f);
    float* dout_f32 = static_cast<float*>(tout_saber.mutable_data());
    tout_saber_int.re_alloc(shape_out, AK_INT32);
    fill_tensor_const(tout_saber_int, 0.f);
    int* dout_int32 = static_cast<int*>(tout_saber_int.mutable_data());
    int* trash_ptr = static_cast<signed int*>(ctx1.get_work_space());
    memset(trash_ptr, 0, wout * sizeof(signed int));
    float* ptr_zero = static_cast<float*>(ctx1.get_work_space()) + wout;
    memset(ptr_zero, 0, w * sizeof(float));
    for (int i = 0; i < g_test_iter; ++i) {
        t1.clear();
        t1.start();
        conv_trans_weights_numc<float>(din, dout_f32, chout, c, ch_n, kernel_size);
        // prepack_input_nxw<float>(din, dout_f32, hei_n, 0, 4, -1, 20, c, w, h, ptr_zero);
        // fill_packed_bias_nxmw_f32(din, dout_f32, c, w, h);
        // conv_trans_weights_numc<int>(din_int32, dout_int32, chout, c, ch_n, kernel_size);
        if (ch_n == 4){
            write_to_output_c4_int32(din_int32, dout_int32, ch_n, hei_n, 0, 4, 0, 2, 0, w * ch_n, \
                chout, hout, wout, true, trash_ptr);
        }
        if (ch_n == 8){
            write_to_output_c8_int32(din_int32, dout_int32, ch_n, hei_n, 0, 4, 0, 2, 0, w * ch_n, \
                chout, hout, wout, true, trash_ptr);
        }

        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
        // print_tensor(tout_basic);
    }
    LOG(INFO) << "saber conv block running time, ave: " << to / g_test_iter << ", min time: " << min_time;
    // print_tensor(tout_saber);


    if (g_compared_result) {
        LOG(INFO) << "run basic conv block for precision comparation";
        tout_basic.re_alloc(shape_out, AK_FLOAT);
        fill_tensor_const(tout_basic, 0.f);
        float* dout = static_cast<float*>(tout_basic.mutable_data());

        tout_basic_int.re_alloc(shape_out, AK_INT32);
        fill_tensor_const(tout_basic_int, 0.f);
        int* dout_32 = static_cast<int*>(tout_basic_int.mutable_data());
        conv_trans_weights_numc_basic<float>(din, dout, chout, c, ch_n, kernel_size);
        // prepack_input_nxw_basic<float>(din, dout, hei_n, 0, 4, -1, 20, c, w, h, ptr_zero);
        // fill_packed_bias_nxmw_basic<float>(din, dout, c, w, h);
        // conv_trans_weights_numc_basic<int>(din_int32, dout_32, chout, c, ch_n, kernel_size);
        write_to_output_nxw_basic<int>(din_int32, dout_32, ch_n, hei_n, 0, 4, 0, 2, 0, w * ch_n, \
    chout, hout, wout, true, trash_ptr);
        // print_tensor(tout_basic);
        double max_ratio = 0;
        double max_diff = 0;
        // tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        tensor_cmp_host(tout_basic_int, tout_saber_int, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            TensorHf4 tdiff(tout_basic_int.valid_shape());
            LOG(INFO) << "biasc result";
            print_tensor(tout_basic_int);
            LOG(INFO) << "saber result";
            print_tensor(tout_saber_int);
            tensor_diff(tout_basic_int, tout_saber_int, tdiff);
            print_tensor(tdiff);
            return SaberInvalidValue;
        }
        max_ratio = 0;
        max_diff = 0;
        // tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            TensorHf4 tdiff(tout_basic.valid_shape());
            LOG(INFO) << "biasc result";
            print_tensor(tout_basic);
            LOG(INFO) << "saber result";
            print_tensor(tout_saber);
            tensor_diff(tout_basic, tout_saber, tdiff);
            print_tensor(tdiff);
            return SaberInvalidValue;
        }
    }
    return SaberSuccess;

}

TEST(TestSaberLite, test_custom) {
    auto flag = test_arm_conv_block_utils(g_num, g_channel, g_height, g_width, g_ch_n, g_hei_n, g_kernel_size, g_threads, g_cluster);
    if (flag == SaberSuccess) {
        LOG(INFO) << "test conv block utils: batchsize: " << g_num << ", channel: " << g_channel << ", h: " << g_height << \
            ", w: " << g_width << ", ch_n: " << g_ch_n << ", hei_n" << g_hei_n <<", kernel_size: " << g_kernel_size << \
            ", threads: " << g_threads << ", cluster: " << g_cluster << " passed!!";
    } else {
        LOG(FATAL) << "test conv block utils: batchsize: " << g_num << ", channel: " << g_channel << ", h: " << g_height << \
            ", w: " << g_width << ", ch_n: " << g_ch_n << ", hei_n" << g_hei_n <<", kernel_size: " << g_kernel_size << \
            ", threads: " << g_threads << ", cluster: " << g_cluster <<  " failed!!";
    }
}


int main(int argc, const char** argv){
    anakin::saber::lite::Env::env_init();
    LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads] [test iter] [compare result]";
    if (argc > 1) {
        g_basic_test = atoi(argv[1]) > 0;
    }
    if (argc > 2) {
        g_cluster = atoi(argv[2]);
    }
    if (argc > 3) {
        g_threads = atoi(argv[3]);
    }
    if (argc > 4) {
    	g_test_iter = atoi(argv[4]);
    }
    if (argc > 5){
    	g_compared_result = atoi(argv[5]);
    }
    if (argc > 6){
    	if (argc < 13) {
            LOG(FATAL) << "usage: ./" << argv[0] << " do_basic_test cluster  threads  test_iter " << \
                " compare_result num  channel  height  width ch_n  hei_n  kernel_size";
            return -1;
        }
    	g_num = atoi(argv[6]);
    	g_channel = atoi(argv[7]);
    	g_height = atoi(argv[8]);
    	g_width = atoi(argv[9]);
        g_ch_n = atoi(argv[10]); //channel num
        g_hei_n = atoi(argv[11]); //height num
        g_kernel_size = atoi(argv[12]);
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
