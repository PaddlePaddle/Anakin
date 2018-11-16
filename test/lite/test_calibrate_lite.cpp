#include "saber/lite/funcs/calibrate_lite.h"
#include "test_lite.h"
using namespace anakin::saber;
using namespace anakin::saber::lite;
int cluster = 0;
int threads = 1;
int iter = 1;
typedef Tensor<CPU> TensorH;
std::vector<float> get_scale_basic(const TensorH& tin, int axis, float scale_factor) {
    int axis_dims = tin.valid_shape()[axis];
    std::vector<float> scale_out;
    scale_out.resize(axis_dims);
    int out_dims = tin.count_valid(0, axis);
    long long inner_dims = tin.count(axis + 1, tin.dims());
    long long inner_size = inner_dims * axis_dims;
    // printf("inner_dims: %d, inner_size: %d \n", inner_dims, inner_size);
    const float* in_data = static_cast<const float*>(tin.data());
#pragma omp parallel for
    for (int c = 0; c < axis_dims; ++c) {
        float max_val = 0.f;
        const float* din = in_data + c * inner_dims;
        for (int j = 0; j < out_dims; ++j) {
            const float* ptr_in = din + j * inner_size;
            for (int i = 0; i < inner_dims; ++i) {
                float read_data = fabsf(ptr_in[i]);
                max_val = (read_data > max_val) ? read_data : max_val;
            }
        }
        // printf("max_val: %d \n", max_val);
        scale_out[c] = max_val / scale_factor;
    }
    return scale_out;
}
void fp32_to_int8_basic(const TensorH& tin, TensorH& tout, int axis, std::vector<float> scale_factor) {
    int outer_size = tin.count_valid(0, axis);
    int inner_size = tin.count_valid(axis, tin.dims());
    const float* din = static_cast<const float*>(tin.data());
    char* dout = static_cast<char*>(tout.mutable_data());
    for (int i = 0; i < outer_size; ++i) {
        float scale = 1.f / scale_factor[i];
        for (int j = 0; j < inner_size; ++j) {
#ifdef __aarch64__
            dout[j] = static_cast<char>(round(din[j] * scale));
#else
            dout[j] = static_cast<char>((din[j] * scale));
#endif
        }
        dout += inner_size;
        din += inner_size;
    }
}
void fp32_to_int8_inplace_basic(const TensorH& tin, int axis, std::vector<float> scale_factor) {
    //! alloc memory
    // int m = tin.num();
    // int k = tin.count_valid(1, tin.dims());
    Tensor<CPU> tout;
    tout.re_alloc(tin.valid_shape(), AK_INT8);
    int outer_size = tin.count_valid(0, axis);
    int inner_size = tin.count_valid(axis, tin.dims());
    // printf("inner_size: %d, outer_size: %d \n", inner_size, outer_size);
    const float* din = static_cast<const float*>(tin.data());
    char* dout = static_cast<char*>(tout.mutable_data());
    for (int i = 0; i < outer_size; ++i) {
        float scale = 1.f / scale_factor[i];
        for (int j = 0; j < inner_size; ++j) {
#ifdef __aarch64__
            dout[j] = static_cast<char>(round(din[j] * scale));
#else
            dout[j] = static_cast<char>((din[j] * scale));
#endif
        }
        dout += inner_size;
        din += inner_size;
    }
    // tin.reshape(Shape(m, k, 1, 1), AK_INT8);
    tin.copy_from(tout);
}
void tensor_to_int8_basic(const Tensor<CPU>& tin, Tensor<CPU>& tout){
    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }
    if (tout.get_dtype() != AK_INT8) {
        tout.set_dtype(AK_INT8);
    }
    tout.reshape(tin.valid_shape());
    //! get scale
    std::vector<float> scale = tin.get_scale();
    // const float* din = static_cast<const float*>(tin.data());
    // char* dout = static_cast<char*>(tout.mutable_data());
    //! convert to int8
    fp32_to_int8_basic(tin, tout, 1, scale);
}
void tensor_to_int8_inplace_basic(const Tensor<CPU>& tin){
    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }
    //! get scale
    std::vector<float> scale = tin.get_scale();
    //! convert to int8
    fp32_to_int8_inplace_basic(tin, 1, scale);
}
bool test_get_scale(int axis, float scale_factor) {
    Shape sh(get_rand(1, 100), get_rand(1, 100), get_rand(1, 512), get_rand(1, 512));
    // Shape sh(4, 32, 112, 112);
    TensorH tin;
    tin.re_alloc(sh, AK_FLOAT);
    fill_tensor_rand(tin, -20, 20);
            LOG(INFO) << "input shape num = " << sh[0];
            LOG(INFO) << "input shape channel = " << sh[1];
            LOG(INFO) << "input shape height = " << sh[2];
            LOG(INFO) << "input shape width = " << sh[3];
    std::vector<float> scale_basic;
    std::vector<float> scale_lite;
            LOG(INFO) << "get_scale_basic compute";
    scale_basic = get_scale_basic(tin, axis, scale_factor);
            LOG(INFO) << "get_tensor_scale compute";
    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;
    for (int i = 0; i < iter; i++){
        t1.clear();
        t1.start();
        get_tensor_scale(tin, scale_lite, axis, scale_factor);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
            LOG(INFO) << "get_tensor_scale running time, ave: " << to / iter << ", min time: " << min_time;
    if (scale_basic.size() != scale_lite.size()) {
                LOG(INFO) << "scale_basic size:" << scale_basic.size() <<", scale_lite size: " << scale_lite.size();
        return false;
    }
    // LOG(INFO) << "basic result";
    // for (int i = 0; i < scale_basic.size(); ++i) {
    //     printf("%.6f  ", scale_basic[i]);
    //     if ((i + 1) % 10 == 0)
    //         printf("\n");
    // }
    // printf("\n");
    // LOG(INFO) << "lite result";
    // for (int i = 0; i < scale_lite.size(); ++i) {
    //     printf("%.6f  ", scale_lite[i]);
    //     if ((i + 1) % 10 == 0)
    //         printf("\n");
    // }
    // printf("\n");
            LOG(INFO) << "diff";
    for (int i = 0; i < scale_basic.size(); ++i) {
        float tmp = scale_basic[i] - scale_lite[i];
        // printf("%.6f  ", tmp);
        // if ((i + 1) % 10 == 0)
        //     printf("\n");
        // if (tmp != 0){
        //     printf("i: %d, tmp: %.6f, a: %.6f, b: %.6f \n", i, tmp, scale_basic[i], scale_lite[i]);
        // }
                CHECK_EQ(fabsf(tmp) < 1e-5f, true) << "compute result error";//scale_basic[i] - scale_lite[i]
        // return false;
    }
            LOG(INFO) << "get_tensor_scale result is right";
    return true;
}
bool test_fp32_to_int8(int axis, float scale_factor, Context ctx){
    Shape sh(get_rand(1, 10), get_rand(1, 50), get_rand(1, 512), get_rand(1, 512));
    // Shape sh(4, 32, 112, 112);
    TensorH tin;
    tin.re_alloc(sh, AK_FLOAT);
    fill_tensor_rand(tin, -20, 20);
            LOG(INFO) << "input shape num = " << sh[0];
            LOG(INFO) << "input shape channel = " << sh[1];
            LOG(INFO) << "input shape height = " << sh[2];
            LOG(INFO) << "input shape width = " << sh[3];
    std::vector<float> scale_basic;
    std::vector<float> scale_lite;
            LOG(INFO) << "get_scale_basic compute";
    scale_basic = get_scale_basic(tin, axis, scale_factor);
            LOG(INFO) << "get_tensor_scale compute";
    get_tensor_scale(tin, scale_lite, axis, scale_factor);
    if (scale_basic.size() != scale_lite.size()) {
        return false;
    }
    for (int i = 0; i < scale_basic.size(); ++i) {
        // float tmp = scale_basic[i] - scale_lite[i];
        // if (tmp != 0){
        //     printf("i: %d, tmp: %.6f \n", i, tmp);
        // }
                CHECK_EQ(fabsf(scale_basic[i] - scale_lite[i]) < 1e-4f, true) << "scale compute result error";
        // return false;
        // if (fabsf(scale_basic[i] - scale_lite[i]) > 1e-5f) {
        //     LOG(INFO) << "scale compute failed";
        //     return false;
        // }
    }
            LOG(INFO) << "scale is right";
    TensorH tout;
    TensorH tout_basic;
    tout.re_alloc(sh, AK_INT8);
    tout_basic.re_alloc(sh, AK_INT8);
            LOG(INFO) << "fp32_to_int8_basic compute";
    fp32_to_int8_basic(tin, tout_basic, axis + 1, scale_lite);
    // print_tensor(tout_basic);
            LOG(INFO) << "trans_fp32_weights_to_int8 compute";
    int outer_size = tin.count_valid(0, axis);
    int inner_size = tin.count_valid(axis, tin.dims());
            LOG(INFO) << "outer_size: " << outer_size << ", inner_size: " << inner_size;
    // fp32_to_int8((const float*)tin.data(), (char*)tout.mutable_data(), scale_lite, outer_size, inner_size);
    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;
    for (int i = 0; i < iter; i++){
        t1.clear();
        t1.start();
        trans_fp32_weights_to_int8(tin, tout, scale_factor, 0, &ctx);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
            LOG(INFO) << "trans_fp32_weights_to_int8 running time, ave: " << to / iter << ", min time: " << min_time;
    // print_tensor(tout);
    double max_ratio = 0;
    double max_diff = 0;
    const double eps = 1e-6f;
    int out_size = tout.valid_size();
    char* ptr_basic = static_cast<char*>(tout_basic.data());
    char* ptr = static_cast<char*>(tout.data());
            LOG(INFO) << "trans_fp32_weights_to_int8 diff, size: " << out_size;
    for (int i = 0; i < out_size; i++){
        int a = ptr[i];
        int b = ptr_basic[i];
        int diff1 = a - b;
        int diff = diff1 < 0 ? -diff1 : diff1;
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
        }
        // if (i != 0 && i % sh[3] == 0)
        //     printf("\n");
        // printf("%d  ", diff);
        // if (diff1 != 0)
        //     printf("i: %d, out: %d, a: %d, b: %d \n", i, diff, a, b);
    }
    // printf("\n");
            LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
            CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
            LOG(INFO)<< "tensor_to_int8";
    tin.set_scale(scale_lite);
    TensorH tout1;
    TensorH tout_basic1;
    tout1.re_alloc(sh, AK_INT8);
    tout_basic1.re_alloc(sh, AK_INT8);
            LOG(INFO) << "tensor_to_int8_basic compute";
    tensor_to_int8_basic(tin, tout_basic1);
            LOG(INFO) << "trans_tensor_to_int8 compute";
    to = 0;
    min_time = 1000000;
    SaberTimer t2;
    for (int i = 0; i < iter; i++){
        t2.clear();
        t2.start();
        trans_tensor_fp32_to_int8(tin, tout1, &ctx);
        t2.end();
        to += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
    }
            LOG(INFO) << "trans_tensor_to_int8 running time, ave: " << to / iter << ", min time: " << min_time;
    ptr_basic = static_cast<char*>(tout_basic1.data());
    ptr = static_cast<char*>(tout1.data());
            LOG(INFO) << "trans_tensor_to_int8 diff, size: " << out_size;
    for (int i = 0; i < out_size; i++){
        int a = ptr[i];
        int b = ptr_basic[i];
        int diff1 = a - b;
        int diff = diff1 < 0 ? -diff1 : diff1;
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
        }
        // if (i != 0 && i % sh[3] == 0)
        //     printf("\n");
        // printf("%d  ", diff);
        // if (diff1 != 0)
        //     printf("i: %d, out: %d, a: %d, b: %d \n", i, diff, a, b);
    }
    return true;
}
bool test_fp32_to_int8_inplace(int axis, float scale_factor, Context ctx){
    Shape sh(get_rand(1, 10), get_rand(1, 50), get_rand(1, 512), get_rand(1, 512));
    // Shape sh(4, 32, 112, 112);
    TensorH tin, tin1, tin2, tin3, tin4;
    tin.re_alloc(sh, AK_FLOAT);
    tin1.re_alloc(sh, AK_FLOAT);
    tin2.re_alloc(sh, AK_FLOAT);
    tin3.re_alloc(sh, AK_FLOAT);
    tin4.re_alloc(sh, AK_FLOAT);
    fill_tensor_rand(tin, -20, 20);
    tin1.copy_from(tin);
    tin2.copy_from(tin);
    tin3.copy_from(tin);
    tin4.copy_from(tin);
            LOG(INFO) << "input shape num = " << sh[0];
            LOG(INFO) << "input shape channel = " << sh[1];
            LOG(INFO) << "input shape height = " << sh[2];
            LOG(INFO) << "input shape width = " << sh[3];
    std::vector<float> scale_basic;
    std::vector<float> scale_lite;
            LOG(INFO) << "get_scale_basic compute";
    scale_basic = get_scale_basic(tin, axis, scale_factor);
            LOG(INFO) << "get_tensor_scale compute";
    get_tensor_scale(tin, scale_lite, axis, scale_factor);
    if (scale_basic.size() != scale_lite.size()) {
        return false;
    }
    for (int i = 0; i < scale_basic.size(); ++i) {
        float tmp = scale_basic[i] - scale_lite[i];
        // if (tmp != 0){
        //     printf("i: %d, tmp: %.6f \n", i, tmp);
        // }
        if (fabsf(scale_basic[i] - scale_lite[i]) > 1e-4f) {
                    LOG(INFO) << "scale compute failed";
            return false;
        }
    }
            LOG(INFO) << "scale is right";
    TensorH tout;
    TensorH tout_basic;
    tout.re_alloc(sh, AK_INT8);
    tout_basic.re_alloc(sh, AK_INT8);
            LOG(INFO) << "fp32_to_int8_inplace_basic compute";
    fp32_to_int8_inplace_basic(tin1, axis + 1, scale_lite);
    // print_tensor(tout_basic);
            LOG(INFO) << "trans_fp32_weights_to_int8_inplace compute";
    // int outer_size = tin.count_valid(0, axis);
    // int inner_size = tin.count_valid(axis, tin.dims());
    // LOG(INFO) << "outer_size: " << outer_size << ", inner_size: " << inner_size;
    // fp32_to_int8((const float*)tin.data(), (char*)tout.mutable_data(), scale_lite, outer_size, inner_size);
    trans_fp32_weights_to_int8_inplace(tin2, scale_factor, 0, &ctx);
    // print_tensor(tout);
    double max_ratio = 0;
    double max_diff = 0;
    const double eps = 1e-6f;
    int out_size = tin2.valid_size();
    char* ptr_basic = static_cast<char*>(tin1.data());
    char* ptr = static_cast<char*>(tin2.data());
            LOG(INFO) << "trans_fp32_weights_to_int8 diff, size: " << out_size;
    for (int i = 0; i < out_size; i++){
        int a = ptr[i];
        int b = ptr_basic[i];
        int diff1 = a - b;
        int diff = diff1 < 0 ? -diff1 : diff1;
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
        }
        // if (i != 0 && i % sh[3] == 0)
        //     printf("\n");
        // printf("%d  ", diff);
        // if (diff1 != 0)
        //     printf("i: %d, out: %d, a: %d, b: %d \n", i, diff, a, b);
    }
    // printf("\n");
            LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
            CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
            LOG(INFO)<< "tensor_to_int8";
    tin3.set_scale(scale_lite);
    tin4.set_scale(scale_lite);
            LOG(INFO) << "tensor_to_int8_inplace_basic compute";
    tensor_to_int8_inplace_basic(tin3);
            LOG(INFO) << "trans_tensor_to_int8 compute";
    trans_tensor_fp32_to_int8_inplace(tin4, &ctx);
    ptr_basic = static_cast<char*>(tin3.data());
    ptr = static_cast<char*>(tin4.data());
            LOG(INFO) << "trans_tensor_to_int8 diff, size: " << out_size;
    for (int i = 0; i < out_size; i++){
        int a = ptr[i];
        int b = ptr_basic[i];
        int diff1 = a - b;
        int diff = diff1 < 0 ? -diff1 : diff1;
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (a + b + eps);
        }
        // if (i != 0 && i % sh[3] == 0)
        //     printf("\n");
        // printf("%d  ", diff);
        // if (diff1 != 0)
        //     printf("i: %d, out: %d, a: %d, b: %d \n", i, diff, a, b);
    }
    return true;
}
TEST(TestSaberLite, test_calibrate_lite) {
    Context ctx1;
    PowerMode mode = SABER_POWER_HIGH;
    ctx1.set_run_mode(mode, threads);
            LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
                LOG(INFO) << "number of threads: " << thread;
#endif
    }
#if 1
            LOG(INFO) << "scale compute";
    for (auto& axis : {0, 1, 2, 3}) {
        for (auto& scale : {63.f, 127.f}) {
            if (test_get_scale(axis, scale)) {
                        LOG(INFO) << "test calibrate get_scale, axis=" << axis << ", scale=" << scale;
            }else{
                        LOG(INFO) << "test calibrate get_scale, axis=" << axis << ", scale=" << scale <<", compute error";
                return;
            }
        }
    }
#endif
            LOG(INFO) << "****************************";
#if 1
            LOG(INFO) << "fp32_to_int8 compute";
    for (auto& axis : {0}) {
        for (auto& scale : {63.f, 127.f}) {
                    LOG(INFO) << "test calibrate get_scale, axis=" << axis << ", scale=" << scale;
            if (test_fp32_to_int8(axis, scale, ctx1)) {
                        LOG(INFO) << "The fp32_to_int8 result is right";
            }
        }
    }
#endif
            LOG(INFO) << "****************************";
#if 1
            LOG(INFO) << "fp32_to_inplace_int8 compute";
    for (auto& axis : {0}) {
        for (auto& scale : {63.f, 127.f}) {
                    LOG(INFO) << "test calibrate get_scale, axis=" << axis << ", scale=" << scale;
            if (test_fp32_to_int8_inplace(axis, scale, ctx1)) {
                        LOG(INFO) << "The fp32_to_inplace_int8 result is right";
            }
        }
    }
#endif
}
int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    Env::env_init();
    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if (argc >= 4) {
        iter = atoi(argv[3]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}