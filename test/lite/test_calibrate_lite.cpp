#include "saber/lite/funcs/calibrate_lite.h"
#include "test_lite.h"
using namespace anakin::saber;
using namespace anakin::saber::lite;
int cluster = 0;
int threads = 1;
int iter = 1;
typedef Tensor<CPU> TensorH;
std::vector<float> get_scale_basic(const TensorH& tin, int axis, float scale_factor) {
    int axis_size = 1;
    if (axis >= 0 && axis < tin.dims()) {
        axis_size = tin.valid_shape()[axis];
    }
    std::vector<float> scale_out(axis_size);
    long long outer_size = tin.count_valid(0, axis);
    long long inner_size = tin.count_valid(axis + 1, tin.dims());
    const float* in_data = static_cast<const float*>(tin.data());
#pragma omp parallel for
    for (int c = 0; c < axis_size; ++c) {
        float max_val = 0.f;
        const float* din = in_data + c * inner_size;
        for (int j = 0; j < outer_size; ++j) {
            const float* ptr_in = din + j * inner_size * axis_size;
            for (int i = 0; i < inner_size; ++i) {
                float read_data = fabsf(ptr_in[i]);
                max_val = (read_data > max_val) ? read_data : max_val;
            }
        }
        scale_out[c] = max_val / scale_factor;
    }
    return scale_out;
}

std::vector<float> get_scale_basic(const float* in_data, int axis_size, \
    long long outer_size, long long inner_size , float scale_factor) {
    std::vector<float> scale_out(axis_size);
#pragma omp parallel for
    for (int c = 0; c < axis_size; ++c) {
        float max_val = 0.f;
        const float* din = in_data + c * inner_size;
        for (int j = 0; j < outer_size; ++j) {
            const float* ptr_in = din + j * inner_size * axis_size;
            for (int i = 0; i < inner_size; ++i) {
                float read_data = fabsf(ptr_in[i]);
                max_val = (read_data > max_val) ? read_data : max_val;
            }
        }
        scale_out[c] = max_val / scale_factor;
    }
    return scale_out;
}

void fp32_to_int8_basic(const float* din, char* dout, const float* scale, \
    int axis_size, long long outer_size, long long inner_size) {
    int loop_size = axis_size * outer_size;
    for (int i = 0; i < loop_size; ++i) {
        float inv_scale = 1.f / scale[i % axis_size];
        for (int j = 0; j < inner_size; ++j) {
            dout[j] = static_cast<char>(roundf(din[j] * inv_scale));
        }
        dout += inner_size;
        din += inner_size;
    }
}

void int8_to_fp32_basic(const char* din, float* dout, const float* scale, \
    int axis_size, long long outer_size, long long inner_size) {
    int loop_size = axis_size * outer_size;
    for (int i = 0; i < loop_size; ++i) {
        float scale_in = scale[i % axis_size];
        for (int j = 0; j < inner_size; ++j) {
            dout[j] = din[j] * scale_in;
        }
        dout += inner_size;
        din += inner_size;
    }
}

void int32_to_fp32_basic(const int* din, float* dout, const float* scale, \
    int axis_size, long long outer_size, long long inner_size) {
    int loop_size = axis_size * outer_size;
    for (int i = 0; i < loop_size; ++i) {
        float scale_in = scale[i % axis_size];
        for (int j = 0; j < inner_size; ++j) {
            dout[j] = din[j] * scale_in;
        }
        dout += inner_size;
        din += inner_size;
    }
}

void int32_to_int8_basic(const int* din, char* dout, const float* scale, \
    int axis_size, long long outer_size, long long inner_size) {
    int loop_size = outer_size * axis_size;
    for (int i = 0; i < loop_size; ++i) {
        float scale_in = scale[i % axis_size];
        for (int j = 0; j < inner_size; ++j) {
            dout[j] = static_cast<char>(roundf(din[j] * scale_in));
        }
        dout += inner_size;
        din += inner_size;
    }
}

bool trans_weights_dtype_basic(Tensor<CPU>& weights, DataType type, float scale_factor, \
    bool is_trans = false) {
    if (weights.get_dtype() == type) {
        return true;
    }
    if (type == AK_FLOAT && weights.get_dtype() == AK_INT8) {
        //! trans int8 weights to fp32 weights
        if (weights.get_scale().size() <= 0) {
            LOGE("ERROR: Trans weights from int8 to fp32, without scale\n");
            return false;
        }
        Tensor<CPU> tmp_tensor;
        tmp_tensor.re_alloc(weights.valid_shape(), AK_FLOAT);
        std::vector<float> scale = weights.get_scale();
        const char* din = static_cast<const char*>(weights.data());
        float* dout = static_cast<float*>(tmp_tensor.mutable_data());

        if (is_trans) {
            //! for deconv
            int axis_size = weights.valid_shape()[0];
            int outer_size = weights.valid_shape()[1];
            int inner_size = weights.valid_shape()[2] * weights.valid_shape()[3];
            int8_to_fp32_basic(din, dout, scale.data(), axis_size, outer_size, inner_size);
        } else {
            //! for conv
            int axis_size = weights.valid_shape()[0];
            int outer_size = 1;
            int inner_size = weights.count_valid(1, weights.dims());
            int8_to_fp32_basic(din, dout, scale.data(), axis_size, outer_size, inner_size);
        }
        weights.re_alloc(weights.valid_shape(), AK_FLOAT);
        weights.copy_from(tmp_tensor);
    } else if (type == AK_INT8 && weights.get_dtype() == AK_FLOAT) {
        //! trans fp32 weights to int8 weights
        Tensor<CPU> tmp_tensor;
        tmp_tensor.re_alloc(weights.valid_shape(), AK_INT8);
        std::vector<float> scale;
        const float* din = static_cast<const float*>(weights.data());
        char* dout = static_cast<char*>(tmp_tensor.mutable_data());
        if (is_trans) {
            //! for deconv, chout and chin in inversed
            //! real layout is: chin, chout, kh, kw
            int axis_size = weights.valid_shape()[0];
            int outer_size = weights.valid_shape()[1];
            int inner_size = weights.valid_shape()[2] * weights.valid_shape()[3];
            scale = get_scale_basic(din, axis_size, outer_size, inner_size, scale_factor);
            fp32_to_int8_basic(din, dout, scale.data(), axis_size, outer_size, inner_size);
        } else {
            //! for conv
            //! layout is: chout, chin, kh, kw
            int axis_size = weights.valid_shape()[0];
            int inner_size = weights.valid_size() / axis_size;
            scale = get_scale_basic(din, axis_size, 1, inner_size, scale_factor);
            fp32_to_int8_basic(din, dout, scale.data(), axis_size, 1, inner_size);
        }
        //! set weights scale
        weights.set_scale(scale);
        weights.re_alloc(weights.valid_shape(), AK_INT8);
        weights.copy_from(tmp_tensor);
    } else {
        LOGE("ERROR: Trans weights fialed, unsupported data type\n");
        return false;
    }
    return true;
}

bool trans_tensor_fp32_to_int8_basic(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
    float input_scale) {
    if (tin.get_dtype() != AK_FLOAT) {
        return false;
    }
    if (tout.get_dtype() != AK_INT8) {
        tout.set_dtype(AK_INT8);
    }
    tout.reshape(tin.valid_shape());
    std::vector<float> scale = {input_scale};

    const float* din = static_cast<const float*>(tin.data());
    char* dout = static_cast<char*>(tout.mutable_data());
    //! convert to int8
    fp32_to_int8_basic(din, dout, scale.data(), 1, 1, tin.valid_size());
    return true;
}

bool trans_tensor_int8_to_fp32_basic(Tensor<CPU>& tin, Tensor<CPU>& tout, \
    float input_scale) {

    if (tin.get_dtype() != AK_INT8) {
        return false;
    }
    if (tout.get_dtype() != AK_FLOAT) {
        tout.set_dtype(AK_FLOAT);
    }
    tout.reshape(tin.valid_shape());

    //! compute scale
    std::vector<float> scale = {input_scale};

    const char* input = (const char*)tin.data();
    float* output = (float*)tout.mutable_data();

    int inner_size = tin.valid_size();

    //! convert to fp32
    int8_to_fp32_basic(input, output, scale.data(), 1, 1, inner_size);
    return true;
}

bool trans_tensor_int32_to_fp32_basic(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
	float input_scale, std::vector<float>& weights_scale) {

    if (tin.get_dtype() != AK_INT32) {
        return false;
    }
    if (tout.get_dtype() != AK_FLOAT) {
        tout.set_dtype(AK_FLOAT);
    }
    tout.reshape(tin.valid_shape());

    //! compute scale
    std::vector<float> scale(weights_scale.size());

    for (int i = 0; i < weights_scale.size(); ++i){
        scale[i] = input_scale * weights_scale[i];
    }

    const int* input = (const int*)tin.data();
    float* output = (float*)tout.mutable_data();

    int outer_size = tin.num();
    int axis_size = tin.channel();
    int inner_size = tin.width() * tin.height();

    //! convert to fp32
    int32_to_fp32_basic(input, output, scale.data(), axis_size, outer_size, inner_size);
    return true;
}

bool trans_tensor_int32_to_int8_basic(Tensor<CPU>& tin, Tensor<CPU>& tout, \
	float input_scale, float output_scale, std::vector<float>& weights_scale) {

    if (tin.get_dtype() != AK_INT32) {
        return false;
    }
    if (tout.get_dtype() != AK_INT8) {
        tout.set_dtype(AK_INT8);
    }
    tout.reshape(tin.valid_shape());

    //! compute scale
    std::vector<float> scale(weights_scale.size());
    for (int i = 0; i < weights_scale.size(); ++i){
        scale[i] = input_scale * weights_scale[i] / output_scale;
    }
    const int* input = (const int*)tin.data();
    char* output = (char*)tout.mutable_data();

    int outer_size = tin.num();
    int inner_size = tin.width() * tin.height();
    //! convert to int8
    int32_to_int8_basic(input, output, scale.data(), tin.channel(), outer_size, inner_size);
    return true;
}

bool trans_fp32_bias_to_int32_basic(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
    float in_scale, std::vector<float> vector_weight_scale) {

    if (tin.get_dtype() != AK_FLOAT || vector_weight_scale.size() != tin.valid_size()) {
        return false;
    }
    tout.set_dtype(AK_INT32);
    tout.reshape(tin.valid_shape());
    const float* in_data = static_cast<const float*>(tin.data());
    int* out_data = static_cast<int*>(tout.mutable_data());
    for (int i = 0; i < tin.valid_size(); ++i) {
        out_data[i] = static_cast<int>(roundf(in_data[i] / in_scale / vector_weight_scale[i]));
    }
    return SaberSuccess;
}
#if 1
TEST(TestSaberLite, test_get_scale) {
    for (auto& n : {1, 2}) {
    for (auto& c : {1, 2, 3, 16}) {
    for (auto& h : {1, 2, 3, 16, 112}) {
    for (auto& axis : {-1, 0, 1, 2, 3}) {
    for (auto& threads : {1, 2}) {
        Context ctx;
        ctx.set_run_mode(SABER_POWER_NO_BIND, threads);
        LOG(INFO) << "input shape: " << n << ", " << c << ", " << h << ", " << h;
        LOG(INFO) << "axis: " << axis << ", threads: " << threads;
        std::vector<float> scale_basic;
        std::vector<float> scale_saber;
        Tensor<CPU> tin;
        tin.re_alloc(Shape(n, c, h, h), AK_FLOAT);
        fill_tensor_rand(tin, -1.f * get_rand(0, 100), 1.f * get_rand(0, 100));
        scale_basic = get_scale_basic(tin, axis, 127.f);
        get_tensor_scale(tin, scale_saber, axis, 127.f);
        CHECK_EQ(scale_basic.size(), scale_saber.size()) << "scale size not euqal";
        for (int i = 0; i < scale_basic.size(); ++i) {
            float tmp = scale_basic[i] - scale_saber[i];
            if (fabsf(tmp) > 1e-5f) {
                print_tensor(tin);
                LOG(FATAL) << "compute result error: " << i \
                    << ", " << scale_basic[i] << ", " << scale_saber[i] << ", max basic: " << \
                    127.f * scale_basic[i] << ", max saber: " << scale_saber[i] * 127.f;
            }
        }
        LOG(INFO) << "get_tensor_scale result is right\n";
    }
    }
    }
    }
    }
}
#endif
#if 1
TEST(TestSaberLite, test_trans_weights) {
    for (auto& n : {1, 5, 32, 128}) {
    for (auto& c : {1, 3, 16, 64}) {
    for (auto& h : {1, 3, 4, 5}) {
    for (auto& trans : {false, true}) {
    for (auto& dtype_in : {AK_INT8, AK_FLOAT}) {
    for (auto& dtype_out : {AK_INT8, AK_FLOAT}) {
    for (auto& threads : {1, 2}) {
        Context ctx;
        ctx.set_run_mode(SABER_POWER_NO_BIND, threads);
        LOG(INFO) << "input shape: " << n << ", " << c << ", " << h << ", " << h;
        LOG(INFO) << "is_trans: " << (trans > 0? "true" : "false") << ", threads: " << threads;
        LOG(INFO) << "in type: " << (dtype_in == AK_INT8? "INT8" : "FP32") << \
            ", out type: " << (dtype_out == AK_INT8? "INT8" : "FP32");
        Tensor<CPU> tin_basic;
        Tensor<CPU> tin_saber;
        Tensor<CPU> tin_ori;
        tin_basic.re_alloc(Shape(n, c, h, h), dtype_in);
        tin_saber.re_alloc(Shape(n, c, h, h), dtype_in);
        tin_ori.re_alloc(Shape(n, c, h, h), dtype_in);
        if (dtype_in == AK_INT8) {
            fill_tensor_rand(tin_basic, -127, 127);
            std::vector<float> scale(n);
            Tensor<CPU> tmp;
            tmp.re_alloc(Shape(n), AK_FLOAT);
            fill_tensor_rand(tmp, 0.1f, 1.f);
            memcpy(scale.data(), tmp.data(), sizeof(float) * n);
            tin_basic.set_scale(scale);
            tin_saber.set_scale(scale);
            tin_ori.set_scale(scale);
        } else {
            fill_tensor_rand(tin_basic, -0.1f * get_rand(0, 100), 0.1f * get_rand(0, 100));
        }
        tin_ori.copy_from(tin_basic);
        tin_saber.copy_from(tin_basic);
        std::vector<float> scale_basic;
        std::vector<float> scale_saber;
        if (!trans_weights_dtype_basic(tin_basic, dtype_out, 127.f, trans)) {
            LOG(INFO) << "trans weights baisc failed\n";
            continue;
        }
        CHECK_EQ(trans_weights_dtype(tin_saber, dtype_out, 127.f, trans), SaberSuccess) << "compute failed";
        scale_basic = tin_basic.get_scale();
        scale_saber = tin_saber.get_scale();
        CHECK_EQ(scale_basic.size(), scale_saber.size()) << "scale size not euqal";
        for (int i = 0; i < scale_basic.size(); ++i) {
            float tmp = scale_basic[i] - scale_saber[i];
            if (fabsf(tmp) > 1e-5f) {
                print_tensor(tin_ori);
                LOG(FATAL) << "compute result error: " << i \
                    << ", " << scale_basic[i] << ", " << scale_saber[i] << ", max basic: " << \
                    127.f * scale_basic[i] << ", max saber: " << scale_saber[i] * 127.f;
            }
        }
        LOG(INFO) << "get_tensor_scale result is right";
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tin_basic, tin_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            if (max_diff > 1e-4f) {
                LOG(WARNING) << "origin tensor";
                print_tensor(tin_ori);
                for (int i = 0; i < tin_ori.get_scale().size(); ++i) {
                    LOG(WARNING) << "scale: " << tin_ori.get_scale()[i];
                }
                LOG(WARNING) << "basic result";
                print_tensor(tin_basic);
                LOG(WARNING) << "saber result";
                print_tensor(tin_saber);
                Tensor<CPU> tdiff(tin_basic.valid_shape(), tin_basic.get_dtype());
                tensor_diff(tin_basic, tin_saber, tdiff);
                print_tensor(tdiff);
                LOG(FATAL) << "trans weights, result error, " << \
                    "input shape: " << n << ", " << c << ", " << h << ", " << h;;
            }
        }
        LOG(INFO) << "test trans weights passed\n";
    }
    }
    }
    }
    }
    }
    }
}
#endif
#if 1
TEST(TestSaberLite, test_fp32_to_int8) {
    for (auto& n : {1, 5, 32, 64}) {
    for (auto& c : {1, 3, 16, 32}) {
    for (auto& h : {1, 3, 28, 150, 224}) {
    for (auto& threads : {1, 2}) {
        Context ctx;
        ctx.set_run_mode(SABER_POWER_NO_BIND, threads);
        LOG(INFO) << "input shape: " << n << ", " << c << ", " << h << ", " << h;
        LOG(INFO) << "threads: " << threads;
        Tensor<CPU> tin;
        tin.re_alloc(Shape(n, c, h, h), AK_FLOAT);
        fill_tensor_rand(tin, -0.1f * get_rand(0, 100), 0.1f * get_rand(0, 100));
        std::vector<float> scale_basic;
        std::vector<float> scale_saber;
        scale_basic = get_scale_basic(tin, -1, 127.f);
        get_tensor_scale(tin, scale_saber, -1, 127.f);
        CHECK_EQ(scale_basic.size(), scale_saber.size()) << "scale size not euqal";
        for (int i = 0; i < scale_basic.size(); ++i) {
            float tmp = scale_basic[i] - scale_saber[i];
            CHECK_EQ(fabsf(tmp) < 1e-5f, true) << "compute result error";
        }
        LOG(INFO) << "get_tensor_scale result is right";

        Tensor<CPU> tout_basic;
        Tensor<CPU> tout_saber;

        tout_basic.re_alloc(tin.valid_shape(), AK_INT8);
        tout_saber.re_alloc(tin.valid_shape(), AK_INT8);

        trans_tensor_fp32_to_int8_basic(tin, tout_basic, scale_basic[0]);
        trans_tensor_fp32_to_int8(tin, tout_saber, scale_saber[0]);

        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            if (max_diff > 1e-4f) {
                LOG(WARNING) << "basic result";
                print_tensor(tout_basic);
                LOG(WARNING) << "saber result";
                print_tensor(tout_saber);
                Tensor<CPU> tdiff(tout_basic.valid_shape(), tout_basic.get_dtype());
                tensor_diff(tout_basic, tout_saber, tdiff);
                print_tensor(tdiff);
                LOG(FATAL) << "trans fp32 to int8, result error" << \
                    ", input shape: " << n << ", " << c << ", " << h << ", " << h;;
            }
        }
        LOG(INFO) << "test trans tensor fp32 to int8 passed\n";
    }
    }
    }
    }
}
#endif
#if 1
TEST(TestSaberLite, test_int8_to_fp32) {
    for (auto& n : {1, 5, 32, 64}) {
    for (auto& c : {1, 3, 16, 32}) {
    for (auto& h : {1, 3, 28, 150, 224}) {
    for (auto& threads : {1, 2}) {
        Context ctx;
        ctx.set_run_mode(SABER_POWER_NO_BIND, threads);
        LOG(INFO) << "input shape: " << n << ", " << c << ", " << h << ", " << h;
        LOG(INFO) << "threads: " << threads;
        Tensor<CPU> tin;
        tin.re_alloc(Shape(n, c, h, h), AK_INT8);
        fill_tensor_rand(tin, -127, 127);

        std::vector<float> scale = {get_rand(0, 100) * 0.1f};
        tin.set_scale(scale);

        Tensor<CPU> tout_basic;
        Tensor<CPU> tout_saber;

        tout_basic.re_alloc(tin.valid_shape(), AK_FLOAT);
        tout_saber.re_alloc(tin.valid_shape(), AK_FLOAT);

        trans_tensor_int8_to_fp32_basic(tin, tout_basic, scale[0]);
        trans_tensor_int8_to_fp32(tin, tout_saber, scale[0]);

        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            if (max_diff > 1e-4f) {
                LOG(WARNING) << "basic result";
                print_tensor(tout_basic);
                LOG(WARNING) << "saber result";
                print_tensor(tout_saber);
                Tensor<CPU> tdiff(tout_basic.valid_shape(), tout_basic.get_dtype());
                tensor_diff(tout_basic, tout_saber, tdiff);
                print_tensor(tdiff);
                LOG(FATAL) << "trans int8 to fp32, result error" << \
                    ", input shape: " << n << ", " << c << ", " << h << ", " << h;;
            }
        }
        LOG(INFO) << "test trans tensor int8 to fp32 passed\n";
    }
    }
    }
    }
}
#endif
#if 1
TEST(TestSaberLite, test_int32_to_fp32) {
    for (auto& n : {1, 5, 32, 64}) {
    for (auto& c : {1, 3, 16, 32}) {
    for (auto& h : {1, 3, 28, 150, 224}) {
    for (auto& threads : {1, 2}) {
        Context ctx;
        ctx.set_run_mode(SABER_POWER_NO_BIND, threads);
        LOG(INFO) << "input shape: " << n << ", " << c << ", " << h << ", " << h;
        LOG(INFO) << "threads: " << threads;
        Tensor<CPU> tin;
        tin.re_alloc(Shape(n, c, h, h), AK_INT32);
        fill_tensor_rand(tin, -127 * 32, 127 * 32);

        std::vector<float> scale_in = {get_rand(0, 100) * 0.01f};
        std::vector<float> scale_w(c);
        for (int i = 0; i < c; ++i) {
            scale_w[i] = get_rand(0, 100) * 0.01f;
        }

        Tensor<CPU> tout_basic;
        Tensor<CPU> tout_saber;

        tout_basic.re_alloc(tin.valid_shape(), AK_FLOAT);
        tout_saber.re_alloc(tin.valid_shape(), AK_FLOAT);

        trans_tensor_int32_to_fp32_basic(tin, tout_basic, scale_in[0], scale_w);
        trans_tensor_int32_to_fp32(tin, tout_saber, scale_in[0], scale_w);

        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            if (max_diff > 1e-4f) {
                LOG(WARNING) << "origin: ";
                print_tensor(tin);
                LOG(WARNING) << "basic result";
                print_tensor(tout_basic);
                LOG(WARNING) << "saber result";
                print_tensor(tout_saber);
                Tensor<CPU> tdiff;
                tdiff.re_alloc(tout_basic.valid_shape(), tout_basic.get_dtype());
                tensor_diff(tout_basic, tout_saber, tdiff);
                LOG(WARNING) << "diff:";
                print_tensor(tdiff);
                LOG(FATAL) << "trans int32 to fp32, result error" << \
                    ", input shape: " << n << ", " << c << ", " << h << ", " << h;;
            }
        }
        LOG(INFO) << "test trans tensor int32 to fp32 passed\n";
    }
    }
    }
    }
}
#endif
#if 1
TEST(TestSaberLite, test_int32_to_int8) {
    for (auto& n : {1, 5, 32, 64}) {
    for (auto& c : {1, 3, 16, 32}) {
    for (auto& h : {1, 3, 28, 150, 224}) {
    for (auto& threads : {1, 2}) {
        Context ctx;
        ctx.set_run_mode(SABER_POWER_NO_BIND, threads);
        LOG(INFO) << "input shape: " << n << ", " << c << ", " << h << ", " << h;
        LOG(INFO) << "threads: " << threads;
        Tensor<CPU> tin;
        tin.re_alloc(Shape(n, c, h, h), AK_INT32);
        fill_tensor_rand(tin, -127 * 32, 127 * 32);

        std::vector<float> scale_in = {1.f / 2.f};
        std::vector<float> scale_out = {1.f};
        std::vector<float> scale_w(c);
        for (int i = 0; i < c; ++i) {
            scale_w[i] = 1 / 16.f;
        }

        Tensor<CPU> tout_basic;
        Tensor<CPU> tout_saber;

        tout_basic.re_alloc(tin.valid_shape(), AK_FLOAT);
        tout_saber.re_alloc(tin.valid_shape(), AK_FLOAT);

        trans_tensor_int32_to_int8_basic(tin, tout_basic, scale_in[0], scale_out[0], scale_w);
        trans_tensor_int32_to_int8(tin, tout_saber, scale_in[0], scale_out[0], scale_w);

        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            if (max_diff > 1e-4f) {
                LOG(WARNING) << "basic result";
                print_tensor(tout_basic);
                LOG(WARNING) << "saber result";
                print_tensor(tout_saber);
                Tensor<CPU> tdiff(tout_basic.valid_shape(), tout_basic.get_dtype());
                tensor_diff(tout_basic, tout_saber, tdiff);
                print_tensor(tdiff);
                LOG(FATAL) << "trans int32 to int8, result error, " << \
                    "input shape: " << n << ", " << c << ", " << h << ", " << h;;
            }
        }
        LOG(INFO) << "test trans tensor int32 to int8 passed\n";
    }
    }
    }
    }
}
#endif
#if 1
TEST(TestSaberLite, test_trans_bias_to_int32) {
    for (auto& size : {1, 5, 32, 64, 128}) {
        for (auto& threads : {1, 2}) {
            Context ctx;
            ctx.set_run_mode(SABER_POWER_NO_BIND, threads);
            LOG(INFO) << "bias size: " << size << ", threads: " << threads;
            Tensor<CPU> tin_basic;
            tin_basic.re_alloc(Shape(size), AK_FLOAT);
            fill_tensor_rand(tin_basic, -1.f, 1.f);

            std::vector<float> scale_in = {1.f / 2.f};
            std::vector<float> scale_w(size);
            for (int i = 0; i < size; ++i) {
                scale_w[i] = 0.1f * get_rand(1, 10);
            }

            Tensor<CPU> tin_saber;
            tin_saber.re_alloc(tin_basic.valid_shape(), AK_FLOAT);
            tin_saber.copy_from(tin_basic);

            trans_fp32_bias_to_int32_basic(tin_basic, tin_basic, scale_in[0], scale_w);
            trans_fp32_bias_to_int32(tin_saber, tin_saber, scale_in[0], scale_w);

            double max_ratio = 0;
            double max_diff = 0;
            tensor_cmp_host(tin_basic, tin_saber, max_ratio, max_diff);
            LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
            if (fabsf(max_ratio) > 1e-3f) {
                if (max_diff > 1e-4f) {
                    LOG(WARNING) << "basic result";
                    print_tensor(tin_basic);
                    LOG(WARNING) << "saber result";
                    print_tensor(tin_saber);
                    Tensor<CPU> tdiff(tin_basic.valid_shape(), tin_saber.get_dtype());
                    tensor_diff(tin_basic, tin_saber, tdiff);
                    print_tensor(tdiff);
                    LOG(FATAL) << "trans bias to int32, result error, size:" << \
                        size << ", threads: " << threads;
                }
            }
            LOG(INFO) << "test trans tensor int32 to int8 passed\n";
        }
    }
}
#endif
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
