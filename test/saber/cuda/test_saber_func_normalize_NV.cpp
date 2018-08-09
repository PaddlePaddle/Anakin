
#include "core/context.h"
#include "funcs/normalize.h"
#include "test_saber_func_normalize_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

template <typename dtype>
void norm_cpu_nchw(const int p, const dtype* scale, const dtype* src, dtype* dst, \
                   bool across_spatial, bool has_scale, bool channel_shared, float eps, \
                   int n, int c, int h, int w) {

    const dtype* src_ptr = src;
    dtype* dst_ptr = dst;

    if (across_spatial) {
        int compute_size = h * w * c;
        int outer_size = n * c * h * w / compute_size;

        for (int i = 0; i < outer_size; ++i) {
            dtype sum = 0;

            for (int j = 0; j < compute_size; ++j) {
                if (p == 1) {
                    sum += fabsf(src_ptr[j]);
                } else {
                    sum += src_ptr[j] * src_ptr[j];
                }
            }

            //LOG(INFO) << "idx: " << i << ", " << "norm: " << sum;

            if (p == 1) {
                sum = 1 / (sum + eps);
            } else {
                sum = 1 / (sqrtf(sum) + eps);
            }

            if (has_scale) { //! with scale
                if (channel_shared) { // scale is shared across channel
                    for (int j = 0; j < compute_size; ++j) {
                        dst_ptr[j] = src_ptr[j] * sum * scale[0];
                    }
                } else {
                    for (int j = 0; j < compute_size; ++j) {
                        int c_idx = j / (h * w);
                        dst_ptr[j] = src_ptr[j] * sum * scale[c_idx];
                    }
                }
            } else { //! without scale
                for (int j = 0; j < compute_size; ++j) {
                    dst_ptr[j] = src_ptr[j] * sum;
                }
            }

            src_ptr += compute_size;
            dst_ptr += compute_size;
        }
    } else {
        int channel_in_size = h * w;

        for (int i = 0; i < n; ++i) {
            const dtype* src_batch_ptr = src_ptr + i * c * h * w;
            dtype* dst_batch_ptr = dst_ptr + i * c * h * w;

            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < w; ++k) {
                    const dtype* src_pixel = src_batch_ptr + 0 * channel_in_size + j * w + k;
                    dtype* dst_pixel = dst_batch_ptr + 0 * channel_in_size + j * w + k;
                    float norm = 0.f;

                    for (int l = 0; l < c; ++l) {
                        if (p == 1) {
                            norm += fabsf(src_pixel[l * channel_in_size]);
                        } else {
                            norm += src_pixel[l * channel_in_size] * src_pixel[l * channel_in_size];
                        }
                    }

                    if (p == 1) {
                        norm = 1.f / (norm + eps);
                    } else {
                        norm = 1.f / (sqrtf(norm) + eps);
                    }

                    for (int l = 0; l < c; ++l) {
                        if (has_scale) {
                            if (channel_shared) {
                                dst_pixel[l * channel_in_size] = \
                                                                 src_pixel[l * channel_in_size] * norm * scale[0];
                            } else {
                                dst_pixel[l * channel_in_size] = \
                                                                 src_pixel[l * channel_in_size] * norm * scale[l];
                            }
                        } else {
                            dst_pixel[l * channel_in_size] = \
                                                             src_pixel[l * channel_in_size] * norm;
                        }
                    }
                }
            }
        }
    }
}


SaberStatus test_norm_continue_nchw(int p, bool across_spatial, \
                                    bool has_scale, bool channel_shared) {

    typedef TargetWrapper<NV> API;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef TensorDf4::Dtype dtype;

    int test_iter = 100;
    float eps = 1e-6f;

    int w_in = 128;
    int h_in = 128;
    int ch_in = 64;
    int num_in = 1;
    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    //! create normalize param
    int ch_scale = channel_shared ? 1 : ch_in;
    Shape sh_slope{1, 1, 1, ch_scale};
    Tensor<X86, AK_FLOAT, NCHW> th_scale(sh_slope);
    TensorDf4 tdscale(sh_slope);

    for (int i = 0; i < ch_scale; ++i) {
        th_scale.mutable_data()[i] = 0.1f * (i + 1);
    }

    tdscale.copy_from(th_scale);
    NormalizeParam<TensorDf4> param;

    if (has_scale) {
        NormalizeParam<TensorDf4> param_tmp(across_spatial, channel_shared, &tdscale, eps, p);
        param = param_tmp;
    } else {
        NormalizeParam<TensorDf4> param_tmp(across_spatial, eps, p);
        param = param_tmp;
    }

    LOG(INFO) << "create normalize param";

    //! create input output tensor
    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in), thout(shape_in);

    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = 1;//-i + thin.size() / 2 / ch_in;
    }

    TensorDf4 tdin, tdout;
    tdin.re_alloc(shape_in);
    SABER_CHECK(tdin.copy_from(thin));
    CUDA_POST_KERNEL_CHECK;
    input_dev_4d.push_back(&tdin);
    output_dev_4d.push_back(&tdout);
    //print_tensor_device(tdin);
    //cudaDeviceSynchronize();
    //CUDA_POST_KERNEL_CHECK;
    //! create process contex
    Context<NV> ctx_dev(0, 1, 1);

    //! create normalize class
    Normalize<NV, AK_FLOAT> norm;
    LOG(INFO) << "normalize compute ouput shape";
    SABER_CHECK(norm.compute_output_shape(input_dev_4d, output_dev_4d, param));
    //LOG(INFO) << "re-alloc tensor buffer";
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());
    Shape va_sh = tdout.valid_shape();
    LOG(INFO) << "shape out 4d: " << va_sh[0] << ", " << va_sh[1] << ", " << \
              va_sh[2] << ", " << va_sh[3];
    CHECK_EQ(va_sh == shape_out, true) << "compute output shape error";

    LOG(INFO) << "normalize initialization";
    SABER_CHECK(norm.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev));

    LOG(INFO) << "normalize compute";
    //! compute result by cpu funcs
    norm_cpu_nchw(p, th_scale.data(), thin.data(), thout.mutable_data(), \
                  across_spatial, has_scale, channel_shared, eps, num_in, ch_in, h_in, w_in);
    //print_tensor_host(thout);
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        SABER_CHECK(norm(input_dev_4d, output_dev_4d, param, ctx_dev));
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    LOG(INFO) << "total time: " << ts << ", avg time: " << ts / test_iter;
    //print_tensor_device(*output_dev_4d[0]);
    //cudaDeviceSynchronize();

    Tensor<X86, AK_FLOAT, NCHW> th_result(shape_in);
    th_result.copy_from(tdout);
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(thout.data(), th_result.data(), thout.valid_size(), max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    //CHECK_LE(fabs(max_ratio), 1.0e-6) << "error result";
    LOG(INFO) << "\n";
    return SaberSuccess;
}

TEST(TestSaberFuncNormalizeNV, test_func_normalize_without_roi_NV) {
    for (auto& sp_flag : {
                false, true
            }) {
        for (auto& scale_flag : {
                    false
                }) {
            for (auto& channel_flag : {
                        false, true
                    }) {
                for (auto& p : {
                            1, 2
                        }) {
                    LOG(WARNING) << "across spatio: " << sp_flag << ", has scale: " << \
                                 scale_flag << ", shared channel: " << channel_flag << ", p:" << p;
                    test_norm_continue_nchw(p, sp_flag, scale_flag, channel_flag);
                }
            }
        }
    }
}

#if 0
TEST(TestSaberFuncNormalizeNV, test_func_normalize_ROI_NV) {

    typedef TargetWrapper<NV> API;

    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 1000;

    int w_in = 10;
    int h_in = 4;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_in_roi{num_in, ch_in, h_in / 2, w_in / 2};
    Shape off1{0, 0, 0, 0};
    Shape off2{0, 0, 2, 5};
    Shape shape_out = shape_in_roi;

    Shape sh_slope{1, 1, 1, ch_in};
    Tensor<X86, AK_FLOAT, NCHW> th_slope(sh_slope);
    TensorDf4 tslop(sh_slope);

    for (int i = 0; i < ch_in; ++i) {
        th_slope.mutable_data()[i] = 0.1f * (i + 1);
    }

    tslop.copy_from(th_slope);

    PreluParam<TensorDf4> param_shared(true, &tslop);
    PreluParam<TensorDf4> param(false, &tslop);

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    std::vector<TensorDf4*> in_4d1, in_4d2;
    std::vector<TensorDf4*> out_4d1, out_4d2;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);

    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = -i + thin.size() / 2 / ch_in;
    }

    TensorDf4 tdin, tdin_roi1, tdin_roi2, tdout, tdout_roi1, tdout_roi2;
    tdin.re_alloc(shape_in);
    tdout.re_alloc(shape_in);
    tdin.copy_from(thin);
    tdin_roi1.share_sub_buffer(tdin, shape_in_roi, off1);
    tdin_roi2.share_sub_buffer(tdin, shape_in_roi, off2);
    in_4d1.push_back(&tdin_roi1);
    in_4d2.push_back(&tdin_roi2);
    out_4d1.push_back(&tdout_roi1);
    out_4d2.push_back(&tdout_roi2);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    Prelu<NV, AK_FLOAT> prelu_dev1;
    Prelu<NV, AK_FLOAT> prelu_dev2;

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    prelu_dev1.compute_output_shape(out_4d1, in_4d1, param);
    prelu_dev2.compute_output_shape(out_4d2, in_4d2, param_shared);

    LOG(INFO) << "re-alloc tensor buffer";
    out_4d1[0]->share_sub_buffer(tdout, shape_in_roi, off1);
    out_4d2[0]->share_sub_buffer(tdout, shape_in_roi, off2);

    CHECK_EQ(out_4d1[0]->valid_shape() == shape_out, true) << "compute shape error";

    LOG(INFO) << "prelu initialization";
    prelu_dev1.init(in_4d1, out_4d1, param, SPECIFY, SABER_IMPL, ctx_dev);
    prelu_dev2.init(in_4d2, out_4d2, param_shared, SPECIFY, SABER_IMPL, ctx_dev);

    LOG(INFO) << "prelu compute";
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        prelu_dev1(in_4d1, out_4d1, param, ctx_dev);
        out_4d1[0]->record_event(ctx_dev.get_compute_stream());
        prelu_dev2(in_4d2, out_4d2, param_shared, ctx_dev);
        out_4d2[0]->record_event(ctx_dev.get_compute_stream());
        out_4d1[0]->sync();
        out_4d2[0]->sync();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    printf("total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    print_tensor_device(tdout);
    cudaDeviceSynchronize();
    TensorDf4 troi(out_4d1[0]->valid_shape());
    troi.copy_from(*out_4d1[0]);
    print_tensor_device(troi);
    cudaDeviceSynchronize();
    troi.copy_from(*out_4d2[0]);
    print_tensor_device(troi);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
}
#endif //if 0
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

