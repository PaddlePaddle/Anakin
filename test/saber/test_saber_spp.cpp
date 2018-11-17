#include "saber/core/context.h"
#include "saber/funcs/spp.h"
#include "saber/funcs/pooling.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>
using namespace anakin::saber;
template<typename dtype, typename TargetType_H>
void pooling_cpu(const std::vector<Tensor<TargetType_H>*>& input,
                 std::vector<Tensor<TargetType_H>*>& output, PoolingParam<TargetType_H>& param) {
    const dtype* src_ptr = static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());

    int in_n = input[0]->num();
    int in_c = input[0]->channel();
    int in_h = input[0]->height();
    int in_w = input[0]->width();
    int size_in_n = in_c * in_h * in_w;
    int size_in_c = in_h * in_w;

    int out_h = output[0]->height();
    int out_w = output[0]->width();
    int size_out_n = in_c * out_h * out_w;
    int size_out_c = out_h * out_w;

    for (int ind_n = 0; ind_n < in_n; ++ind_n) {
        for (int ind_c = 0; ind_c < in_c; ++ind_c) {
            for (int ind_h = 0; ind_h < out_h; ++ind_h) {
                int sh = ind_h * param.stride_h;
                int eh = sh + param.window_h;

                if (param.pad_h > 0) {
                    sh = (sh - param.pad_h) < 0 ? 0 : sh - param.pad_h;
                    eh = (eh - param.pad_h) > in_h ? in_h : eh - param.pad_h;
                }

                for (int ind_w = 0; ind_w < out_w; ++ind_w) {
                    int sw = ind_w * param.stride_w;
                    int ew = sw + param.window_w;

                    if (param.pad_w > 0) {
                        sw = (sw - param.pad_w) < 0 ? 0 : sw - param.pad_w;
                        ew = (ew - param.pad_w) > in_w ? in_w : ew - param.pad_w;
                    }

                    dtype result = dtype(0);

                    int dst_ind = ind_n * size_out_n + ind_c * size_out_c + ind_h * out_w + ind_w;

                    for (int kh = sh; kh < eh; ++kh) {
                        for (int kw = sw; kw < ew; ++kw) {
                            int src_ind = ind_n * size_in_n + ind_c * size_in_c + kh * in_w + kw;

                            if (kh == sh && kw == sw) {
                                result = src_ptr[src_ind];
                            } else {
                                if (param.pooling_type == Pooling_max) {
                                    result = result >= src_ptr[src_ind] ? result : src_ptr[src_ind];
                                }

                                if (param.pooling_type == Pooling_average_include_padding) {
                                    result += src_ptr[src_ind];
                                }

                                if (param.pooling_type == Pooling_average_exclude_padding) {
                                    result += src_ptr[src_ind];
                                }
                            }

                        }
                    }

                    if (param.pooling_type == Pooling_average_include_padding) {
                        result /= param.window_h * param.window_w;
                    }

                    if (param.pooling_type == Pooling_average_exclude_padding) {
                        result /= (ew - sw) * (eh - sh);
                    }

                    dst_ptr[dst_ind] = result;

                }
            }
        }

    }
}
template <typename dtype, typename TargetType_D, typename TargetType_H>
void spp_cpu(const std::vector<Tensor<TargetType_H>*>& input,
             std::vector<Tensor<TargetType_H>*>& output, \
             SPPParam<TargetType_D>& param) {
    int pyramid_height = param.pyramid_height;
    typedef Pooling<TargetType_H, AK_FLOAT> pool_t;
    pool_t* pool = new pool_t[pyramid_height];
    dtype* out_data = (dtype*)output[0]->mutable_data();
    int spatial_size = output[0]->width() * output[0]->height();

    for (int i = 0; i < pyramid_height; ++i) {
        int out_w = pow(2, i);
        int out_h = pow(2, i);
        int in_w = input[0]->valid_shape()[3];
        int in_h = input[0]->valid_shape()[2];
        int window_w = std::ceil(in_w / (double)out_w);
        int window_h = std::ceil(in_h / (double)out_h);
        int pad_w = (window_w * out_w - in_w + 1) / 2;
        int pad_h = (window_h * out_h - in_h + 1) / 2;
        Tensor<TargetType_H> poolout;
        std::vector<Tensor<TargetType_H>*> pool_out;
        pool_out.push_back(&poolout);
        PoolingParam<TargetType_H> pool_param(window_h, window_w, pad_h, pad_w, window_h, window_w,
                                              param.pool_type);
        pool[i].compute_output_shape(input, pool_out, pool_param);
        Shape sh = pool_out[0]->valid_shape();
        pool_out[0]->re_alloc(sh, AK_FLOAT);
        pooling_cpu<dtype, TargetType_H>(input, pool_out, pool_param);
        int valid_size = pool_out[0]->valid_size();
        int spatial_size_out = pool_out[0]->width() * pool_out[0]->height();
        dtype* in_data = (dtype*)pool_out[0]->mutable_data();
        int offset = (pow(4, i) - 1) / 3;

        for (int i = 0; i < valid_size; ++i) {
            int idx = i / spatial_size_out;
            int out_index = idx * spatial_size + i % spatial_size_out;
            out_data[out_index + offset] = in_data[i];
        }
    }

    delete [] pool;
}
TEST(TestSaberFunc, test_func_scale) {
#ifdef USE_CUDA
    LOG(INFO) << "NV test......";
    TestSaberBase<NV, NVHX86, AK_FLOAT, Spp, SPPParam> testbase;

    for (auto num : {1, 3, 11}) {
    for (auto c : {1, 3, 11}) {
    for (auto h : {16, 32, 64}) {
    for (auto w : {16, 32, 64}) {
    for (auto pyramid_height : {1, 2, 3}) {
    for (auto pool_type : {
            Pooling_max, Pooling_average_exclude_padding, Pooling_average_include_padding
        }) {
        SPPParam<NV> param(pyramid_height, pool_type);
        testbase.set_param(param);
        testbase.set_input_shape(Shape({num, c, h, w}));
        testbase.run_test(spp_cpu<float, NV, NVHX86>, 0.0001);
    }
    }
    }
    }
    }
    }

    LOG(INFO) << "NV test end.";
#endif

#ifdef USE_X86_PLACE
    LOG(INFO) << "x86 test......";

    do {
        TestSaberBase<X86, X86, AK_FLOAT, Spp, SPPParam> testbase;

    for (auto num : {1, 3, 11}) {
    for (auto c : {1, 3, 11}) {
    for (auto h : {16, 32, 64}) {
    for (auto w : {16, 32, 64}) {
    for (auto pyramid_height : {1, 2, 3}) {
    for (auto pool_type : {
            Pooling_max, Pooling_average_exclude_padding, Pooling_average_include_padding
    }) {
        SPPParam<X86> param(pyramid_height, pool_type);
        testbase.set_param(param);
        testbase.set_input_shape(Shape({num, c, h, w}));
        testbase.run_test(spp_cpu<float, X86, X86>, 0.0001);
    }
    }
    }
    }
    }
    }

    } while (0);

    LOG(INFO) << "x86 test end.";

#endif
}
int main(int argc, const char** argv) {
    // initial logger
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
