
#include "core/context.h"
#include "funcs/normalize.h"
#include "test_saber_func.h"
#include "tensor_op.h"
#include "saber_types.h"
#include "saber/core/tensor_op.h"
#include <vector>

using namespace anakin::saber;

template <typename dtype>
void norm_cpu_nchw(int p, const dtype* scale, const dtype* src, dtype* dst, \
                   bool across_spatial, bool has_scale,bool channel_shared, float eps, \
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
                sum = 1 / sqrtf(sum+eps);
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
                    const dtype* src_pixel = src_batch_ptr + j * w + k;
                    dtype* dst_pixel = dst_batch_ptr  + j * w + k;
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
                        norm = 1.f / sqrtf(norm+eps);
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

template <typename TargetType,typename TargetType_H>
SaberStatus test_norm_continue_nchw(int p, bool across_spatial, bool has_scale,bool channel_shared,\
                            int w_in,int h_in,int ch_in,int num_in)
{
    
    typedef TargetWrapper<TargetType> API;
    typedef Tensor<TargetType> TensorD;
    typedef float dtype;

    int test_iter = 100;
    float eps = 1e-6f;

    //LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    //! create normalize param
    int ch_scale = channel_shared ? 1 : ch_in;
    Shape sh_slope({1, 1, 1, ch_scale});
    Tensor<TargetType_H> th_scale(sh_slope);
    TensorD tdscale;
    tdscale.re_alloc(sh_slope,AK_FLOAT);
    for (int i = 0; i < ch_scale; ++i) {
        static_cast<dtype *>(th_scale.mutable_data())[i] = 0.1f * (i + 1);/**/
    }

    tdscale.copy_from(th_scale);
    NormalizeParam<TargetType> param;

    if (has_scale) {
        NormalizeParam<TargetType> param_tmp(across_spatial, channel_shared, &tdscale, eps, p);
        param = param_tmp;
    } else {
        NormalizeParam<TargetType> param_tmp(across_spatial, eps, p);
        param = param_tmp;
    }

    //LOG(INFO) << "create normalize param";

    //! create input output tensor
    Shape shape_in({num_in, ch_in, h_in, w_in});
    Shape shape_out = shape_in;

    std::vector<TensorD*> input_dev;
    std::vector<TensorD*> output_dev;

    Tensor<TargetType_H> thin(shape_in), thout(shape_out);

    for (int i = 0; i < thin.size(); ++i) {
        static_cast<dtype*>(thin.mutable_data())[i] = 1;
    }

    TensorD tdin, tdout;
    tdin.re_alloc(shape_in,AK_FLOAT);
    SABER_CHECK(tdin.copy_from(thin));
#ifdef USE_CUDA
    CUDA_POST_KERNEL_CHECK;
#endif
    input_dev.push_back(&tdin);
    output_dev.push_back(&tdout);
    
    //! create process contex
    Context<TargetType> ctx_dev(0, 1, 1);

    //! create normalize class
    Normalize<TargetType,AK_FLOAT> norm;
    //LOG(INFO) << "normalize compute ouput shape";
    SABER_CHECK(norm.compute_output_shape(input_dev, output_dev, param));
    
    output_dev[0]->re_alloc(output_dev[0]->valid_shape(),AK_FLOAT);
    
    Shape va_sh = tdout.valid_shape();
    //LOG(INFO) << "shape out 4d: " << va_sh[0] << ", " << va_sh[1] << ", " << va_sh[2] << ", " << va_sh[3];
    CHECK_EQ(va_sh == shape_out, true) << "compute output shape error";

    //LOG(INFO) << "normalize initialization";
    SABER_CHECK(norm.init(input_dev, output_dev, param, SPECIFY, SABER_IMPL, ctx_dev));

    //LOG(INFO) << "normalize compute";
    //! compute result by cpu funcs
    norm_cpu_nchw(p, static_cast<dtype*>(th_scale.data()), static_cast<dtype*>(thin.data()), static_cast<dtype*>(thout.mutable_data()), \
                  across_spatial, has_scale, channel_shared, eps, num_in, ch_in, h_in, w_in);
    
    SaberTimer<TargetType> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        SABER_CHECK(norm(input_dev, output_dev, param, ctx_dev));
        //output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());/**/
        typename TensorD::API::stream_t stream = ctx_dev.get_compute_stream();
        output_dev[0]->record_event(stream);
        output_dev[0]->sync();
    }
#ifdef USE_CUDA
    CUDA_POST_KERNEL_CHECK;
#endif
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    
    Tensor<TargetType_H> th_result(shape_in);
    th_result.copy_from(tdout);
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host<float>(static_cast<float*>(thout.data()), static_cast<float*>(th_result.data()), thout.valid_size(), max_ratio, max_diff);
    //LOG(INFO) << "total time: " << ts << ", avg time: " << ts / test_iter;
    //LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    
   // LOG(INFO) << "\n";
    if(max_ratio<=0.000001){
        //LOG(INFO)<<"TEST PASSED!!!"<<"\n";
        return SaberSuccess;
    }
    else{
        //LOG(INFO)<<"TEST FAILED!!!"<<"\n";
        return SaberUnKownError;
    }
    return SaberSuccess;
}

TEST(TestSaberFunc, test_func_normalize) {
    bool scale_flag=false;
    int total_count=2 * 2 * 2 * 3 * 3 * 2 * 2;
    int pass_count=0;
    for (bool sp_flag : {false, true}) {
            for (bool channel_flag : {false, true}) {
                for (int p : {1, 2}) {
                    for(int w_in:{64,128,256}){
                        for(int h_in: {64,128,256}){
                            for(int ch_in:{32,64}){
                                for(int num_in:{1,2}){
                        
                                    LOG(WARNING) << "across spatio: " << sp_flag << ", has scale: " << \
                                    scale_flag << ", shared channel: " << channel_flag << ", p:" << p;
                                                            #ifdef USE_CUDA
                                    Env<NV>::env_init();
                                    Env<NVHX86>::env_init();
                                    SaberStatus status=test_norm_continue_nchw<NV,NVHX86>(p, sp_flag, scale_flag, channel_flag,w_in,h_in,ch_in,num_in);
                                                            #endif
                                                            #ifdef USE_X86_PLACE
                                    Env<X86>::env_init();
                                    SaberStatus status=test_norm_continue_nchw<X86,X86>(p, sp_flag, scale_flag, channel_flag,w_in,h_in,ch_in,num_in);
                                                            #endif
                                    if(status==SaberUnKownError)
                                    {
                                        LOG(INFO)<<"TEST FAILED!!!"<<"\n";
                                        LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
                                                                    ch_in << ", height=" << h_in << ", width=" << w_in;
                                        
                                        //LOG(INFO) << "total time: " << ts << ", avg time: " << ts / test_iter;
                                        
                                    }
                                    else
                                    {
                                        LOG(INFO)<<"TEST PASSED!!!"<<"\n";
                                        ++pass_count;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        
    }
    LOG(INFO)<<"Tested total "<<total_count<<"\n";
    LOG(INFO)<<"Passed "<<pass_count<<"\n";
    
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

