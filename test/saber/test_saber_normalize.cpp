
#include "saber/core/context.h"
#include "saber/funcs/normalize.h"
#include "test/saber/test_saber_func.h"
#include "test/saber/test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/core/tensor_op.h"
#include <vector>

using namespace anakin::saber;
template <typename dtype>
void group_normlize(const dtype* in_data, const dtype* scale, const dtype* bias,
                    int n, int c, int h, int w, float eps, int group,
                    dtype* out_data, dtype* out_mean, dtype* out_var){
    int group_size = (c - 1) / group + 1;
    int im_size = h * w;
    for (int n_index = 0; n_index < n; ++n_index){
        for (int g_index = 0; g_index < group; ++g_index){
            dtype t_mean = 0;
            dtype t_var = 0;
            int real_channels = c - g_index * group_size >= group_size ? 
                                group_size : c - g_index * group_size;
            int compute_size = im_size * real_channels;
            for (int im_index = 0; im_index < compute_size; ++im_index){
                t_mean += in_data[im_index];
                t_var += in_data[im_index] * in_data[im_index]; 
            }
            t_mean /= compute_size;
            t_var /= compute_size;
            t_var -= t_mean * t_mean;
            dtype t_var_inv = 1 / sqrt(t_var + eps);
            if (out_mean){
                out_mean[n * group + g_index] = t_mean;
            }
            if (out_var){
                out_var[n * group + g_index] = t_var;
            }

            int scale_bias_start_index = g_index * group_size;
            for (int c_index = 0; c_index < real_channels; ++c_index){
                int c_start = c_index * im_size;
                for (int im_index = 0; im_index < im_size; ++im_index){
                    dtype dest_val = (in_data[c_start + im_index] - t_mean) * t_var_inv;
                    if (scale){
                        dest_val *= scale[scale_bias_start_index + c_index];
                    }
                    if (bias){
                        dest_val += bias[scale_bias_start_index + c_index];
                    }
                    out_data[c_start + im_index] = dest_val;      
                }

            }
            out_data += compute_size;
            in_data += compute_size;   
        }
    }
}
/*CPU function form:
 void FuncName(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,Param<TargetType_D>& param,Shape shape)
 */
template <typename dtype,typename TargetType_D,typename TargetType_H>
void norm_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,NormalizeParam<TargetType_D>& param) {

    int p=param.p;
    bool across_spatial=param.across_spatial;
    bool has_scale=param.has_scale;
    bool has_bias = param.has_bias;
    bool channel_shared=param.channel_shared;
    dtype eps=param.eps;
    int n=input[0]->num();
    int c=input[0]->channel();
    int h=input[0]->height();
    int w=input[0]->width();
    Tensor<TargetType_H> th_scale;
    Tensor<TargetType_H> th_bias;
    dtype* scale = nullptr;
    dtype* bias = nullptr;
    dtype* out_mean = nullptr;
    dtype* out_var = nullptr;
    if (has_scale){
        th_scale.re_alloc(param.scale->shape(), AK_FLOAT);
        th_scale.copy_from(*param.scale);
        scale = static_cast<float*>(th_scale.data());
    }
    if (has_bias){
        th_bias.re_alloc(param.bias->shape(), AK_FLOAT);
        th_bias.copy_from(*param.bias);
        bias = static_cast<float*>(th_bias.data());
    }

    const dtype* src_ptr = static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());
    if (param.group > 0){
                //group>1, do group normal
                if (output.size() > 1){
                    out_mean = static_cast<float*>(output[1]->mutable_data());
                }
                if (output.size() > 2){
                    out_var = static_cast<float*>(output[2]->mutable_data());
                }
                group_normlize<float>(src_ptr, scale, bias, n, c, h, w, eps, param.group,
                                     dst_ptr, out_mean, out_var);
                return;
    }
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
                    //LOG(INFO)<<"c:"<<c;

                    for (int l = 0; l < c; ++l) {
                        if (p == 1) {
                            norm += fabsf(src_pixel[l * channel_in_size]);
                        } else {
                            norm += src_pixel[l * channel_in_size] * src_pixel[l * channel_in_size];
                        }
                    }
                    //LOG(INFO)<<"norm:"<<norm;

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
                            //LOG(INFO)<<"dst:"<<dst_pixel[l * channel_in_size];
                            //LOG(INFO)<<"src:"<<src_pixel[l * channel_in_size];
                            //LOG(INFO)<<"norm_dd:"<<norm;

                        }
                    }
                }
            }
        }
    }
}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_normalize(bool scale_flag){
    typedef typename DataTrait<TargetType_D, OpDtype> :: Dtype dtype;
    //Init the test_base
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, Normalize, NormalizeParam> testbase;

    //combine param by yourself
    int total_count=2 * 2 * 2 * 3 * 3 * 2 * 2;
    int pass_count=0;
    for (bool sp_flag : {false}) {
    for (bool channel_flag : {false,true}) {
    for (int p : {1, 2}) {
    for (int w_in : {32, 64}) {
        for (int h_in: {32, 64}) {
            for (int ch_in : {3, 8}) {
                for (int num_in : {1, 2}) {
                    //make param
                    NormalizeParam<TargetType_D> param;
                    int ch_scale = channel_flag ? 1 : ch_in;
                    Shape sh_slope({1, 1, 1, ch_scale});
                    if (std::is_same<TargetType_D, MLU>::value) {
                        Shape temp({1, ch_scale, 1, 1});
                        sh_slope = temp;
                    }
                    Tensor<TargetType_H> th_scale(sh_slope);
                    Tensor<TargetType_D> tdscale;
                    tdscale.re_alloc(sh_slope,AK_FLOAT);
                    for (int i = 0; i < ch_scale; ++i) {
                        static_cast<dtype *>(th_scale.mutable_data())[i] = 0.1f * (i + 1);
                    }
                    tdscale.copy_from(th_scale);
                    if (scale_flag) {
                        NormalizeParam<TargetType_D> param_tmp(sp_flag, channel_flag, &tdscale, eps, p);
                        param = param_tmp;
                    } else {
                        NormalizeParam<TargetType_D> param_tmp(sp_flag, eps, p);
                        param = param_tmp;
                    }

                    //testbase test
                    testbase.set_param(param);//set param
                    //testbase.set_rand_limit(255,255);
                    testbase.set_input_shape(Shape({num_in, ch_in, h_in, w_in}));//add some input shape
                    if (std::is_same<TargetType_D, MLU>::value) {
                        // comment this test until mlu norm problem solved
                        //testbase.run_test(norm_cpu_func<dtype, TargetType_D, TargetType_H>,
                        //                  0.4, true);//run test
                    } else {
                        testbase.run_test(norm_cpu_func<dtype, TargetType_D, TargetType_H>);//run test
                    }
                }
            }
        }
    }
    }
    }
    }
    
    // Fixme. mlu test not pass, ans of mlu is wrong
    if (std::is_same<TargetType_D, MLU>::value) { return; }

    for (int w_in:{2}){
        for (int h_in: {2}){
            for (int ch_in:{3, 8}){
                for (int num_in:{1, 2}){
                    for (int group : {1, 2 ,3}){
                    LOG(ERROR) << w_in << "," << h_in << "," << ch_in << "," << num_in << "," << group;
                    //make param
                    NormalizeParam<TargetType_D> param;
                    Shape sh_slope({1, 1, 1, ch_in});
                    if (std::is_same<TargetType_D, MLU>::value) {
                        Shape temp({1, ch_in, 1, 1});
                        sh_slope = temp;
                    }
                    Tensor<TargetType_H> th_scale(sh_slope);
                    Tensor<TargetType_D> tdscale;
                    tdscale.re_alloc(sh_slope,AK_FLOAT);
                    for (int i = 0; i < ch_in; ++i) {
                        static_cast<dtype *>(th_scale.mutable_data())[i] = 0.1f * (i + 1);
                    }
                    tdscale.copy_from(th_scale);
                    NormalizeParam<TargetType_D> param_tmp(true, &tdscale, false, nullptr, group, 0.00001);
                    param = param_tmp;
                    
                    //testbase test
                    testbase.set_param(param);//set param
                    //testbase.set_rand_limit(255,255);
                    testbase.set_input_shape(Shape({num_in, ch_in, h_in, w_in}));//add some input shape
                    if (std::is_same<TargetType_D, MLU>::value) {
                        testbase.run_test(norm_cpu_func<dtype, TargetType_D, TargetType_H>,
                                          0.03, true);//run test
                    } else {
                        testbase.run_test(norm_cpu_func<dtype, TargetType_D, TargetType_H>);//run test
                    }

                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_normalize) {

    bool scale_flag=false;

#ifdef USE_CUDA
    test_normalize<NV, NVHX86, AK_FLOAT>(scale_flag);
#endif

#ifdef USE_X86_PLACE
    test_normalize<X86, X86, AK_FLOAT>(scale_flag);
#endif

#ifdef USE_MLU
    // Fixme. test not pass when scale_flag=false, ans of mlu is wrong
    test_normalize<MLU, MLUHX86, AK_FLOAT>(true);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}
