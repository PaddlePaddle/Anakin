/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "saber/core/context.h"
#include "saber/funcs/normalize.h"
#include "test/saber/test_saber_func.h"
#include "test/saber/test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/core/tensor_op.h"
#include <vector>

using namespace anakin::saber;
/*CPU function form:
 void FuncName(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,Param<TargetType_D>& param,Shape shape)
 */
template <typename dtype,typename TargetType_D,typename TargetType_H>
void norm_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,NormalizeParam<TargetType_D>& param) {
    int p=param.p;
    bool across_spatial=param.across_spatial;
    bool has_scale=param.has_scale;
    bool channel_shared=param.channel_shared;
    dtype eps=param.eps;
    int n=input[0]->num();
    int c=input[0]->channel();
    int h=input[0]->height();
    int w=input[0]->width();
    Tensor<TargetType_H> th_scale;
    const dtype* scale;
    if(has_scale){
        th_scale.re_alloc(param.scale->shape(),AK_FLOAT);
        th_scale.copy_from(*param.scale);
        scale=static_cast<dtype*>(th_scale.data());
    }
    const dtype* src_ptr = static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());
    
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
void test_normalize(){
    typedef typename DataTrait<TargetType_D, OpDtype> :: Dtype dtype;
    //Init the test_base
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, Normalize, NormalizeParam> testbase;
    
    //combine param by yourself
    bool scale_flag=false;
    int total_count=2 * 2 * 2 * 3 * 3 * 2 * 2;
    int pass_count=0;
    for (bool sp_flag : {false}){
        for (bool channel_flag : {false,true}) {
            for (int p : {1, 2}) {
                for(int w_in: {32, 64}){
                    for(int h_in: {32, 64}){
                        for(int ch_in: {3, 8}){
                            for(int num_in: {1, 2}){
                                //make param
                                NormalizeParam<TargetType_D> param;
                                int ch_scale = channel_flag ? 1 : ch_in;
                                Shape sh_slope({1, 1, 1, ch_scale});
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
                                testbase.set_rand_limit(-255, 255);
                                testbase.set_input_shape(Shape({num_in, ch_in, h_in, w_in}));//add some input shape
                                testbase.run_test(norm_cpu_func<dtype, TargetType_D, TargetType_H>);//run test
                                
                                
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_normalize) {
#ifdef USE_CUDA
    test_normalize<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    test_normalize<X86, X86, AK_FLOAT>();
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_normalize<AMD, AMDHX86, AK_FLOAT>();
#endif
}



int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    
    return 0;
}
