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
#include "saber/funcs/conv_pooling.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "conv_func_helper.h"
#include "saber/funcs/funcs_utils.h"

#include <vector>

using namespace anakin::saber;
#ifdef AMD_GPU
#define BASIC_TEST true
#else
#define BASIC_TEST false
#endif
template<typename dtype,typename TargetType_D,typename TargetType_H>
void pooling_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
        std::vector<Tensor<TargetType_H>*>& output,
        PoolingParam<TargetType_D>& param)
{
    const dtype* src_ptr=static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr=static_cast<dtype*>(output[0]->mutable_data());

    int in_n=input[0]->num();
    int in_c=input[0]->channel();
    int in_h=input[0]->height();
    int in_w=input[0]->width();
    int size_in_n=in_c*in_h*in_w;
    int size_in_c=in_h*in_w;

    int out_h=output[0]->height();
    int out_w=output[0]->width();
    int size_out_n=in_c*out_h*out_w;
    int size_out_c=out_h*out_w;

    for(int ind_n=0;ind_n<in_n;++ind_n){
        for(int ind_c=0;ind_c<in_c;++ind_c){
            for(int ind_h=0;ind_h<out_h;++ind_h){
                int sh=ind_h*param.stride_h;
                int eh=sh+param.window_h;

                sh=(sh-param.pad_h)<0?0:sh-param.pad_h;
                eh=(eh-param.pad_h)>in_h?in_h:eh-param.pad_h;

                for(int ind_w=0;ind_w<out_w;++ind_w){
                    int sw=ind_w*param.stride_w;
                    int ew=sw+param.window_w;
                    sw=(sw-param.pad_w)<0?0:sw-param.pad_w;
                    ew=(ew-param.pad_w)>in_w?in_w:ew-param.pad_w;

                    dtype result;

                    int dst_ind=ind_n*size_out_n+ind_c*size_out_c+ind_h*out_w+ind_w;
                    for(int kh=sh;kh<eh;++kh){
                        for(int kw=sw;kw<ew;++kw){
                            int src_ind=ind_n*size_in_n+ind_c*size_in_c+kh*in_w+kw;
                            if (kh == sh && kw == sw){
                                result = src_ptr[src_ind];
                            } else {
                                if (param.pooling_type==Pooling_max){
                                    result=result>=src_ptr[src_ind]?result:src_ptr[src_ind];
                                }
                                if (param.pooling_type==Pooling_average_include_padding){
                                    result+=src_ptr[src_ind];
                                }
                                if (param.pooling_type==Pooling_average_exclude_padding){
                                    result+=src_ptr[src_ind];
                                }
                            }

                        }
                    }
                    if(param.pooling_type==Pooling_average_include_padding){
                        result/=param.window_h*param.window_w;
                    }
                    if(param.pooling_type==Pooling_average_exclude_padding){
                        result/=(ew-sw)*(eh-sh);
                    }
                    dst_ptr[dst_ind]=result;

                }
            }
        }

    }
}

template<typename dtype,typename TargetType_D,typename TargetType_H>
void conv_pool_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
        std::vector<Tensor<TargetType_H>*>& output,
        ConvPoolingParam<TargetType_D>& param) {

    int group = param.conv_param.group;
    int input_num = input[0]->num();
    int input_channel = input[0]->channel();
    int input_height = input[0]->height();
    int input_width = input[0]->width();
    int output_channel = output[0]->channel();
    int output_height = output[0]->height();
    int output_width = output[0]->width();
    int stride_h = param.conv_param.stride_h;
    int stride_w = param.conv_param.stride_w;
    int dilation_h = param.conv_param.dilation_h;
    int dilation_w = param.conv_param.dilation_w;
    int pad_h = param.conv_param.pad_h;
    int pad_w = param.conv_param.pad_w;
    int kernel_h = param.conv_param.weight()->height();
    int kernel_w = param.conv_param.weight()->width();
    bool bias_term = param.conv_param.bias()->valid_size() > 0;
    bool with_relu = param.conv_param.activation_param.has_active;

    Tensor<TargetType_H> weights_host;
    Tensor<TargetType_H> bias_host;
    weights_host.re_alloc(param.conv_param.weight()->valid_shape(), AK_FLOAT);
    weights_host.copy_from(*(param.conv_param.weight()));
    bias_host.re_alloc(param.conv_param.bias()->valid_shape(), AK_FLOAT);
    bias_host.copy_from(*(param.conv_param.bias()));
    const dtype* bias_ptr = bias_term ? (const float*)bias_host.data() : nullptr;
    Shape conv_shape = conv_compute_shape(input[0]->valid_shape(), param.conv_param);
    Tensor<TargetType_H> conv_output;
    conv_output.re_alloc(conv_shape, AK_FLOAT);
    std::vector<Tensor<TargetType_H>* > conv_output_v;
    conv_output_v.push_back(&conv_output);
    conv_basic_check<TargetType_H>(*input[0], conv_output,
            (const dtype*)weights_host.data(), bias_ptr,
            group, kernel_w, kernel_h, stride_w, stride_h,
            dilation_w, dilation_h, pad_w, pad_h, bias_term,
            with_relu);
    pooling_cpu_func<float, TargetType_D, TargetType_H>(conv_output_v, output, param.pooling_param);
}


template <typename TargetType, typename TargetType_H>
void test_conv_pool() {
    Env<TargetType>::env_init();
    Env<TargetType_H>::env_init();
    TestSaberBase<TargetType, TargetType_H, AK_FLOAT, ConvPooling, ConvPoolingParam> testbase;

    std::vector<int> kernel_h_v{3};
    std::vector<int> kernel_w_v{3};
    std::vector<int> pad_h_v{1};
    std::vector<int> pad_w_v{1};
    std::vector<int> stride_h_v{1};
    std::vector<int> stride_w_v{1};
    std::vector<int> dilation_h_v{1};
    std::vector<int> dilation_w_v{1};
    std::vector<int> group_v{1};
    std::vector<int> in_h_v{12, 16};
    std::vector<int> in_w_v{16, 18};
    std::vector<int> input_num_v{1, 3};
    std::vector<int> input_channels_v{17, 4};
    std::vector<int> output_channels_v{4, 17};
    std::vector<bool> bias_term_v{true, false};
    if (BASIC_TEST) {
    for (auto kernel_h : kernel_h_v)
    for (auto kernel_w : kernel_w_v)
    for (auto pad_h : pad_h_v)
    for (auto pad_w : pad_w_v)
    for (auto stride_h : stride_h_v)
    for (auto stride_w : stride_w_v)
    for (auto dilation_h : dilation_h_v)
    for (auto dilation_w : dilation_w_v)
    for (auto bias_term : bias_term_v)
    for (auto in_channels : input_channels_v)
    for (auto out_channels : output_channels_v)
    for (auto group : group_v) {

        Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
        Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
        Tensor<TargetType> weights_dev;
        Tensor<TargetType> bias_dev;

        weights_dev.re_alloc(weights_s, AK_FLOAT);
        fill_tensor_rand(weights_dev, -5.f, 5.0f);
        if (bias_term) {
            bias_dev.re_alloc(bias_s, AK_FLOAT);
            fill_tensor_rand(bias_dev, -5.0f, 5.0f);
        }
        ConvParam<TargetType> conv_param(group, pad_h, pad_w,
                                         stride_h, stride_w,
                                         dilation_h, dilation_w,
                                         &weights_dev, &bias_dev);
        PoolingParam<TargetType> pool_param(2, 2, 0, 0, 2, 2, Pooling_max);
        ConvPoolingParam<TargetType> param(conv_param, pool_param);

        for (auto input_num : input_num_v)
        for (auto height : in_h_v)
        for (auto width : in_w_v) {
            // open random fill output!
            testbase.set_random_output(true);
            testbase.set_param(param);//set param
            testbase.set_input_shape(Shape({input_num,in_channels,height,width},
                                           Layout_NCHW));//add some input shape
            testbase.run_test(conv_pool_cpu_func<float, TargetType, TargetType_H>, 1e-3);//run test
        }
    }
    }
}

TEST(TestSaberFunc, test_saber_conv_results) {

    #ifdef USE_X86_PLACE
    test_conv_pool<X86, X86>();
    #endif
    #ifdef AMD_GPU
    test_conv_pool<AMD, AMDHX86>();
    #endif
}

template<typename TargetType, typename TargetType_H>
int test_conv_pool_results(int group,
        int input_num, int in_channels, int height, int width,
        int out_channels, int kernel_h, int kernel_w,
        int stride_h, int stride_w, int dilation_h, int dilation_w,
        int pad_h, int pad_w, bool bias_term, bool with_relu,
        SaberImplStrategy strategy, ImplEnum imp) {

    LOG(INFO)<< " conv param: "
         << " input_num = " << input_num
         << " in_channels = " << in_channels
         << " height = " << height
         << " width = " << width
         << " group = " << group
         << " pad_h = " << pad_h
         << " pad_w = " << pad_w
         << " stride_h = " << stride_h
         << " stride_w = " << stride_w
         << " dilation_h = " << dilation_h
         << " dilation_w = " << dilation_w
         << " kernel_h = " << kernel_h
         << " kernel_w = " << kernel_w
         << " out_channels = " << out_channels
         << " bias_term = " << (bias_term ? "true" : "false")
         << " with_relu = " << (with_relu ? "true" : "false");

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);
    fill_tensor_rand(input_dev, -10.0f, 10.0f);
    input_host.copy_from(input_dev);
//    input_dev.set_scale({10.1f / 128});
//    LOG(INFO) << input_dev.get_scale()[0];

    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);
    fill_tensor_rand(weights_dev, -10.0f, 10.0f);
    weights_host.copy_from(weights_dev);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;
    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
        fill_tensor_rand(bias_dev, -10.0f, 10.0f);
        bias_host.copy_from(bias_dev);
    }
    Tensor<TargetType> output_dev;
    Tensor<TargetType_H> output_host;
    Tensor<TargetType_H> check_host;

    Context<TargetType> ctx1(0, 1, 1);

    ConvParam<TargetType> param(group, pad_h, pad_w,
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                &weights_dev, &bias_dev);

    PoolingParam<TargetType> pool_param(2, 2, 0, 0, 2, 2, Pooling_max);

    ConvPoolingParam<TargetType> conv_pooling_param(param, pool_param);
    ConvPooling<TargetType, AK_FLOAT> conv_pool;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    std::vector<Tensor<TargetType_H>* > input_h_v;
    std::vector<Tensor<TargetType_H>* > output_h_v;

    input_h_v.push_back(&input_host);
    output_h_v.push_back(&check_host);

    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
    conv_pool.compute_output_shape(input_v, output_v, conv_pooling_param);
    output_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    check_host.re_alloc(output_dev.valid_shape(), AK_FLOAT);

    conv_pool_cpu_func<float, TargetType, TargetType_H>(
            input_h_v, output_h_v, conv_pooling_param);
    conv_pool.init(input_v, output_v, conv_pooling_param, strategy, imp, ctx1);
    conv_pool.trans_weights(*param.mutable_weight(), *param.mutable_bias(),
                       param.pad_h, param.pad_w, param.dilation_h, param.dilation_w,
                       param.stride_h, param.stride_w, param.group, imp);

    conv_pool(input_v, output_v, conv_pooling_param, ctx1);

    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    output_host.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    output_host.copy_from(output_dev);


    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
                    check_host.valid_size(), max_ratio, max_diff);

    if (max_ratio > 1e-3) {
            print_tensor_valid(output_host);
            print_tensor_valid(check_host);
        LOG(FATAL) << " max_ratio = " << max_ratio << " max_diff = " << max_diff;
    }
    return 0;
}
TEST(TestSaberFunc, test_saber_cuda_conv) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();

    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0,1};
    std::vector<int> pad_w_v{0,1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1, 2};
    std::vector<int> dilation_w_v{1, 2};
    std::vector<int> in_channels_v{ 4, 8};
    std::vector<int> out_channels_v{4, 8};
//    std::vector<int> group_v{1, 2, 32};
    std::vector<int> in_h_v{24, 36};
    std::vector<int> in_w_v{24, 36};
    std::vector<int> input_num_v{1, 3};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{true, false};
    if (BASIC_TEST) {
    for (auto input_num : input_num_v) {
    for (auto out_channels : out_channels_v) {
    for (auto in_channels : in_channels_v) {
    for (auto kernel_h : kernel_h_v) {
    for (auto kernel_w : kernel_w_v) {
    for (auto height : in_h_v) {
    for (auto width : in_w_v) {
    for (auto stride_h : stride_h_v) {
    for (auto stride_w : stride_w_v) {
    for (auto dilation_h : dilation_h_v) {
    for (auto dilation_w : dilation_w_v) {
    for (auto pad_h : pad_h_v) {
    for (auto pad_w : pad_w_v) {
    for (auto bias_term : bias_term_v) {
    for (auto with_relu : with_relu_v) {
        test_conv_pool_results<NV, NVHX86>(1,
                                           input_num,
                                           in_channels,
                                           height,
                                           width,
                                           out_channels,
                                           kernel_h,
                                           kernel_w,
                                           stride_h, stride_w,
                                           dilation_h,
                                           dilation_w,
                                           pad_h, pad_w,
                                           bias_term,
                                           with_relu,
                                           SPECIFY,
                                           VENDER_IMPL);
        test_conv_pool_results<NV, NVHX86>(1,
                                           input_num,
                                           in_channels,
                                           height,
                                           width,
                                           out_channels,
                                           kernel_h,
                                           kernel_w,
                                           stride_h, stride_w,
                                           dilation_h,
                                           dilation_w,
                                           pad_h, pad_w,
                                           bias_term,
                                           with_relu,
                                           SPECIFY,
                                           SABER_IMPL);
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
#endif
}


int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
