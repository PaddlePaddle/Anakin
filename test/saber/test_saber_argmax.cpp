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

#include "core/context.h"
#include "funcs/argmax.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

bool out_max_val = false;
bool has_axis = false;
int top_k = 3;
int axis = 1;
int num_in = 1;
int ch_in = 32;
int h_in = 112;
int w_in = 112;
int test_iter = 10;
bool compare_result = true;
bool get_time = false;

//void argmax_nv_basic(Tensor<NVHX86>& tensor_in, Tensor<NVHX86>& tensor_out, ArgmaxParam<NV> param){
template <typename dtype, typename TargetType_D, typename TargetType_H>
void argmax_nv_basic(const std::vector<Tensor<TargetType_H>*>& tensor_in,std::vector<Tensor<TargetType_H>*>& tensor_out,ArgmaxParam<TargetType_D>& param){
    int num = tensor_in[0]->num();
    int channel = tensor_in[0]->channel();
    int height = tensor_in[0]->height();
    int width = tensor_in[0]->width();

    int ch_out = tensor_out[0]->channel();
    int w_out = tensor_out[0]->width();
    int h_out = tensor_out[0]->height();

    int top = param.top_k;
    bool has_ax = param.has_axis;
    int ax = param.axis;
    bool out_max = param.out_max_val;

    //LOG(INFO) << "basic compute";
    //LOG(INFO) << "has_axis: "<<   has_ax << ", ax:" << ax << ", out_max_val: "<<out_max;
  //  const float* din = (const float*)tensor_in[0]->data();
  //  float* dout = (float*)tensor_out[0]->mutable_data();
    const dtype* din = (const dtype*)tensor_in[0]->data();
    dtype* dout = (dtype*)tensor_out[0]->mutable_data();
    int in_channel = channel * height * width;
    int out_channel = ch_out * w_out * h_out;

    if (has_ax){//nchw
        auto shape = tensor_in[0]->valid_shape();
        int stride = shape.count(ax+1, shape.dims());
        int out_stride = shape.count(1, ax);
        int out_ss = tensor_out[0]->valid_shape().count(ax, shape.dims());
        int in_ss = shape.count(ax, shape.dims());
       // LOG(INFO) << "stride: "<<stride << ", out_stride: " << out_stride;
        int size = shape[ax];
        if(size < top){
            LOG(INFO) << "input data size less than topk";
            return; 
        }
        for (int n = 0; n < num * out_stride; n++){
            for(int k = 0; k < stride; k ++){
                const dtype* din_ch = din + n * in_ss + k;
                std::vector< std::pair<dtype, int> > vec;
                vec.resize(size);
                for (int i = 0; i < size; i++){
                    vec[i] = std::make_pair(din_ch[i*stride], i);
                }
                 //sort
                std::partial_sort(vec.begin(), vec.begin() + top, vec.end(), std::greater< std::pair<float, int> >());
                //out
                dtype* dout_ch = dout + n * out_ss + k;
                for(int i = 0; i < top ;i ++){
                    if(out_max)
                        dout_ch[i*stride] = vec[i].first;
                    else
                        dout_ch[i*stride] = vec[i].second;
                }
            }
        }
    }else{//all  
        if(in_channel < top){
            LOG(INFO) << "input data size less than topk";
            return; 
        }
        for (int n = 0; n < num; n++){
            const dtype* din_ch = din + n * in_channel;
            std::vector< std::pair<dtype, int> > vec;
            vec.resize(in_channel);
            for (int i = 0; i < in_channel; i++){
                vec[i] = std::make_pair(din_ch[i], i);
            }
            //sort
            std::partial_sort(vec.begin(), vec.begin() + top, vec.end(), std::greater< std::pair<float, int> >());
            //out
            if(out_max){
                dtype* dout_ch = dout + n * out_channel;
                dtype* dout_index = dout_ch;
                dtype* dout_data = dout_ch + top;
                for (int i = 0; i < top; i++){
                    dout_data[i] = vec[i].first;
                    dout_index[i] = vec[i].second;
                    //LOG(INFO) << "max_data: " <<dout_data[i] << ", max_index: "<<dout_index[i];
                }
            }else{
                dtype* dout_data = dout + n * out_channel;
                for (int i = 0; i < top; i++){
                    dout_data[i] = vec[i].second;
                   // LOG(INFO) << "max_data: " <<vec[i].first << ", max_index: "<< dout_data[i];
                }
            }
            vec.clear();
        }
    }
}
template <DataType Dtype,typename TargetType_D,typename TargetType_H>
void test_model(){
    
    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;
    bool out_max = out_max_val;
    int topk = top_k;
    bool has = has_axis;
    int ax = axis;
    
    TestSaberBase<TargetType_D, TargetType_H, Dtype, Argmax, ArgmaxParam> testbase;  
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_shape2({1, 32, 17, 32}, Layout_NCHW);
    
   // typename NV TargetD;
    ArgmaxParam<TargetType_D> argmax_param(out_max, topk, has, ax);//has axis
    ArgmaxParam<TargetType_D> argmax1(false, 3, true, 1);//has axis
    ArgmaxParam<TargetType_D> argmax2(false, 3, false, 1);//has axis
    ArgmaxParam<TargetType_D> argmax3(true, 3, true, 2);//has axis
    ArgmaxParam<TargetType_D> argmax4(true, 3, false, 1);//has axis

   // test_argmax<TargetD, TargetH, OpType>(input_shape, argmax_param);
    for(auto shape: {input_shape, input_shape2}){
        for (auto param : {argmax1, argmax2, argmax3, argmax4, argmax_param}) {
           // test_argmax<TargetD, TargetH, OpType>(shape, param);
	    testbase.set_param(param);//set param
            testbase.set_input_shape(shape);//add some input shape
            testbase.run_test(argmax_nv_basic<float, TargetType_D, TargetType_H>);//run test
                               
        }
    }


}

TEST(TestSaberFunc, test_func_argmax) {
   // LOG(INFO) << "topk: " << topk << ", has_axis: " << has << ", axis: " << ax << ", out_max_val: " << out_max;
#ifdef USE_CUDA
    //Init the test_base
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    //Env<X86>::env_init();
    test_model<AK_FLOAT, X86, X86>();
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_model<AK_FLOAT, AMD, AMDHX86>();
#endif
}

int main(int argc, const char** argv) {

    if (argc >= 2) {
        top_k = atoi(argv[1]);
    }
    if (argc >= 3) {
        has_axis = atoi(argv[2]) > 0;
    }
    if (argc >= 4) {
        axis = atoi(argv[3]);
    }
    if (argc >= 5) {
        out_max_val = atoi(argv[4]) > 0;
    }
    if(argc >= 6) {
        if (argc < 9) {
            LOG(ERROR) << "usage: ./" << argv[0] << \
                " top_k has_axis axis out_max_val num ch_in h_in w_in";
            return 0;
        }
        num_in = atoi(argv[5]);
        ch_in = atoi(argv[6]);
        h_in = atoi(argv[7]);
        w_in = atoi(argv[8]);
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

