#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/topk_pooling.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
int g_num = 1;
int g_channel = 2;
int g_height = 16;
int g_width = 32;

template <typename dtype>
void get_topk(std::vector<dtype>& src,
        int top_k, int real_k, dtype* dst) {
    for (int k = 0; k < real_k; k++) {
        float max_data = -1e10;
        int max_index = -1;
        for (int i = 0; i < src.size(); i++) {
            if (max_data < src[i]) {
                max_index = i;
                max_data = src[i];
            }
        }
        src[max_index] = -1e10;
        dst[k] = max_data;
    }
    for (int k = real_k; k < top_k; k++) {
       dst[k] = (dtype) 0.f;
    }
}

template <typename dtype,typename TargetType_D,typename TargetType_H>
void topk_pooling_basic(const std::vector<Tensor<TargetType_H>*>& inputs, std::vector<Tensor<TargetType_H>*>& outputs, TopKPoolingParam<TargetType_D>& param){

    CHECK_EQ(inputs.size(), 2) <<"topk pooling need two inputs";
    auto height_offset = inputs[1]->get_seq_offset()[0];
    auto width_offset = inputs[0]->get_seq_offset()[0];

    const dtype* input_data = (const dtype*)inputs[0]->data();
    dtype* output_data = (dtype*) outputs[0]->data();
    int channel = inputs[0]->channel();
    int height_stride = inputs[0]->height();
    int width_stride = inputs[0]->width();
    int num = inputs[0]->num();
    int top_k = param.top_k;
    int feat_map_num = param.feat_map_num;
    CHECK_EQ(feat_map_num, channel) <<"feat map num is not valid";

    Shape output_shape(std::vector<int>{num, channel*top_k, 1, 1});
    outputs[0]->reshape(output_shape);

    for (int i = 0; i < num; i++) {
        int height = height_offset[i + 1] - height_offset[i];
        int width = width_offset[i + 1] - width_offset[i];
        int real_k = top_k < height * width  ? top_k : height * width;
        int feat_map_size = height_stride * width_stride;
        for (int c = 0; c < channel; c++) {
            dtype* tmp_out_data = output_data + (i * channel + c) * top_k;
            const dtype* tmp_in_data = input_data + (i * channel + c) * feat_map_size;
            std::vector<dtype> vec;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    auto value =  tmp_in_data[h * width_stride + w];
                    vec.push_back(value);
                }
            }
            get_topk(vec, top_k, real_k, tmp_out_data);
        }
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
}

template <DataType Dtype,typename TargetType_D,typename TargetType_H>
void test_model(){

    int num = g_num;
    int channel = g_channel;
    int height = g_height;
    int width = g_width;

    TestSaberBase<TargetType_D, TargetType_H, Dtype, TopKPooling, TopKPoolingParam> testbase(2,1);
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    
    Shape input_shape2({2, 3, 32, 24}, Layout_NCHW);
    //test example
    for (auto shape: {input_shape, input_shape2}) {
        for (auto top_k: {1, 3, 5}) {
            int feat_map_num = shape[1];
            LOG(ERROR)<<"topk:"<<top_k<<"feat_map_num:"<<feat_map_num;
            TopKPoolingParam<TargetType_D> param(top_k, feat_map_num);
            testbase.set_param(param);//set param
            testbase.set_input_shape(shape);
            Tensor<TargetType_D> input_0(shape);
            Tensor<TargetType_D> input_1(shape);
            fill_tensor_rand(input_0, -1, 1);
            fill_tensor_rand(input_1, -1, 1);
            std::vector<std::vector<int>> height_seq_offset;
            std::vector<std::vector<int>> width_seq_offset;
            height_seq_offset.resize(1);
            width_seq_offset.resize(1);
            int cumsum_width = 0;
            int cumsum_height = 0;
            height_seq_offset[0].push_back(cumsum_height);
            width_seq_offset[0].push_back(cumsum_width);
            for (int i = 0; i < shape[0]; i++) {
                int cur_width = std::rand() % shape[3] + 1;
                int cur_height = std::rand() % shape[2] + 1;
                cumsum_width += cur_width;
                cumsum_height += cur_height; 
                height_seq_offset[0].push_back(cumsum_height);
                width_seq_offset[0].push_back(cumsum_width);
            }
            
            input_0.set_seq_offset(width_seq_offset);
            input_1.set_seq_offset(height_seq_offset);
            std::vector<Tensor<TargetType_D>*> input_vec;
            input_vec.push_back(&input_0);
            input_vec.push_back(&input_1);
            testbase.add_custom_input (input_vec);
            testbase.run_test(topk_pooling_basic<float, TargetType_D, TargetType_H>);//run test
        }
    }
}
TEST(TestSaberFunc, test_func_activation) {
   
#ifdef USE_CUDA
   //Init the test_base
    LOG(ERROR)<<"testing cuda";
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    LOG(ERROR)<<"testing x86";
    test_model<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_model<AK_FLOAT, ARM, ARM>();
#endif
#ifdef USE_BM
   // Env<BM>::env_init();
    //test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    //if(argc >= 3) {
    //    num_in = atoi(argv[2]);
    //    ch_in = atoi(argv[3]);
    //    h_in = atoi(argv[4]);
    //    w_in = atoi(argv[5]);
    //}
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

