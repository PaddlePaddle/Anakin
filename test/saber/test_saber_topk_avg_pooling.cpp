#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/topk_avg_pooling.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
int g_num = 1;
int g_channel = 2;
int g_height = 5;
int g_width = 5;

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

template <typename OpDataType,typename TargetType_D,typename TargetType_H>
void topk_avg_pooling_basic(const std::vector<Tensor<TargetType_H>*>& inputs, std::vector<Tensor<TargetType_H>*>& outputs, TopKAvgPoolingParam<TargetType_D>& param){

    CHECK_EQ(inputs.size(), 3) <<"topk pooling need three inputs";
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    auto height_offset = inputs[1]->get_seq_offset()[0];
    auto width_offset = inputs[2]->get_seq_offset()[0];

    const OpDataType* input_data = (const OpDataType*)inputs[0]->data();
    OpDataType* output_data = (OpDataType*) outputs[0]->data();

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height_stride = inputs[0]->height();
    int width_stride = inputs[0]->width();

    int feat_map_num = param.feat_map_num;
    CHECK_EQ(feat_map_num, channel) <<"feat map num is not valid";
    int dim0 = 0;
    if (param.is_pooling_by_row) {
        dim0 = inputs[1]->num();
        CHECK_EQ(dim0, height_offset[height_offset.size() - 1]);
        outputs[0]->set_seq_offset(inputs[1]->get_seq_offset());
    } else {
        dim0 = inputs[2]->num();
        CHECK_EQ(dim0, width_offset[width_offset.size() - 1]);
        outputs[0]->set_seq_offset(inputs[2]->get_seq_offset());
    }  
    int num_k = param.top_ks.size();
    int max_k = param.top_ks[num_k - 1];
    auto offset = outputs[0]->get_seq_offset()[0];
    Shape output_shape(std::vector<int>{offset[offset.size() - 1], channel*num_k, 1, 1});
    outputs[0]->reshape(output_shape);
    
    
    for (int i = 0; i < num; i++) {
        int height = height_offset[i + 1] - height_offset[i];
        int width = width_offset[i + 1] - width_offset[i];
        int feat_map_size = height_stride * width_stride;
        std::vector<OpDataType> vec;
        std::vector<OpDataType> topk_value;
        std::vector<OpDataType> sum;
        topk_value.resize(max_k);
        sum.resize(max_k);
        if (param.is_pooling_by_row) {
            int real_k = max_k < width  ? max_k : width;
            for (int h = 0; h < height; h++) {
                for (int c = 0; c < channel; c++) {
                    auto tmp_in_data = input_data + ((i *channel + c) * height_stride + h) * width_stride;
                    auto tmp_out_data = output_data + ((height_offset[i] + h) * channel  + c) * num_k;
                    vec.clear();
                    for (int w = 0; w < width; w++) {
                        vec.push_back(tmp_in_data[w]);
                    }
                    get_topk(vec, max_k, real_k, &topk_value[0]);
                    sum[0] = topk_value[0];
                    for (int m = 1; m < max_k; m++) {
                        sum[m] = sum[m-1] + topk_value[m];
                    }
                    for (int m = 0; m < param.top_ks.size(); m++) {
                        tmp_out_data[m] = sum[param.top_ks[m] - 1] / param.top_ks[m];
                    }
                }
            }
        } else {
            int real_k = max_k < height  ? max_k : height;
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channel; c++) {
                    auto tmp_in_data = input_data + ((i *channel + c) * height_stride) * width_stride + w;
                    auto tmp_out_data = output_data + ((width_offset[i] + w ) * channel +  c) * num_k;
                    vec.clear();
                    for (int h = 0; h < height; h++) {
                        vec.push_back(tmp_in_data[h * width_stride]);
                    }
                    get_topk(vec, max_k, real_k,  &topk_value[0]);
                    sum[0] = topk_value[0];
                    for (int m = 1; m < max_k; m++) {
                        sum[m] = sum[m-1] + topk_value[m];
                    }
                    for (int m = 0; m < param.top_ks.size(); m++) {
                        tmp_out_data[m] = sum[param.top_ks[m] - 1] / param.top_ks[m];
                    }
                }
            }
        }
    }

}


template <DataType Dtype,typename TargetType_D,typename TargetType_H>
void test_model(){

    int num = g_num;
    int channel = g_channel;
    int height = g_height;
    int width = g_width;

    TestSaberBase<TargetType_D, TargetType_H, Dtype, TopKAvgPooling, TopKAvgPoolingParam> testbase(3,1);
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    
    Shape input_shape2({2, 3, 7, 8}, Layout_NCHW);
    //test example
    for (auto shape: {input_shape, input_shape2}) {
        int feat_map_num = shape[1];
        std::vector<int> top_ks = {1, 2, 3, 4, 5};
        TopKAvgPoolingParam<TargetType_D> param(top_ks, feat_map_num, true);
        testbase.set_param(param);//set param
        testbase.set_input_shape(shape);
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
            //int cur_height = std::rand() % shape[2] + 1;
            int cur_height = shape[2];
            cumsum_width += cur_width;
            cumsum_height += cur_height; 
            height_seq_offset[0].push_back(cumsum_height);
            width_seq_offset[0].push_back(cumsum_width);
        }
        Tensor<TargetType_D> input_0(shape);
        Shape shape_1({cumsum_height, 10, 1, 1});
        Shape shape_2({cumsum_width, 10, 1, 1});
        Tensor<TargetType_D> input_1(shape_1);
        Tensor<TargetType_D> input_2(shape_2);
        fill_tensor_rand(input_0, -1, 1);
        fill_tensor_rand(input_1, -1, 1);
        fill_tensor_rand(input_2, -1, 1);
        
        input_0.set_seq_offset(width_seq_offset);
        input_1.set_seq_offset(height_seq_offset);
        input_2.set_seq_offset(width_seq_offset);
        std::vector<Tensor<TargetType_D>*> input_vec;
        input_vec.push_back(&input_0);
        input_vec.push_back(&input_1);
        input_vec.push_back(&input_2);
        testbase.add_custom_input (input_vec);
        testbase.run_test(topk_avg_pooling_basic<float, TargetType_D, TargetType_H>);//run test
    }
}
TEST(TestSaberFunc, test_func_activation) {
   
#ifdef USE_CUDA
   //Init the test_base
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
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
    //    g_num = atoi(argv[2]);
    //    g_channel = atoi(argv[3]);
    //    g_height = atoi(argv[4]);
    //    g_width = atoi(argv[5]);
    //}
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

