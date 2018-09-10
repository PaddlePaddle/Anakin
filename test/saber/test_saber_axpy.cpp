#include "core/context.h"
#include "funcs/axpy.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

int num_in = 2;
int ch_in = 32;
int h_in = 112;
int w_in = 112;
int test_iter = 1;
//void axpy_nv_basic(Tensor<NVHX86>& tensor_in, const float* scale, const float* bias, \
                Tensor<NVHX86>& tensor_out, AxpyParam<NV> param){
template <typename dtype,typename TargetType_D,typename TargetType_H>
void axpy_nv_basic(const std::vector<Tensor<TargetType_H>*>& inputs,std::vector<Tensor<TargetType_H>*>& outputs,AxpyParam<TargetType_D>& param){
    Tensor<TargetType_H>* scale_in = inputs[0];
    Tensor<TargetType_H>* tensor_in = inputs[1];
    Tensor<TargetType_H>* bias_in = inputs[2];
    Tensor<TargetType_H>* tensor_out = outputs[0];
    int num = tensor_in->num();
    int channel = tensor_in->channel();
    int height = tensor_in->height();
    int width = tensor_in->width();

    dtype* dout = (dtype*)tensor_out->mutable_data();
    const dtype* din =(const dtype*)tensor_in->data();
    const dtype* scale =(const dtype*)scale_in->data();
    const dtype* bias =(const dtype*)bias_in->data();
    int in_channel = channel * height * width;
    int size = height * width;
/*
    for (int i = 0; i < num; i++){
        const dtype* din_ptr = din + i * in_channel;
        const dtype* bias_ptr = bias + i * in_channel;
        const dtype* scale_ptr = scale + i * channel;
        dtype* dout_ptr = dout + i * in_channel;
        for(int j = 0; j < channel; j++){
            LOG(INFO) << "scale: ";
            LOG(INFO) << scale_ptr[j];
            const dtype* din_ch_ptr = din_ptr + j * size;
            dtype* dout_ch_ptr = dout_ptr + j * size;
            const dtype* bias_ch_ptr = bias_ptr + j * size;
            LOG(INFO) << "din :";
            for (int k = 0; k < size; k++){
                LOG(INFO) << din_ch_ptr[k];
            }
            LOG(INFO) << "bias :";
            for (int k = 0; k < size; k++){
                 LOG(INFO) << bias_ch_ptr[k];
            }
        }
    }
*/
    for (int i = 0; i < num; i++){
        const dtype* din_ptr = din + i * in_channel;
        const dtype* bias_ptr = bias + i * in_channel;
        const dtype* scale_ptr = scale + i * channel;
        dtype* dout_ptr = dout + i * in_channel;
        for(int j = 0; j < channel; j++){
            const dtype* din_ch_ptr = din_ptr + j * size;
            dtype* dout_ch_ptr = dout_ptr + j * size;
            const dtype* scale_ch_ptr = scale_ptr + j;
            const dtype* bias_ch_ptr = bias_ptr + j * size;
           // LOG(INFO) << "dout :";
            for (int k = 0; k < size; k++){
                dout_ch_ptr[k] = din_ch_ptr[k] * scale_ch_ptr[0] + bias_ch_ptr[k];
               // LOG(INFO) << dout_ch_ptr[k];
            }
        }
    }
}

template <DataType Dtype,typename TargetType_D,typename TargetType_H>
void test_model(){

    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;

    TestSaberBase<TargetType_D, TargetType_H, Dtype, Axpy, AxpyParam> testbase(3,1);
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_shape2({1, 3, 17, 42}, Layout_NCHW);
    AxpyParam<TargetType_D> param;

    for(auto shape: {input_shape, input_shape2}){
        testbase.set_param(param);//set param
        //testbase.set_rand_limit(255,255);
        std::vector<Shape> shape_v;
        //LOG(INFO) << "shape" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3];
        shape_v.push_back(Shape({shape[0],shape[1],1,1}, Layout_NCHW));//scale
        shape_v.push_back(shape);//x
        shape_v.push_back(shape);//y
        testbase.set_input_shape(shape_v);//add some input shape
        testbase.run_test(axpy_nv_basic<float, TargetType_D, TargetType_H>);//run test
    }

}

TEST(TestSaberFunc, test_func_axpy) {
   
#ifdef USE_CUDA
   //Init the test_base
   test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>();
#endif
}


int main(int argc, const char** argv) {

    if(argc >= 2) {
        if (argc < 5) {
            LOG(ERROR) << "usage: ./" << argv[0] << "test_iter " << \
                " compare_result get_time num ch_in h_in w_in" ;
            return 0;
        }
        num_in = atoi(argv[1]);
        ch_in = atoi(argv[2]);
        h_in = atoi(argv[3]);
        w_in = atoi(argv[4]);
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

