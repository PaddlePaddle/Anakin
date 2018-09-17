#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/activation.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
int active = 1;
int num_in = 1;
int ch_in = 2;
int h_in = 3;
int w_in = 5;
template <typename dtype,typename TargetType_D,typename TargetType_H>
void activation_basic(const std::vector<Tensor<TargetType_H>*>& inputs, std::vector<Tensor<TargetType_H>*>& outputs, ActivationParam<TargetType_D>& param){

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();

    dtype* dout = (dtype*)outputs[0]->mutable_data();
    const dtype* din = (const dtype*)inputs[0]->data();
    size_t count = inputs[0]->valid_size();
    int size = height * width;
    
    switch (param.active){
         //x > 0 ? x : 0
        case Active_relu:
            for (size_t i = 0; i < count; i++){
                dout[i] = din[i] > 0 ? din[i] : 0;
            }
            break;
        // sigmoid: 1/(exp(-x) + 1)
        case Active_sigmoid:

           for (size_t i = 0; i < count; i++){
                dout[i] = 1.0f / (exp(-din[i]) + 1.0f);
            }
            break;
        // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        case Active_tanh:
            for (size_t i = 0; i < count; i++){
                dout[i] =  tanh(din[i]);//(exp(din[i]) - exp(-din[i])) / (exp(din[i]) + exp(-din[i]));
            }
            break;
        
        // stanh : b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}
        case Active_stanh:
            for (size_t i = 0; i < count; i++){
                dtype val = din[i] * param.negative_slope;
                dout[i] =  param.coef * tanh(val);
            }
            break;

        // x > 0 ? x : 0;
        // x < threshold ? x : threshold
        case Active_clipped_relu:
             for (size_t i = 0; i < count; i++){
                 const dtype threshold = param.coef;
                dout[i] = din[i] > 0 ? (din[i] < threshold ? din[i] : threshold) : 0;
             }
            break;

        //elu:  x > 0 ? x : coef * (exp(x) - 1)
        case Active_elu:
            for (size_t i = 0; i < count; i++){
                dout[i] =  din[i] > 0 ? din[i] : param.coef * (exp(din[i]) - 1);
            }
            break;


        //prelu: x > 0 ? x : slope[c] * x
        case Active_prelu:
            auto prelu_param  = param.prelu_param;
            for (int n = 0; n < num; n++){
                const dtype *in_ptr = din + n * channel * size;
                dtype *out_ptr = dout + n * channel * size;
              //  const dtype *slope_ptr = nullptr;
                Tensor<TargetType_D>* slop_dev;
                slop_dev = prelu_param.slope;
                Shape shape = slop_dev->valid_shape();
                Tensor<TargetType_H>* slop_host;//(shape);
               // LOG(INFO) << "slop_dev: " << shape[0] << ", " << shape[2];  
                //slop_host->set_shape(shape);
                slop_host = new Tensor<TargetType_H>(shape);
                //LOG(INFO) << "slop_dev: " << slop_dev->valid_size();
                slop_host->copy_from(*slop_dev);
                //LOG(INFO) << "slop_host: " << slop_host->valid_size();
                const dtype *slope_ptr = (const dtype*)slop_host->data();
              // const dtype *slope_ptr = (const dtype*)prelu_param.slope->data();
                for (int c = 0; c < channel; c++){
                    const dtype *in_ch_ptr = in_ptr + c * size;
                    dtype *out_ch_ptr = out_ptr + c * size;
                    dtype slope = prelu_param.channel_shared ?  slope_ptr[0] : slope_ptr[c];
                    for (int k = 0; k < size; k++){
                        out_ch_ptr[k] = in_ch_ptr[k] > 0 ? in_ch_ptr[k] : in_ch_ptr[k] * slope;
                    }
                }
                delete slop_host;
            }
            break; 
    }
}

template <DataType Dtype,typename TargetType_D,typename TargetType_H>
void test_model(){

    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;

    TestSaberBase<TargetType_D, TargetType_H, Dtype, Activation, ActivationParam> testbase(1,1);
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    
    Shape input_shape2({2, 2, 12, 22}, Layout_NCHW);
    //test example
    for(auto shape: {input_shape, input_shape2}){
        for(auto act: {1, 2, 3, 4, 5, 9, 10, active}){
            LOG(INFO) << "================ active: " << act;
            for(auto neg_slope: {-1.0, 0.5}){
                for(auto coef: {1.0, 0.5}){
                    for(auto has: {true, false}){
                        if(act == 10){
                            for(auto shared: {true, false}){
                                Shape slope_shape({1, shape[1], 1, 1}, Layout_NCHW);
                                Tensor<TargetType_D> slope_tensor;
                                slope_tensor.re_alloc(slope_shape, Dtype);
                                fill_tensor_rand(slope_tensor, -1.0, 1.0);
                                PreluParam<TargetType_D> prelu(shared, &slope_tensor);
                                ActivationParam<TargetType_D> param(act, neg_slope, coef, prelu, has);
                                testbase.set_param(param);//set param
                                testbase.set_input_shape(shape);
                                testbase.run_test(activation_basic<float, TargetType_D, TargetType_H>);//run test
                               // LOG(INFO) << "NV run end";
                            }
                            
                        }else{
                            PreluParam<TargetType_D> prelu(false, nullptr);
                            if(act == 2) neg_slope = 0.f;//relu
                            ActivationParam<TargetType_D> param(act, neg_slope, coef, prelu, has);
                            //LOG(INFO) << "neg_slope: " << neg_slope << ", coef: " << coef << ", has: " << has;
                            testbase.set_param(param);//set param
                            testbase.set_input_shape(shape);
                            testbase.run_test(activation_basic<float, TargetType_D, TargetType_H>);//run test
                           // LOG(INFO) << "NV run end";
                        }
                    }
                }
            }
        }
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
    test_model<AK_FLOAT, ARM, ARM>();
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_model<AK_FLOAT, AMD, AMDHX86>();
#endif
#ifdef USE_BM_PLACE
//    Env<BM>::env_init();
//    test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    if (argc >= 2) {
        active = atoi(argv[1]);
    }
    if(argc >= 3) {
        if (argc < 6) {
            LOG(ERROR) << "usage: ./" << argv[0] << "axis " << \
                " num ch_in h_in w_in" ;
            return 0;
        }
        num_in = atoi(argv[2]);
        ch_in = atoi(argv[3]);
        h_in = atoi(argv[4]);
        w_in = atoi(argv[5]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

