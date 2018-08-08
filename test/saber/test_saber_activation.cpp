#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/activation.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include <vector>
#include<cmath>

using namespace anakin::saber;
int axis_in = 1;
int num_in = 1;
int ch_in = 32;
int h_in = 112;
int w_in = 112;
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
/*
    for (int i = 0; i < inputs.size(); i++){
        const dtype* din = (dtype*)inputs[i]->data();
        LOG(INFO) << "i: " << i;
        for(int j = 0; j < inputs[i]->count_valid(0, 4); j++){
            LOG(INFO) << "j: "<< j << ", data: " << din[j];
        }
    }
*/
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
                dout[i] =  tanh(din[i];//((exp(din[i]) - exp(-din[i])) / (exp(din[i]) + exp(-din[i]));
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
                dout[i] = din[i] > 0 ? (din[i] < threshold ? din[i] : threshold : 0;
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
                const dtpye *in_ptr = din + n * channel * size;
                dtpye *out_ptr = dout + n * channel * size;
                for (int c = 0; c < channel; c++){
                    const dtpye *in_ch_ptr = in_ptr + c * size;
                    dtpye *out_ch_ptr = out_ptr + c * size;
                    dtpye *slope_ptr = (dtpye*)prelu_param.slope->data();
                    dtpye slope = prelu_param.channel_shared ?  slope_ptr[0]: slope_ptr[c];
                    for (int k = 0; k < size; k++){
                        out_ch_ptr[k] = in_ch_ptr[k] > 0 ? in_ch_ptr[k] : in_ch_ptr[k] * slope;
                    }
                }
            }
            break; 
    }
}

TEST(TestSaberFunc, test_func_activation) {
    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;
    int axis1 = axis_in;
   
#ifdef USE_CUDA
   //Init the test_base
    TestSaberBase<NV, NVHX86, AK_FLOAT, Activation, ActivationParam> testbase(1,1);
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_shape2({2, 2, 12, 22}, Layout_NCHW);

    for(auto shape: {input_shape, input_shape2}){
        for(auto act: {1, 2, 3, 4, 5, 9, 10, active}){
            for(auto neg_slope: {-1, 0.5, 1}){
                for(auto coef: {-1, 0.5, 1}){
                    for(auto has: {true, false}){
                        if(act == 10){
                            for(auto shared: {true, false}){
                                Shape slope_shape({1, shape[1], 1, 1}, Layout_NCHW);
                                Tensor<NV> slope_tensor;
                                slope_tensor.re_alloc(slope_shape, AK_FLOAT);
                                fill_tensor_rand(slope_tensot, -1.0, 1.0);
                                PreluParam<NV> prelu(shares, &slope_tensor);
                                ActivationParam<NV> param(act, neg_slope, coef, prelu, has_active);
                                testbase.set_param(param);//set param
                                testbase.set_input_shape(shape);
                                testbase.run_test(cactivation_basic<float, NV, NVHX86>);//run test
                                LOG(INFO) << "NV run end";
                            }
                            
                        }else{
                            PreluParam<NV> prelu(false, nullptr);
                            ActivationParam<NV> param(act, neg_slope, coef, prelu, has_active);
                            testbase.set_param(param);//set param
                            testbase.set_input_shape(shape);
                            testbase.run_test(cactivation_basic<float, NV, NVHX86>);//run test
                            LOG(INFO) << "NV run end";
                        }
                    }
                }
            }
        }
    }

#endif
#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, Concat, ConcatParam> testbase(2,1);
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_shape2({2, 2, 12, 22}, Layout_NCHW);

    for(auto shape: {input_shape, input_shape2}){
        for(auto axis: {0,1,2,3, axis1}){
            ConcatParam<X86> param(axis);
            testbase.set_param(param);//set param
            //testbase.set_rand_limit(255,255);
            std::vector<Shape> shape_v;
            shape_v.push_back(shape);
            Shape shin = shape;
            shin[axis] = 2;
            shape_v.push_back(shin);
            Shape shin2 = shape;
            shin2[axis] = 4;
            shape_v.push_back(shin2);
            testbase.set_input_shape(shape_v);//add some input shape
            testbase.run_test(concat_nv_basic<float, X86, X86>);//run test
            LOG(INFO) << "X86 run end";
        }
    }
#endif

#ifdef USE_ARM_PLACE
   //Init the test_base
    TestSaberBase<ARM, ARM, AK_FLOAT, Concat, ConcatParam> testbase(2,1);
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_shape2({2, 2, 12, 22}, Layout_NCHW);

    for(auto shape: {input_shape, input_shape2}){
        for(auto axis: {0,1,2,3, axis1}){
            ConcatParam<ARM> param(axis);
            testbase.set_param(param);//set param
            //testbase.set_rand_limit(255,255);
            std::vector<Shape> shape_v;
            shape_v.push_back(shape);
            Shape shin = shape;
            shin[axis] = 2;
            shape_v.push_back(shin);
            Shape shin2 = shape;
            shin2[axis] = 4;
            shape_v.push_back(shin2);
            testbase.set_input_shape(shape_v);//add some input shape
            testbase.run_test(concat_nv_basic<float, ARM, ARM>);//run test
            LOG(INFO) << "ARM run end";
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

