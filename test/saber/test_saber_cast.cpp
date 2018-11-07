#include "core/context.h"
#include "funcs/cast.h"
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
int inType = 1;
int outType = 1;

template <typename dtype, typename TargetType_D, typename TargetType_H>
void cast_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                std::vector<Tensor<TargetType_H>*>& outputs, \
                CastParam<TargetType_D>& param) {
    // int num = tensor_in.num();
    // int channel = tensor_in.channel();
    // int height = tensor_in.height();
    // int width = tensor_in.width();
    Tensor<TargetType_H>* tensor_in = inputs[0];
    Tensor<TargetType_H>* tensor_out = outputs[0];
    int num = tensor_in->num();
    int channel = tensor_in->channel();
    int height = tensor_in->height();
    int width = tensor_in->width();

    int size = channel * height * width;

    //LOG(INFO) << "in_type: "<<param.in_type << ", out_type: "<<param.out_type;
    if (param.in_type == param.out_type) {
        outputs[0]->copy_from(*inputs[0]);
        return;
    }

    if (tensor_in->get_dtype() == 1) { //AK_FLOAT
        const float* in_data = (const float*)tensor_in->data();
        int* out_data = (int*)tensor_out->mutable_data();

        for (int i = 0; i < num; i++) {
            const float* din_ptr = in_data + i * size;
            int* dout_ptr = out_data + i * size;

            for (int j = 0; j < size; j++) {
                dout_ptr[j] = static_cast<int>(din_ptr[j]);
            }
        }

        return;
    }

    if (tensor_in->get_dtype() == 5) { //AK_INT32
        const int* in_data = (const int*)tensor_in->data();
        float* out_data = (float*)tensor_out->mutable_data();

        for (int i = 0; i < num; i++) {
            const int* din_ptr = in_data + i * size;
            float* dout_ptr = out_data + i * size;

            for (int j = 0; j < size; j++) {
                dout_ptr[j] = static_cast<float>(din_ptr[j]);
            }
        }
    }

    return;
}
template <typename TargetType_D, typename TargetType_H>
void test_model() {
    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;

    typedef Tensor<TargetType_H> TensorH;
    typedef Tensor<TargetType_D> TensorD;
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_shape2({1, 32, 17, 32}, Layout_NCHW);

    for (auto shape : {
                input_shape, input_shape2
            }) {
        for (auto a : {
                    1, 5, inType
                }) {
            TensorH input_host;
            TensorD input_dev;

            if (a == 1) {
                float min = -100.f;
                float max = 100.f;
                input_host.re_alloc(shape, AK_FLOAT);
                input_dev.re_alloc(shape, AK_FLOAT);
                fill_tensor_rand(input_host, min, max);
            }

            if (a == 5) {
                float min = -100.f;
                float max = 100.f;
                input_host.re_alloc(shape, AK_INT32);
                input_dev.re_alloc(shape, AK_INT32);
                fill_tensor_rand(input_host, min, max);
            }

            input_dev.copy_from(input_host);
            std::vector<TensorD*> input_dt;
            input_dt.push_back(&input_dev);

            for (auto b : {
                        1, 5, outType
                    }) {
                CastParam<TargetType_D> param(a, b);

                if (b == 1) {
                    TestSaberBase<TargetType_D, TargetType_H, AK_FLOAT, Cast, CastParam> testbase(1, 1);
                    testbase.set_param(param);
                    testbase.add_custom_input(input_dt);
                    testbase.run_test(cast_basic<float, TargetType_D, TargetType_H>, 2.1e-5f);
                }

                if (b == 5) {
                    TestSaberBase<TargetType_D, TargetType_H, AK_INT32, Cast, CastParam> testbase(1, 1);
                    testbase.set_param(param);
                    testbase.add_custom_input(input_dt);
                    testbase.run_test(cast_basic<int, TargetType_D, TargetType_H>, 2.1e-5f);
                }
            }
        }
    }
    /*
     for(auto shape: {input_shape, input_shape2}){
         for(auto a: {1, 5, inType}){
             TensorH input_host;
             TensorD input_dev;
             if(a == 1){
                 float min = -100.f;
                 float max = 100.f;
                 input_host.re_alloc(shape, AK_FLOAT);
                 input_dev.re_alloc(shape, AK_FLOAT);
                 fill_tensor_rand(input_host,min, max);
             }else{
                 int min = -100;
                 int max = 100;
                 input_host.re_alloc(shape, AK_INT32);
                 input_dev.re_alloc(shape, AK_INT32);
                 fill_tensor_rand(input_host, min, max);
             }
             input_dev.copy_from(input_host);
             std::vector<TensorD*> input_dt;
             input_dt.push_back(&input_dev);
             for(auto b: {1, 5, outType}){
                 CastParam<NV> param(a, b);
                 if(b == 1){
                     TestSaberBase<NV, NVHX86, AK_FLOAT, Cast, CastParam> testbase(1,1);
                     testbase.set_param(param);//set param
                     testbase.add_custom_input(input_dt);
                     testbase.run_test(cast_nv_basic<float, NV, NVHX86>);//run test
                 }else{
                     TestSaberBase<NV, NVHX86, AK_INT32, Cast, CastParam> testbase(1,1);
                     testbase.set_param(param);//set param
                     testbase.add_custom_input(input_dt);
                     testbase.run_test(cast_nv_basic<int, NV, NVHX86>);//run test
                 }

             }
         }
     }
     */
}
TEST(TestSaberFunc, test_func_cast) {
    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;

#ifdef USE_CUDA
    //Init the test_base
    test_model<NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    //Env<X86>::env_init();
    test_model<X86, X86>();
#endif
}


int main(int argc, const char** argv) {

    if (argc >= 2) {
        inType = atoi(argv[1]);
    }

    if (argc >= 3) {
        outType = atoi(argv[2]);
    }

    if (argc >= 4) {
        if (argc < 7) {
            LOG(ERROR) << "usage: ./" << argv[0] << "num ch_in h_in w_in" ;
            return 0;
        }

        num_in = atoi(argv[3]);
        ch_in = atoi(argv[4]);
        h_in = atoi(argv[5]);
        w_in = atoi(argv[6]);
    }

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

