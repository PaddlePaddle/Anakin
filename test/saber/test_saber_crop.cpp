#include "core/context.h"
#include "saber/core/tensor_op.h"
#include "funcs/crop.h"
#include "test_saber_base.h"
#include "saber_types.h"
#include "tensor_op.h"
#include <vector>


using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void norm_cpu_nchw(const std::vector<Tensor<TargetType_H>*>& inputs,std::vector<Tensor<TargetType_H>*>& outputs,CropParam<TargetType_D>& param) {
    int _c_off = 0;
    int _h_off = 0;
    int _w_off = 0;
    int _c_end = 0;
    int _h_end = 0;
    int _w_end = 0;
    CHECK_EQ(param.shape.size(), 4);
    if (param.axis == 1) {
        CHECK_EQ(param.offset.size(), 3);
        _c_off = param.offset[0];
        _h_off = param.offset[1];
        _w_off = param.offset[2];
        _c_end = param.shape[1]+_c_off;
        _h_end = param.shape[2]+_h_off;
        _w_end = param.shape[3]+_w_off;
    } else if (param.axis == 2) {
        CHECK_EQ(param.offset.size(), 2);
        _c_off = 0;
        _h_off = param.offset[0];
        _w_off = param.offset[1];
        _c_end = param.shape[1];
        _h_end = param.shape[2]+_h_off;
        _w_end = param.shape[3]+_w_off;
    } else if (param.axis == 3) {
        CHECK_EQ(param.offset.size(), 1);
        _c_off = 0;
        _h_off = 0;
        _w_off = param.offset[0];
        _c_end = param.shape[1];
        _h_end = param.shape[2];
        _w_end = param.shape[3]+_w_off;
    }
    int num = inputs[0] -> num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    float* ptr_in = (float*)inputs[0]->data();
    float* ptr_out = (float*)outputs[0]->mutable_data();
    for(int i =0; i < num; ++i){
        int offset_n = i * in_c * in_h * in_w;
        for(int j=_c_off; j < _c_end; ++j){
            int offset_c = offset_n + j * in_h * in_w;
            for(int k=_h_off; k < _h_end; ++k){
                int offset_h = offset_c + k * in_w;
                for(int l=_w_off; l < _w_end; ++l){
                   ptr_out[0]=ptr_in[offset_h + l];
                   ptr_out++;
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_crop) {
#ifdef USE_CUDA
    //Init the test_base
    TestSaberBase<NV,NVHX86,AK_FLOAT,Crop,CropParam> testbase_nv;
    LOG(INFO)<<"ENVEND";
#endif
#ifdef USE_X86_PLACE
    //Init the test_base
    TestSaberBase<X86,X86,AK_FLOAT,Crop,CropParam> testbase_x86;
    LOG(INFO)<<"ENVEND";
#endif
    //combine param by yourself
    std::vector<int> offset = {2, 2};
    std::vector<int> shape = {1, 3, 4, 5};
    for(int w_in:{32,64,128,512}){
        for(int h_in: {32,64,128,512}){
            for(int ch_in:{3,8,16,32}){
                for(int num_in:{1,2,32,64}){
                    #ifdef USE_CUDA
//                    //make param
//                    CropParam<NV> param_nv( /*axis_in*/2, /*offset*/offset, /*shape_in*/shape);
//                    //testbase test
//                    testbase_nv.set_param(param_nv);//set param
//                    //testbase.set_rand_limit(255,255);
//                    testbase_nv.set_input_shape(Shape({num_in,ch_in,h_in,w_in}));
//                    testbase_nv.run_test(norm_cpu_nchw<float,NV,NVHX86>);//run test
                    #endif
                    #ifdef USE_X86_PLACE
                    //make param
                    CropParam<X86> param_x86( /*axis_in*/2, /*offset*/offset, /*shape_in*/shape);
                    //testbase test
                    testbase_x86.set_param(param_x86);//set param
                    //testbase.set_rand_limit(255,255);
                    testbase_x86.set_input_shape(Shape({num_in,ch_in,h_in,w_in}));
                    testbase_x86.run_test(norm_cpu_nchw<float,X86,X86>);//run test
                    #endif
                }
            }
        }
    }
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

