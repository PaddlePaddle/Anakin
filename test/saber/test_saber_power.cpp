#include <vector>
#include <cmath>

#include "saber/core/context.h"
#include "saber/funcs/power.h"
#include "test/saber/test_saber_func.h"
#include "test/saber/test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"

using namespace anakin::saber;
/*CPU function form:
 void FuncName(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,Param<TargetType_D>& param,Shape shape)
 */
template <typename dtype,typename TargetType_D,typename TargetType_H>
void power_cpu_func(const std::vector<Tensor<TargetType_H>*>& input, std::vector<Tensor<TargetType_H>*>& output, PowerParam<TargetType_D>& param) {
    dtype p = param.power;
    dtype scale = param.scale;
    dtype shift = param.shift;
    
    const dtype* src_ptr = input[0] -> data();
    dtype* dst_ptr = output[0] -> mutable_data();
    
    for (int i=0; i < input[0] -> valid_size(); ++i){
        dst_ptr[i] = pow(src_ptr[i]* scale +shift, p);
    }
}

TEST(TestSaberFunc, test_func_normalize) {
#ifdef USE_CUDA
    //Init the test_base
    TestSaberBase<NV, NVHX86, AK_FLOAT, Power, PowerParam> testbase;
    for (float p : {0, 1, 2}){
        for (float scale : {0.5, 1.0, 2.0}){
            for (float shift : {0, 1, 2}){
                PowerParam<NV> param(p, scale, shift);
                
                for (int n : {1, 2}){
                    for (int c : {1, 3}){
                        for (int h: {32, 64}){
                            for (int w : {32, 64}){
                                testbase.set_param(param);
                                testbase.set_input_shape(Shape({n, c, h, w}));
                                testbase.run_test(power_cpu_func<float, NV, NVHX86>);
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
