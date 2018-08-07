
#include "saber/funcs/reshape.h"

#include <vector>
#include <algorithm>

#include "saber/core/context.h"
#include "saber/saber_types.h"
#include "saber/core/tensor_op.h"
#include "test/saber/test_saber_func.h"
#include "test/saber/test_saber_base.h"
#include "saber/core/tensor_op.h"

using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void reshape_cpu_func(const std::vector<Tensor<TargetType_H>*>& input, std::vector<Tensor<TargetType_H>*>& output, ReshapeParam<TargetType_D>& param) {
    Shape in_shape = input[0] -> valid_shape();
    Shape param_shape = param.shape_params;
    Shape out_shape;
    out_shape.resize(param_shape.size());
    int infer_axis = 0;
    int num_axis = 1;
    int infer_count=0;
    for (int i=0; i < param_shape.size(); ++i){
        CHECK_LT(i, in_shape.size()) << "param index exceed input dims";
        if ( param_shape[i] == 0){
            out_shape[i] = in_shape[i];
            num_axis *= out_shape[i];
        } else if (param_shape[i] == -1){
            infer_axis = i;
            ++infer_count;
        } else {
            out_shape[i] = param_shape[i];
            num_axis *= out_shape[i];
        }
        CHECK_EQ(infer_count, 1) << "infer axises cannot exceed 1";
        if (infer_axis >= 0){
            out_shape[infer_axis] = input[0] -> valid_size() / num_axis;
        }
    }
    //output[0] -> copy_from(*input[0]);
    output[0] -> set_shape(out_shape);
}
 
TEST(TestSaberFunc, test_func_reshape) {
#ifdef USE_CUDA
    //Init the test_base
    TestSaberBase<NV, NVHX86, AK_FLOAT, Reshape, ReshapeParam> testbase;
    
    for (int rs0 : {-1, 2}){
        for (int rs1 : {-1, 4}){
            for (int rs2 : {-1, 8}){
                for (int rs3 : {-1, 16}){
                    std::vector<int> new_shape{rs0, rs1, rs2, rs3};
                    if (std::count(new_shape.begin(), new_shape.end(), -1) == 1){
                        ReshapeParam<NV> param(new_shape);
                        for (int n : {1, 2}){
                            for (int c : {1, 4}){
                                for (int h: {32, 64}){
                                    for (int w : {32, 64}){
                                        testbase.set_param(param);
                                        testbase.set_input_shape(Shape({n, c, h, w}));
                                        testbase.run_test(reshape_cpu_func<float, NV, NVHX86>);
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

