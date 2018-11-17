
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
    LOG(INFO) << "reshape_cpu_func start";
    LOG(INFO) << "reshape_cpu_func: param.layout: " << param.layout;
    Shape in_shape = input[0] -> valid_shape();
    std::vector<int> param_shape = param.shape_params;
    Shape out_shape;
    out_shape.resize(param_shape.size());
    out_shape.set_layout(param.layout);
    int infer_axis = -1;
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
    }
    CHECK_LE(infer_count, 1) << "infer axises cannot exceed 1";
    if (infer_axis >= 0){
        out_shape[infer_axis] = input[0] -> valid_size() / num_axis;
    }
    output[0] -> set_shape(out_shape);
    LOG(INFO) << "reshape_cpu_func: out_shape.layout: " << out_shape.get_layout();
}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_reshape(){
    
    typedef typename DataTrait<TargetType_D, OpDtype>:: Dtype dtype;
    //Init the test_base
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, Reshape, ReshapeParam> testbase;
    auto param_check = [](std::vector<int> new_shape, std::vector<int> in_shape) -> bool {
        CHECK_EQ(new_shape.size(), in_shape.size()) << "invalid check";
        int new_count=1;
        int in_count=1;
        for(int i=0; i<new_shape.size(); ++i){
            if (new_shape[i] > 0){
                in_count *= in_shape[i];
                if (new_shape[i] !=-1){
                    new_count *= new_shape[i];
                }
            }
        }
        return new_count <= in_count;
    };


    //test shape contain -1 and 0
    for (int rs0 : {0, -1, 2}){
        for (int rs1 : {0, -1, 4}){
            for (int rs2 : {0, -1, 8}){
                for (int rs3 : {0, -1, 16}){
                    std::vector<int> new_shape{rs0, rs1, rs2, rs3};
                    if (std::count(new_shape.begin(), new_shape.end(), -1) == 1){
                        ReshapeParam<TargetType_D> param(new_shape, Layout_NCHW);
                        LOG(INFO) << "new_shape:"<<rs0<<" "<<rs1<<" "<<rs2<<" "<<rs3;
                        for (int n : {1, 2}){
                            for (int c : {1, 4}){
                                for (int h: {32, 64}){
                                    for (int w : {32, 64}){
                                        Shape in_shape({n, c, h, w});
                                        if (param_check(new_shape, in_shape)){
                                            testbase.set_param(param);
                                            testbase.set_input_shape(in_shape);
                                            testbase.run_test(reshape_cpu_func<dtype, TargetType_D, TargetType_H>);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }//for rs0

    //test shape normal
    std::vector<Shape> new_shapes;
    std::vector<Shape> in_shapes;
    new_shapes.emplace_back(Shape({1, 1, 3, 64}, Layout_NCHW));
    in_shapes.emplace_back(Shape({1, 3, 4, 16}, Layout_NCHW));
    new_shapes.emplace_back(Shape({1, 4, 3, 64}, Layout_NCHW));
    in_shapes.emplace_back(Shape({1, 1, 1, 3*64*4}, Layout_NCHW));
    new_shapes.emplace_back(Shape({2, 2, 3, 64}, Layout_NCHW));
    in_shapes.emplace_back(Shape({1, 2, 1, 2*3*64}, Layout_NCHW));
    new_shapes.emplace_back(Shape({32, 3*32, 64}, Layout_NHW));
    in_shapes.emplace_back(Shape({1, 3, 64, 32*32}, Layout_NCHW));
    for (int i=0; i<new_shapes.size(); ++i){
        ReshapeParam<TargetType_D> param(new_shapes[i], new_shapes[i].get_layout());
        testbase.set_param(param);
        testbase.set_input_shape(in_shapes[i]);
        testbase.run_test(reshape_cpu_func<dtype, TargetType_D, TargetType_H>);
    }
}
 
TEST(TestSaberFunc, test_func_reshape) {
#ifdef USE_CUDA
    test_reshape<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    test_reshape<X86, X86, AK_FLOAT>();
#endif
        
}


    
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

