
#include "saber/core/context.h"
#include "saber/funcs/reshape.h"
#include "test_saber_func_reshape_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncReshapeX86, test_func_reshape) {

    Env<X86>::env_init();

    typedef TargetWrapper<X86> X86_API;

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
//    typedef Tensor<X86, AK_FLOAT, HW> TensorHf2;

    typedef TensorHf4::Dtype dtype;

    int w_in = 8;
    int h_in = 8;
    int ch_in = 4;
    int num_in = 2;

    std::vector<int> shape_param_4d = {0, 0, -1, 16};
//    std::vector<int> shape_param_2d = {-1, 64};

//////    ReshapeParam<TensorHf2> param_host_2d(shape_param_2d);
    ReshapeParam<TensorHf4> param_host_4d(shape_param_4d);

    LOG(INFO) << "Reshape param: ";
    LOG(INFO) << " input size, num=" << num_in << ", channel=" << ch_in << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) << "4d reshape params = " << shape_param_4d[0] << ", " << shape_param_4d[1] << ", " << shape_param_4d[2] << \
              ", " << shape_param_4d[3];
//    LOG(INFO) << "2d reshape params = " << shape_param_4d[0] << ", " << shape_param_4d[1];

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out_4d(num_in, ch_in, 4, 16);
//    Shape shape_out_2d(8, 64);

    TensorHf4 thost_in, thost_out_4d;
////    TensorHf2 thost_out_2d;
    
    thost_in.re_alloc(shape_in);

    for (int i = 0; i < thost_in.size(); ++i){
        thost_in.mutable_data()[i] = static_cast<dtype>(i);
    }

    // start Reshape & doInfer
    Context<X86> ctx_host;

    std::vector<TensorHf4*> input_host_4d;
    std::vector<TensorHf4*> output_host_4d;
////    std::vector<TensorHf2*> output_host_2d;

    input_host_4d.push_back(&thost_in);

    Reshape<X86, AK_FLOAT> host_reshape_4d;
////    Reshape<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, HW, NCHW, HW> host_reshape_2d;

    LOG(INFO) << "reshape compute output shape";
    typedef std::vector<Shape> Shape_v;

    output_host_4d.push_back(&thost_out_4d);
//    output_host_2d.push_back(&thost_out_2d);
    host_reshape_4d.compute_output_shape(input_host_4d, output_host_4d, param_host_4d);
////    host_reshape_2d.compute_output_shape(output_host_2d, input_host_4d, param_host_2d);

    LOG(INFO) << "shape out 4d: " << shape_out_4d[0] << ", " << shape_out_4d[1]<< ", " << \
              shape_out_4d[2] << ", " << shape_out_4d[3];
//    LOG(INFO) << "shape out 2d: " << shape_out_2d[0] << ", " << shape_out_2d[1];

    thost_out_4d.re_alloc(shape_out_4d);
//    thost_out_2d.re_alloc(shape_out_2d);


    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "reshape initialization";
    host_reshape_4d.init(input_host_4d, output_host_4d, param_host_4d, SPECIFY, VENDER_IMPL, ctx_host);
////    host_reshape_2d.init(input_host_4d, output_host_2d, param_host_2d, SPECIFY, VENDER_IMPL, ctx_host);

    LOG(INFO) << "reshape compute";
    host_reshape_4d(input_host_4d, output_host_4d, param_host_4d, ctx_host);
////    host_reshape_2d(input_host_4d, output_host_2d, param_host_2d, ctx_host);

    print_tensor_host(*output_host_4d[0]);
//    print_tensor_host(*output_host_2d[0]);

}

int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

