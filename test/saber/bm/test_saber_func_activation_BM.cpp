#include "core/context.h"
#include "funcs/activation.h"
#include "test_saber_func_BM.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor& t0) {

    LOG(INFO) << name << " valid shape is ["
              << t0.valid_shape()[0] << ", "
              << t0.valid_shape()[1] << ", "
              << t0.valid_shape()[2] << ", "
              << t0.valid_shape()[3] << "].";

    LOG(INFO) << name << " real shape is ["
              << t0.shape()[0] << ", "
              << t0.shape()[1] << ", "
              << t0.shape()[2] << ", "
              << t0.shape()[3] << "].";

    LOG(INFO) << name << " offset is ["
              << t0.offset()[0] << ", "
              << t0.offset()[1] << ", "
              << t0.offset()[2] << ", "
              << t0.offset()[3] << "].";
}

TEST(TestSaberFuncBM, test_func_constructor) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 1;
    int img_h = 8;
    int img_w = 8;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    int sign = -1;
    for (int i = 0; i < img_host.size(); ++i) {
	sign = i % 2 ? -1 : 1;
        img_host.mutable_data()[i] = (float)(0.05 * (i & 0x1f) * sign);
    }

    img_dev.copy_from(img_host);
    TensorDf4 output_dev;
    print_tensor_device(img_dev);

    // start Reshape & doInfer

    Context<BM> ctx1(0, 1, 1);

    ActivationParam<TensorDf4> param(Active_relu);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Activation<BM, AK_BM, AK_BM, AK_BM, NCHW> act;
    act.compute_output_shape(input, output, param);
    output_dev.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    act.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    act(input, output, param, ctx1);

    print_tensor_device(output_dev);
}

int main(int argc, const char** argv) {
    Env<BM>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

