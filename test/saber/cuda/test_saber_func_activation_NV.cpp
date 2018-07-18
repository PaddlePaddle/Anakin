#include "core/context.h"
#include "funcs/activation.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
typedef Tensor<X86> TensorH;
typedef Tensor<NV> TensorD;
template <typename ActParam>
void test_activation(Shape input_big_shape, Shape input_shape,
                     ActParam param, Shape offset, bool is_share_from) {
    Context<NV> ctx(0, 1, 1);

    TensorD big_input;
    TensorD small_input;
    TensorD big_output;
    TensorD small_output;

    big_input.re_alloc(input_big_shape, AK_FLOAT);
    big_output.re_alloc(input_big_shape, AK_FLOAT);
    small_input.set_shape(input_shape, input_shape);
    small_output.set_shape(input_shape, input_shape);
    TensorH host_big_input(input_big_shape);
    fill_tensor_rand(host_big_input, -1, 1);
    big_input.copy_from(host_big_input);
    //fill_tensor_device_rand(big_input, -1, 1);

    if (is_share_from) {
        small_input.share_from(big_input);
        small_output.share_from(big_output);
    } else {
        small_input.share_sub_buffer(big_input, input_shape, offset);
        small_output.share_sub_buffer(big_output, input_shape, offset);
    }

    TensorD output_dev;
    // start Reshape & doInfer

    std::vector<TensorD*> inputs;
    std::vector<TensorD*> outputs;

    inputs.push_back(&small_input);
    outputs.push_back(&small_output);

    Activation<NV, AK_FLOAT> act;

    act.compute_output_shape(inputs, outputs, param);
    // init assume output tensor has been reshpaed by user.
    act.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);
    act(inputs, outputs, param, ctx);
    cudaStream_t cuda_stream = ctx.get_compute_stream();
    outputs[0]->record_event(cuda_stream);
    outputs[0]->sync();
    print_tensor(big_output);
    print_tensor(big_input);
    if (param.prelu_param.slope) {
        print_tensor((*param.prelu_param.slope));
    }
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
}

TEST(TestSaberFuncNV, test_func_activation) {
    int num = 1;
    int channel = 2;
    int height = 5;
    int width = 4;

    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_big_shape({num, channel, height+1, width+1}, Layout_NCHW);
    Shape offset_0({0, 0, 0, 0}, Layout_NCHW);
    Shape offset_1({0, 0, 1, 1}, Layout_NCHW);
    Shape slope_shape_0({1, channel, 1, 1}, Layout_NCHW);
    Shape slope_shape_1({1, 1, 1, 1}, Layout_NCHW);
    TensorD prelu_slope_0;
    prelu_slope_0.reshape(slope_shape_0);
    PreluParam<NV> prelu_0(false, &prelu_slope_0);

    TensorD prelu_slope_1;
    prelu_slope_1.reshape(slope_shape_1);
    PreluParam<NV> prelu_1(true, &prelu_slope_1);
    fill_tensor_rand(prelu_slope_0, 0, 1);
    fill_tensor_rand(prelu_slope_1, 0, 1);

    ActivationParam<NV> param_elu(Active_elu, 0.1f, 0.1f);
    ActivationParam<NV> param_relu(Active_relu, 0.0f, 0.0f);
    ActivationParam<NV> param_sigmoid(Active_sigmoid, 0.1f, 0.1f);
    ActivationParam<NV> param_tanh(Active_tanh, 0.1f, 0.1f);
    ActivationParam<NV> param_prelu_0(Active_prelu, 0.f, 0.f, prelu_0);
    ActivationParam<NV> param_prelu_1(Active_prelu, 0.f, 0.f, prelu_1);

    for (ActivationParam<NV> param : {param_elu, param_relu, param_sigmoid, param_tanh, param_prelu_0, param_prelu_1}) {
        //for (ActivationParam<TensorD> param : {param_sigmoid}) {
        for (auto share_from : {false, true}) {
            for (auto offset: {offset_0, offset_1}) {
                test_activation(input_big_shape,
                                input_shape, param, offset, share_from);
            }
        }
    }
}

int main(int argc, const char** argv) {
    Env<NV>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

