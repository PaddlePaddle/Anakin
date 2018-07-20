#include "saber/core/context.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/activation.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include <vector>

using namespace anakin::saber;

template <typename targetType, typename targetType_H>
void test_activation(Shape input_big_shape, Shape input_shape,
                     ActivationParam<targetType> param, Shape offset, bool is_share_from) {
    typedef Tensor<targetType_H> TensorH;
    typedef Tensor<targetType> TensorD;
    Context<targetType> ctx(0, 1, 1);

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

    Activation<targetType, AK_FLOAT> act;

    act.compute_output_shape(inputs, outputs, param);
    // init assume output tensor has been reshpaed by user.
    act.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);
    act(inputs, outputs, param, ctx);
    typename TensorD::API::stream_t stream = ctx.get_compute_stream();
    outputs[0]->record_event(stream);
    outputs[0]->sync();
    print_tensor(big_output);
    print_tensor(big_input);
    if (param.prelu_param.slope) {
        print_tensor((*param.prelu_param.slope));
    }
#ifdef USE_CUDA
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
#endif
}

template <typename targetType, typename targetType_H>
void test_accuracy(int num, int channel, int height, int width) {

    typedef Tensor<targetType_H> TensorH;
    typedef Tensor<targetType> TensorD;

    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_big_shape({num, channel, height+1, width+1}, Layout_NCHW);
    Shape offset_0({0, 0, 0, 0}, Layout_NCHW);
    Shape offset_1({0, 0, 1, 1}, Layout_NCHW);
    Shape slope_shape_0({1, channel, 1, 1}, Layout_NCHW);
    Shape slope_shape_1({1, 1, 1, 1}, Layout_NCHW);
    TensorD prelu_slope_0;
    prelu_slope_0.reshape(slope_shape_0);
    PreluParam<targetType> prelu_0(false, &prelu_slope_0);

    TensorD prelu_slope_1;
    prelu_slope_1.reshape(slope_shape_1);
    PreluParam<targetType> prelu_1(true, &prelu_slope_1);
    fill_tensor_rand(prelu_slope_0, 0, 1);
    fill_tensor_rand(prelu_slope_1, 0, 1);

    ActivationParam<targetType> param_elu(Active_elu, 0.1f, 0.1f);
    ActivationParam<targetType> param_relu(Active_relu, 0.0f, 0.0f);
    ActivationParam<targetType> param_sigmoid(Active_sigmoid, 0.1f, 0.1f);
    ActivationParam<targetType> param_tanh(Active_tanh, 0.1f, 0.1f);
    ActivationParam<targetType> param_prelu_0(Active_prelu, 0.f, 0.f, prelu_0);
    ActivationParam<targetType> param_prelu_1(Active_prelu, 0.f, 0.f, prelu_1);

    for (ActivationParam<targetType> param : {param_elu, param_relu, param_sigmoid, param_tanh, param_prelu_0, param_prelu_1}) {
        //for (ActivationParam<TensorD> param : {param_sigmoid}) {
        for (auto share_from : {false, true}) {
            for (auto offset: {offset_0, offset_1}) {
                test_activation<targetType, targetType_H>(input_big_shape,
                                input_shape, param, offset, share_from);
            }
        }
    }
}

TEST(TestSaberFunc, test_func_activation) {
    int num = 1;
    int channel = 2;
    int height = 5;
    int width = 4;
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    test_accuracy<NV, NVHX86>(num, channel, height, width);
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    test_accuracy<X86, X86>(num, channel, height, width);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

