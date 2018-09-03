
#include "core/context.h"
#include "funcs/im2sequence.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
template<typename TensorType>
void test_im2sequence(std::vector<TensorType*>& inputs, std::vector<TensorType*>& outputs,
                      Im2SequenceParam<Tensor<NV, AK_FLOAT, NCHW>>& param, bool get_time, Context<NV>& ctx) {
    /*prepare data*/
    Im2Sequence<NV, AK_FLOAT> im2sequence;
    im2sequence.compute_output_shape(inputs, outputs, param);
    outputs[0]->re_alloc(outputs[0]->valid_shape());
    // init assume output tensor has been reshpaed by user.
    im2sequence.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);
    CUDA_CHECK(cudaPeekAtLastError());

    /*warm up*/
    im2sequence(inputs, outputs, param, ctx);
    outputs[0]->record_event(ctx.get_compute_stream());
    outputs[0]->sync();
    CUDA_CHECK(cudaPeekAtLastError());

    /*test time*/
    if (get_time) {
        SaberTimer<NV> my_time;
        my_time.start(ctx);

        for (int i = 0; i < 100; i++) {
            im2sequence(inputs, outputs, param, ctx);
            outputs[0]->record_event(ctx.get_compute_stream());
            outputs[0]->sync();
        }

        my_time.end(ctx);
        LOG(INFO) << "aveage time" << my_time.get_average_ms() / 100;
    }

    CUDA_CHECK(cudaPeekAtLastError());
    print_tensor_device(*inputs[0]);
    cudaDeviceSynchronize();
    print_tensor_device(*outputs[0]);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_func_constructor) {
    Env<NV>::env_init();
    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int n = 10;
    int c = 64;
    int h = 32;
    int w = 32;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape valid_shape(n, c, h, w);

    TensorHf4 in_host;
    TensorDf4 in_dev;
    TensorDf4 out_dev;
    in_host.re_alloc(valid_shape);
    in_dev.re_alloc(valid_shape);

    /*prepare input data*/
    auto data = in_host.mutable_data();

    for (int i = 0; i < in_host.size(); ++i) {
        data[i] = std::rand() * 1.0f / RAND_MAX - 0.5;
    }

    in_dev.copy_from(in_host);

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&in_dev);
    outputs.push_back(&out_dev);

    // start Reshape & doInfer
    Context<NV> ctx(0, 1, 1);
    Im2SequenceParam<TensorDf4> param(3, 3, 0, 0, 0, 0, 2, 2, 1, 1);

    for (auto get_time : {
                true
            }) {
        test_im2sequence<TensorDf4>(inputs, outputs,
                                    param, get_time, ctx);
    }
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

