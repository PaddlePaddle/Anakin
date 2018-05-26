
#include "core/context.h"
#include "funcs/axpy.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
template<typename TensorType>
void test_axpy(std::vector<TensorType*>& inputs, std::vector<TensorType*>& outputs,
               AxpyParam<Tensor<NV, AK_FLOAT, NCHW>>& param, bool get_time, Context<NV>& ctx) {
    /*prepare data*/
    Axpy<NV, AK_FLOAT> axpy;
    axpy.compute_output_shape(inputs, outputs, param);
    outputs[0]->re_alloc(outputs[0]->valid_shape());
    // init assume output tensor has been reshpaed by user.
    axpy.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);
    CUDA_CHECK(cudaPeekAtLastError());

    /*warm up*/
    axpy(inputs, outputs, param, ctx);
    outputs[0]->record_event(ctx.get_compute_stream());
    outputs[0]->sync();

    /*test time*/
    if (get_time) {
        SaberTimer<NV> my_time;
        my_time.start(ctx);

        for (int i = 0; i < 100; i++) {
            axpy(inputs, outputs, param, ctx);
            outputs[0]->record_event(ctx.get_compute_stream());
            outputs[0]->sync();
        }

        my_time.end(ctx);
        //LOG(INFO)<<"aveage time"<<my_time.get_average_ms()/100;
    }

    CUDA_CHECK(cudaPeekAtLastError());
    print_tensor_device(*inputs[0]);
    print_tensor_device(*inputs[1]);
    print_tensor_device(*inputs[2]);
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

    int n = 1;
    int c = 5;
    int h = 8;
    int w = 8;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape valid_shape(n, c, h, w);
    Shape scale_shape(n, c, 1, 1);

    TensorHf4 in_host_scale;
    TensorHf4 in_host_x;
    TensorHf4 in_host_y;
    TensorDf4 in_dev_scale;
    TensorDf4 in_dev_x;
    TensorDf4 in_dev_y;
    TensorDf4 out_dev;
    in_host_scale.re_alloc(scale_shape);
    in_host_x.re_alloc(valid_shape);
    in_host_y.re_alloc(valid_shape);
    in_dev_scale.re_alloc(scale_shape);
    in_dev_x.re_alloc(valid_shape);
    in_dev_y.re_alloc(valid_shape);

    /*prepare input data*/
    auto data = in_host_scale.mutable_data();

    for (int i = 0; i < in_host_scale.size(); ++i) {
        data[i] = std::rand() * 1.0f / RAND_MAX - 0.5;
    }

    in_dev_scale.copy_from(in_host_scale);
    data = in_host_x.mutable_data();

    for (int i = 0; i < in_host_x.size(); ++i) {
        data[i] = i;
    }

    in_dev_x.copy_from(in_host_x);
    data = in_host_y.mutable_data();

    for (int i = 0; i < in_host_y.size(); ++i) {
        data[i] = i;
    }

    in_dev_y.copy_from(in_host_y);

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&in_dev_scale);
    inputs.push_back(&in_dev_x);
    inputs.push_back(&in_dev_y);
    outputs.push_back(&out_dev);

    // start Reshape & doInfer
    Context<NV> ctx(0, 1, 1);
    AxpyParam<TensorDf4> param;

    for (auto get_time : {
                false
            }) {
        test_axpy<TensorDf4>(inputs, outputs,
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

