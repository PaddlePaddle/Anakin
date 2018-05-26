
#include "core/context.h"
#include "funcs/mvn.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
template<typename TensorType>
void test_mvn(std::vector<TensorType*>& inputs_big, std::vector<TensorType*>& outputs_big,
              std::vector<TensorType*>& inputs, std::vector<TensorType*>& outputs,
              Shape input_offset, Shape output_offset,
              bool input_share, bool output_share, bool input_share_sub, bool output_share_sub,
              MvnParam<Tensor<NV, AK_FLOAT, NCHW>>& param, bool get_time, Context<NV>& ctx) {
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    /*prepare data*/
    Mvn<NV, AK_FLOAT> mvn;
    mvn.compute_output_shape(inputs, outputs, param);
    outputs[0]->set_shape(outputs[0]->valid_shape(), outputs[0]->valid_shape());
    inputs[0]->set_shape(inputs[0]->valid_shape(), inputs[0]->valid_shape());

    if (output_share && output_share_sub) {
        outputs[0]->share_sub_buffer(*outputs_big[0], outputs[0]->valid_shape(), output_offset);
    } else if (output_share) {
        outputs[0]->share_from(*outputs_big[0]);
    } else {
        outputs[0]->re_alloc(outputs[0]->valid_shape());
    }

    if (input_share && input_share_sub) {
        inputs[0]->share_sub_buffer(*inputs_big[0], inputs[0]->valid_shape(), input_offset);
    } else if (input_share) {
        inputs[0]->share_from(*inputs_big[0]);
    } else {
        inputs[0]->re_alloc(inputs[0]->valid_shape());
        cudaMemcpy(inputs[0]->mutable_data(), inputs_big[0]->data(),
                   sizeof(float) * (inputs[0]->valid_size()), cudaMemcpyDeviceToDevice);
    }

    // init assume output tensor has been reshpaed by user.
    mvn.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);

    /*warm up*/
    mvn(inputs, outputs, param, ctx);
    outputs[0]->record_event(ctx.get_compute_stream());
    outputs[0]->sync();

    /*test time*/
    if (get_time) {
        SaberTimer<NV> my_time;
        my_time.start(ctx);

        for (int i = 0; i < 100; i++) {
            mvn(inputs, outputs, param, ctx);
            outputs[0]->record_event(ctx.get_compute_stream());
            outputs[0]->sync();
        }

        my_time.end(ctx);
        LOG(INFO) << "aveage time" << my_time.get_average_ms() / 100;
    }

    CUDA_CHECK(cudaPeekAtLastError());
    Shape valid_shape = outputs[0]->valid_shape();
    printf("shape: %d, %d, %d, %d\n", valid_shape[0], valid_shape[1],
           valid_shape[2], valid_shape[3]);
    TensorDf4 out_valid(outputs[0]->valid_shape());
    TensorHf4 out_valid_host(outputs[0]->valid_shape());
    CUDA_CHECK(cudaPeekAtLastError());
    out_valid.copy_from(*outputs[0]);
    out_valid_host.copy_from(out_valid);
    CUDA_CHECK(cudaPeekAtLastError());
    printf("%f, %f\n", out_valid_host.data()[0], out_valid_host.data()[1]);

    //print_tensor_device(*inputs[0]);
    //cudaDeviceSynchronize();
    //print_tensor_device(*outputs[0]);
    //cudaDeviceSynchronize();
    //print_tensor_device(out_valid);
    //cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_func_mvn) {
    Env<NV>::env_init();
    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int n = 1;
    int c = 3;
    int h = 320;
    int w = 1080;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape real_shape(n + 1, c + 1, h + 1, w + 1);
    Shape valid_shape(n, c, h, w);
    Shape input_offset(0, 0, 0, 0);
    Shape output_offset(0, 0, 0, 0);

    TensorHf4 in_host_big;
    TensorDf4 in_dev_big;
    TensorHf4 out_host_big;
    TensorDf4 out_dev_big;

    TensorDf4 in_dev;
    TensorDf4 output_dev;

    in_host_big.re_alloc(real_shape);
    in_dev_big.re_alloc(real_shape);
    out_host_big.re_alloc(real_shape);
    out_dev_big.re_alloc(real_shape);

    /*prepare input data*/
    auto data = in_host_big.mutable_data();

    for (int i = 0; i < in_host_big.size(); ++i) {
        data[i] = 0x7f & i;
    }

    in_dev_big.copy_from(in_host_big);

    in_dev.set_shape(valid_shape);

    std::vector<TensorDf4*> inputs_big;
    std::vector<TensorDf4*> outputs_big;
    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs_big.push_back(&in_dev_big);
    outputs_big.push_back(&out_dev_big);
    inputs.push_back(&in_dev);
    outputs.push_back(&output_dev);

    // start Reshape & doInfer
    Context<NV> ctx(0, 1, 1);
    //normalize_variance_in,  across_channels_in,  eps_in
    MvnParam<TensorDf4> param1(false, false, 1e-6);
    MvnParam<TensorDf4> param2(true, false, 1e-6);

    for (auto param : {
                param2
            }) {
        for (auto input_share : {
                    false
                }) {
            for (auto output_share : {
                        false
                    }) {
                for (auto input_share_sub : {
                            false
                        }) {
                    for (auto output_share_sub : {
                                false
                            }) {
                        for (auto get_time : {
                                    true
                                }) {
                            LOG(INFO) << input_share << "," << output_share << "," << input_share_sub << "," << output_share_sub
                                      << "," << get_time;
                            test_mvn<TensorDf4>(inputs_big, outputs_big,
                                                inputs, outputs,
                                                input_offset, output_offset,
                                                input_share, output_share, input_share_sub, output_share_sub,
                                                param, get_time, ctx);
                        }
                    }
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

