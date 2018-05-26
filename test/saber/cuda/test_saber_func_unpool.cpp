
#include "core/context.h"
#include "funcs/unpool.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
template<typename TensorType>
void test_unpool(std::vector<TensorType*>& inputs_big, std::vector<TensorType*>& outputs_big,
                 std::vector<TensorType*>& inputs, std::vector<TensorType*>& outputs,
                 Shape input_offset, Shape output_offset,
                 bool input_share, bool output_share, bool input_share_sub, bool output_share_sub,
                 PoolingParam<TensorType>& param, bool get_time, Context<NV>& ctx) {
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    /*prepare data*/
    Unpool<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> unpool;
    unpool.compute_output_shape(inputs, outputs, param);

    outputs[0]->set_shape(outputs[0]->valid_shape(), outputs[0]->valid_shape());
    inputs[0]->set_shape(inputs[0]->valid_shape(), inputs[0]->valid_shape());
    inputs[1]->set_shape(inputs[1]->valid_shape(), inputs[1]->valid_shape());

    if (output_share && output_share_sub) {
        outputs[0]->share_sub_buffer(*outputs_big[0], outputs[0]->valid_shape(), output_offset);
    } else if (output_share) {
        outputs[0]->share_from(*outputs_big[0]);
    } else {
        outputs[0]->re_alloc(outputs[0]->valid_shape());
    }

    if (input_share && input_share_sub) {
        inputs[0]->share_sub_buffer(*inputs_big[0], inputs[0]->valid_shape(), input_offset);
        inputs[1]->share_sub_buffer(*inputs_big[1], inputs[1]->valid_shape(), input_offset);
    } else if (input_share) {
        inputs[0]->share_from(*inputs_big[0]);
        inputs[1]->share_from(*inputs_big[1]);
    } else {
        inputs[0]->re_alloc(inputs[0]->valid_shape());
        cudaMemcpy(inputs[0]->mutable_data(), inputs_big[0]->data(),
                   sizeof(float) * (inputs[0]->valid_size()), cudaMemcpyDeviceToDevice);
        inputs[1]->re_alloc(inputs[1]->valid_shape());
        cudaMemcpy(inputs[1]->mutable_data(), inputs_big[1]->data(),
                   sizeof(float) * (inputs[1]->valid_size()), cudaMemcpyDeviceToDevice);
    }

    // init assume output tensor has been reshpaed by user.
    unpool.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);

    /*warm up*/
    unpool(inputs, outputs, param, ctx);
    outputs[0]->record_event(ctx.get_compute_stream());
    outputs[0]->sync();

    /*test time*/
    if (get_time) {
        SaberTimer<NV> my_time;
        my_time.start(ctx);

        for (int i = 0; i < 100; i++) {
            unpool(inputs, outputs, param, ctx);
            outputs[0]->record_event(ctx.get_compute_stream());
            outputs[0]->sync();
        }

        my_time.end(ctx);
        //LOG(INFO)<<"aveage time"<<my_time.get_average_ms()/100;
    }

    CUDA_CHECK(cudaPeekAtLastError());
    Shape valid_shape = outputs[0]->valid_shape();
    //printf("shape: %d, %d, %d, %d\n", valid_shape[0], valid_shape[1],valid_shape[2], valid_shape[3]);
    TensorDf4 out_valid_0(outputs[0]->valid_shape());
    CUDA_CHECK(cudaPeekAtLastError());
    out_valid_0.copy_from(*outputs[0]);
    CUDA_CHECK(cudaPeekAtLastError());

    print_tensor_device(*inputs[0]);
    print_tensor_device(*inputs[1]);
    cudaDeviceSynchronize();
    //print_tensor_device(*outputs[0]);
    //cudaDeviceSynchronize();
    print_tensor_device(out_valid_0);
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
    int c = 3;
    int h = 4;
    int w = 4;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape real_shape(n, c, h, w);
    Shape valid_shape(n, c, h, w);
    Shape input_offset(0, 0, 0, 0);
    Shape output_offset(0, 0, 0, 0);

    TensorHf4 in_host_big_0;
    TensorHf4 in_host_big_1;
    TensorDf4 in_dev_big_0;
    TensorDf4 in_dev_big_1;
    TensorHf4 out_host_big;
    TensorDf4 out_dev_big_0;

    TensorDf4 in_dev_0;
    TensorDf4 in_dev_1;
    TensorDf4 output_dev_0;

    in_host_big_0.re_alloc(real_shape);
    in_host_big_1.re_alloc(real_shape);
    in_dev_big_0.re_alloc(real_shape);
    in_dev_big_1.re_alloc(real_shape);
    out_host_big.re_alloc(real_shape);
    out_dev_big_0.re_alloc(real_shape);

    /*prepare input data*/
    auto data = in_host_big_0.mutable_data();

    for (int i = 0; i < in_host_big_0.size(); ++i) {
        data[i] = 0x7f & i;
    }

    PoolingParam<TensorDf4> param(2, 2, 0, 0, 2, 2, Pooling_max, false);
    int out_h = (h - 1) * param.stride_h + param.window_h - 2 * param.pad_h;
    int out_w = (w - 1) * param.stride_w + param.window_w - 2 * param.pad_w;
    auto index = in_host_big_1.mutable_data();
    int count  = 0;

    for (int n_id = 0; n_id < n; n_id++) {
        for (int c_id = 0; c_id < c; c_id++) {
            for (int h_id = 0; h_id < h; h_id++) {
                for (int w_id = 0; w_id < w; w_id++) {
                    index[count] = (h_id * param.stride_h + rand() % param.window_h) * out_w +
                                   (w_id * param.stride_w + rand() % param.window_w);
                    count++;
                }
            }
        }
    }

    in_dev_big_0.copy_from(in_host_big_0);
    in_dev_big_1.copy_from(in_host_big_1);


    in_dev_0.set_shape(valid_shape);
    in_dev_1.set_shape(valid_shape);

    std::vector<TensorDf4*> inputs_big;
    std::vector<TensorDf4*> outputs_big;
    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs_big.push_back(&in_dev_big_0);
    inputs_big.push_back(&in_dev_big_1);
    outputs_big.push_back(&out_dev_big_0);
    inputs.push_back(&in_dev_0);
    inputs.push_back(&in_dev_1);
    outputs.push_back(&output_dev_0);
    // start Reshape & doInfer
    Context<NV> ctx(0, 1, 1);
    std::vector<int> offset = {2, 2};

    /*window, pad, stride*/
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
                                false
                            }) {
                        LOG(INFO) << input_share << "," << output_share << "," << input_share_sub << "," << output_share_sub
                                  << "," << get_time;
                        test_unpool<TensorDf4>(inputs_big, outputs_big,
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

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

