
#include "core/context.h"
#include "funcs/ctc_align.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
template<typename TensorType>
void test_ctc_align(std::vector<TensorType*>& inputs, std::vector<TensorType*>& outputs,
               CtcAlignParam<Tensor<NV, AK_FLOAT, NCHW>>& param, bool get_time, Context<NV>& ctx) {
    /*prepare data*/
    CtcAlign<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> ctc_align;
    ctc_align.compute_output_shape(inputs, outputs, param);
    outputs[0]->re_alloc(outputs[0]->valid_shape());
    // init assume output tensor has been reshpaed by user.
    ctc_align.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);
    CUDA_CHECK(cudaPeekAtLastError());

    /*warm up*/
    ctc_align(inputs, outputs, param, ctx);
    outputs[0]->record_event(ctx.get_compute_stream());
    outputs[0]->sync();

    /*test time*/
    if (get_time) {
        SaberTimer<NV> my_time;
        my_time.start(ctx);
        for (int i = 0; i < 100; i++) {
            ctc_align(inputs, outputs, param, ctx);
            outputs[0]->record_event(ctx.get_compute_stream());
            outputs[0]->sync();
        }
        my_time.end(ctx);
        //LOG(INFO)<<"aveage time"<<my_time.get_average_ms()/100;
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

    int n = 100;
    int c = 1;
    int h = 1;
    int w = 1;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape valid_shape(n, c, h, w);

    TensorHf4 in_host_x;
    TensorDf4 in_dev_x;
    TensorDf4 out_dev;
    in_host_x.re_alloc(valid_shape);
    in_dev_x.re_alloc(valid_shape);

    /*prepare input data*/
    auto data = in_host_x.mutable_data();
    for (int i = 0; i < in_host_x.size(); ++i) {
        data[i] = std::rand() % 10;
    }
    in_dev_x.copy_from(in_host_x);
    std::vector<int> seq_offset = {0, 10, 19, 32, 40, 60, 78, 90, 100};

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&in_dev_x);
    outputs.push_back(&out_dev);
    inputs[0]->set_seq_offset(seq_offset);

    // start Reshape & doInfer
    Context<NV> ctx(0, 1, 1);
    CtcAlignParam<Tensor<NV, AK_FLOAT, NCHW>> param(0, 1);
    for (auto get_time: {false}) {
        test_ctc_align<TensorDf4>(inputs, outputs,
                param, get_time, ctx);
    }
}

int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

