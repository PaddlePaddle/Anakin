#include "core/context.h"
#include "funcs/axpy.h"
#include "test_saber_func.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

int num_in = 1;
int ch_in = 4;
int h_in = 2;
int w_in =4;
int test_iter = 100;
bool compare_result = true;
bool get_time = false;
void axpy_nv_basic(Tensor<NVHX86>& tensor_in, const float* scale, const float* bias, \
                Tensor<NVHX86>& tensor_out, AxpyParam<NV> param){
    int num = tensor_in.num();
    int channel = tensor_in.channel();
    int height = tensor_in.height();
    int width = tensor_in.width();

    //LOG(INFO) << "basic compute";
    //LOG(INFO) << "has_axis: "<<   has_ax << ", out_max_val: "<<out_max;

    float* dout = (float*)tensor_out.mutable_data();
    int in_channel = channel * height * width;
    int size = height * width;
    for (int i = 0; i < num; i++){
        const float* din_ptr = din + i * in_channel;
        const float* bias_ptr = bias + i * in_channel;
        float* dout_ptr = dout + i * in_channel;
        for(int j = 0; j < channel; j++){
            const float* din_ch_ptr = din_ptr + j * size;
            const float* dout_ch_ptr = dout_ptr + j * size;
            const float* scale_ptr = scale + j;
            const float* bias_ptr = scale + j;
            for (int k = 0; k < size; k++){
                dout_ch_ptr[k] = din_ch_ptr[k] * scale_ptr[0] + bias_ptr[k];
            }
        }
    }
}
template<typename Dtype>
void tensor_diff(const Dtype* src1, const Dtype* src2, Dtype* des, int size) {
    for (int i = 0; i < size; ++i) {
        des[i] = src1[i] - src2[i];
    }
}

template <typename TargetD, typename TargetH, DataType OpType>
void test_axpy(Shape input_shape, AxpyParam<TargetD> param, Shape offset, bool is_share_from) {

    typedef typename DataTrait<TargetD, OpType>::Dtype Dtype;

    typedef Tensor<TargetH> TensorH;
    typedef Tensor<TargetD> TensorD;
    Context<TargetD> ctx(0, 1, 1);

    TensorD dev_input;
    TensorD dev_scale_input;
    TensorD dev_bias_input;
    //TensorD dev_output;

    dev_input.set_shape(input_shape, input_shape);
    dev_bias_input.set_shape(input_shape, input_shape);
    int channel = input_shape.channel();
    Shape scale_shape{1, channel, 1, 1};
    dev_scale_input.size(scale_shape, scale_shape);

    TensorH host_input(input_shape);
    fill_tensor_rand(host_input, -1, 1);
    dev_input.copy_from(host_input);

    TensorH host_scale(scale_shape);
    fill_tensor_rand(host_scale, -1, 1);
    dev_input.copy_from(host_scale);

    TensorH host_bias(input_shape);
    fill_tensor_rand(host_bias, -1, 1);
    dev_input.copy_from(host_bias);
    //fill_tensor_device_rand(big_input, -1, 1);
/*
    LOG(INFO) << "is_share_from: " << is_share_from;
    if (is_share_from) {
        small_input.share_from(big_input);
        small_output.share_from(big_output);
    } else {
        small_input.share_sub_buffer(big_input, input_shape, offset);
        small_output.share_sub_buffer(big_output, input_shape, offset);
    }
*/
    TensorD dev_output;
    // start Reshape & doInfer

    std::vector<TensorD*> inputs;
    std::vector<TensorD*> outputs;

    inputs.push_back(&dev_input);
    inputs.push_back(&dev_scale_input);
    inputs.push_back(&dev_bias_input);
    outputs.push_back(&dev_output);

    Axpy<TargetD, OpType> axpy;

    LOG(INFO) << "num: " << inputs[0]->num();
    LOG(INFO) << "chin: " << inputs[0]->channel();
    LOG(INFO) << "hin: " << inputs[0]->height();
    LOG(INFO) << "win: " << inputs[0]->width();
    LOG(INFO) << "compute_output_shape";
    axpy.compute_output_shape(inputs, outputs, param);

    LOG(INFO) << "num_out: " << outputs[0]->num();
    LOG(INFO) << "chout: " << outputs[0]->channel();
    LOG(INFO) << "hout: " << outputs[0]->height();
    LOG(INFO) << "wout: " << outputs[0]->width();

    LOG(INFO) << "run axpy  cuda for precision comparation";
    LOG(INFO) << "init";
    // init assume output tensor has been reshpaed by user.
    axpy.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);
    //argmax.init(inputs, outputs, param, RUNTIME, VENDER_IMPL, ctx);
    LOG(INFO) << "compute";
    axpy(inputs, outputs, param, ctx);
    typename TensorD::API::stream_t stream = ctx.get_compute_stream();
    outputs[0]->record_event(stream);
    outputs[0]->sync();
    //print_tensor(big_output);
    //print_tensor(big_input);

    /*test time*/
    if (get_time) {
        SaberTimer<TargetD> my_time;
        my_time.start(ctx);

        for (int i = 0; i < test_iter; i++) {
            axpy(inputs, outputs, param, ctx);
            outputs[0]->record_event(ctx.get_compute_stream());
            outputs[0]->sync();
        }

        my_time.end(ctx);
        LOG(INFO) << "axpy cuda aveage time " << my_time.get_average_ms() / test_iter;
    }

    if (compare_result) {
        LOG(INFO) << "run axpy  basic for precision comparation";

        TensorH tout_basic(outputs[0]->valid_shape());

        TensorH tin_saber(inputs[0]->valid_shape());
        tin_saber.copy_from(*inputs[0]);

      //  LOG(INFO) << "tin";
      //  print_tensor(tin_saber);

        const float* scale_ptr = host_scale.data();
        const float* bias_ptr = host_bias.data();
        SaberTimer<TargetD> my_time;
        my_time.start(ctx);
        for (int i = 0; i < test_iter; ++i) {
            axpy_nv_basic(tin_saber, scale_ptr, bias_ptr, tout_basic, param);
        }
        my_time.end(ctx);
        LOG(INFO) << "axpy basic aveage time " << my_time.get_average_ms() / test_iter;

        //fast_free(work_space_data);
        LOG(INFO) << "basic";
        print_tensor(tout_basic);

        double max_ratio = 0;
        double max_diff = 0;

        TensorH tout_saber(outputs[0]->valid_shape());
        tout_saber.copy_from(*outputs[0]);
        LOG(INFO) << "saber";
        print_tensor(tout_saber);

        TensorH tdiff(tout_basic.valid_shape());

        int size1 = tout_basic.valid_size();
        int size2 = tout_saber.valid_size();
            
        CHECK_EQ(size1, size2) << "wrong shape";
        //LOG(INFO) << "tdiff";

        const Dtype* din = (const Dtype*)tout_basic.data();
        const Dtype* dout = (const Dtype*)tout_saber.data();
        Dtype* diff = (Dtype*)tdiff.mutable_data();
        int size = tout_basic.valid_size();
        LOG(INFO) << "diff";
        tensor_diff(din, dout, diff, size);
        //print_tensor_host(tdiff);
        tensor_cmp_host((const Dtype*)tout_basic.data(), (const Dtype*)tout_saber.data(), tout_basic.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        CHECK_EQ(fabsf(max_ratio) < 1e-3f, true) << "compute result error";
    }

    //print_tensor_device(*outputs[0]);

#ifdef USE_CUDA
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
#endif
}

template <typename TargetD, typename TargetH, DataType OpType>
void test_accuracy(int num, int channel, int height, int width) {

    typedef Tensor<TargetH> TensorH;
    typedef Tensor<TargetD> TensorD;
    typedef typename DataTrait<TargetD, OpType>::Dtype Dtype;

    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_big_shape({num, channel, height+1, width+1}, Layout_NCHW);
    Shape offset_0({0, 0, 0, 0}, Layout_NCHW);
    Shape offset_1({0, 0, 1, 1}, Layout_NCHW);

    AxpyParam<TargetD> axpy_param();//has axis


    for (ArgmaxParam<TargetD> param : {axpy_param}) {
        //for (ActivationParam<TensorD> param : {param_sigmoid}) {
        for (auto share_from : {false, true}) {
            for (auto offset: {offset_0, offset_1}) {
                test_argmax<TargetD, TargetH, OpType>(input_shape, param, offset, share_from);
            }
        }
    }
}

TEST(TestSaberFunc, test_func_argmax) {
    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;
   
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    test_accuracy<NV, NVHX86, AK_FLOAT>(num, channel, height, width, out_max, topk, has, ax);
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    test_accuracy<X86, X86, AK_FLOAT>(num, channel, height, width, out_max, topk, has, ax);
#endif
}


int main(int argc, const char** argv) {

 /*   if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    */
    if (argc >= 2) {
        test_iter = atoi(argv[1]);
    }
    if (argc >= 3) {
        compare_result = atoi(argv[2]) > 0;
    }
    if (argc >= 4) {
        get_time = atoi(argv[3]) > 0;
    }
    if(argc >= 5) {
        if (argc < 8) {
            LOG(ERROR) << "usage: ./" << argv[0] << "test_iter " << \
                " compare_result get_time num ch_in h_in w_in" ;
            return 0;
        }
        num_in = atoi(argv[4]);
        ch_in = atoi(argv[5]);
        h_in = atoi(argv[6]);
        w_in = atoi(argv[7]);
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

