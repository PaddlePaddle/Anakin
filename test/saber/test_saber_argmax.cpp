#include "core/context.h"
#include "funcs/argmax.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

bool out_max_val = false;
bool has_axis = false;
int top_k = 3;
int axis = 1;
int num_in = 1;
int ch_in = 32;
int h_in = 112;
int w_in = 112;
int test_iter = 10;
bool compare_result = true;
bool get_time = false;

//void argmax_nv_basic(Tensor<NVHX86>& tensor_in, Tensor<NVHX86>& tensor_out, ArgmaxParam<NV> param){
template <typename dtype,typename TargetType_D,typename TargetType_H>
void argmax_nv_basic(const std::vector<Tensor<TargetType_H>*>& tensor_in,std::vector<Tensor<TargetType_H>*>& tensor_out,ArgmaxParam<TargetType_D>& param){
    int num = tensor_in[0]->num();
    int channel = tensor_in[0]->channel();
    int height = tensor_in[0]->height();
    int width = tensor_in[0]->width();

    int ch_out = tensor_out[0]->channel();
    int w_out = tensor_out[0]->width();
    int h_out = tensor_out[0]->height();

    int top = param.top_k;
    bool has_ax = param.has_axis;
    int ax = param.axis;
    bool out_max = param.out_max_val;

    //LOG(INFO) << "basic compute";
    //LOG(INFO) << "has_axis: "<<   has_ax << ", ax:" << ax << ", out_max_val: "<<out_max;
  //  const float* din = (const float*)tensor_in[0]->data();
  //  float* dout = (float*)tensor_out[0]->mutable_data();
    const dtype* din = (const dtype*)tensor_in[0]->data();
    dtype* dout = (dtype*)tensor_out[0]->mutable_data();
    int in_channel = channel * height * width;
    int out_channel = ch_out * w_out * h_out;

    if (has_ax){//nchw
        auto shape = tensor_in[0]->valid_shape();
        int stride = shape.count(ax+1, shape.dims());
        int out_stride = shape.count(1, ax);
        int out_ss = tensor_out[0]->valid_shape().count(ax, shape.dims());
        int in_ss = shape.count(ax, shape.dims());
       // LOG(INFO) << "stride: "<<stride << ", out_stride: " << out_stride;
        int size = shape[ax];
        if(size < top){
            LOG(INFO) << "input data size less than topk";
            return; 
        }
        for (int n = 0; n < num * out_stride; n++){
            for(int k = 0; k < stride; k ++){
                const dtype* din_ch = din + n * in_ss + k;
                std::vector< std::pair<dtype, int> > vec;
                vec.resize(size);
                for (int i = 0; i < size; i++){
                    vec[i] = std::make_pair(din_ch[i*stride], i);
                }
                 //sort
                std::partial_sort(vec.begin(), vec.begin() + top, vec.end(), std::greater< std::pair<float, int> >());
                //out
                dtype* dout_ch = dout + n * out_ss + k;
                for(int i = 0; i < top ;i ++){
                    if(out_max)
                        dout_ch[i*stride] = vec[i].first;
                    else
                        dout_ch[i*stride] = vec[i].second;
                }
            }
        }
    }else{//all  
        if(in_channel < top){
            LOG(INFO) << "input data size less than topk";
            return; 
        }
        for (int n = 0; n < num; n++){
            const dtype* din_ch = din + n * in_channel;
            std::vector< std::pair<dtype, int> > vec;
            vec.resize(in_channel);
            for (int i = 0; i < in_channel; i++){
                vec[i] = std::make_pair(din_ch[i], i);
            }
            //sort
            std::partial_sort(vec.begin(), vec.begin() + top, vec.end(), std::greater< std::pair<float, int> >());
            //out
            if(out_max){
                dtype* dout_ch = dout + n * out_channel;
                dtype* dout_data = dout_ch;
                dtype* dout_index = dout_ch + top;
                for (int i = 0; i < top; i++){
                    dout_data[i] = vec[i].first;
                    dout_index[i] = vec[i].second;
                    //LOG(INFO) << "max_data: " <<dout_data[i] << ", max_index: "<<dout_index[i];
                }
            }else{
                dtype* dout_data = dout + n * out_channel;
                for (int i = 0; i < top; i++){
                    dout_data[i] = vec[i].second;
                   // LOG(INFO) << "max_data: " <<vec[i].first << ", max_index: "<< dout_data[i];
                }
            }
            vec.clear();
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
void test_argmax(Shape input_shape, ArgmaxParam<TargetD> param) {

    typedef typename DataTrait<TargetD, OpType>::Dtype Dtype;
    typedef Tensor<TargetH> TensorH;
    typedef Tensor<TargetD> TensorD;
    Context<TargetD> ctx(0, 1, 1);

    TensorD dev_input;
    TensorD dev_output;

    dev_input.re_alloc(input_shape, OpType);//Dtype);

    dev_output.re_alloc(input_shape, OpType);//Dtype);

    TensorH host_input(input_shape);
    fill_tensor_rand(host_input, -1, 1);
    dev_input.copy_from(host_input);

    std::vector<TensorD*> inputs;
    std::vector<TensorD*> outputs;

    inputs.push_back(&dev_input);
    outputs.push_back(&dev_output);

    Argmax<TargetD, OpType> argmax;

    LOG(INFO) << "num: " << inputs[0]->num();
    LOG(INFO) << "chin: " << inputs[0]->channel();
    LOG(INFO) << "hin: " << inputs[0]->height();
    LOG(INFO) << "win: " << inputs[0]->width();
    LOG(INFO) << "topk: " << param.top_k;
    LOG(INFO) << "has_axis: " << param.has_axis;
    LOG(INFO) << "axis: " << param.axis ;
    LOG(INFO) << "out_max_val: " << param.out_max_val;
    LOG(INFO) << "compute_output_shape";
    argmax.compute_output_shape(inputs, outputs, param);

    LOG(INFO) << "num_out: " << outputs[0]->num();
    LOG(INFO) << "chout: " << outputs[0]->channel();
    LOG(INFO) << "hout: " << outputs[0]->height();
    LOG(INFO) << "wout: " << outputs[0]->width();


    LOG(INFO) << "run argmax cuda for precision comparation";
    LOG(INFO) << "init";
    // init assume output tensor has been reshpaed by user.
    argmax.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx);
    //argmax.init(inputs, outputs, param, RUNTIME, VENDER_IMPL, ctx);
    LOG(INFO) << "compute";
    argmax(inputs, outputs, param, ctx);
    typename TensorD::API::stream_t stream = ctx.get_compute_stream();
    outputs[0]->record_event(stream);
    outputs[0]->sync();
    //print_tensor(dev_output);
    //print_tensor(dev_input);

    /*test time*/
    if (get_time) {
        SaberTimer<TargetD> my_time;
        my_time.start(ctx);

        for (int i = 0; i < test_iter; i++) {
            argmax(inputs, outputs, param, ctx);
            outputs[0]->record_event(ctx.get_compute_stream());
            outputs[0]->sync();
        }

        my_time.end(ctx);
        LOG(INFO) << "argmax cuda aveage time " << my_time.get_average_ms() / test_iter;
    }

    if (compare_result) {
        LOG(INFO) << "run argmax  basic for precision comparation";

        TensorH tout_basic(outputs[0]->valid_shape());

        TensorH tin_saber(inputs[0]->valid_shape());
        tin_saber.copy_from(*inputs[0]);

        //LOG(INFO) << "tin";
        //print_tensor(tin_saber);

        //ArgmaxParam<NVHX86> argmax_param(param);
        SaberTimer<TargetD> my_time;
        my_time.start(ctx);
        for (int i = 0; i < test_iter; ++i) {
            argmax_nv_basic(tin_saber, tout_basic, param);
        }
        my_time.end(ctx);
        LOG(INFO) << "argmax basic aveage time " << my_time.get_average_ms() / test_iter;

        //fast_free(work_space_data);
        // LOG(INFO) << "basic";
        //print_tensor(tout_basic);

        double max_ratio = 0;
        double max_diff = 0;

        TensorH tout_saber(outputs[0]->valid_shape());
        tout_saber.copy_from(*outputs[0]);
       // LOG(INFO) << "saber";
        //print_tensor(tout_saber);

        TensorH tdiff(tout_basic.valid_shape());

        int size1 = tout_basic.valid_size();
        int size2 = tout_saber.valid_size();
            
        CHECK_EQ(size1, size2) << "wrong shape";
        //LOG(INFO) << "tdiff";

        const Dtype* din = (const Dtype*)tout_basic.data();
        const Dtype* dout = (const Dtype*)tout_saber.data();
        Dtype* diff = (Dtype*)tdiff.mutable_data();
        int size = tout_basic.valid_size();
     //   LOG(INFO) << "diff";
      //  tensor_diff(din, dout, diff, size);
        //print_tensor_host(tdiff);
        tensor_cmp_host((const Dtype*)tout_basic.data(), (const Dtype*)tout_saber.data(), tout_basic.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        CHECK_EQ(fabsf(max_ratio) < 1e-3f, true) << "compute result error";
        if (max_ratio < 1.5e-1) {
            LOG(INFO) << " PASS!!! max_ratio = " << max_ratio << " max_diff = " << max_diff;
        } else {
            LOG(FATAL) << "FAIL!!! max_ratio = " << max_ratio << " max_diff = " << max_diff
            << "argmax param: "
           << "topk: " << param.top_k
           << "has_axis: " << param.has_axis
           << "axis: " << param.axis 
           << "out_max_val: " << param.out_max_val
           << "num: " << inputs[0]->num() << "chin: " << inputs[0]->channel()
           << "hin: " << inputs[0]->height()
           << "win: " << inputs[0]->width();
        }
    }

#ifdef USE_CUDA
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
#endif
}

template <typename TargetD, typename TargetH, DataType OpType>
void test_accuracy(int num, int channel, int height, int width, \
     bool out_max,int topk, bool has, int ax) {

    typedef Tensor<TargetH> TensorH;
    typedef Tensor<TargetD> TensorD;
    typedef typename DataTrait<TargetD, OpType>::Dtype Dtype;

    Shape input_shape({num, channel, height, width}, Layout_NCHW);

    ArgmaxParam<TargetD> argmax_param(out_max, topk, has, ax);//has axis
    ArgmaxParam<TargetD> argmax1(false, 3, true, 1);//has axis
    ArgmaxParam<TargetD> argmax2(false, 3, false, 1);//has axis
    ArgmaxParam<TargetD> argmax3(true, 3, true, 2);//has axis
    ArgmaxParam<TargetD> argmax4(true, 3, false, 1);//has axis

   // test_argmax<TargetD, TargetH, OpType>(input_shape, argmax_param);

    for(auto shape: {input_shape}){
        for (auto param : {argmax1,argmax2, argmax3, argmax4, argmax_param}) {
            test_argmax<TargetD, TargetH, OpType>(shape, param);
        }
    }
}

TEST(TestSaberFunc, test_func_argmax) {
    int num = num_in;
    int channel = ch_in;
    int height = h_in;
    int width = w_in;
    bool out_max = out_max_val;
    int topk = top_k;
    bool has = has_axis;
    int ax = axis;
   // LOG(INFO) << "topk: " << topk << ", has_axis: " << has << ", axis: " << ax << ", out_max_val: " << out_max;
#ifdef USE_CUDA
   // Env<NV>::env_init();
    //Env<NVHX86>::env_init();
    //test_accuracy<NV, NVHX86, AK_FLOAT>(num, channel, height, width, out_max, topk, has, ax);
    //Init the test_base
    TestSaberBase<NV,NVHX86,AK_FLOAT,Argmax,
    ArgmaxParam> testbase;
    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape input_shape2({1, 32, 17, 32}, Layout_NCHW);
   // typename NV TargetD;
    ArgmaxParam<NV> argmax_param(out_max, topk, has, ax);//has axis
    ArgmaxParam<NV> argmax1(false, 3, true, 1);//has axis
    ArgmaxParam<NV> argmax2(false, 3, false, 1);//has axis
    ArgmaxParam<NV> argmax3(true, 3, true, 2);//has axis
    ArgmaxParam<NV> argmax4(true, 3, false, 1);//has axis

   // test_argmax<TargetD, TargetH, OpType>(input_shape, argmax_param);

    for(auto shape: {input_shape, input_shape2}){
        for (auto param : {argmax1,argmax2, argmax3, argmax4, argmax_param}) {
           // test_argmax<TargetD, TargetH, OpType>(shape, param);
             testbase.set_param(param);//set param
            //testbase.set_rand_limit(255,255);
            testbase.set_input_shape(shape);//add some input shape
            testbase.run_test(argmax_nv_basic<float,NV,NVHX86>);//run test
                               
        }
    }


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
    if (argc >= 5) {
        top_k = atoi(argv[4]);
    }
    if (argc >= 6) {
        has_axis = atoi(argv[5]) > 0;
    }
    if (argc >= 7) {
        axis = atoi(argv[6]);
    }
    if (argc >= 8) {
        out_max_val = atoi(argv[7]) > 0;
    }
    if(argc >= 9) {
        if (argc < 12) {
            LOG(ERROR) << "usage: ./" << argv[0] << " test_iter " << \
                " compare_result get_time top_k has_axis axis out_max_val num ch_in h_in w_in";
            return 0;
        }
        num_in = atoi(argv[8]);
        ch_in = atoi(argv[9]);
        h_in = atoi(argv[10]);
        w_in = atoi(argv[11]);
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

