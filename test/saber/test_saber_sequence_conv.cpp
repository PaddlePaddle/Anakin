
#include <vector>

#include "saber/funcs/sequence_conv.h"
#include "test_saber_func.h"
#include "test_util.h"
#include "tensor_op.h"
#include "debug.h"
#ifdef USE_X86_PLACE
#include <mkl_service.h>
#endif

using namespace anakin::saber;
using namespace std;

template <typename Dtype>
static void im2col_2d_ocf(const Dtype* in, int start,int stride, int pad_up, int pad_down, int kernel_size,
                          Dtype* out, int seq_length, int hidden_size) {
    for (int out_row = 0; out_row < seq_length; ++out_row) {
        for (int col = 0; col < kernel_size; ++col) {
            int index = out_row + col - pad_up+start;
            int out_index = (out_row * kernel_size + col) * hidden_size;

            if (index < 0 || index >= seq_length) {
                for (int hidden_index = 0; hidden_index < hidden_size; ++hidden_index){
                    out[out_index + hidden_index] = 0;
                }
            } else{
                for (int hidden_index = 0; hidden_index < hidden_size; ++hidden_index){
                    out[out_index + hidden_index] = in[index * hidden_size + hidden_index];
                }
            }
        }
    }
}

template<typename Dtype>
static void gemm_naive(int m,int n,int k,const float alpha,const Dtype * a, const Dtype*b ,const float beta,Dtype *c){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            Dtype acc=0;
            for(int inner=0;inner<k;inner++){
                acc+=alpha*a[i*k+inner]*b[inner*n+j];
            }
            c[i*n+j]=acc+beta*c[i*n+j];
        }
    }
}

template <typename TensorHost,typename TargetType>
static void compute_ref_sequence_conv(std::vector<TensorHost*> &inputs, std::vector<TensorHost*> &outputs, SequenceConvParam<TargetType> &param){

    int hidden_size = param.filter_tensor->height() / param.context_length;
    int feature_size = param.filter_tensor->width();
    int up_pad = std::max(0, -param.context_start);
    int down_pad = std::max(0, param.context_start + param.context_length - 1);
    int hidden_kernel_size = hidden_size * param.context_length;
    int start_word=std::max(0,param.context_start);
    TensorHost* in_data = inputs[0];
    TensorHost* out_data = outputs[0];
    std::vector<std::vector<int>> offset_vec_vec = in_data->get_seq_offset();
    std::vector<int> offset = offset_vec_vec[offset_vec_vec.size()-1];
    out_data->set_seq_offset(offset_vec_vec);

    int word_num = offset[offset.size() - 1];
    Shape shape_im2col({1,1,1,word_num*param.filter_tensor->height()});
    TensorHost temp_im2col_tensor(shape_im2col);

    LOG(INFO)<<"pad_up = "<<up_pad;
    for (int i = 0; i < offset.size() - 1; ++i) {
        int start = offset[i];
        int seq_length = offset[i + 1] - offset[i];
        im2col_2d_ocf(static_cast<const float *>(in_data->data()) + hidden_size * start,start_word, param.context_stride, up_pad, down_pad,
                      param.context_length, static_cast<float *>(temp_im2col_tensor.mutable_data()) + hidden_kernel_size * start, seq_length,
                      hidden_size);
    }

    gemm_naive(word_num, feature_size, hidden_kernel_size, 1.f, static_cast<const float*>(temp_im2col_tensor.data()),
         static_cast<const float*>(param.filter_tensor->data()), 0.f, static_cast<float*>(out_data->mutable_data()));

};

//#define COMPARE_WITH_OUT
template <typename HOST,typename DEVICE>
void sequence_ut(std::vector<int> offsets = {0, 3,13,22,30,50},int hidden_size = 2,
        int context_length = 3, int feature_size = 5,int context_start=-1,
        int perf_iter=0,ImplEnum test_mode=SABER_IMPL){

    typedef Tensor<HOST> TensorHf4;
    typedef Tensor<DEVICE> TensorDf4;
    Context<DEVICE> ctx_dev(0, 1, 1);

    int word_num=offsets[offsets.size()-1];
    Shape shape_weight({1, 1, context_length*hidden_size,feature_size},Layout_NCHW);
    Shape shape_x({word_num, hidden_size, 1, 1},Layout_NCHW);
    Shape shape_out({word_num,feature_size,1,1},Layout_NCHW);

    TensorHf4 host_x(shape_x);
    TensorHf4 host_weight(shape_weight);
    TensorHf4 host_out(shape_out);
    TensorDf4 dev_x(shape_x);
    TensorDf4 dev_weight(shape_weight);
    TensorDf4 dev_out(shape_out);
#ifdef COMPARE_WITH_OUT
    readTensorData(host_weight, "host_w");
    readTensorData(host_x, "host_x");
#else
    fill_tensor_rand(host_weight,-1,1);
    fill_tensor_rand(host_x,-1,1);
#endif
    dev_weight.copy_from(host_weight);
    dev_x.copy_from(host_x);

    host_x.set_seq_offset({offsets});
    dev_x.set_seq_offset({offsets});
    SequenceConvParam<DEVICE> param(&dev_weight,context_length,context_start);

    SequenceConv<DEVICE, AK_FLOAT> seqconv_op;

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&dev_x);
    outputs.push_back(&dev_out);

    SABER_CHECK(seqconv_op.init(inputs, outputs, param, SPECIFY, test_mode, ctx_dev));
    SABER_CHECK(seqconv_op.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape(),outputs[0]->get_dtype());

#ifndef COMPARE_WITH_OUT
    SABER_CHECK(seqconv_op(inputs, outputs, param, ctx_dev));
    outputs[0]->record_event(ctx_dev.get_compute_stream());
    outputs[0]->sync();

    if(perf_iter>0) {
        SaberTimer<DEVICE> t1;
        t1.start(ctx_dev);
        for (int i = 0; i < perf_iter; ++i) {
            SABER_CHECK(seqconv_op(inputs, outputs, param, ctx_dev));
            outputs[0]->record_event(ctx_dev.get_compute_stream());
            outputs[0]->sync();
        }
        t1.end(ctx_dev);
        LOG(INFO) << "!!saber care: iter = " << perf_iter << " , total time: " << t1.get_average_ms() <<
                  "avg time : " << t1.get_average_ms() / perf_iter << " args [" << offsets[offsets.size() - 1]
                  << "," << offsets.size() - 1 << "," << hidden_size << "]";
    }
    host_out.copy_from(dev_out);
#endif

    TensorHf4 compare_g(shape_out);

//    write_tensorfile(host_out, "host_g.txt");
//    write_tensorfile(compare_g, "host_correct.txt");

    std::vector<TensorHf4*> inputs_ref;
    std::vector<TensorHf4*> outputs_ref;
    inputs_ref.push_back(&host_x);
    outputs_ref.push_back(&compare_g);
    SequenceConvParam<HOST> param_ref(&host_weight,context_length,context_start);
    compute_ref_sequence_conv(inputs_ref,outputs_ref,param_ref);
#ifdef COMPARE_WITH_OUT
    host_out.copy_from(compare_g);
    readTensorData(compare_g, "host_correct");
    write_tensorfile(host_out, "host_g.txt");
    write_tensorfile(compare_g, "host_correct.txt");
#endif
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host((const float*)host_out.data(), (const float*)compare_g.data(), host_out.valid_size(), maxratio, maxdiff);
    if (abs(maxratio) <= 0.001||abs(maxdiff)<0.001) {
        LOG(INFO) << "passed  " << maxratio<<","<<maxdiff<<",?="<<abs(maxratio);
    } else {
        CHECK(false) << "failed : ratio " << maxratio<<","<<maxdiff;
    }
}

#ifdef USE_X86_PLACE
TEST(TestSaberFunc, test_func_sequence_conv_x86) {
    Env<X86>::env_init();
#ifdef USE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    mkl_set_num_threads(1);
    sequence_ut<X86,X86>({0, 6, 11,20},20,3,50,-1);
    sequence_ut<X86,X86>({0, 6, 15,30},20,4,50,0);
    sequence_ut<X86,X86>({0, 6, 15,30},20,2,50,1);
    sequence_ut<X86,X86>({0, 6, 15,30},20,3,50,2);

}
#endif

#ifdef NVIDIA_GPU
TEST(TestSaberFunc, test_func_sequence_conv_nv) {
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    sequence_ut<NVHX86,NV>({0, 6, 11,20},20,3,50,-1);
    sequence_ut<NVHX86,NV>({0, 6, 15,30},20,4,50,0);
    sequence_ut<NVHX86,NV>({0, 6, 15,30},20,2,50,1);
    sequence_ut<NVHX86,NV>({0, 6, 15,30},20,3,50,2);
}
#endif

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}