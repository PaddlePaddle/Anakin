
#include "saber/core/context.h"
#include "saber/funcs/sequence_pool.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>
#include <cmath>
using namespace anakin::saber;

template <typename dtype>
static void seq_pool_average(dtype* dst, const dtype* src_in,
                  const int slice_num, const int slice_size) {
    dtype sum = 0.f;
    for (int i = 0; i < slice_size; ++i) {
        sum = src_in[i];
        for (int s = 1; s < slice_num; ++s) {
            dtype src_in_read = src_in[s * slice_size +i];
            sum += src_in_read;
        }
        dst[i] = sum / slice_num;
    }
}

template <typename dtype>
static void seq_pool_sum(dtype* dst, const dtype* src_in,
                  const int slice_num, const int slice_size) {
    dtype sum = 0.f;
    for (int i = 0; i < slice_size; ++i) {
        sum = src_in[i];
        for (int s = 1; s < slice_num; ++s) {
            dtype src_in_read = src_in[s * slice_size +i];
            sum += src_in_read;
        }
        dst[i] = sum;
    }
}

template <typename dtype>
static void seq_pool_sqrt(dtype* dst, const dtype* src_in,
                  const int slice_num, const int slice_size) {
    dtype sqrt_len = sqrtf(slice_num);
    dtype sum = 0.f;
    for (int i = 0; i < slice_size; ++i) {
        sum = src_in[i];
        for (int s = 1; s < slice_num; ++s) {
            dtype src_in_read = src_in[s * slice_size +i];
            sum += src_in_read;
        }
        dst[i] = sum / sqrt_len;
    }
}

template <typename dtype>
static void seq_pool_max(dtype* dst, const dtype* src_in,
                  const int slice_num, const int slice_size) {
    dtype max = 0.f;
    for (int i = 0; i < slice_size; ++i) {
        max = src_in[i];
        for (int s = 1; s < slice_num; ++s) {
            dtype src_in_read = src_in[s * slice_size +i];
            if (max < src_in_read) {
                max = src_in_read;
            }
        }
        dst[i] = max;
    }
}

template <typename dtype>
static void seq_pool_first(dtype* dst, const dtype* src_in,
                  const int slice_num, const int slice_size) {
    memcpy(dst, src_in, sizeof(dtype)* slice_size);
}

template <typename dtype>
static void seq_pool_last(dtype* dst, const dtype* src_in,
                  const int slice_num, const int slice_size) {
    memcpy(dst, src_in + slice_size * (slice_num - 1), sizeof(dtype)* slice_size);
}

template <typename dtype>
static void seq_pool_unknow(dtype* dst, const dtype* src_in,
                  const int slice_num, const int slice_size) {
    LOG(ERROR) << " UNKNOWN seq pool type";
}

template <typename dtype,typename TargetType_D,typename TargetType_H>
void sequence_pool_cpu(const std::vector<Tensor<TargetType_H>*>& inputs,std::vector<Tensor<TargetType_H>*>& outputs,\
                SequencePoolParam<TargetType_D>& param){
    CHECK_EQ(inputs[0]->channel(), outputs[0]->channel());
    CHECK_EQ(inputs[0]->height(), outputs[0]->height());
    CHECK_EQ(inputs[0]->width(), outputs[0]->width());
    std::vector<int> seq_offset = inputs[0]->get_seq_offset()[0];
    int slice_size = outputs[0]->channel()
                     * outputs[0]->height()
                     * outputs[0]->width();
    dtype* dst_ptr = (dtype*)outputs[0]->mutable_data();
    const dtype* src_ptr = (const dtype*)inputs[0]->data();
    int batch_size = seq_offset.size() - 1;
    for (int i = 0; i < seq_offset.size()-1; ++i) {
        int slice_num = seq_offset[i+1] - seq_offset[i];
        switch(param.sequence_pool_type){
            case Sequence_pool_average:
                seq_pool_average(dst_ptr, src_ptr, slice_num, slice_size);
                break;
            case Sequence_pool_sum:
                seq_pool_sum(dst_ptr, src_ptr, slice_num, slice_size);
                break;
            case Sequence_pool_sqrt:
                seq_pool_sqrt(dst_ptr, src_ptr, slice_num, slice_size);
                break;
            case Sequence_pool_max:
                seq_pool_max(dst_ptr, src_ptr, slice_num, slice_size);
                break;
            case Sequence_pool_first:
                seq_pool_first(dst_ptr, src_ptr, slice_num, slice_size);
                break;
            case Sequence_pool_last:
                seq_pool_last(dst_ptr, src_ptr, slice_num, slice_size);
                break;
            case Sequence_pool_unknow:
                seq_pool_unknow(dst_ptr, src_ptr, slice_num, slice_size);
                break;
        }
        dst_ptr += slice_size;
        src_ptr += slice_size * slice_num;
    }

}

static void ge_seq_offset(int num, std::vector<int>& seq_offset){
    int seg_num = 4;
    seq_offset.resize(seg_num + 1);
    for (int i = 0; i < seg_num; ++i){
        seq_offset[i] = i * num / seg_num;
    }
    seq_offset[seg_num] = num;
}

TEST(TestSaberFunc, test_func_sequence_pool){

#ifdef USE_CUDA
    LOG(INFO)<<"NV test......";
    //Init the test_base
    TestSaberBase<NV,NVHX86,AK_FLOAT,SequencePool, SequencePoolParam> testbase;
    for (auto num:{4,10,32}){
        for (auto c:{1, 30, 128}){
            for (auto h:{2, 32}){
                for (auto w:{2, 32}){
                    for (auto seq_type:{Sequence_pool_average, Sequence_pool_sum,\
                                       Sequence_pool_max, Sequence_pool_sqrt,\
                                       Sequence_pool_first, Sequence_pool_last}){
                        Tensor<NV> input;
                        input.re_alloc(Shape({num, c, h, w}), AK_FLOAT);
                        std::vector<int> seq_offset;
                        ge_seq_offset(num, seq_offset);
                        std::vector<std::vector<int>> vseq_offset;
                        vseq_offset.push_back(seq_offset);
                        input.set_seq_offset(vseq_offset);
                        std::vector<Tensor<NV>*> inputs;
                        inputs.push_back(&input);
                        testbase.add_custom_input(inputs);
                        SequencePoolParam<NV> param(seq_type);
                        testbase.set_param(param);
                        testbase.run_test(sequence_pool_cpu<float, NV, NVHX86>);
                    }
                }
            }
        }
    }
    LOG(INFO)<<"NV test end.";
#endif

#ifdef USE_X86_PLACE
    LOG(INFO)<<"x86 test......";
    do
    {
        //Init the test_base
        TestSaberBase<X86,X86,AK_FLOAT,SequencePool, SequencePoolParam> testbase;
        for (auto num:{4,10,32}){
            for (auto c:{1, 30, 128}){
                for (auto h:{2, 32}){
                    for (auto w:{2, 32}){
                        for (auto seq_type:{Sequence_pool_average, Sequence_pool_sum,\
                                           Sequence_pool_max, Sequence_pool_sqrt,\
                                           Sequence_pool_first, Sequence_pool_last}){
                            Tensor<X86> input;
                            input.re_alloc(Shape({num, c, h, w}), AK_FLOAT);
                            std::vector<int> seq_offset;
                            ge_seq_offset(num, seq_offset);
                            std::vector<std::vector<int>> vseq_offset;
                            vseq_offset.push_back(seq_offset);
                            input.set_seq_offset(vseq_offset);
                            std::vector<Tensor<X86>*> inputs;
                            inputs.push_back(&input);
                            testbase.add_custom_input(inputs);
                            SequencePoolParam<X86> param(seq_type);
                            testbase.set_param(param);
                            testbase.run_test(sequence_pool_cpu<float, X86, X86>);
                        }
                    }
                }
            }
        }
    }while(0);
    LOG(INFO)<<"x86 test end.";

#endif
}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
