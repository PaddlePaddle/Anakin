#include "core/common.h"
#include "saber/funcs/impl/cuda/saber_sequence_pool.h"
#include "saber/saber_funcs_param.h"
#include "cuda.h"
namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void seq_pool_average_kernel(Dtype* dst, const Dtype* src_in,const int batch_size,
                             const int* seq_offset, const int slice_size){
    const int batch_id=blockIdx.x;
    const int tid=threadIdx.x;
    if(tid<slice_size){
        Dtype sum=0;
        int slice_num=seq_offset[batch_id+1]-seq_offset[batch_id];

        const Dtype* data_in=src_in+seq_offset[batch_id]*slice_size;
        Dtype* data_out=dst+batch_id*slice_size;
        for(int i=0;i<slice_num;i++){
            sum+=data_in[i*slice_size+tid];
        }
        data_out[tid]=sum/slice_num;
    }
}


template <typename dtype>
void seq_pool_average(dtype* dst, const dtype* src_in,const int batch_size,
                      const int* seq_offset, const int slice_size) {
    int grid_dim=batch_size;
    int block_dim=slice_size;
    CHECK_LE(block_dim,1024)<<"slice_size should <= 1024";
    seq_pool_average_kernel<<<grid_dim,block_dim>>>(dst,src_in,batch_size,seq_offset,slice_size);
}


template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberSequencePool<NV, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::init(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
SequencePoolParam<OpTensor>& param, Context<NV>& ctx) {

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;
    kernel_direct_map = {
        {
            Sequence_pool_unknow, [](
                DataType_out* , const DataType_in* ,const int ,
                const int* , const int) {
                LOG(ERROR) << " UNKNOWN seq pool type";
            }
        },

        {Sequence_pool_average, seq_pool_average<DataType_in>},
//        {Sequence_pool_sum, seq_pool_sum<DataType_in>},
//        {Sequence_pool_sqrt, seq_pool_sqrt<DataType_in>},
//        {Sequence_pool_max, seq_pool_max<DataType_in>},

//        {
//            Sequence_pool_last, [](
//                DataType_in * dst, const DataType_in * src_in,
//                const int slice_num, const int slice_size) {
//                memcpy(dst, src_in + slice_size * (slice_num - 1),
//                       sizeof(DataType_in)* slice_size);
//            }
//        },
//        {
//            Sequence_pool_first, [](
//                DataType_in * dst, const DataType_in * src_in,
//                const int slice_num, const int slice_size) {
//                memcpy(dst, src_in, sizeof(DataType_in)* slice_size);
//            }
//        },

    };
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberSequencePool<NV, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::create(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                SequencePoolParam<OpTensor>& param,
Context<NV>& ctx) {
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    this->_ctx = &ctx;

    return SaberSuccess;
}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberSequencePool<NV, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
SequencePoolParam<OpTensor>& param) {
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    CHECK_EQ(inputs[0]->channel(), outputs[0]->channel());
    CHECK_EQ(inputs[0]->height(), outputs[0]->height());
    CHECK_EQ(inputs[0]->width(), outputs[0]->width());

    std::vector<int> seq_offset = inputs[0]->get_seq_offset();
    int slice_size = outputs[0]->channel()
                     * outputs[0]->height()
                     * outputs[0]->width();

    DataType_in* dst_ptr = outputs[0]->mutable_data();
    const DataType_out* src_ptr = inputs[0]->data();
    int batch_size=seq_offset.size()-1;

    _seq_offset.try_expand_size(seq_offset.size());
    CUDA_CHECK(cudaMemcpyAsync(_seq_offset.mutable_data(),seq_offset.data(),sizeof(int)*seq_offset.size(),cudaMemcpyDeviceToDevice,this->_ctx->get_compute_stream()));

    kernel_direct_map[param.sequence_pool_type](dst_ptr,src_ptr,batch_size,_seq_offset.data(),slice_size);

    std::vector<int> offset_new(batch_size + 1);

    for (int i = 0; i <= batch_size; ++i) {
        offset_new[i] = i;
    }

    outputs[0]->set_seq_offset(offset_new);
    return SaberSuccess;

}

template class SaberSequencePool<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}
} // namespace anakin
