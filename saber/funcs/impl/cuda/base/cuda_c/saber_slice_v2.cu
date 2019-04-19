#include "saber/funcs/impl/cuda/saber_slice_v2.h"

namespace anakin{

namespace saber{

template <typename dtype>
__global__ void slice_v2_impl_cuda(const int count, const dtype* in_data,
                                const int* in_stride_data, 
                                const int* out_shape_data,
                                const int* starts_data,
                                const int* axes_data,
                                const int dims,
                                const int start_size,
                                const int in_outer_stride,
                                const int out_outer_stride,
                                const int inner,
                                dtype* out_data) {
    CUDA_KERNEL_LOOP(tid, count) {
        int inner_id = tid % inner;
        int out_id = tid / out_outer_stride;
        int in_offset  = inner_id + out_id * in_outer_stride;
        int new_i = tid / inner;
        for (int k = start_size - 1; k >= 0; k--) {
            int axes_id = axes_data[k];
            int cur_id = new_i % out_shape_data[axes_id];
            in_offset += (cur_id + starts_data[k]) * in_stride_data[axes_id];
            new_i /= out_shape_data[axes_id];
        }
        
        out_data[tid] = in_data[in_offset];
    }
}

template <DataType OpDtype>
SaberStatus SaberSliceV2<NV, OpDtype>::create(const std::vector<Tensor<NV>*>& inputs,
                    std::vector<Tensor<NV>*>& outputs,
                    SliceV2Param<NV> &param,
                    Context<NV> &ctx) {
    auto starts = param.starts;
    auto ends = param.ends;
    auto axes = param.axes;
    CHECK_EQ(axes.size(), starts.size()) << "the size of axes and starts are not equal ";
    CHECK_EQ(ends.size(), starts.size()) << "the size of starts and ends are not valid";
    std::vector<int> starts_h;
    std::vector<int> ends_h;
    starts_h.resize(starts.size());
    ends_h.resize(ends.size());
    Shape output_shape = inputs[0]->valid_shape();
    for (int i = 0; i < starts.size(); i++) {
        int dim_value = output_shape[axes[i]];
        int start = starts[i] < 0 ? starts[i] + dim_value : starts[i];
        int end = ends[i] < 0 ? ends[i] + dim_value : ends[i];
        start = std::max(start, 0);
        start = std::min(start, dim_value);
        end = std::max(end, 0);
        end = std::min(end, dim_value);
        output_shape[axes[i]] = end - start;
        starts_h[i] = start;
        ends_h[i] = end;
    }
    auto in_stride = inputs[0]->get_stride();
    auto out_stride = outputs[0]->get_stride();
    Shape stride_shape({inputs[0]->dims(), 1, 1, 1}, Layout_NCHW);
    _in_stride_d.re_alloc(stride_shape, AK_INT32);
    _out_shape_d.re_alloc(stride_shape, AK_INT32);
    int starts_size = param.starts.size();
    Shape start_shape({starts_size, 1, 1, 1}, Layout_NCHW);
    _starts_d.re_alloc(start_shape, AK_INT32);
    _axes_d.re_alloc(start_shape, AK_INT32);
    int* in_stride_data = (int*)_in_stride_d.mutable_data();
    int* out_shape_data = (int*)_out_shape_d.mutable_data();
    int* starts_data = (int*)_starts_d.mutable_data();
    int* axes_data = (int*)_axes_d.mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    cudaMemcpyAsync(in_stride_data, &in_stride[0], sizeof(int) * in_stride.size(),            cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(out_shape_data, &output_shape[0], sizeof(int) * output_shape.size() ,           cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(starts_data, &starts_h[0], sizeof(int) * starts_size, 
            cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(axes_data, &param.axes[0], sizeof(int) * starts_size,
            cudaMemcpyHostToDevice, cuda_stream);
    return SaberSuccess;
}


template <DataType OpDtype>
SaberStatus SaberSliceV2<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    SliceV2Param<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();
    //! inputs only has one tensor
    Shape shape_in = inputs[0]->valid_shape();

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    int* in_stride_data = (int*)_in_stride_d.mutable_data();
    int* out_shape_data = (int*)_out_shape_d.mutable_data();
    int* starts_data = (int*)_starts_d.mutable_data();
    int* axes_data = (int*)_axes_d.mutable_data();
    const int count = outputs[0]->valid_size();
    int inner = inputs[0]->count_valid(param.axes.back() + 1, inputs[0]->dims());
    int out_outer_stride = outputs[0]->count_valid(param.axes[0], outputs[0]->dims());
    int in_outer_stride = inputs[0]->count_valid(param.axes[0], inputs[0]->dims());
    int start_size = param.starts.size();
    slice_v2_impl_cuda<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
            count, in_data, in_stride_data, out_shape_data,
            starts_data, axes_data, inputs[0]->dims(), start_size, 
            in_outer_stride, out_outer_stride,
            inner, out_data);
    return SaberSuccess;

}
DEFINE_OP_TEMPLATE(SaberSliceV2, SliceV2Param, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSliceV2, SliceV2Param, NV, AK_INT8);
} //namespace anakin

} //namespace anakin
