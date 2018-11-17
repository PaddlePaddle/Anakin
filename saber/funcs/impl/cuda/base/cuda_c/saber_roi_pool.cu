#include "saber/funcs/impl/cuda/saber_roi_pool.h"
#include "cuda_fp16.h"
#include <cfloat>

namespace anakin {

namespace saber {

template <typename Dtype>
__global__ void ker_roi_pool_fwd(Dtype * out_data, \
                    Dtype* out_index,
                    const Dtype* in_data,
                    const Dtype* in_rois,
                    const int in_n_stride,
                    const int in_c_stride,
                    const int in_h_stride, 
                    const int in_w_stride,
                    const int out_n_stride,
                    const int out_c_stride,
                    const int out_h_stride,
                    const int out_w_stride,
                    const Dtype spatial_scale,
                    const int in_n, 
                    const int in_c,
                    const int in_h,
                    const int in_w,
                    const int roi_num, 
                    const int roi_size,
                    const int out_h,
                    const int out_w,
                    const int num_threads)
{
    CUDA_KERNEL_LOOP(tid, num_threads){
        int n = (tid / out_n_stride);
        int c = (tid / out_c_stride) % in_c;
        int h = (tid / out_h_stride) % out_h;
        int w = (tid / out_w_stride) % out_w;
        const Dtype* cur_roi  = in_rois + n * roi_size;
        int roi_batch_id = cur_roi[0];
        int roi_start_w = round(cur_roi[1] * spatial_scale);
        int roi_start_h = round(cur_roi[2] * spatial_scale);
        int roi_end_w = round(cur_roi[3] * spatial_scale);
        int roi_end_h = round(cur_roi[4] * spatial_scale);
        int roi_width = roi_end_w - roi_start_w + 1;
        int roi_height = roi_end_h - roi_start_h + 1;
        Dtype pool_w_rate = (Dtype)roi_width / out_w;
        Dtype pool_h_rate = (Dtype)roi_height / out_h;

        int h_start = static_cast<int>(floor(static_cast<Dtype>(h) * pool_h_rate));
        int w_start = static_cast<int>(floor(static_cast<Dtype>(w) * pool_w_rate));
        int h_end = static_cast<int>(ceil(static_cast<Dtype>(h + 1) * pool_h_rate));
        int w_end = static_cast<int>(ceil(static_cast<Dtype>(w + 1) * pool_w_rate));
        h_start = fminf(fmaxf(h_start + roi_start_h, 0), in_h);
        h_end = fminf(fmaxf(h_end + roi_start_h, 0), in_h);
        w_start = fminf(fmaxf(w_start + roi_start_w, 0), in_w);
        w_end = fminf(fmaxf(w_end + roi_start_w, 0), in_w);
        bool is_empty = (h_end <= h_start) || (w_end <= w_start);
        Dtype max_val = is_empty ? 0 : -FLT_MAX;
        int max_idx = -1;
        const Dtype* in_tmp  =
            in_data + roi_batch_id * in_n_stride + c * in_c_stride;
        for (int h_id = h_start; h_id < h_end; ++h_id) {
            for (int w_id = w_start; w_id < w_end; ++w_id) {
                int input_data_index = h_id * in_h_stride + w_id * in_w_stride;
                Dtype data = in_tmp[input_data_index]; 
                if (data > max_val) {
                    max_val = data;
                    max_idx = input_data_index;
                }
            }
        }
        out_data[tid] = max_val;
        if (out_index) {
            out_index[tid] = max_idx;
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberRoiPool<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    RoiPoolParam<NV>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    const OpDataType* in_rois = (const OpDataType*)inputs[1]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* out_index = nullptr;
    if (outputs.size() == 2) {
        out_index = (OpDataType*)outputs[1]->mutable_data();
    }
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int in_n = inputs[0]->num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        ker_roi_pool_fwd<OpDataType>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                 out_data, out_index, in_data, in_rois, \
                 _in_n_stride, _in_c_stride, _in_h_stride, _in_w_stride,\
                 _out_n_stride, _out_c_stride, _out_h_stride, _out_w_stride,\
                 param.spatial_scale,
                 in_n, in_c, in_h, in_w,
                 out_n, 5, out_h, out_w, count);
    }
    return SaberSuccess;
}

}
}
