
#include "saber/funcs/impl/x86/saber_resize.h"
namespace anakin {
namespace saber {

template<typename dtype>
void resize_bilinear_custom_kernel(const int w_out, const int h_out,
                                 const int n_in, const int c_in,
                                 const int dst_stride_w,
                                 const int dst_stride_h,
                                 const int dst_stride_channel,
                                 const int dst_stride_batch,
                                 const int w_in, const int h_in,
                                 const int src_stride_w,
                                 const int src_stride_h,
                                 const int src_stride_channel,
                                 const int src_stride_batch,
                                 const float scale_w, const float scale_h,
                                 const dtype* src, dtype* dst){

#pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < h_out; ++h) {
        for (int w = 0; w < w_out; ++w) {
            dtype fw = w * scale_w;
            dtype fh = h * scale_h;
            int w_start = (int)fw;
            int w_end = (int)fw + 1;
            int h_start = (int)fh;
            int h_end = (int)fh + 1;
            fw -= w_start;
            fh -= h_start;
            const dtype w00 = (1.0 - fh) * (1.0 - fw);
            const dtype w01 = fw * (1.0 - fh);
            const dtype w10 = fh * (1.0 - fw);
            const dtype w11 = fw * fh;

            for (int n = 0; n < n_in; ++n) {
                for (int c = 0; c < c_in; ++c) {
                    int src_index = n * src_stride_batch + c * src_stride_channel;
                    dtype tl = src[src_index + w_start * src_stride_w + h_start * src_stride_h];
                    dtype tr = w_end >= w_in ? 0 : src[src_index + w_end * src_stride_w + h_start * src_stride_h];
                    dtype bl = h_end >= h_in ? 0 : src[src_index + w_start * src_stride_w + h_end * src_stride_h];
                    dtype br = (w_end >= w_in)
                               || (h_end >= h_in) ? 0 : src[src_index + w_end * src_stride_w + h_end * src_stride_h];
                    int dst_index = n * dst_stride_batch + c * dst_stride_channel + h * dst_stride_h + w * dst_stride_w;
                    dst[dst_index] = static_cast<dtype>(w00 * tl + w01 * tr + w10 * bl + w11 * br);
                }
            }
        }
    }
}

template<typename dtype>
void resize_bilinear_align_kernel(const int w_out, const int h_out,
                                 const int n_in,const int c_in,
                                 const int dst_stride_w,
                                 const int dst_stride_h,
                                 const int dst_stride_channel,
                                 const int dst_stride_batch,
                                 const int w_in, const int h_in,
                                 const int src_stride_w,
                                 const int src_stride_h,
                                 const int src_stride_channel,
                                 const int src_stride_batch,
                                 const float scale_w, const float scale_h,
                                 const dtype* src, dtype* dst){

    float scale_w_new = (float)(w_in - 1) / (w_out - 1);
    float scale_h_new = (float)(h_in - 1) / (h_out - 1);
#pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < h_out; ++h) {
        for (int w = 0; w < w_out; ++w) {
            dtype fw = w * scale_w_new;
            dtype fh = h * scale_h_new;
            int w_start = (int)fw;
            int w_id = w_start < w_in - 1 ? 1 : 0;
            int w_end = (int)fw + w_id;
            int h_start = (int)fh;
            int h_id = h_start < h_in - 1 ? 1 : 0;
            int h_end = (int)fh + h_id;
            fw -= w_start;
            fh -= h_start;
            const dtype w00 = (1.0 - fh) * (1.0 - fw);
            const dtype w01 = fw * (1.0 - fh);
            const dtype w10 = fh * (1.0 - fw);
            const dtype w11 = fw * fh;

            for (int n = 0; n < n_in; ++n) {
                for (int c = 0; c < c_in; ++c) {
                    int src_index = n * src_stride_batch + c * src_stride_channel;
                    dtype tl = src[src_index + w_start * src_stride_w + h_start * src_stride_h];
                    dtype tr = src[src_index + w_end * src_stride_w + h_start * src_stride_h];
                    dtype bl = src[src_index + w_start * src_stride_w + h_end * src_stride_h];
                    dtype br = src[src_index + w_end * src_stride_w + h_end * src_stride_h];
                    int dst_index = n * dst_stride_batch + c * dst_stride_channel + h * dst_stride_h + w * dst_stride_w;
                    dst[dst_index] = static_cast<dtype>(w00 * tl + w01 * tr + w10 * bl + w11 * br);
                }
            }
        }
    }
}

template<typename dtype>
void resize_bilinear_no_align_kernel(const int w_out, const int h_out,
                                 const int n_in,const int c_in,
                                 const int dst_stride_w,
                                 const int dst_stride_h,
                                 const int dst_stride_channel,
                                 const int dst_stride_batch,
                                 const int w_in, const int h_in,
                                 const int src_stride_w,
                                 const int src_stride_h,
                                 const int src_stride_channel,
                                 const int src_stride_batch,
                                 const float scale_w, const float scale_h,
                                 const dtype* src, dtype* dst){
    
    float scale_w_new = (float)w_in / w_out;
    float scale_h_new = (float)h_in / h_out;
#pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < h_out; ++h) {
        for (int w = 0; w < w_out; ++w) {
            dtype fw = scale_w_new * (w + 0.5f) - 0.5f;
            dtype fh = scale_h_new * (h + 0.5f) - 0.5f;
            fw = fw < 0 ? 0 : fw;
            fh = fh < 0 ? 0 : fh;
            int w_start = (int)fw;
            int w_id = w_start < w_in - 1 ? 1 : 0;
            int w_end = (int)fw + w_id;
            int h_start = (int)fh;
            int h_id = h_start < h_in - 1 ? 1 : 0;
            int h_end = (int)fh + h_id;
            fw -= w_start;
            fh -= h_start;
            const dtype w00 = (1.0 - fh) * (1.0 - fw);
            const dtype w01 = fw * (1.0 - fh);
            const dtype w10 = fh * (1.0 - fw);
            const dtype w11 = fw * fh;

            for (int n = 0; n < n_in; ++n) {
                for (int c = 0; c < c_in; ++c) {
                    int src_index = n * src_stride_batch + c * src_stride_channel;
                    dtype tl = src[src_index + w_start * src_stride_w + h_start * src_stride_h];
                    dtype tr = src[src_index + w_end * src_stride_w + h_start * src_stride_h];
                    dtype bl = src[src_index + w_start * src_stride_w + h_end * src_stride_h];
                    dtype br = src[src_index + w_end * src_stride_w + h_end * src_stride_h];
                    int dst_index = n * dst_stride_batch + c * dst_stride_channel + h * dst_stride_h + w * dst_stride_w;
                    dst[dst_index] = static_cast<dtype>(w00 * tl + w01 * tr + w10 * bl + w11 * br);
                }
            }
        }
    }
}
template<typename dtype, bool align>
void resize_nearest_kernel(const int w_out, const int h_out,
                                 const int n_in,const int c_in,
                                 const int dst_stride_w,
                                 const int dst_stride_h,
                                 const int dst_stride_channel,
                                 const int dst_stride_batch,
                                 const int w_in, const int h_in,
                                 const int src_stride_w,
                                 const int src_stride_h,
                                 const int src_stride_channel,
                                 const int src_stride_batch,
                                 const float scale_w, const float scale_h,
                                 const dtype* src, dtype* dst){
    
    float scale_w_new = (float)(w_in - 1) / (w_out - 1);
    float scale_h_new = (float)(h_in - 1) / (h_out - 1);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < h_out; ++h) {
        for (int w = 0; w < w_out; ++w) {

            int near_x = static_cast<int>(scale_w_new * w + 0.5);
            int near_y = static_cast<int>(scale_h_new * h + 0.5);
            near_x = near_x < 0 ? 0 : near_x;
            near_y = near_y < 0 ? 0 : near_y;
            

            for (int n = 0; n < n_in; ++n) {
                for (int c = 0; c < c_in; ++c) {
                    int src_index = n * src_stride_batch + c * src_stride_channel;
                    int dst_index = n * dst_stride_batch + c * dst_stride_channel + h * dst_stride_h + w * dst_stride_w;
                    dst[dst_index] = src[src_index + near_y * src_stride_h + near_x * src_stride_w];
                }
            }
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberResize<X86, OpDtype>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    ResizeParam<X86>& param) {
    typedef typename DataTrait<X86, OpDtype>::Dtype InDataType;
    typedef typename DataTrait<X86, OpDtype>::Dtype OutDataType;
    typedef typename DataTrait<X86, OpDtype>::Dtype dtype;

    int w_out = outputs[0]->width();
    int h_out = outputs[0]->height();
    int c_out = outputs[0]->channel();
    int n_out = outputs[0]->num();

    if (inputs.size() > 1){
        int* out_size_data = static_cast<int*>(inputs[1]->data());
        h_out = out_size_data[0];
        w_out = out_size_data[1];
        outputs[0]->reshape(Shape({n_out, c_out, h_out, w_out}));
    }

    int w_in = inputs[0]->width();
    int h_in = inputs[0]->height();
    int c_in = inputs[0]->channel();
    int n_in = inputs[0]->num();

    int num_idx = inputs[0]->num_index();
    int channel_idx = inputs[0]->channel_index();
    int height_idx = inputs[0]->height_index();
    int width_idx = inputs[0]->width_index();

    int dims = inputs[0]->dims();

    CHECK_EQ(c_in, c_out) << "input channel should = output channel";
    CHECK_EQ(c_in, c_out) << "input batch size should = output batch size";


    const InDataType* src = (const InDataType*)inputs[0]->data();
    OutDataType* dst = (OutDataType*)outputs[0]->mutable_data();
    Shape src_real_shape;
    Shape dst_real_shape;

    if (inputs[0]->is_continue_mem()) {
        src_real_shape = inputs[0]->valid_shape();
    } else {
        src_real_shape = inputs[0]->shape();
    }

    if (outputs[0]->is_continue_mem()) {
        dst_real_shape = outputs[0]->valid_shape();
    } else {
        dst_real_shape = outputs[0]->shape();
    }

    float scale_w = 0.f;
    float scale_h = 0.f;
    if (param.out_width != -1 && param.out_height != -1){
        scale_w = (float)param.out_width / w_in;
        scale_h = (float)param.out_height / h_in;
    } else {
        scale_w = param.width_scale;
        scale_h = param.height_scale;
    }
    int src_stride_w = src_real_shape.count(width_idx + 1);
    int src_stride_h = src_real_shape.count(height_idx + 1);
    int src_stride_channel = src_real_shape.count(channel_idx + 1);
    int src_stride_batch = src_real_shape.count(num_idx + 1);
    int dst_stride_w = dst_real_shape.count(width_idx + 1);
    int dst_stride_h = dst_real_shape.count(height_idx + 1);
    int dst_stride_channel = dst_real_shape.count(channel_idx + 1);
    int dst_stride_batch = dst_real_shape.count(num_idx + 1);

    switch (param.resize_type){
        case BILINEAR_ALIGN:
            resize_bilinear_align_kernel<dtype>(w_out, h_out, n_in, c_in, dst_stride_w, dst_stride_h, \
                                 dst_stride_channel, dst_stride_batch, w_in, h_in, src_stride_w, src_stride_h, \
                                 src_stride_channel, src_stride_batch, 1.f / scale_w, 1.f / scale_h, src, dst);
            break;
        case BILINEAR_NO_ALIGN:
            resize_bilinear_no_align_kernel<dtype>(w_out, h_out, n_in, c_in, dst_stride_w, dst_stride_h, \
                                     dst_stride_channel, dst_stride_batch, w_in, h_in, src_stride_w, src_stride_h, \
                                     src_stride_channel, src_stride_batch, 1.f / scale_w, 1.f / scale_h, src, dst);
            break;
        case RESIZE_CUSTOM:
            resize_bilinear_custom_kernel<dtype>(w_out, h_out, n_in, c_in, dst_stride_w, dst_stride_h, \
                                 dst_stride_channel, dst_stride_batch, w_in, h_in, src_stride_w, src_stride_h, \
                                 src_stride_channel, src_stride_batch, 1.f / scale_w, 1.f / scale_h, src, dst);
            break;
        case NEAREST_ALIGN:
            resize_nearest_kernel<dtype, true>(w_out, h_out, n_in, c_in, dst_stride_w, dst_stride_h, \
                                 dst_stride_channel, dst_stride_batch, w_in, h_in, src_stride_w, src_stride_h, \
                                 src_stride_channel, src_stride_batch, 1.f / scale_w, 1.f / scale_h, src, dst);
            break;
        default:
            LOG(FATAL) << "Unsupport resize type: " << (int)param.resize_type;
    }


    return SaberSuccess;
}

template class SaberResize<X86, AK_FLOAT>;

}
} // namespace anakin
