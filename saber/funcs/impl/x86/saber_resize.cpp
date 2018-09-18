
#include "saber/funcs/impl/x86/saber_resize.h"
namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberResize<X86, OpDtype>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ResizeParam<X86> &param)
{
    typedef typename DataTrait<X86, OpDtype>::Dtype InDataType;
    typedef typename DataTrait<X86, OpDtype>::Dtype OutDataType;
    typedef typename DataTrait<X86, OpDtype>::Dtype dtype;

    int w_out = outputs[0]->width();
    int h_out = outputs[0]->height();
    int c_out = outputs[0]->channel();
    int n_out = outputs[0]->num();

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

    int src_stride_w = src_real_shape.count(width_idx + 1);
    int src_stride_h = src_real_shape.count(height_idx + 1);
    int src_stride_channel = src_real_shape.count(channel_idx + 1);
    int src_stride_batch = src_real_shape.count(num_idx + 1);
    int dst_stride_w = dst_real_shape.count(width_idx + 1);
    int dst_stride_h = dst_real_shape.count(height_idx + 1);
    int dst_stride_channel = dst_real_shape.count(channel_idx + 1);
    int dst_stride_batch = dst_real_shape.count(num_idx + 1);
    float scale_w = 1. / param.width_scale;
    float scale_h = 1. / param.height_scale;
    for(int n = 0; n < n_in; ++n){
        for(int c = 0; c < c_in; ++c){
            int src_index = n * src_stride_batch + c * src_stride_channel;
            for(int h = 0; h < h_out; ++h){
                for(int w = 0; w < w_out; ++w){
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
                    dtype tl = src[src_index + w_start * src_stride_w + h_start * src_stride_h];
                    dtype tr = w_end >= w_in ? 0 : src[src_index + w_end * src_stride_w + h_start * src_stride_h];
                    dtype bl = h_end >= h_in ? 0 : src[src_index + w_start * src_stride_w + h_end * src_stride_h];
                    dtype br = (w_end >= w_in) || (h_end >= h_in) ? 0 : src[src_index + w_end * src_stride_w + h_end * src_stride_h];
                    int dst_index = n * dst_stride_batch + c * dst_stride_channel + h * dst_stride_h + w * dst_stride_w;
                    dst[dst_index] = static_cast<dtype>(w00 * tl + w01 * tr + w10 * bl + w11 * br);
                }
            }
        }
    }
    return SaberSuccess;
}

template class SaberResize<X86, AK_FLOAT>;

}
} // namespace anakin
