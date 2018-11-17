#include "saber/funcs/impl/x86/saber_im2sequence.h"

namespace anakin {

namespace saber {

/**
 * @brief Extract image patches from input tensor to a tensor with the shape 
 *                     [batch_size * output_h * ouput_w, window_h * window_w * channels]
 *                  output_h = (padding_up + padding_down + input_h - window_h)/strid_h + 1;
 *                  output_w = (padding_left + padding_right + input_w - windwo_w)/strid_w + 1;
 * @tparam OpDtype 
 */
template <DataType OpDtype>
SaberStatus SaberIm2Sequence<X86, OpDtype>::dispatch(\
        const std::vector<Tensor<X86> *>& inputs, \
        std::vector<Tensor<X86> *>& outputs, \
        Im2SequenceParam<X86>& param) {
     
     //brief for each channel:
     //get patches[kernel_extern_w * kernel_extern_h] to dst tensor util the channel has been finished.
    int out_rows_id = 0;
    int old_row;
    int out_cols = outputs[0]->channel();
    const OpDataType* input_ptr = (const OpDataType*)inputs[0]->data();
    OpDataType* output_ptr = (OpDataType*)outputs[0]->mutable_data();
    int H_pad = H + param.pad_up + param.pad_down;
    int W_pad = W + param.pad_left + param.pad_right;
    int wd_id = 0;
    int wd_num_each_channel = output_height * output_width;
    int wd_size = param.window_h * param.window_w;
    int m = 0; //the id which is mapped to the j th element of i th window
    int input_id;
    int st_id;
    int get_stride_h = param.dilation_h ? param.dilation_h : 1;
    int get_stride_w = param.dilation_w ? param.dilation_w : 1;
    for (int i = 0; i < N; i++) {
        wd_id = 0;
        out_rows_id = i * wd_num_each_channel + wd_id % wd_num_each_channel;
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < H_pad - kernel_extern_h + 1; k += param.stride_h) {
                for (int l = 0; l < W_pad - kernel_extern_w + 1; l += param.stride_w) {
                    m = 0;
                    //consider dilation.
                    for (int wd_h = k; wd_h < k + kernel_extern_h; wd_h += get_stride_h) {
                        for (int wd_w = l; wd_w < l + kernel_extern_w; wd_w += get_stride_w) {
                            input_id = i * C * H_pad * W_pad + j * H_pad * W_pad + wd_h * W_pad + wd_w;
                            st_id = out_rows_id * out_cols + j * wd_size + m;
                            output_ptr[st_id] = input_ptr[input_id];    
                            m++;
                        }
                    }
                    wd_id++;
                    out_rows_id = i * wd_num_each_channel + wd_id % wd_num_each_channel;
                }
            }
        }
    } 

    return SaberSuccess;
}
template class SaberIm2Sequence<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberIm2Sequence, Im2SequenceParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberIm2Sequence, Im2SequenceParam, X86, AK_INT8);
}
}
