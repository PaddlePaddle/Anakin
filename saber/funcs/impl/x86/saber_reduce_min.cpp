#include "saber/funcs/impl/x86/saber_reduce_min.h"

namespace anakin {
namespace saber {

template <typename dtype>
void reduce_n(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {
    
    int hw_size = height_in * width_in;
    int chw_size = channel_in * hw_size;
    int data_index, src_index, src_index0;
    for (int c = 0; c < channel_in; ++c) {
        for (int h = 0; h < height_in; ++h) {
            for (int w = 0; w < width_in; ++w) {
                data_index = c * hw_size + h * width_in + w;
                dst[data_index] = src[data_index];
                for (int n = 1; n < num_in; ++n) {
                    src_index = n * chw_size + data_index;
                    dst[data_index] = dst[data_index] < src[src_index]? dst[data_index] : src[src_index];
                }
            }
        }
    }
}

template <typename dtype>
void reduce_c(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {

    int hw_size = height_in * width_in;
    int chw_size = hw_size * channel_in;  
    int data_index, src_index0, src_index;
    for (int n = 0; n < num_in; ++n) {
        for (int h = 0; h < height_in; ++h) {
            for (int w = 0; w < width_in; ++w) {
                data_index = n * hw_size + h * width_in + w;
                src_index0 = n * chw_size + h * width_in + w; 
                dst[data_index] = src[src_index0];
                for (int c = 1; c < channel_in; ++c) {
                    src_index = src_index0 + c * hw_size;
                    dst[data_index] = dst[data_index] < src[src_index]? dst[data_index] : src[src_index];
                }
            }
        }
    }
}

template <typename dtype>
void reduce_h(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {

    int cw_size = channel_in * width_in;
    int chw_size = cw_size * height_in;
    int hw_size = height_in * width_in;
    int data_index, src_index, src_index0;
    for (int n = 0; n < num_in; ++n) {
        for (int c = 0; c < channel_in; ++c) {
            for (int w = 0; w < width_in; ++w) {
                data_index = n * cw_size + c * width_in + w;
                src_index0 = n * chw_size + c * hw_size + w;
                dst[data_index] = src[src_index0];
                for (int h = 1; h < height_in; ++h) {
                    src_index = src_index0 + h * width_in;
                    dst[data_index] = dst[data_index] < src[src_index]? dst[data_index] : src[src_index];
                }
            }
        }
    }
}

template <typename dtype>
void reduce_w(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {

    int ch_size = channel_in * height_in;
    int hw_size = height_in * width_in;
    int chw_size = ch_size * width_in;
    int data_index, src_index0, src_index;
    for (int n = 0; n < num_in; ++n) {
        for (int c = 0; c < channel_in; ++c) {
            for (int h = 0; h < height_in; ++h) {
                data_index = n * ch_size + c * height_in + h;
                src_index0 = n * chw_size + c * hw_size + h * width_in;
                dst[data_index] = src[src_index0];
                for (int w = 1; w < width_in; ++w) {
                    src_index = src_index0 + w;
                    dst[data_index] = dst[data_index] < src[src_index] ? dst[data_index] : src[src_index];
                }
            }
        }
    }
}

template <typename dtype>
void reduce_all(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {

    dtype min = src[0];
    int src_index;
    int n_id, c_id;
    for (int n = 0; n < num_in; ++n) {
        n_id = n * channel_in * height_in * width_in;
        for (int c = 0; c < channel_in; ++c) {
            c_id = c * height_in * width_in;
            for (int h = 0; h < height_in; ++h) {
                for (int w = 0; w < width_in; ++w) {
                    src_index = n_id + c_id + h * width_in + w;
                    min = src[src_index] < min? src[src_index] : min;
                }
            }
        }
    }
    dst[0] = min;
}

template <typename dtype>
void reduce_nc(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {
    
    //reduce n first. 
    Shape shape_tmp({1, channel_in, height_in, width_in});
    Tensor<X86> tensor_tmp(shape_tmp);
    dtype* tmp_out = (dtype*)tensor_tmp.mutable_data();
    reduce_n<dtype>(src, tmp_out, num_in, channel_in, height_in, width_in);
    reduce_c<dtype>(tmp_out, dst, 1, channel_in, height_in, width_in);
}

template <typename dtype>
void reduce_ch(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {
    //reduce c first
    Shape shape_tmp({num_in, 1, height_in, width_in});
    Tensor<X86> tensor_tmp(shape_tmp);
    dtype* tmp_out = (dtype*)tensor_tmp.mutable_data();
    reduce_c<dtype>(src, tmp_out, num_in, channel_in, height_in, width_in);
    reduce_h<dtype>(tmp_out, dst, num_in, 1, height_in, width_in); 
}

template <typename dtype>
void reduce_hw(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in) {
    //reduce h first
    Shape shape_tmp({num_in, channel_in, 1, width_in});
    Tensor<X86> tensor_tmp(shape_tmp);
    dtype* tmp_out = (dtype*)tensor_tmp.mutable_data();
    reduce_h<dtype>(src, tmp_out, num_in, channel_in, height_in, width_in);
    reduce_w<dtype>(tmp_out, dst, num_in, channel_in, 1, width_in); 
}

template <DataType OpDtype>
SaberStatus SaberReduceMin<X86, OpDtype>::dispatch(const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ReduceMinParam<X86>& param) {

    const OpDataType* input_ptr = (const OpDataType*)inputs[0]->data();
    OpDataType* output_ptr = (OpDataType*)outputs[0]->mutable_data();

    if (_reduce_dim.empty()) {
        //reduce all.
        reduce_all<OpDataType>(input_ptr, output_ptr, _n, _c, _h, _w);
    }else {
        if (_reduce_dim.size() == 1) {
            switch (_reduce_dim[0]) {
                case 0: reduce_n<OpDataType>(input_ptr, output_ptr, _n, _c, _h, _w); break;
                case 1: reduce_c<OpDataType>(input_ptr, output_ptr, _n, _c, _h, _w); break;
                case 2: reduce_h<OpDataType>(input_ptr, output_ptr, _n, _c, _h, _w); break;
                case 3: reduce_w<OpDataType>(input_ptr, output_ptr, _n, _c, _h, _w); break;
                default: LOG(FATAL) << "error!!!";
            }
        }else if (_reduce_dim.size() == 2) {
            if (_reduce_dim[0] == 0 && _reduce_dim[1] == 1) {
                reduce_nc<OpDataType>(input_ptr, output_ptr, _n, _c, _h, _w);
            }else if (_reduce_dim[0] == 1 && _reduce_dim[1] == 2) {
                reduce_ch<OpDataType>(input_ptr, output_ptr, _n, _c, _h, _w);
            }else if (_reduce_dim[0] == 2 && _reduce_dim[1] == 3) {
                reduce_hw<OpDataType>(input_ptr, output_ptr, _n, _c, _h, _w);
            }else {
                LOG(FATAL) <<"invalid reduce_dim!!";
            }
        } else {
            LOG(FATAL) << "reduce_dim's size over than 2, which is not supported now!!";
        }
    }
    

    return SaberSuccess;
}

template class SaberReduceMin<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberReduceMin, ReduceMinParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberReduceMin, ReduceMinParam, X86, AK_INT8);

} // namespace saber.
} // namespace anakin.