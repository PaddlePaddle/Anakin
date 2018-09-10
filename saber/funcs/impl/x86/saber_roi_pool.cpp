#include "saber/funcs/impl/x86/saber_roi_pool.h"
#include "cuda_fp16.h"
#include <limits>

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberRoiPool<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    RoiPoolParam<X86>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    const OpDataType* in_rois = (const OpDataType*)inputs[1]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* out_index = nullptr;
    if (outputs.size() == 2) {
        out_index = (OpDataType*)outputs[1]->mutable_data();
    }
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    float spatial_scale = param.spatial_scale;
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        for(int n = 0; n < out_n; ++n){
            int in_index_n = in_rois[n * 5] * _in_n_stride;
            int in_w_start = round(in_rois[n * 5 + 1] * spatial_scale);
            int in_h_start = round(in_rois[n * 5 + 2] * spatial_scale);
            int in_w_end = round(in_rois[n * 5 + 3] * spatial_scale);
            int in_h_end = round(in_rois[n * 5 + 4] * spatial_scale);
            float roi_rate_w = (float)(in_w_end - in_w_start + 1) / out_w;
            float roi_rate_h = (float)(in_h_end - in_h_start + 1) / out_h;
            for(int c = 0; c < out_c; ++c){
                int in_index = in_index_n + c * _in_c_stride;
                for(int h = 0; h < out_h; ++h){
                    for(int w = 0; w < out_w; ++w){
                        int w_start = floor(w * roi_rate_w) + in_w_start;
                        int h_start = floor(h * roi_rate_h) + in_h_start;
                        int w_end = ceil((w+1) * roi_rate_w) + in_w_start;
                        int h_end = ceil((h+1) * roi_rate_h) + in_h_start;
                        w_end = w_end > in_w ? in_w : w_end;
                        h_end = h_end > in_h ? in_h : h_end;
                        int out_id = n * _out_n_stride + c * _out_c_stride + h * _out_h_stride + w * _out_w_stride;
                        bool is_empty = (h_start >= h_end) || (w_start >= w_end);
                        float max = is_empty ? 0.0f : std::numeric_limits<float>::min();
						int max_idx = -1;
                        for(int j = h_start; j < h_end; ++j){
                            for(int i = w_start; i < w_end; ++i){
								int in_id = in_index + i * _in_w_stride + j * _in_h_stride;
                                float data_in = in_data[in_id];
                                if(data_in > max){
                                    max = data_in;
				    				max_idx = in_id;
								}
                            }
                        }
                        out_data[out_id] = max;
                        if (out_index) {
                            out_index[out_id] = max_idx;
                        }
                    }
                }
            }
        }
    }
    return SaberSuccess;
}

}
}
