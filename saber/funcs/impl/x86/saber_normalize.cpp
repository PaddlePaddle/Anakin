#include "saber/funcs/impl/x86/saber_normalize.h"

namespace anakin{
namespace saber{

template class SaberNormalize<X86, AK_FLOAT>;
    
template <>
SaberStatus SaberNormalize<X86, AK_FLOAT>::\
        dispatch(const std::vector<Tensor<X86> *>& inputs,
                        std::vector<Tensor<X86> *>& outputs,
                        NormalizeParam<X86> &param){
            int p = param.p;
            bool across_spatial = param.across_spatial;
            bool has_scale = param.has_scale;
            bool channel_shared = param.channel_shared;
            float eps = param.eps;
            int n = inputs[0]->num();
            int c = inputs[0]->channel();
            int h = inputs[0]->height();
            int w = inputs[0]->width();
            Tensor<X86> th_scale;
            const float* scale;
            if(has_scale){
                th_scale.re_alloc(param.scale->shape(), AK_FLOAT);
                th_scale.copy_from(*param.scale);
                scale=static_cast<float*>(th_scale.data());
            }
            const float* src_ptr = static_cast<const float*>(inputs[0]->data());
            float* dst_ptr = static_cast<float*>(outputs[0]->mutable_data());
            
            if (across_spatial) {
                int compute_size = h * w * c;
                int outer_size = n * c * h * w / compute_size;
                
                for (int i = 0; i < outer_size; ++i) {
                    float sum = 0;
                    
                    for (int j = 0; j < compute_size; ++j) {
                        if (p == 1) {
                            sum += fabsf(src_ptr[j]);
                        } else {
                            sum += src_ptr[j] * src_ptr[j];
                        }
                    }
                    
                    if (p == 1) {
                        sum = 1 / (sum + eps);
                    } else {
                        sum = 1 / sqrtf(sum+eps);
                    }
                    
                    if (has_scale) { //! with scale
                        if (channel_shared) { // scale is shared across channel
                            for (int j = 0; j < compute_size; ++j) {
                                dst_ptr[j] = src_ptr[j] * sum * scale[0];
                            }
                        } else {
                            for (int j = 0; j < compute_size; ++j) {
                                int c_idx = j / (h * w);
                                dst_ptr[j] = src_ptr[j] * sum * scale[c_idx];
                            }
                        }
                    } else { //! without scale
                        for (int j = 0; j < compute_size; ++j) {
                            dst_ptr[j] = src_ptr[j] * sum;
                        }
                    }
                    
                    src_ptr += compute_size;
                    dst_ptr += compute_size;
                }
            } else {
                int channel_in_size = h * w;
                
                for (int i = 0; i < n; ++i) {
                    const float* src_batch_ptr = src_ptr + i * c * h * w;
                    float* dst_batch_ptr = dst_ptr + i * c * h * w;
                    
                    for (int j = 0; j < h; ++j) {
                        for (int k = 0; k < w; ++k) {
                            const float* src_pixel = src_batch_ptr + j * w + k;
                            float* dst_pixel = dst_batch_ptr  + j * w + k;
                            float norm = 0.f;
                            //LOG(INFO)<<"c:"<<c;
                            
                            for (int l = 0; l < c; ++l) {
                                if (p == 1) {
                                    norm += fabsf(src_pixel[l * channel_in_size]);
                                } else {
                                    norm += src_pixel[l * channel_in_size] * src_pixel[l * channel_in_size];
                                }
                            }
                            //LOG(INFO)<<"norm:"<<norm;
                            
                            if (p == 1) {
                                norm = 1.f / (norm + eps);
                            } else {
                                norm = 1.f / sqrtf(norm+eps);
                            }
                            
                            for (int l = 0; l < c; ++l) {
                                if (has_scale) {
                                    if (channel_shared) {
                                        dst_pixel[l * channel_in_size] = \
                                        src_pixel[l * channel_in_size] * norm * scale[0];
                                    } else {
                                        dst_pixel[l * channel_in_size] = \
                                        src_pixel[l * channel_in_size] * norm * scale[l];
                                    }
                                } else {
                                    dst_pixel[l * channel_in_size] = \
                                    src_pixel[l * channel_in_size] * norm;
                                    
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
