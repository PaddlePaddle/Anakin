#include "saber/funcs/impl/arm/saber_pad2d.h"

namespace anakin{

namespace saber{
void pad_constant(const float* din, float* dout, int n, int c, int h, int w, const int pad_top, \
                    const int pad_bottom, const int pad_left, const int pad_right, const float pad_value){
    int h_in = h - pad_top - pad_bottom;
    int w_in = w - pad_left - pad_right;
    int spatial_size_out = w * h;
    int spatial_size_in = h_in * w_in;
#pragma omp parallel for
    for (int s = 0; s < n * c; ++s){
        const float* din_s = din + s * spatial_size_in;
        float* dout_s = dout + s * spatial_size_out;
        int top_loop = (w * pad_top) >> 3;
        int top_loop_remain = (w * pad_top) & 7;
        float32x4_t vpad_value = vdupq_n_f32(pad_value);
        // process top
        for (int i = 0; i < top_loop; ++i){
            vst1q_f32(dout_s, vpad_value);
            vst1q_f32(dout_s + 4, vpad_value);
            dout_s += 8;
        }
        for (int i = 0; i < top_loop_remain; ++i){
            *dout_s++ = pad_value;
        }
        // process med
        int left_loop = pad_left >> 2;
        int left_loop_remain = pad_left & 3;
        int med_loop = w_in >> 3;
        int med_loop_remain = w_in & 7;
        for (int i = 0; i < left_loop; ++i){
            vst1q_f32(dout_s, vpad_value);
            dout_s += 4;
        }

        for (int i = 0; i < left_loop_remain; ++i){
            *dout_s++ = pad_value;
        }

        for (int i = 0; i < med_loop; ++i){
            float32x4_t val = vld1q_f32(din_s);
            float32x4_t val1 = vld1q_f32(din_s + 4);
            vst1q_f32(dout_s, val);
            vst1q_f32(dout_s + 4, val1);
            dout_s += 8;
            din_s += 8;
        }
        for (int i = 0; i < med_loop_remain; ++i){
            float val = *din_s++;
            *dout_s++ = val;
        }

        int loop = (pad_right + pad_left) >> 2;
        int loop_remain = (pad_right + pad_left) & 3;
        for (int j = 0; j < h_in - 1; ++j){
            for (int i = 0; i < loop; ++i){
                vst1q_f32(dout_s, vpad_value);
                dout_s += 4;
            }

            for (int i = 0; i < loop_remain; ++i){
                *dout_s++ = pad_value;
            }

            for (int i = 0; i < med_loop; ++i){
                float32x4_t val = vld1q_f32(din_s);
                float32x4_t val1 = vld1q_f32(din_s + 4);
                vst1q_f32(dout_s, val);
                vst1q_f32(dout_s + 4, val1);
                dout_s += 8;
                din_s += 8;
            }

            for (int i = 0; i < med_loop_remain; ++i){
                *dout_s++ = *din_s++;
            }
        }
        int right_loop = pad_right >> 2;
        int right_loop_remain = pad_right & 3;

        for (int i = 0; i < right_loop; ++i){
            vst1q_f32(dout_s, vpad_value);
            dout_s += 4;
        }

        for (int i = 0; i < right_loop_remain; ++i){
            *dout_s++ = pad_value;
        }
        //process bottom
        int bottom_loop = (pad_bottom * w) >> 3;
        int bottom_loop_remain = (pad_bottom * w) & 7;
        for (int i = 0; i < bottom_loop; ++i){
            vst1q_f32(dout_s, vpad_value);
            vst1q_f32(dout_s + 4, vpad_value);
            dout_s += 8;
        }
        for (int i = 0; i < bottom_loop_remain; ++i){
            *dout_s++ = pad_value;
        }
    }
}

void pad_edge(const float* din, float* dout, int n, int c, int h, int w, const int pad_top, \
                const int pad_bottom, const int pad_left, const int pad_right, const float pad_value) {
    int h_in = h - pad_top - pad_bottom;
    int w_in = w - pad_left - pad_right;
    int spatial_size_out = w * h;
    int spatial_size_in = h_in * w_in;
#pragma omp parallel for
    for (int s = 0; s < n * c; ++s){
        const float* din_s = din + s * spatial_size_in;
        float* dout_s = dout + s * spatial_size_out;

        // process med
        int left_loop = pad_left >> 2;
        int right_loop = pad_right >> 2;
        int med_loop = w_in >> 3;
        int med_loop_remain = w_in & 7;
        int left_loop_remain = pad_left & 3;
        int right_loop_remain = pad_right & 3;
        float* dout_med = dout_s + w * pad_top;
        for (int j = 0; j < h_in; ++j){
            float edge_val = din_s[0];
            float32x4_t vedge = vdupq_n_f32(edge_val);
            for (int i = 0; i < left_loop; ++i){
                vst1q_f32(dout_med, vedge);
                dout_med += 4;
            }
            for (int i = 0; i < left_loop_remain; ++i){
                *dout_med++ = edge_val;
            }
            for (int i = 0; i < med_loop; ++i){
                float32x4_t val = vld1q_f32(din_s);
                float32x4_t val1 = vld1q_f32(din_s + 4);
                vst1q_f32(dout_med, val);
                vst1q_f32(dout_med + 4, val1);
                din_s += 8;
                dout_med += 8;
            }
            for (int i = 0; i < med_loop_remain; ++i){
                *dout_med++ = *din_s++;
            }
            edge_val = din_s[-1];
            vedge = vdupq_n_f32(edge_val);
            for (int i = 0; i < right_loop; ++i){
                vst1q_f32(dout_med, vedge);
                dout_med += 4;
            }
            for (int i = 0; i < right_loop_remain; ++i){
                *dout_med++ = edge_val;
            }
        }

        //process bottom
        float* dout_bottom = dout_med;
        for (int i = 0; i < pad_bottom; ++i){
            memcpy(dout_bottom, dout_s + w * (pad_top + h_in - 1), w * sizeof(float));
            dout_bottom += w;
        }

        //process top
        float* dout_top = dout_s;
        for (int i = 0; i < pad_top; ++i){
            memcpy(dout_top, dout_s + w * pad_top, w * sizeof(float));
            dout_top += w;
        }
    }
}

void pad_reflect(const float* din, float* dout, int n, int c, int h, int w, const int pad_top, \
            const int pad_bottom, const int pad_left, const int pad_right, const float pad_value) {
    int h_in = h - pad_top - pad_bottom;
    int w_in = w - pad_left - pad_right;
    int spatial_size_out = w * h;
    int spatial_size_in = h_in * w_in;
#pragma omp parallel for
    for (int s = 0; s < n * c; ++s){
        const float* din_s = din + s * spatial_size_in;
        float* dout_s = dout + s * spatial_size_out;

        // process med
        int left_loop = pad_left >> 2;
        int right_loop = pad_right >> 2;
        int med_loop = w_in >> 3;
        int med_loop_remain = w_in & 7;
        int left_loop_remain = pad_left & 3;
        int right_loop_remain = pad_right & 3;
        float* dout_med = dout_s + w * pad_top;
        for (int j = 0; j < h_in; ++j) {


#ifdef __aarch64__
            for (int i = 0; i < left_loop; ++i){
                float32x4_t val = vld1q_f32(din_s + left_loop_remain + ((left_loop - i - 1) << 2) + 1);
                val = vrev64q_f32(val);
                float32x2_t low = vget_low_f32(val);
                float32x2_t high = vget_high_f32(val);
                float32x2_t tmp = low;
                low = high;
                high = tmp;
                float32x4_t val1 = vcombine_f32(low, high);
                vst1q_f32(dout_med, val1);
                dout_med += 4;
            }
#else
            const float* din_s_ptr = din_s + left_loop_remain + ((left_loop - 1) << 2) + 1;
            int cnt = left_loop;
            if (cnt > 0) {
                asm volatile(
                    "1:    \n"
                    "vld1.32 {d0-d1}, [%[din_s]]  \n"
                    "subs %[cnt], #1       \n"
                    "sub %[din_s], #16   \n"
                    "vrev64.32 q1, q0   \n"
                    "vswp d2, d3        \n"
                    "vst1.32 {d2-d3}, [%[dout_med]]!\n"
                    "bne 1b \n"
                    : [din_s] "+r"(din_s_ptr), [dout_med] "+r"(dout_med), [cnt] "+r"(cnt)
                    :
                    :"q0", "q1"
                    );
            }
#endif //__aarch64__
            for (int i = 0; i < left_loop_remain; ++i){
                *dout_med++ = *(din_s + left_loop_remain - i);
            }
            for (int i = 0; i < med_loop; ++i){
                float32x4_t val = vld1q_f32(din_s);
                float32x4_t val1 = vld1q_f32(din_s + 4);
                vst1q_f32(dout_med, val);
                vst1q_f32(dout_med + 4, val1);
                din_s += 8;
                dout_med += 8;
            }
            for (int i = 0; i < med_loop_remain; ++i){
                *dout_med++ = *din_s++;
            }
#ifdef __aarch64__
            for (int i = 0; i < right_loop; ++i){
                float32x4_t val = vld1q_f32(din_s - ((i + 1) << 2) - 1);
                val = vrev64q_f32(val);
                float32x2_t low = vget_low_f32(val);
                float32x2_t high = vget_high_f32(val);
                float32x2_t tmp = low;
                low = high;
                high = tmp;
                float32x4_t val1 = vcombine_f32(low, high);
                vst1q_f32(dout_med, val1);
                dout_med += 4;
            }
#else
            din_s_ptr = din_s - 5;
            cnt = right_loop;
            if (cnt > 0) {
                asm volatile(
                    "1:    \n"
                    "vld1.32 {d0-d1}, [%[din_s]]  \n"
                    "subs %[cnt], #1     \n"
                    "sub %[din_s], #16    \n"
                    "vrev64.32 q1, q0    \n"
                    "vswp d2, d3         \n"
                    "vst1.32 {d2-d3}, [%[dout_med]]!\n"
                    "bne 1b \n"
                    : [din_s] "+r"(din_s_ptr), [dout_med] "+r"(dout_med), [cnt] "+r"(cnt)
                    :
                    :"q0", "q1"
                    );
            }
#endif //__aarch64__
            const float* remain = din_s - (right_loop << 2) - 2;
            for (int i = 0; i < right_loop_remain; ++i){
                *dout_med++ = *remain--;
            }
        }

        //process bottom
        float* dout_bottom = dout_med;
        float* dout_bottom_reflect = dout_med - (w << 1);
        for (int i = 0; i < pad_bottom; ++i){
            memcpy(dout_bottom, dout_bottom_reflect, w * sizeof(float));
            dout_bottom += w;
            dout_bottom_reflect -= w;
        }

        //process top
        float* dout_top = dout_s;
        float* dout_top_reflect = dout_s + w * (pad_top << 1);
        for (int i = 0; i < pad_top; ++i){
            memcpy(dout_top, dout_top_reflect, w * sizeof(float));
            dout_top += w;
            dout_top_reflect -= w;
        }
    }
}
template <>
SaberStatus SaberPad2D<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        Pad2DParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif

    const float* din = (const float*)inputs[0]->data();
    float* dout = (float*)outputs[0]->mutable_data();

    if (_mode == PAD_CONSTANT){
        pad_constant(din, dout, outputs[0]->num(), outputs[0]->channel(), outputs[0]->height(), outputs[0]->width(),\
         _pad_h[0], _pad_h[1], _pad_w[0], _pad_w[1], _pad_value);
    } else if (_mode == PAD_EDGE){
        pad_edge(din, dout, outputs[0]->num(), outputs[0]->channel(), outputs[0]->height(), outputs[0]->width(),\
         _pad_h[0], _pad_h[1], _pad_w[0], _pad_w[1], _pad_value);
    } else if (_mode == PAD_REFLECT){
        pad_reflect(din, dout, outputs[0]->num(), outputs[0]->channel(), outputs[0]->height(), outputs[0]->width(),\
         _pad_h[0], _pad_h[1], _pad_w[0], _pad_w[1], _pad_value);
    } else {
        LOG(ERROR) <<"ERROR: unknown pad mode " << _mode;
    }
#ifdef ENABLE_OP_TIMER
   this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Pad2D : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("Pad2D", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberPad2D, Pad2DParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPad2D, Pad2DParam, ARM, AK_INT8);
//template class SaberPad2D<ARM, AK::FLOAT>;

} //namespace anakin

} //namespace anakin
