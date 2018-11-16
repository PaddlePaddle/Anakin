#include "saber/lite/funcs/calibrate_lite.h"
#include "saber/lite/funcs/neon/impl/sgemm_prepacked_int8.h"
#include "saber/lite/core/tensor_op_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberStatus get_tensor_scale_n(const float* in_data, std::vector<float>& scale_out, \
    int axis_dims, long long inner_dims, float scale_factor) {
    int cnt = inner_dims / 16;
    int remain = inner_dims % 16;
#pragma omp parallel for
    for (int c = 0; c < axis_dims; ++c) {//num
        float32x4_t vmax_val = vdupq_n_f32(0.f);
        float max_value = 0.f;
        const float* ptr_in =  in_data + c * inner_dims;//channel*width*height

        if (cnt > 0) {
            int cnt_loop = cnt;
#ifdef __aarch64__
            asm volatile(
                    "ld1 {v0.4s, v1.4s}, [%[in]], #32               \n"
                    "ld1 {v2.4s, v3.4s}, [%[in]], #32               \n"
                "1:                                                 \n"
                    "fabs v4.4s, v0.4s                              \n"
                    "fabs v5.4s, v1.4s                              \n"
                    "fabs v6.4s, v2.4s                              \n"
                    "fabs v7.4s, v3.4s                              \n"

                    "fmax v0.4s, v4.4s, v5.4s                     \n"
                    "fmax v1.4s, v6.4s, v7.4s                     \n"

                    "PRFM PLDL1KEEP, [%[in]]                      \n"

                    "fmax v2.4s, v0.4s, v1.4s                     \n"
                    "ld1 {v0.4s, v1.4s}, [%[in]], #32               \n"

                    "fmax %[max_val].4s, v2.4s, %[max_val].4s     \n"
                    "subs %[cnt], %[cnt], #1                      \n"

                    "ld1 {v2.4s, v3.4s}, [%[in]], #32               \n"

                    "bne    1b                                 \n"
            : [in] "+r" (ptr_in), [cnt] "+r" (cnt_loop), [max_val] "+w" (vmax_val)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"
            );
#else
            asm volatile(
                    "vld1.32   {d0-d3}, [%[in]]!              @ load 8 float\n"
                    "vld1.32   {d4-d7}, [%[in]]!              @ load 8 float\n"
                "1:                                                @ main loop\n"
                    "vabs.f32 q4, q0                           @ abs \n"
                    "vabs.f32 q5, q1                           @ abs \n"
                    "vabs.f32 q6, q2                           @ abs \n"
                    "vabs.f32 q7, q3                           @ abs \n"

                    "vmax.f32 q0, q4, q5                           @ max \n"
                    "vmax.f32 q1, q6, q7                           @ max \n"

                    "pld [%[in]]                                   @ preload data \n"
                    "vmax.f32 q2, q0, q1                           @ max \n"

                    "vld1.32   {d0-d3}, [%[in]]!              @ load 8 float\n"

                    "vmax.f32 %q[max_val], q2, %q[max_val]     @ max \n"
                    "subs %[cnt], #1                           @ loop count -1\n"
                    "vld1.32   {d4-d7}, [%[in]]!              @ load 8 float\n"

                    "bne    1b                                 @ jump to main loop\n"

            : [in] "+r" (ptr_in), [cnt] "+r" (cnt_loop), [max_val] "+w" (vmax_val)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"
            );
            // max = max_val[0];
#endif
            float32x2_t vmax_p = vpmax_f32(vget_high_f32(vmax_val), vget_low_f32(vmax_val));
            float max0 = vget_lane_f32(vmax_p, 0);
            float max1 = vget_lane_f32(vmax_p, 1);
            float max2 = max0 > max1 ? max0 : max1;
            max_value = max_value > max2 ? max_value : max2;
        }
        ptr_in = in_data + c * inner_dims + 16 * cnt;
        for (int i = 0; i < remain; ++i) {
            float data = fabsf(*(ptr_in++));
            max_value = fmaxf(max_value, data);
        }
        scale_out[c] = max_value / scale_factor;
    }
    return SaberSuccess;
}

SaberStatus get_tensor_scale_chw(const float* in_data, std::vector<float>& scale_out, \
    int axis_dims, int outer_dims, long long inner_dims, long long inner_size, float scale_factor) {
    int cnt = inner_dims / 16;
    int remain = inner_dims % 16;
#pragma omp parallel for
    for (int c = 0; c < axis_dims; ++c) {
        float32x4_t vmax_val = vdupq_n_f32(0.f);
        float max_value = 0.f;
        const float* din = in_data + c * inner_dims;
        for (int j = 0; j < outer_dims; ++j) {
            const float* ptr_in = din + j * inner_size;
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ld1 {v0.4s, v1.4s}, [%[in]], #32               \n"
                    "ld1 {v2.4s, v3.4s}, [%[in]], #32               \n"
                    "1:                                                 \n"
                    "fabs v4.4s, v0.4s                              \n"
                    "fabs v5.4s, v1.4s                              \n"
                    "fabs v6.4s, v2.4s                              \n"
                    "fabs v7.4s, v3.4s                              \n"

                    "fmax v0.4s, v4.4s, v5.4s                     \n"
                    "fmax v1.4s, v6.4s, v7.4s                     \n"

                    "PRFM PLDL1KEEP, [%[in]]                      \n"

                    "fmax v2.4s, v0.4s, v1.4s                     \n"
                    "ld1 {v0.4s, v1.4s}, [%[in]], #32               \n"

                    "fmax %[max_val].4s, v2.4s, %[max_val].4s     \n"
                    "subs %[cnt], %[cnt], #1                      \n"

                    "ld1 {v2.4s, v3.4s}, [%[in]], #32               \n"

                    "bne    1b                                 \n"
                : [in] "+r" (ptr_in), [cnt] "+r" (cnt_loop), [max_val] "+w" (vmax_val)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"
                );
#else
                asm volatile(
                    "vld1.32   {d0-d3}, [%[in]]!              @ load 8 float\n"
                    "vld1.32   {d4-d7}, [%[in]]!              @ load 8 float\n"
                    "1:                                                @ main loop\n"
                    "vabs.f32 q4, q0                           @ abs \n"
                    "vabs.f32 q5, q1                           @ abs \n"
                    "vabs.f32 q6, q2                           @ abs \n"
                    "vabs.f32 q7, q3                           @ abs \n"

                    "vmax.f32 q0, q4, q5                           @ max \n"
                    "vmax.f32 q1, q6, q7                           @ max \n"

                    "pld [%[in]]                                   @ preload data \n"
                    "vmax.f32 q8, q0, q1                           @ max \n"

                    "vld1.32   {d0-d3}, [%[in]]!              @ load 8 float\n"

                    "vmax.f32 %q[max_val], q8, %q[max_val]     @ max \n"
                    "subs %[cnt], #1                           @ loop count -1\n"
                    "vld1.32   {d4-d7}, [%[in]]!              @ load 8 float\n"

                    "bne    1b                                 @ jump to main loop\n"

                : [in] "+r" (ptr_in), [cnt] "+r" (cnt_loop), [max_val] "+w" (vmax_val)
                :
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"
                );
#endif
                float32x2_t vmax_p = vpmax_f32(vget_high_f32(vmax_val), vget_low_f32(vmax_val));
                float max0 = vget_lane_f32(vmax_p, 0);
                float max1 = vget_lane_f32(vmax_p, 1);
                float max2 = max0 > max1 ? max0 : max1;
                max_value = max_value > max2 ? max_value : max2;
            }
            ptr_in = din + j * inner_size + 16 * cnt;
            for (int i = 0; i < remain; ++i) {
                float data = fabsf(*(ptr_in++));
                max_value = fmaxf(max_value, data);
            }
        }
        scale_out[c] = max_value / scale_factor;
    }
    return SaberSuccess;
}

SaberStatus get_tensor_scale(const Tensor<CPU>& tin, std::vector<float>& scale_out, \
    int axis, float scale_factor) {

    int axis_dims = 1;
    if (axis >= 0) {
        axis_dims = tin.valid_shape()[axis];
    }
    scale_out.resize(axis_dims);
    int outer_dims = 1;
    if (axis >= 0) {
        outer_dims = tin.count_valid(0, axis);
    }
    long long inner_dims = tin.count_valid(axis + 1, tin.dims());
    long long inner_size = inner_dims * axis_dims;
    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }
    const float* in_data = static_cast<const float*>(tin.data());
    if (axis == 0){
        return get_tensor_scale_n(in_data, scale_out, axis_dims, inner_dims, scale_factor);
    }else{
        return get_tensor_scale_chw(in_data, scale_out, axis_dims, outer_dims, inner_dims, inner_size, scale_factor);
    }
}

SaberStatus get_tensor_scale_inplace(Tensor<CPU>& tin, int axis, float scale_factor) {
    int axis_dims = 1;
    if (axis >= 0) {
        axis_dims = tin.valid_shape()[axis];
    }
    std::vector<float> scale_out;
    scale_out.resize(axis_dims);
    int out_dims = 1;
    if (axis >= 0) {
        out_dims = tin.count_valid(0, axis);
    }
    long long inner_dims = tin.count_valid(axis + 1, tin.dims());
    long long inner_size = inner_dims * axis_dims;
    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }
    const float* in_data = static_cast<const float*>(tin.data());
    if (axis <= 0){
        get_tensor_scale_n(in_data, scale_out, axis_dims, inner_dims, scale_factor);
    }else{
        get_tensor_scale_chw(in_data, scale_out, axis_dims, out_dims, inner_dims, inner_size, scale_factor);
    }
    tin.set_scale(scale_out);
    return SaberSuccess;
}

void fp32_to_int8(const float* din, char* dout, std::vector<float> scale, int axis_size, int outer_size, int inner_size) {

    int cnt = inner_size / 16;
    int remain = inner_size & 15;
#pragma omp parallel for
    for (int i = 0; i < outer_size; ++i) {
        float scale_inv = 1.f / scale[i % axis_size];
        const float* din_ptr = din + i * inner_size;
        char* dout_ptr = dout + i * inner_size;
        float32x4_t vscale = vdupq_n_f32(scale_inv);
#ifdef __aarch64__
        if (cnt > 0){
            int cnt_loop = cnt;
            asm volatile(
                "1:                                               \n"
                "ld1 {v0.4s, v1.4s}, [%[in]], #32                  \n"
                "ld1 {v2.4s, v3.4s}, [%[in]], #32                  \n"

                "fmul v4.4s, v0.4s, %[scale].4s                      \n"
                "fmul v5.4s, v1.4s, %[scale].4s                      \n"
                "fmul v6.4s, v2.4s, %[scale].4s                      \n"
                "fmul v7.4s, v3.4s, %[scale].4s                      \n"

                "subs %[cnt], %[cnt], #1                    \n"

                "FCVTAS v0.4s, v4.4s                        \n"
                "FCVTAS v1.4s, v5.4s                        \n"
                "FCVTAS v2.4s, v6.4s                        \n"
                "FCVTAS v3.4s, v7.4s                        \n"

                "sqxtn    v4.4h, v0.4s                      \n"
                "sqxtn2   v4.8h, v1.4s                      \n"
                "sqxtn    v5.4h, v2.4s                      \n"
                "sqxtn2    v5.8h, v3.4s                     \n"

                "sqxtn    v0.8b, v4.8h                      \n"
                "sqxtn2    v0.16b, v5.8h                    \n"
                "st1 {v0.4s}, [%[out]], #16                 \n"
                "bne    1b                                \n"

            : [in] "+r" (din_ptr), [out] "+r" (dout_ptr), [cnt] "+r" (cnt_loop)
            : [scale] "w" (vscale)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            );
        }
        for (int j = 0; j < remain; ++j) {
            *(dout_ptr++) = static_cast<char>(round(*(din_ptr++) * scale_inv));
        }
#else
        if (cnt > 0) {
            int cnt_loop = cnt;
            asm volatile(
            "1:                                                @ main loop\n"
                    "vld1.32   {q0, q1}, [%[in]]!              @ load 8 float\n"
                    "vld1.32   {q2, q3}, [%[in]]!              @ load 8 float\n"

                    "vmul.f32      q4, q0,  %q[scale]          @ mul scale\n"
                    "vmul.f32      q5, q1,  %q[scale]          @ mul scale\n"
                    "vmul.f32      q6, q2,  %q[scale]          @ mul scale\n"
                    "vmul.f32      q7, q3,  %q[scale]          @ mul scale\n"

                    "subs          %[cnt], #1                  @ loop count -1\n"

                    "vcvt.s32.f32  q0, q4                      @ convert to int32\n"
                    "vcvt.s32.f32  q1, q5                      @ convert to int32\n"
                    "vcvt.s32.f32  q2, q6                      @ convert to int32\n"
                    "vcvt.s32.f32  q3, q7                      @ convert to int32\n"

                    "vqmovn.s32     d8, q0                      @ convert to int16\n"
                    "vqmovn.s32     d9, q1                      @ convert to int16\n"
                    "vqmovn.s32     d10, q2                     @ convert to int16\n"
                    "vqmovn.s32     d11, q3                     @ convert to int16\n"

                    "vqmovn.s16     d0, q4                      @ convert to int8\n"
                    "vqmovn.s16     d1, q5                      @ convert to int8\n"

                    "vst1.32   {q0}, [%[out]]!                 @ save to output\n"

                    "bne    1b                                 @ jump to main loop\n"

            : [in] "+r" (din_ptr), [out] "+r" (dout_ptr), [cnt] "+r" (cnt_loop)
            : [scale] "w" (vscale)
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
        }
        for (int j = 0; j < remain; ++j) {
#ifdef __aarch64__
            *(dout_ptr++) = static_cast<char>(roundf(*(din_ptr++) * scale_inv));
#else
            *(dout_ptr++) = static_cast<char>((*(din_ptr++) * scale_inv));
#endif //__aarch_64__
        }
#endif //__aarch64__
    }
}

void int32_to_fp32(const int* in, float* out, std::vector<float>& scale, int channel, int inner_size, int outer_size){
    
    int cnt = inner_size >> 3;
#pragma omp parallel for
    for (int n = 0; n < outer_size; ++n){
        float in_scale = scale[n % channel];
        const int* input_channel = in + n * inner_size;
        float* output_channel = out + n * inner_size;
        float32x4_t vscale = vdupq_n_f32(in_scale);
        int i = 0;
        int loop = cnt;
        if (loop > 0){
#ifdef __aarch64__
            asm volatile(
                 "1:                                 \n"
                 "ld1     {v0.4s}, [%[in]], #16      \n"
                 "ld1     {v3.4s}, [%[in]], #16      \n"
                 
                 "scvtf   v1.4s, v0.4s               \n"
                 "scvtf   v4.4s, v3.4s               \n"
                 
                 "fmul    v2.4s, v1.4s, %[scale].4s  \n"
                 "fmul    v5.4s, v4.4s, %[scale].4s  \n"
                 
                 "st1     {v2.4s}, [%[out]], #16     \n"
                 "st1     {v5.4s}, [%[out]], #16     \n"
                 "subs    %[loop], %[loop], #1       \n"
                 "bne     1b                         \n"
                 :[loop] "+r" (loop), [in] "+r" (input_channel), [out] "+r" (output_channel)
                 :[scale] "w" (vscale)
                 :"v0", "v1", "v2", "v3", "v4", "v5"
                 );
#else
            asm volatile(
                 "1:                                     \n"
                 "vld1.s32       {d0-d1}, [%[in]]!       \n"
                 "vld1.s32       {d6-d7}, [%[in]]!       \n"
                 
                 "vcvt.f32.s32   q1, q0                  \n"
                 "vcvt.f32.s32   q4, q3                  \n"
                 
                 "vmul.f32       q2, q1, %q[scale]       \n"
                 "vmul.f32       q5, q4, %q[scale]       \n"
                 
                 "vst1.f32       {d4, d5}, [%[out]]!     \n"
                 "vst1.f32       {d10, d11}, [%[out]]!   \n"
                 "subs           %[loop], #1             \n"
                 "bne            1b                      \n"
                 :[loop] "+r" (loop), [in] "+r" (input_channel), [out] "+r" (output_channel)
                 :[scale] "w" (vscale)
                 :"q0", "q1", "q2", "q3", "q4", "q5"
                 );
#endif //__aarch64__
        }
        i = cnt << 3;
        for (; i < inner_size; ++i){
            *(output_channel++) = float(*(input_channel++)) * in_scale;
        }
    }
    
}

void int32_to_int8(const int* in, char* out, std::vector<float>&scale, int channel, int inner_size, int outer_size){
    
    int cnt = inner_size >> 4;
#pragma omp parallel for
    for (int n = 0; n < outer_size; ++n) {
        float in_scale = scale[n % channel];
        const int* input_channel = in + n * inner_size;
        char* output_channel = out + n * inner_size;
        float32x4_t vscale = vdupq_n_f32(in_scale);
        int i = 0;
        int loop = cnt;
        if (loop > 0) {
#ifdef __aarch64__
            asm volatile(
                 "1:                                        \n"
                 "ld1     {v0.4s, v1.4s}, [%[in]], #32      \n"
                 "ld1     {v2.4s, v3.4s}, [%[in]], #32      \n"
                 
                 "scvtf   v4.4s, v0.4s                      \n"
                 "scvtf   v5.4s, v1.4s                      \n"
                 "scvtf   v6.4s, v2.4s                      \n"
                 "scvtf   v7.4s, v3.4s                      \n"
                 
                 "fmul    v0.4s, v4.4s, %[scale].4s         \n"
                 "fmul    v1.4s, v5.4s, %[scale].4s         \n"
                 "fmul    v2.4s, v6.4s, %[scale].4s         \n"
                 "fmul    v3.4s, v7.4s, %[scale].4s         \n"
                 
                 "fcvtas  v4.4s, v0.4s                      \n"
                 "fcvtas  v5.4s, v1.4s                      \n"
                 "fcvtas  v6.4s, v2.4s                      \n"
                 "fcvtas  v7.4s, v3.4s                      \n"
                 
                 "sqxtn   v0.4h, v4.4s                      \n"
                 "sqxtn2  v0.8h, v5.4s                      \n"
                 "sqxtn   v1.4h, v6.4s                      \n"
                 "sqxtn2  v1.8h, v7.4s                      \n"
                 
                 "sqxtn   v2.8b, v0.8h                      \n"
                 "sqxtn2  v2.16b, v1.8h                     \n"
                 
                 "st1     {v2.16b}, [%[out]], #16           \n"
                 "subs    %[loop], %[loop], #1              \n"
                 "bne     1b                                \n"
                 :[loop] "+r" (loop), [in] "+r" (input_channel), [out] "+r" (output_channel)
                 :[scale] "w" (vscale)
                 :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                 );
#else
            asm volatile(
                 "1:                                     \n"
                 "vld1.s32       {q0, q1}, [%[in]]!      \n"
                 "vld1.s32       {q2, q3}, [%[in]]!      \n"
                 
                 "vcvt.f32.s32   q4, q0                  \n"
                 "vcvt.f32.s32   q5, q1                  \n"
                 "vcvt.f32.s32   q6, q2                  \n"
                 "vcvt.f32.s32   q7, q3                  \n"
                 
                 "vmul.f32       q0, q4, %q[scale]       \n"
                 "vmul.f32       q1, q5, %q[scale]       \n"
                 "vmul.f32       q2, q6, %q[scale]       \n"
                 "vmul.f32       q3, q7, %q[scale]       \n"
                 
                 "vcvt.s32.f32   q4, q0                  \n"
                 "vcvt.s32.f32   q5, q1                  \n"
                 "vcvt.s32.f32   q6, q2                  \n"
                 "vcvt.s32.f32   q7, q3                  \n"
                 
                 "vqmovn.s32     d0, q4                  \n"
                 "vqmovn.s32     d1, q5                  \n"
                 "vqmovn.s32     d2, q6                  \n"
                 "vqmovn.s32     d3, q7                  \n"
                 
                 "vqmovn.s16     d4, q0                  \n"
                 "vqmovn.s16     d5, q1                  \n"
                 
                 "vst1.s8        {d4, d5}, [%[out]]!     \n"
                 "subs           %[loop], #1             \n"
                 "bne            1b                      \n"
                 :[loop] "+r" (loop), [in] "+r" (input_channel), [out] "+r" (output_channel)
                 :[scale] "w" (vscale)
                 :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                 );
#endif //__aarch64__
        }
        i = cnt << 4;
        for (; i < inner_size; ++i){
#ifdef __aarch64__
            output_channel[i] = static_cast<char>(roundf(input_channel[i] * in_scale));
#else
            output_channel[i] = static_cast<char>((input_channel[i] * in_scale));
#endif //__aarch64__
        }
    }
    
}

void int8_to_fp32(const char* in, float* out, std::vector<float>&scale, int channel, int inner_size, int outer_size){
    
    int cnt = inner_size >> 3;
#pragma omp parallel for
    for (int n = 0; n < outer_size; ++n){
        float in_scale = scale[n % channel];
        const char* input_channel = in + n * inner_size;
        float* output_channel = out + n * inner_size;
        float32x4_t vscale = vdupq_n_f32(in_scale);
        int i = 0;
        int loop = cnt;
        if (loop > 0){
#ifdef __aarch64__
            asm volatile(
                 "1:                                 \n"
                 "ld1     {v0.8b}, [%[in]], #8       \n"
                 "sshll   v1.8h, v0.8b, #0           \n"
                 
                 "sshll   v2.4s, v1.4h, #0           \n"
                 "sshll2  v3.4s, v1.8h, #0           \n"
                 
                 "scvtf   v4.4s, v2.4s               \n"
                 "scvtf   v5.4s, v3.4s               \n"
                 
                 
                 "fmul    v6.4s, v4.4s, %[scale].4s  \n"
                 "fmul    v7.4s, v5.4s, %[scale].4s  \n"
                 
                 "st1     {v6.4s}, [%[out]], #16     \n"
                 "st1     {v7.4s}, [%[out]], #16     \n"
                 
                 "subs    %[loop], %[loop], #1       \n"
                 "bne     1b                         \n"
                 :[loop] "+r" (loop), [in] "+r" (input_channel), [out] "+r" (output_channel)
                 :[scale] "w" (vscale)
                 :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                 );
#else
            asm volatile(
                 "1:                                   \n"
                 "vld1.s8       d0, [%[in]]!           \n"
                 "vmovl.s8      q1, d0                 \n"
                 
                 "vmovl.s16     q2, d2                 \n"
                 "vmovl.s16     q3, d3                 \n"
                 
                 "vcvt.f32.s32  q4, q2                 \n"
                 "vcvt.f32.s32  q5, q3                 \n"
                 
                 "vmul.f32      q6, q4, %q[scale]      \n"
                 "vmul.f32      q7, q5, %q[scale]      \n"
                 
                 "vst1.f32      {d12, d13}, [%[out]]!  \n"
                 "vst1.f32      {d14, d15}, [%[out]]!  \n"
                 
                 "subs          %[loop], #1            \n"
                 "bne           1b                     \n"
                 :[loop] "+r" (loop), [in] "+r" (input_channel), [out] "+r" (output_channel)
                 :[scale] "w" (vscale)
                 :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                 );
#endif //__aarch64__
        }
        i = cnt << 3;
        for (; i < inner_size; ++i) {
            *(output_channel++) = *(input_channel++) * in_scale;
        }
    }
    
}

SaberStatus trans_fp32_weights_to_int8(const Tensor<CPU>& tin, Tensor<CPU>& tout, float scale_factor, int axis, Context* ctx) {
    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }
    if (tout.get_dtype() != AK_INT8) {
        tout.set_dtype(AK_INT8);
    }
    tout.reshape(tin.valid_shape());
    std::vector<float> scale;
    get_tensor_scale(tin, scale, axis, scale_factor);
    tout.set_scale(scale);
    int axis_size = tin.valid_shape()[axis];
    int outer_size = tin.count_valid(0, axis + 1);
    int inner_size = tin.count_valid(axis + 1, tin.dims());
    const float* din = static_cast<const float*>(tin.data());
    char* dout = static_cast<char*>(tout.mutable_data());

    fp32_to_int8(din, dout, scale, axis_size, outer_size, inner_size);

    return SaberSuccess;
}

SaberStatus trans_fp32_weights_to_int8_inplace(Tensor<CPU>& tin, float scale_factor, int axis, Context* ctx) {
    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }

    //! alloc memory
    int axis_size = tin.valid_shape()[axis];
    int outer_size = tin.count_valid(0, axis + 1);
    int inner_size = tin.count_valid(axis + 1, tin.dims());

    Tensor<CPU> tmp;
    tmp.re_alloc(tin.valid_shape(), AK_INT8);

    //! inverse the scale
    std::vector<float> scale;
    get_tensor_scale(tin, scale, axis, scale_factor);

    tin.set_scale(scale);

    //! scale to int8
    const float* din = static_cast<const float*>(tin.data());
    char* dout_tmp = static_cast<char*>(tmp.mutable_data());

    fp32_to_int8(din, dout_tmp, scale, axis_size, outer_size, inner_size);

    tin.set_dtype(AK_INT8);
    tin.copy_from(tmp);
    return SaberSuccess;
}

SaberStatus trans_fp32_weights_to_int8_gemm(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
    float scale_factor, bool is_trans, int group, Context* ctx) {

    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }
    if (tout.get_dtype() != AK_INT8) {
        tout.set_dtype(AK_INT8);
    }
    //! alloc memory
    int m = tin.num() / group;
    int k = tin.count_valid(1, tin.dims());
    int axis = 0;
    int axis_size = tin.num();
    int inner_size = k;
    int outer_size = tin.num();
    //! get scale
    std::vector<float> scale;
    if (is_trans) {
        m = tin.channel() / group;
        k = tin.num() * tin.height() * tin.width();
        axis = 1;
        axis_size = tin.channel();
        inner_size = tin.width() * tin.height();
        outer_size = tin.channel() * tin.num();
    }
    get_tensor_scale(tin, scale, axis, scale_factor);

    Tensor<CPU> tmp;
    tmp.re_alloc(tin.valid_shape(), AK_INT8);
    tout.set_scale(scale);

    int hblock = get_hblock_int8(ctx->get_arch());
    int m_round = hblock * ((m + hblock - 1) / hblock);
    int group_size_round_up = ((m_round * k + 15) / 16) * 16;
    tout.reshape(Shape(group_size_round_up * group, 1, 1, 1));

    //! scale to int8
    const float* din = static_cast<const float*>(tin.data());
    char* dout_tmp = static_cast<char*>(tmp.mutable_data());

    fp32_to_int8(din, dout_tmp, scale, axis_size, outer_size, inner_size);

    char* dout = static_cast<char*>(tout.mutable_data());
    int lda = k;
    if (is_trans) {
        lda = m;
    }
    for (int g = 0; g < group; ++g) {
        const char* weights_group = dout_tmp + g * m * k;
        char* weights_trans_ptr = dout + g * group_size_round_up;
        prepackA_int8(weights_trans_ptr, weights_group, lda, 0, m_round, 0, k, is_trans, ctx);
    }
    return SaberSuccess;
}


SaberStatus trans_fp32_weights_to_int8_inplace_gemm(Tensor<CPU>& tin, float scale_factor, \
    bool is_trans, int group, Context* ctx) {

    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }

    //! alloc memory
    int m = tin.num() / group;
    int k = tin.count_valid(1, tin.dims());
    int axis = 0;
    int axis_size = tin.num();
    int inner_size = k;
    int outer_size = tin.num();
    //! get scale
    std::vector<float> scale;
    if (is_trans) {
        m = tin.channel() / group;
        k = tin.num() * tin.height() * tin.width();
        axis = 1;
        axis_size = tin.channel();
        inner_size = tin.width() * tin.height();
        outer_size = tin.channel() * tin.num();
    }
    get_tensor_scale(tin, scale, axis, scale_factor);

    Tensor<CPU> tmp;
    tmp.re_alloc(tin.valid_shape(), AK_INT8);

    tin.set_scale(scale);

    int hblock = get_hblock_int8(ctx->get_arch());
    int m_round = hblock * ((m + hblock - 1) / hblock);
    int group_size_round_up = ((m_round * k + 15) / 16) * 16;
    //! scale to int8
    const float* din = static_cast<const float*>(tin.data());
    char* dout_tmp = static_cast<char*>(tmp.mutable_data());

    fp32_to_int8(din, dout_tmp, scale, axis_size, outer_size, inner_size);

    tin.set_dtype(AK_INT8);
    tin.reshape(Shape(group_size_round_up, group, 1, 1));

    char* dout = static_cast<char*>(tin.mutable_data());
    int lda = k;
    if (is_trans) {
        lda = m;
    }
    for (int g = 0; g < group; ++g) {
        const char* weights_group = dout_tmp + g * m * k;
        char* weights_trans_ptr = dout + g * group_size_round_up;
        prepackA_int8(weights_trans_ptr, weights_group, lda, 0, m_round, 0, k, is_trans, ctx);
    }
    return SaberSuccess;
}

SaberStatus trans_tensor_fp32_to_int8(const Tensor<CPU>& tin, Tensor<CPU>& tout, Context* ctx) {
    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }
    if (tout.get_dtype() != AK_INT8) {
        tout.set_dtype(AK_INT8);
    }
    tout.reshape(tin.valid_shape());

    //! get scale
    std::vector<float> scale = tin.get_scale();

    const float* din = static_cast<const float*>(tin.data());
    char* dout = static_cast<char*>(tout.mutable_data());
    //! convert to int8
    fp32_to_int8(din, dout, scale, 1, 1, tin.valid_size());
    return SaberSuccess;
}
SaberStatus trans_tensor_fp32_to_int8_inplace(Tensor<CPU>& tin, Context* ctx) {
    if (tin.get_dtype() != AK_FLOAT) {
        return SaberInvalidValue;
    }

    Tensor<CPU> tmp;
    tmp.re_alloc(tin.valid_shape(), AK_INT8);

    //! get scale
    std::vector<float> scale = tin.get_scale();

    const float* din = static_cast<const float*>(tin.data());
    char* dout = static_cast<char*>(tmp.mutable_data());
    //! convert to int8
    fp32_to_int8(din, dout, scale, 1, 1, tin.valid_size());

    tin.set_dtype(AK_INT8);
    tin.reshape(tin.valid_shape());
    tin.copy_from(tmp);
    return SaberSuccess;
}
SaberStatus trans_tensor_int32_to_fp32(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
                                       float input_scale, std::vector<float>& weights_scale, Context* ctx){
    
    if (tin.get_dtype() != AK_INT32) {
        return SaberInvalidValue;
    }
    if (tout.get_dtype() != AK_FLOAT) {
        tout.set_dtype(AK_FLOAT);
    }
    tout.reshape(tin.valid_shape());

    //! compute scale
    std::vector<float> scale(weights_scale.size());

    for (int i = 0; i < weights_scale.size(); ++i){
        scale[i] = input_scale * weights_scale[i];
    }

    const int* input = (const int*)tin.data();
    float* output = (float*)tout.mutable_data();
    
    int outer_size = tin.channel() * tin.num();
    int inner_size = tin.width() * tin.height();
    
    //! convert to fp32
    int32_to_fp32(input, output, scale, tin.channel(), inner_size, outer_size);
    return SaberSuccess;
}
SaberStatus trans_tensor_int32_to_fp32_inplace(Tensor<CPU>& tin, float input_scale, \
                                               std::vector<float>& weights_scale, Context* ctx){
    
    if (tin.get_dtype() != AK_INT32) {
        return SaberInvalidValue;
    }
    
    Tensor<CPU> tmp;
    tmp.re_alloc(tin.valid_shape(), AK_FLOAT);
    
    //! compute scale
    std::vector<float> scale(weights_scale.size());
    for (int i = 0; i < weights_scale.size(); ++i){
        scale[i] = input_scale * weights_scale[i];
    }
    
    const int* input = (const int*)tin.data();
    float* output = (float*)tmp.mutable_data();
    
    int outer_size = tin.channel() * tin.num();
    int inner_size = tin.width() * tin.height();
    
    //! convert to fp32
    int32_to_fp32(input, output, scale, tin.channel(), inner_size, outer_size);
    
    tin.set_dtype(AK_FLOAT);
    tin.reshape(tin.valid_shape());
    tin.copy_from(tmp);
    return SaberSuccess;
}
SaberStatus trans_tensor_int32_to_int8(Tensor<CPU>& tin, Tensor<CPU>& tout, \
                                       float input_scale, std::vector<float>& weights_scale, Context* ctx){
    
    if (tin.get_dtype() != AK_INT32) {
        return SaberInvalidValue;
    }
    if (tout.get_dtype() != AK_INT8) {
        tout.set_dtype(AK_INT8);
    }
    tout.reshape(tin.valid_shape());
    
    //! compute scale
    std::vector<float> out_scale = tout.get_scale();
    std::vector<float> scale(weights_scale.size());
    for (int i = 0; i < weights_scale.size(); ++i){
        scale[i] = input_scale * weights_scale[i] / out_scale[0];
    }
    const int* input = (const int*)tin.data();
    char* output = (char*)tout.mutable_data();
    
    int outer_size = tin.channel() * tin.num();
    int inner_size = tin.width() * tin.height();
    //! convert to int8
    int32_to_int8(input, output, scale, tin.channel(), inner_size, outer_size);
    return SaberSuccess;
}
SaberStatus trans_tensor_int32_to_int8_inplace(Tensor<CPU>& tin, float input_scale,\
                                               std::vector<float>& weights_scale, Context* ctx){
    
    if (tin.get_dtype() != AK_INT32) {
        return SaberInvalidValue;
    }
    
    Tensor<CPU> tmp;
    tmp.re_alloc(tin.valid_shape(), AK_INT8);
    
    //! compute scale
    std::vector<float> out_scale = tin.get_scale();
    std::vector<float> scale(weights_scale.size());
    for (int i = 0; i < weights_scale.size(); ++i){
        scale[i] = input_scale * weights_scale[i] / out_scale[0];
    }
    
    const int* input = (const int*)tin.data();
    char* output = (char*)tin.mutable_data();
    
    int outer_size = tin.channel() * tin.num();
    int inner_size = tin.width() * tin.height();
    
    //! convert to int8
    int32_to_int8(input, output, scale, tin.channel(), inner_size, outer_size);
    tin.set_dtype(AK_INT8);
    tin.reshape(tin.valid_shape());
    tin.copy_from(tmp);
    return SaberSuccess;
}
SaberStatus trans_tensor_int8_to_fp32(Tensor<CPU>& tin, Tensor<CPU>& tout, \
                                      float input_scale, Context* ctx){
    
    if (tin.get_dtype() != AK_INT8) {
        return SaberInvalidValue;
    }
    if (tout.get_dtype() != AK_FLOAT) {
        tout.set_dtype(AK_FLOAT);
    }
    tout.reshape(tin.valid_shape());
    
    //! compute scale
    std::vector<float> scale(tin.channel());
    for (int i = 0; i < scale.size(); ++i){
        scale[i] = input_scale;
    }
    
    const char* input = (const char*)tin.data();
    float* output = (float*)tout.mutable_data();
    
    int outer_size = tin.channel() * tin.num();
    int inner_size = tin.width() * tin.height();
    
    //! convert to fp32
    int8_to_fp32(input, output, scale, tin.channel(), inner_size, outer_size);
    return SaberSuccess;
}
SaberStatus trans_tensor_int8_to_fp32_inplace(Tensor<CPU>& tin, float input_scale,\
    Context* ctx){
    
    if (tin.get_dtype() != AK_INT8) {
        return SaberInvalidValue;
    }
    Tensor<CPU> tmp;
    tmp.re_alloc(tin.valid_shape(), AK_FLOAT);

    //! compute scale
    std::vector<float> scale(tin.channel());
    for (int i = 0; i < scale.size(); ++i){
        scale[i] = input_scale;
    }
    
    const char* input = (const char*)tin.data();
    float* output = (float*)tmp.mutable_data();
    
    int outer_size = tin.channel() * tin.num();
    int inner_size = tin.width() * tin.height();
    
    //! convert to fp32
    int8_to_fp32(input, output, scale, tin.channel(), inner_size, outer_size);
    tin.set_dtype(AK_FLOAT);
    tin.reshape(tin.valid_shape());
    tin.copy_from(tmp);
    return SaberSuccess;
}

SaberStatus trans_fp32_bias_to_int32(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
        float in_scale, std::vector<float> vector_weight_scale, Context* ctx) {
    if (tin.get_dtype() != AK_FLOAT || vector_weight_scale.size() != tin.valid_size()) {
        return SaberInvalidValue;
    }
    tout.set_dtype(AK_INT32);
    tout.reshape(tin.valid_shape());
    const float* in_data = static_cast<const float*>(tin.data());
    int* out_data = static_cast<int*>(tout.mutable_data());

    for (int i = 0; i < tin.valid_size(); ++i) {
        out_data[i] = static_cast<int>(roundf(in_data[i] / in_scale / vector_weight_scale[i]));
    }
    return SaberSuccess;
}

SaberStatus trans_fp32_bias_to_int32_inplace(Tensor<CPU>& tin, \
        float in_scale, std::vector<float> vector_weight_scale, Context* ctx) {
    if (tin.get_dtype() != AK_FLOAT || vector_weight_scale.size() != tin.valid_size()) {
        return SaberInvalidValue;
    }
    const float* in_data = static_cast<const float*>(tin.data());
    tin.set_dtype(AK_INT32);
    int* out_data = static_cast<int*>(tin.mutable_data());

    for (int i = 0; i < tin.valid_size(); ++i) {
        out_data[i] = static_cast<int>(roundf(in_data[i] / in_scale / vector_weight_scale[i]));
    }
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} //namespace anakin