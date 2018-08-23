#include "saber/funcs/impl/arm/saber_eltwise_active.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{
void eltwise_prod_relu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size, \
      int channel_size, int channel, float* slop_ptr, bool channel_shared) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

    int cnt = size >> 3;
    int remain = size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vmulq_f32(va0, vb0);
        uint32x4_t vmask1 = vcgtq_f32(vout1, vzero);//vout1 > 0
        float32x4_t vrst1 = vbslq_f32(vmask1, vout1, vzero);
        vst1q_f32(out_ptr, vrst1);
        float32x4_t vout2 = vmulq_f32(va1, vb1);
        uint32x4_t vmask2 = vcgtq_f32(vout2, vzero);//vout2 > 0
        float32x4_t vrst2 = vbslq_f32(vmask2, vout2, vzero);
        vst1q_f32(out_ptr + 4, vrst2);
        a_ptr += 8;
        b_ptr += 8;
        out_ptr += 8;
    }
#else
    int loop_cnt = cnt;
    if (loop_cnt > 0) {
        asm volatile(
        "prod_loop:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vmul.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vmul.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "vcgt.f32  q0, q8, %q[vzero]            @ q8 > 0 \n"
                "vcgt.f32  q1, q9, %q[vzero]            @ q9 > 0 \n"
                "vbif.f32  q8, %q[vzero], q0            @ bif \n"
                "vbif.f32  q9,  %q[vzero], q1           @ bif \n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!        @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!        @ store data\n"
                "bne       prod_loop                    @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__
    for (; remain > 0; remain--) {
        float tmp = *(a_ptr++) * (*(b_ptr++));
        *(out_ptr++) = tmp > 0 ? tmp : 0.f;
    }
}

void eltwise_sum_relu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size, \
     int channel_size, int channel, float* slop_ptr, bool channel_shared) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

    int cnt = size >> 3;
    int remain = size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vaddq_f32(va0, vb0);
        uint32x4_t vmask1 = vcgtq_f32(vout1, vzero);//vout1 > 0
        float32x4_t vrst1 = vbslq_f32(vmask1, vout1, vzero);
        vst1q_f32(out_ptr, vrst1);
        float32x4_t vout2 = vaddq_f32(va1, vb1);
        uint32x4_t vmask2 = vcgtq_f32(vout2, vzero);//vout1 > 0
        float32x4_t vrst2 = vbslq_f32(vmask2, vout2, vzero);
        vst1q_f32(out_ptr + 4, vrst2);
        a_ptr += 8;
        b_ptr += 8;
        out_ptr += 8;
    }
#else
    int loop_cnt = cnt;
    if (loop_cnt > 0) {
        asm volatile(
        "sum_loop:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vadd.f32  q8, q0, q1                   @ q8 = q0 + q1\n"
                "vadd.f32  q9, q2, q3                   @ q9 = q2 + q3\n"
                "vcgt.f32  q0, q8, %q[vzero]            @ q8 > 0 \n"
                "vcgt.f32  q1, q9, %q[vzero]            @ q9 > 0 \n"
                "vbif.f32  q8, %q[vzero], q0            @ bsl \n"
                "vbif.f32  q9,  %q[vzero], q1           @ bsl \n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       sum_loop                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        float tmp = *(a_ptr++) + (*(b_ptr++));
        *(out_ptr++) = tmp > 0 ? tmp : 0.f;
    }
}

void eltwise_sub_relu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size, \
    int channel_size, int channel, float* slop_ptr, bool channel_shared) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

    int cnt = size >> 3;
    int remain = size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vsubq_f32(va0, vb0);
        uint32x4_t vmask1 = vcgtq_f32(vout1, vzero);//vout1 > 0
        float32x4_t vrst1 = vbslq_f32(vmask1, vout1, vzero);
        vst1q_f32(out_ptr, vrst1);
        float32x4_t vout2 = vsubq_f32(va1, vb1);
        uint32x4_t vmask2 = vcgtq_f32(vout2, vzero);//vout1 > 0
        float32x4_t vrst2 = vbslq_f32(vmask2, vout2, vzero);
        vst1q_f32(out_ptr + 4, vrst2);
        a_ptr += 8;
        b_ptr += 8;
        out_ptr += 8;
    }
#else
    int loop_cnt = cnt;
    if (loop_cnt > 0) {
        asm volatile(
        "sub_loop:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vsub.f32  q8, q0, q1                   @ q8 = q0 - q1\n"
                "vsub.f32  q9, q2, q3                   @ q9 = q2  q3\n"
                "vcgt.f32  q0, q8, %q[vzero]            @ q8 > 0 \n"
                "vcgt.f32  q1, q9, %q[vzero]            @ q9 > 0 \n"
                "vbif.f32  q8, %q[vzero], q0            @ bif \n"
                "vbif.f32  q9, %q[vzero], q1            @ bif \n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       sub_loop                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        float tmp = *(a_ptr++) - (*(b_ptr++));
         *(out_ptr++) = tmp > 0 ? tmp : 0.f;
    }
}
void eltwise_sum_coeff_relu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size, \
    int channel_size, int channel, float* slop_ptr, bool channel_shared) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

    int cnt = size >> 3;
    int remain = size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vcoef0 = vdupq_n_f32(coeff[0]);
    float32x4_t vcoef1 = vdupq_n_f32(coeff[1]);
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vmulq_f32(va0, vcoef0);
        vout1 = vmlaq_f32(vout1, vb0, vcoef1);
        uint32x4_t vmask1 = vcgtq_f32(vout1, vzero);//vout1 > 0
        float32x4_t vrst1 = vbslq_f32(vmask1, vout1, vzero);
        vst1q_f32(out_ptr, vrst1);
        float32x4_t vout2 = vmulq_f32(va1, vcoef0);
        vout2 = vmlaq_f32(vout2, vb1, vcoef1);
        uint32x4_t vmask2 = vcgtq_f32(vout2, vzero);//vout1 > 0
        float32x4_t vrst2 = vbslq_f32(vmask2, vout2, vzero);
        vst1q_f32(out_ptr + 4, vrst2);
        a_ptr += 8;
        b_ptr += 8;
        out_ptr += 8;
    }
#else
    int loop_cnt = cnt;
    if (loop_cnt > 0) {
        asm volatile(
        "sum_coeff_loop:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vmul.f32  q8, q0, %q[vcoef0]           @ q8 = q0 * coef0 \n"
                "vmul.f32  q9, q2, %q[vcoef0]           @ q9 = q1 * coef0 \n"
                "vmla.f32  q8, q1, %q[vcoef1]           @ q8 = q8 + q1 * coef1\n"
                "vmla.f32  q9, q3, %q[vcoef1]           @ q9 = q9 + q1 * vcoef1 \n"
                "vcgt.f32  q0, q8, %q[vzero]            @ q8 > 0 \n"
                "vcgt.f32  q1, q9, %q[vzero]            @ q9 > 0 \n"
                "vbif.f32  q8, %q[vzero], q0            @ bif \n"
                "vbif.f32  q9, %q[vzero], q1            @ bif \n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       sum_coeff_loop               @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), \ 
            [vcoef0] "+w" (vcoef0), [vcoef1] "+w" (vcoef1), [vzero] "+w" (vzero)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        float tmp = *(a_ptr++) * coeff[0] + (*(b_ptr++)) * coeff[1];
        *(out_ptr++) = tmp > 0 ? tmp : 0.f;
    }
}

void eltwise_max_relu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size, \
     int channel_size, int channel, float* slop_ptr, bool channel_shared) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

    int cnt = size >> 3;
    int remain = size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vmaxq_f32(va0, vb0);
        uint32x4_t vmask1 = vcgtq_f32(vout1, vzero);//vout1 > 0
        float32x4_t vrst1 = vbslq_f32(vmask1, vout1, vzero);
        vst1q_f32(out_ptr, vrst1);
        float32x4_t vout2 = vmaxq_f32(va1, vb1);
        uint32x4_t vmask2 = vcgtq_f32(vout2, vzero);//vout1 > 0
        float32x4_t vrst2 = vbslq_f32(vmask2, vout2, vzero);
        vst1q_f32(out_ptr + 4, vrst2);
        a_ptr += 8;
        b_ptr += 8;
        out_ptr += 8;
    }
#else
    int loop_cnt = cnt;
    if (loop_cnt > 0) {
        asm volatile(
        "max_loop:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vmax.f32  q8, q0, q1                   @ q8 = max(q0, q1)\n"
                "vmax.f32  q9, q2, q3                   @ q9 = max(q2, q3)\n"
                "vcgt.f32  q0, q8, %q[vzero]            @ q8 > 0 \n"
                "vcgt.f32  q1, q9, %q[vzero]            @ q9 > 0 \n"
                "vbif.f32  q8, %q[vzero], q0            @ bif \n"
                "vbif.f32  q9, %q[vzero], q1            @ bif \n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       max_loop                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
       float tmp = std::max(*(a_ptr++), *(b_ptr++));
       *(out_ptr++) = tmp > 0 ? tmp : 0.f;
    }
}

//prelu
void eltwise_prod_prelu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int num, \
    int channel, int channel_size, float* slop_ptr, bool channel_shared) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int size = channel * channel_size;
    for(int n = 0; n < num; n++){
        const float* dina_ptr = din_a + n * size;
        const float* dinb_ptr = din_b + n * size;
        float* dout_ptr = dout + n * size;
#pragma omp parallel for
        for(int c = 0; c < channel; c++){
            const float* a_ptr = dina_ptr + c * channel_size;
            const float* b_ptr = dinb_ptr + c * channel_size;
            float* out_ptr = dout_ptr + c * channel_size;
            float slope_val = channel_shared ? slop_ptr[0] : slop_ptr[c];
            float32x4_t vslope = vdupq_n_f32(slope_val);
#ifdef __aarch64__
            for (int i = 0; i < cnt; ++i) {
                float32x4_t va0 = vld1q_f32(a_ptr);
                float32x4_t vb0 = vld1q_f32(b_ptr);
                float32x4_t va1 = vld1q_f32(a_ptr + 4);
                float32x4_t vb1 = vld1q_f32(b_ptr + 4);

                float32x4_t vsum0 = vmulq_f32(va0, vb0);
                float32x4_t vsum1 = vmulq_f32(va1, vb1);

                uint32x4_t vmask0 = vcltq_f32(vsum0, vzero);//vsum1 <= vzero
                float32x4_t vout0 = vmulq_f32(vsum0, vslope);//vsum1 * vslope

                uint32x4_t vmask1 = vcltq_f32(vsum1, vzero);//vsum2 <= vzero
                float32x4_t vout1 = vmulq_f32(vsum1, vslope);//vsum2 * vslope

                float32x4_t vout_sel0 = vbslq_f32(vmask0, vout0, vsum0);
                float32x4_t vout_sel1 = vbslq_f32(vmask1, vout1, vsum1);
            
                a_ptr += 8;
                b_ptr += 8;

                vst1q_f32(out_ptr, vout_sel0);
                vst1q_f32(out_ptr + 4, vout_sel1);

                out_ptr += 8;
            }
    
    #else
            int loop_cnt = cnt;
            if (loop_cnt > 0) {
                asm volatile(
                "1:                                         @ main loop start point\n"
                    "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                    "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                    "vmul.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                    "vmul.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                    "pld [%[a_ptr], #256]                   @ preload data\n"
                    "pld [%[b_ptr], #256]                   @ preload data\n"

                    "vclt.f32   q0, q8, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q1, q8, %q[vslope]          @vmul q8 * vslope\n"

                    "vclt.f32   q2, q9, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q3, q9, %q[vslope]          @vmul q8 * vslope\n"

                    "vbit.32    q8, q1, q0                  @vbit q0, q1, q0\n"
                    "vbit.32    q9, q3, q2                  @vbit q0, q1, q0\n"

                    "subs      %[loop_cnt], #1              @ loop --\n"
                    "vst1.f32 {d16-d17}, [%[out_ptr]]!        @ store data\n"
                    "vst1.f32 {d18-d19}, [%[out_ptr]]!        @ store data\n"
                    "bne       1b                    @ top_loop \n"
                :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
                [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero), [vslope] "+w" (vslope)
                :
                :"q0", "q1", "q2", "q3", "q8", "q9"
                );
            }
#endif //__aarch64__

            for (; remain > 0; remain--) {
                float tmp = *(a_ptr++) * (*(b_ptr++));
                *(out_ptr++) = tmp > 0 ? tmp : tmp * slope_val;
            }
        }
    }
}

void eltwise_sum_prelu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int num, \
     int channel, int channel_size, float* slop_ptr, bool channel_shared) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int size = channel * channel_size;
    for(int n = 0; n < num; n++){
        const float* dina_ptr = din_a + n * size;
        const float* dinb_ptr = din_b + n * size;
        float* dout_ptr = dout + n * size;
#pragma omp parallel for
        for(int c =0; c < channel; c++){
            const float* a_ptr = dina_ptr + c * channel_size;
            const float* b_ptr = dinb_ptr + c * channel_size;
            float* out_ptr = dout_ptr + c * channel_size;
            float slope_val = channel_shared ? slop_ptr[0] : slop_ptr[c];
            float32x4_t vslope = vdupq_n_f32(slope_val);
#ifdef __aarch64__
            for (int i = 0; i < cnt; ++i) {
                float32x4_t va0 = vld1q_f32(a_ptr);
                float32x4_t vb0 = vld1q_f32(b_ptr);
                float32x4_t va1 = vld1q_f32(a_ptr + 4);
                float32x4_t vb1 = vld1q_f32(b_ptr + 4);

                float32x4_t vsum0 = vaddq_f32(va0, vb0);
                float32x4_t vsum1 = vaddq_f32(va1, vb1);

                uint32x4_t vmask0 = vcltq_f32(vsum0, vzero);//vsum1 <= vzero
                float32x4_t vout0 = vmulq_f32(vsum0, vslope);//vsum1 * vslope

                uint32x4_t vmask1 = vcltq_f32(vsum1, vzero);//vsum2 <= vzero
                float32x4_t vout1 = vmulq_f32(vsum1, vslope);//vsum2 * vslope

                float32x4_t vout_sel0 = vbslq_f32(vmask0, vout0, vsum0);
                float32x4_t vout_sel1 = vbslq_f32(vmask1, vout1, vsum1);
            
                a_ptr += 8;
                b_ptr += 8;

                vst1q_f32(out_ptr, vout_sel0);
                vst1q_f32(out_ptr + 4, vout_sel1);

                out_ptr += 8;
            }
    
#else
            int loop_cnt = cnt;
            if (loop_cnt > 0) {
                asm volatile(
                "2:                                         @ main loop start point\n"
                    "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                    "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                    "vadd.f32  q8, q0, q1                   @ q8 = q0 + q1\n"
                    "vadd.f32  q9, q2, q3                   @ q9 = q2 + q3\n"
                    "pld [%[a_ptr], #256]                   @ preload data\n"
                    "pld [%[b_ptr], #256]                   @ preload data\n"

                    "vclt.f32   q0, q8, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q1, q8, %q[vslope]          @vmul q8 * vslope\n"

                    "vclt.f32   q2, q9, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q3, q9, %q[vslope]          @vmul q8 * vslope\n"

                    "vbit.32    q8, q1, q0                  @vbit q0, q1, q0\n"
                    "vbit.32    q9, q3, q2                  @vbit q0, q1, q0\n"

                    "subs      %[loop_cnt], #1              @ loop --\n"
                    "vst1.f32 {d16-d17}, [%[out_ptr]]!        @ store data\n"
                    "vst1.f32 {d18-d19}, [%[out_ptr]]!        @ store data\n"
                    "bne       2b                    @ top_loop \n"
                :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
                [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero), [vslope] "+w" (vslope)
                :
                :"q0", "q1", "q2", "q3", "q8", "q9"
                );
            }
#endif //__aarch64__

            for (; remain > 0; remain--) {
                float tmp = *(a_ptr++) + (*(b_ptr++));
                *(out_ptr++) = tmp > 0 ? tmp : tmp * slope_val;
            }
        }
    }
}

void eltwise_sub_prelu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int num, \
    int channel, int channel_size, float* slop_ptr, bool channel_shared) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int size = channel * channel_size;
    for(int n = 0; n < num; n++){
        const float* dina_ptr = din_a + n * size;
        const float* dinb_ptr = din_b + n * size;
        float* dout_ptr = dout + n * size;
#pragma omp parallel for
        for(int c = 0; c < channel; c++){
            const float* a_ptr = dina_ptr + c * channel_size;
            const float* b_ptr = dinb_ptr + c * channel_size;
            float* out_ptr = dout_ptr + c * channel_size;
            float slope_val = channel_shared ? slop_ptr[0] : slop_ptr[c];
            float32x4_t vslope = vdupq_n_f32(slope_val);
#ifdef __aarch64__
            for (int i = 0; i < cnt; ++i) {
                float32x4_t va0 = vld1q_f32(a_ptr);
                float32x4_t vb0 = vld1q_f32(b_ptr);
                float32x4_t va1 = vld1q_f32(a_ptr + 4);
                float32x4_t vb1 = vld1q_f32(b_ptr + 4);

                float32x4_t vsum0 = vsubq_f32(va0, vb0);
                float32x4_t vsum1 = vsubq_f32(va1, vb1);

                uint32x4_t vmask0 = vcltq_f32(vsum0, vzero);//vsum1 <= vzero
                float32x4_t vout0 = vmulq_f32(vsum0, vslope);//vsum1 * vslope

                uint32x4_t vmask1 = vcltq_f32(vsum1, vzero);//vsum2 <= vzero
                float32x4_t vout1 = vmulq_f32(vsum1, vslope);//vsum2 * vslope

                float32x4_t vout_sel0 = vbslq_f32(vmask0, vout0, vsum0);
                float32x4_t vout_sel1 = vbslq_f32(vmask1, vout1, vsum1);
            
                a_ptr += 8;
                b_ptr += 8;

                vst1q_f32(out_ptr, vout_sel0);
                vst1q_f32(out_ptr + 4, vout_sel1);

                out_ptr += 8;
            }
    
#else
            int loop_cnt = cnt;
            if (loop_cnt > 0) {
                asm volatile(
                "3:                                         @ main loop start point\n"
                    "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                    "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                    "vsub.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                    "vsub.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                    "pld [%[a_ptr], #256]                   @ preload data\n"
                    "pld [%[b_ptr], #256]                   @ preload data\n"

                    "vclt.f32   q0, q8, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q1, q8, %q[vslope]          @vmul q8 * vslope\n"

                    "vclt.f32   q2, q9, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q3, q9, %q[vslope]          @vmul q8 * vslope\n"

                    "vbit.32    q8, q1, q0                  @vbit q0, q1, q0\n"
                    "vbit.32    q9, q3, q2                  @vbit q0, q1, q0\n"

                    "subs      %[loop_cnt], #1              @ loop --\n"
                    "vst1.f32 {d16-d17}, [%[out_ptr]]!        @ store data\n"
                    "vst1.f32 {d18-d19}, [%[out_ptr]]!        @ store data\n"
                    "bne       3b                    @ top_loop \n"
                :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
                [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero), [vslope] "+w" (vslope)
                :
                :"q0", "q1", "q2", "q3", "q8", "q9"
                );
            }
#endif //__aarch64__

            for (; remain > 0; remain--) {
                float tmp = *(a_ptr++) - (*(b_ptr++));
                *(out_ptr++) = tmp > 0 ? tmp : tmp * slope_val;
            }
        }
    }
}

void eltwise_sum_coeff_prelu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int num, \
     int channel, int channel_size, float* slop_ptr, bool channel_shared) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vcoef0 = vdupq_n_f32(coeff[0]);
    float32x4_t vcoef1 = vdupq_n_f32(coeff[1]);
    int size = channel * channel_size;
    for(int n = 0; n < num; n++){
        const float* dina_ptr = din_a + n * size;
        const float* dinb_ptr = din_b + n * size;
        float* dout_ptr = dout + n * size;
#pragma omp parallel for
        for(int c = 0; c < channel; c++){
            const float* a_ptr = dina_ptr + c * channel_size;
            const float* b_ptr = dinb_ptr + c * channel_size;
            float* out_ptr = dout_ptr + c * channel_size;
            float slope_val = channel_shared ? slop_ptr[0] : slop_ptr[c];
            float32x4_t vslope = vdupq_n_f32(slope_val);
#ifdef __aarch64__
            for (int i = 0; i < cnt; ++i) {
                float32x4_t va0 = vld1q_f32(a_ptr);
                float32x4_t vb0 = vld1q_f32(b_ptr);
                float32x4_t va1 = vld1q_f32(a_ptr + 4);
                float32x4_t vb1 = vld1q_f32(b_ptr + 4);
                float32x4_t vsum0 = vmulq_f32(va0, vcoef0);
                vsum0 = vmlaq_f32(vsum0, vb0, vcoef1);

                float32x4_t vsum1 = vmulq_f32(va1, vcoef0);
                vsum1 = vmlaq_f32(vsum1, vb1, vcoef1);

                uint32x4_t vmask0 = vcltq_f32(vsum0, vzero);//vsum1 <= vzero
                float32x4_t vout0 = vmulq_f32(vsum0, vslope);//vsum1 * vslope

                uint32x4_t vmask1 = vcltq_f32(vsum1, vzero);//vsum2 <= vzero
                float32x4_t vout1 = vmulq_f32(vsum1, vslope);//vsum2 * vslope

                float32x4_t vout_sel0 = vbslq_f32(vmask0, vout0, vsum0);
                float32x4_t vout_sel1 = vbslq_f32(vmask1, vout1, vsum1);

                vst1q_f32(out_ptr, vout_sel0);
                vst1q_f32(out_ptr + 4, vout_sel1);
                a_ptr += 8;
                b_ptr += 8;
                out_ptr += 8;
            }
#else
            int loop_cnt = cnt;
            if (loop_cnt > 0) {
                asm volatile(
                "4:                                         @ main loop start point\n"
                    "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                    "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                    "vmul.f32  q8, q0, %q[vcoef0]           @ q8 = q0 * coef0 \n"
                    "vmul.f32  q9, q2, %q[vcoef0]           @ q9 = q1 * coef0 \n"
                    "vmla.f32  q8, q1, %q[vcoef1]           @ q8 = q8 + q1 * coef1\n"
                    "vmla.f32  q9, q3, %q[vcoef1]           @ q9 = q9 + q1 * vcoef1 \n"

                    "vclt.f32   q0, q8, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q1, q8, %q[vslope]          @vmul q8 * vslope\n"

                    "vclt.f32   q2, q9, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q3, q9, %q[vslope]          @vmul q8 * vslope\n"

                    "vbit.32    q8, q1, q0                  @vbit q0, q1, q0\n"
                    "vbit.32    q9, q3, q2                  @vbit q0, q1, q0\n"

                    "subs      %[loop_cnt], #1              @ loop --\n"
                    "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                    "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                    "bne       4b               @ top_loop \n"
                :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
                [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), \ 
                [vcoef0] "+w" (vcoef0), [vcoef1] "+w" (vcoef1), [vzero] "+w" (vzero), [vslope] "+w" (vslope)
                :
                :"q0", "q1", "q2", "q3", "q8", "q9"
                );
            }
#endif //__aarch64__

            for (; remain > 0; remain--) {
                float tmp = *(a_ptr++) * coeff[0] + (*(b_ptr++)) * coeff[1];
                *(out_ptr++) = tmp > 0 ? tmp : tmp * slope_val;
            }
        }
    }
}

void eltwise_max_prelu(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int num, \
     int channel, int channel_size, float* slop_ptr, bool channel_shared) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int size = channel * channel_size;
    for(int n = 0; n < num; n++){
        const float* dina_ptr = din_a + n * size;
        const float* dinb_ptr = din_b + n * size;
        float* dout_ptr = dout + n * size;
#pragma omp parallel for
        for(int c = 0; c <channel; c++){
            const float* a_ptr = dina_ptr + c * channel_size;
            const float* b_ptr = dinb_ptr + c * channel_size;
            float* out_ptr = dout_ptr + c * channel_size;
            float slope_val = channel_shared ? slop_ptr[0] : slop_ptr[ c];
            float32x4_t vslope = vdupq_n_f32(slope_val);
#ifdef __aarch64__
            for (int i = 0; i < cnt; ++i) {
                float32x4_t va0 = vld1q_f32(a_ptr);
                float32x4_t vb0 = vld1q_f32(b_ptr);
                float32x4_t va1 = vld1q_f32(a_ptr + 4);
                float32x4_t vb1 = vld1q_f32(b_ptr + 4);

                float32x4_t vsum0 = vmaxq_f32(va0, vb0);
                float32x4_t vsum1 = vmaxq_f32(va1, vb1);

                uint32x4_t vmask0 = vcltq_f32(vsum0, vzero);//vsum1 <= vzero
                float32x4_t vout0 = vmulq_f32(vsum0, vslope);//vsum1 * vslope

                uint32x4_t vmask1 = vcltq_f32(vsum1, vzero);//vsum2 <= vzero
                float32x4_t vout1 = vmulq_f32(vsum1, vslope);//vsum2 * vslope

                float32x4_t vout_sel0 = vbslq_f32(vmask0, vout0, vsum0);
                float32x4_t vout_sel1 = vbslq_f32(vmask1, vout1, vsum1);
            
                a_ptr += 8;
                b_ptr += 8;

                vst1q_f32(out_ptr, vout_sel0);
                vst1q_f32(out_ptr + 4, vout_sel1);

                out_ptr += 8;
            }
    
#else
            int loop_cnt = cnt;
            if (loop_cnt > 0) {
                asm volatile(
                "5:                                         @ main loop start point\n"
                    "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                    "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                    "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                    "vmax.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                    "vmax.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                    "pld [%[a_ptr], #256]                   @ preload data\n"
                    "pld [%[b_ptr], #256]                   @ preload data\n"

                    "vclt.f32   q0, q8, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q1, q8, %q[vslope]          @vmul q8 * vslope\n"

                    "vclt.f32   q2, q9, %q[vzero]           @vcle q8 <= vzero\n"
                    "vmul.f32   q3, q9, %q[vslope]          @vmul q8 * vslope\n"

                    "vbit.32    q8, q1, q0                  @vbit q0, q1, q0\n"
                    "vbit.32    q9, q3, q2                  @vbit q0, q1, q0\n"

                    "subs      %[loop_cnt], #1              @ loop --\n"
                    "vst1.f32 {d16-d17}, [%[out_ptr]]!        @ store data\n"
                    "vst1.f32 {d18-d19}, [%[out_ptr]]!        @ store data\n"
                    "bne       5b                    @ top_loop \n"
                :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
                [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero), [vslope] "+w" (vslope)
                :
                :"q0", "q1", "q2", "q3", "q8", "q9"
                );
            }
#endif //__aarch64__

            for (; remain > 0; remain--) {
                float tmp = *a_ptr > *b_ptr ? *a_ptr : *b_ptr;
                a_ptr++;
                b_ptr++;
                *(out_ptr++) = tmp > 0 ? tmp : tmp * slope_val;
            }
        }
    }
}


template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberEltwiseActive<ARM, OpDtype, inDtype, outDtype, \
LayOutType_op, LayOutType_in, LayOutType_out>::create(\
    const std::vector<DataTensor_in*>& inputs,\
        std::vector<DataTensor_out*>& outputs,\
        EltwiseActiveParam<OpTensor> &param, \
        Context<ARM> &ctx) {
    this->_ctx = &ctx;
    _coeff = param.eltwise_param.coeff;
    Shape sh_out_saber = outputs[0]->valid_shape();
    for (int i = 0; i < inputs.size(); i ++){
        Shape sh_in_saber = inputs[i]->valid_shape();
        if (sh_out_saber != sh_in_saber){
                    LOG(INFO) << "input shape is not same with output shape ";
            return SaberInvalidValue;
        }
    }
    if(param.has_activation){
        if(param.activation_param.active == 2){//Active_relu = 2
            switch (param.eltwise_param.operation) {
                case Eltwise_prod:
                   _impl = eltwise_prod_relu;
                   //printf("prod\n");
                   break;
                case Eltwise_sum:
                   if (param.eltwise_param.coeff[0] == 1 && param.eltwise_param.coeff[1] == 1)
                        _impl = eltwise_sum_relu;
                    else if (param.eltwise_param.coeff[0] == 1 && param.eltwise_param.coeff[1] == -1)
                        _impl = eltwise_sub_relu;
                    else
                        _impl = eltwise_sum_coeff_relu;
                    break;
                case Eltwise_max:
                    _impl = eltwise_max_relu;
                    break;
                default:
                    LOG(ERROR) << "unknown eltwise type!!";
                    return SaberUnKownError;
            }
        }
        if(param.activation_param.active == 10){
            PreluParam<OpTensor> prelu = param.activation_param.prelu_param; 
            _channel_shared = prelu.channel_shared;
            _slop_ptr = prelu.slope->data();
            _channel_size = inputs[0]->width() * inputs[0]->height();
            _channel = inputs[0]->channel();
            switch (param.eltwise_param.operation) {
                case Eltwise_prod:
                   _impl = eltwise_prod_prelu;
                   //printf("prod\n");
                   break;
                case Eltwise_sum:
                   if (param.eltwise_param.coeff[0] == 1 && param.eltwise_param.coeff[1] == 1)
                        _impl = eltwise_sum_prelu;
                    else if (param.eltwise_param.coeff[0] == 1 && param.eltwise_param.coeff[1] == -1)
                        _impl = eltwise_sub_prelu;
                    else
                        _impl = eltwise_sum_coeff_prelu;
                    break;
                case Eltwise_max:
                    _impl = eltwise_max_prelu;
                    break;
                default:
                    LOG(ERROR) << "unknown eltwise type!!";
                    return SaberUnKownError;
            }
        }
        //todo
    }
   /* switch (param.eltwise_param.operation) {
        case Eltwise_prod:
            _impl = eltwise_prod_relu;
            break;
        case Eltwise_sum:
            if (param.coeff[0] == 1 && param.coeff[1] == 1)
                _impl = eltwise_sum_relu;
            else if (param.coeff[0] == 1 && param.coeff[1] == -1)
                _impl = eltwise_sub_relu;
            else
                _impl = eltwise_sum_coeff_relu;
            break;
        case Eltwise_max:
            _impl = eltwise_max_relu;
            break;
        default:
            LOG(ERROR) << "unknown eltwise type!!";
            return SaberUnKownError;
    }
    */
    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberEltwiseActive<ARM, OpDtype, inDtype, outDtype, \
LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
    const std::vector<DataTensor_in*>& inputs, \
    std::vector<DataTensor_out*>& outputs, \
    EltwiseActiveParam<OpTensor> &param) {

    const float* din_a = inputs[0]->data();
    const float* din_b = inputs[1]->data();
    float* dout = outputs[0]->mutable_data();

  // printf("threads compute begin:device.id: %d \n", this->_ctx.get_device_id());
   int threads = 1;
    this->_ctx->get_mode(threads);
   // printf("threads: %d\n", threads);
    int size = outputs[0]->valid_size();
    int num = size / threads;
  // printf("threads: %d, size: %d, num: %d\n", threads, size, num);
    if(param.has_activation && param.activation_param.active == 10){
       // LOG(INFO) << "Active_prelu";
        int numin = size / _channel_size;
        numin = numin / _channel;
        _impl(din_a, din_b, dout, _coeff, numin, _channel, _channel_size, _slop_ptr, _channel_shared);
        for (int i = 2; i < inputs.size(); ++i) {
            din_a = inputs[i]->data();
            _impl(din_a, dout, dout, _coeff, numin,  _channel, _channel_size, _slop_ptr, _channel_shared);
        }
    }else{
      //  LOG(INFO) << "Active_relu";
#pragma omp parallel for 
        for(int i = 0; i < size; i+=num){
            const float* din0_ptr = din_a + i;
            const float* din1_ptr = din_b + i;
            float* dout_ptr = dout + i;
            _impl(din0_ptr, din1_ptr, dout_ptr, _coeff, num, _channel, _channel_size,  _slop_ptr, _channel_shared);
        }
       // LOG(INFO) << "num: " << num;
        //_impl(din_a, din_b, dout, _coeff, size);
        for (int i = 2; i < inputs.size(); ++i) {
            din_a = inputs[i]->data();
            _impl(din_a, dout, dout, _coeff, size,  _channel, _channel_size, _slop_ptr, _channel_shared);
        }

    }
    return SaberSuccess;
}

template class SaberEltwiseActive<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE