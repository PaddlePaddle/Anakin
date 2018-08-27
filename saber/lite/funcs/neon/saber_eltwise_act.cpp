#include "saber/lite/funcs/saber_eltwise_act.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
void eltwise_prod_relu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
    int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr);

template <typename Dtype>
void eltwise_sum_relu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr);

template <typename Dtype>
void eltwise_max_relu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr);

template <typename Dtype>
void eltwise_prod_prelu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
    int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr);

template <typename Dtype>
void eltwise_sum_prelu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr);

template <typename Dtype>
void eltwise_max_prelu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr);


template <>
void eltwise_prod_relu(const float* din_a, const float* din_b, float* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int total = num * channel;

#pragma omp parallel for
    for(int n = 0; n < total; n++){
        float* out_ptr = dout + n * channel_size;
        const float* a_ptr = din_a + n * channel_size;
        const float* b_ptr = din_b + n * channel_size;
#ifdef __aarch64__
        for (int i = 0; i < cnt; ++i) {
            float32x4_t va0 = vld1q_f32(a_ptr);
            float32x4_t vb0 = vld1q_f32(b_ptr);
            float32x4_t va1 = vld1q_f32(a_ptr + 4);
            float32x4_t vb1 = vld1q_f32(b_ptr + 4);
            float32x4_t vout1 = vmulq_f32(va0, vb0);
            float32x4_t vout2 = vmulq_f32(va1, vb1);
            float32x4_t vout1_relu = vmaxq_f32(vout1, vzero);
            float32x4_t vout2_relu = vmaxq_f32(vout2, vzero);
            vst1q_f32(out_ptr, vout1_relu);
            vst1q_f32(out_ptr + 4, vout2_relu);
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
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vmax.f32 q8, q8, %q[vzero]             @ relu \n"
                "vmax.f32 q9, q9, %q[vzero]             @ relu \n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       prod_loop                    @ top_loop \n"
            :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
            :[vzero] "w" (vzero)
            :"q0", "q1", "q2", "q3", "q8", "q9"
            );
        }
#endif //__aarch64__

        for (; remain > 0; remain--) {
            float out = *(a_ptr++) * (*(b_ptr++));
            *(out_ptr++) = out > 0.f? out : 0.f;
        }
    }
}

void eltwise_sum_relu(const float* din_a, const float* din_b, float* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int total = num * channel;

#pragma omp parallel for
    for(int n = 0; n < total; n++){
        float* out_ptr = dout + n * channel_size;
        const float* a_ptr = din_a + n * channel_size;
        const float* b_ptr = din_b + n * channel_size;
#ifdef __aarch64__
        for (int i = 0; i < cnt; ++i) {
            float32x4_t va0 = vld1q_f32(a_ptr);
            float32x4_t vb0 = vld1q_f32(b_ptr);
            float32x4_t va1 = vld1q_f32(a_ptr + 4);
            float32x4_t vb1 = vld1q_f32(b_ptr + 4);
            float32x4_t vout1 = vaddq_f32(va0, vb0);
            float32x4_t vout2 = vaddq_f32(va1, vb1);
            float32x4_t vout1_relu = vmaxq_f32(vout1, vzero);
            float32x4_t vout2_relu = vmaxq_f32(vout2, vzero);
            vst1q_f32(out_ptr, vout1_relu);
            vst1q_f32(out_ptr + 4, vout2_relu);
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
                "vadd.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vadd.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vmax.f32 q8, q8, %q[vzero]             @ relu \n"
                "vmax.f32 q9, q9, %q[vzero]             @ relu \n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       sum_loop                     @ top_loop \n"
            :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
            :[vzero] "w" (vzero)
            :"q0", "q1", "q2", "q3", "q8", "q9"
            );
        }
#endif //__aarch64__

        for (; remain > 0; remain--) {
            float out = *(a_ptr++) + (*(b_ptr++));
            *(out_ptr++) = out > 0.f? out : 0.f;
        }
    }
}

void eltwise_max_relu(const float* din_a, const float* din_b, float* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, float* slop_ptr) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int total = num * channel;

#pragma omp parallel for
    for(int n = 0; n < total; n++){
        float* out_ptr = dout + n * channel_size;
        const float* a_ptr = din_a + n * channel_size;
        const float* b_ptr = din_b + n * channel_size;
#ifdef __aarch64__
        for (int i = 0; i < cnt; ++i) {
            float32x4_t va0 = vld1q_f32(a_ptr);
            float32x4_t vb0 = vld1q_f32(b_ptr);
            float32x4_t va1 = vld1q_f32(a_ptr + 4);
            float32x4_t vb1 = vld1q_f32(b_ptr + 4);
            float32x4_t vout1 = vmaxq_f32(va0, vb0);
            vst1q_f32(out_ptr, vout1);
            float32x4_t vout2 = vmaxq_f32(va1, vb1);
            vst1q_f32(out_ptr + 4, vout2);
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
                "vmax.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vmax.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vmax.f32 q8, q8, %q[vzero]             @ relu \n"
                "vmax.f32 q9, q9, %q[vzero]             @ relu \n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       max_loop                     @ top_loop \n"
            :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
            :[vzero] "w" (vzero)
            :"q0", "q1", "q2", "q3", "q8", "q9"
            );
        }
#endif //__aarch64__

        for (; remain > 0; remain--) {
            *(out_ptr++) = std::max(*(a_ptr++), *(b_ptr++));
        }
    }
}

//prelu
void eltwise_prod_prelu(const float* din_a, const float* din_b, float* dout, const int num, \
    int channel, int channel_size, std::vector<float> coeff, bool channel_shared, float* slop_ptr) {

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

void eltwise_sum_prelu(const float* din_a, const float* din_b, float* dout, const int num, \
    int channel, int channel_size, std::vector<float> coeff, bool channel_shared, float* slop_ptr) {

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

void eltwise_max_prelu(const float* din_a, const float* din_b, float* dout, const int num, \
    int channel, int channel_size, std::vector<float> coeff, bool channel_shared, float* slop_ptr) {

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

SaberEltwiseAct::SaberEltwiseAct(const ParamBase *param) {
    _param = (const EltwiseActParam*)param;
    this->_flag_param = true;
}

SaberEltwiseAct::~SaberEltwiseAct() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberEltwiseAct::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const EltwiseActParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberEltwiseAct::load_param(std::istream &stream, const float *weights) {
    int type;
    int size;
    std::vector<float> coef;

    stream >> type >> size;
    coef.resize(size);
    for (int i = 0; i < size; ++i) {
        stream >> coef[i];
    }
    EltwiseType etype = static_cast<EltwiseType>(type);
   // EltwiseParam* elt_param = new EltwiseParam(etype, coef);

    int act_type;
    float slop;
    float act_coef;
    bool channel_shared;
    int w_offset;

    stream >> act_type >> slop >> act_coef >> channel_shared >> w_offset;
    ActiveType atype = static_cast<ActiveType>(act_type);
   // ActivationParam* act_param = new ActivationParam(atype, slop, act_coef, channel_shared, weights + w_offset);

    _param = new EltwiseActParam(etype, coef, atype, slop, act_coef, channel_shared, weights + w_offset);
    this->_flag_create_param = true;
    this->_flag_param  =true;
    return SaberSuccess;
}

SaberStatus SaberEltwiseAct::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_param) {
        printf("load eltwise param first\n");
        return SaberNotInitialized;
    }

    for (int i = 1; i < inputs.size(); ++i) {
        LCHECK_EQ(inputs[0]->num(), inputs[i]->num(), "input size must be the same");
        LCHECK_EQ(inputs[0]->channel(), inputs[i]->channel(), "input size must be the same");
        LCHECK_EQ(inputs[0]->height(), inputs[i]->height(), "input size must be the same");
        LCHECK_EQ(inputs[0]->width(), inputs[i]->width(), "input size must be the same");
    }

    Shape output_shape = inputs[0]->valid_shape();
    return outputs[0]->set_shape(output_shape);
}

//template <typename Dtype>
SaberStatus SaberEltwiseAct::init(\
    const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
        std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) {

    if (!this->_flag_param) {
        printf("load eltwise param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;
    Shape sh_out_saber = outputs[0]->valid_shape();
    for (int i = 0; i < inputs.size(); i ++){
        Shape sh_in_saber = inputs[i]->valid_shape();
        if (sh_out_saber != sh_in_saber){
            printf("input shape is not same with output shape\n");
            return SaberInvalidValue;
        }
    }
    if(_param->_has_activation){
        if(_param->_activation_param._act_type == 2){//relu
            switch(_param->_eltwise_param._elt_type){
                case Eltwise_prod:
                    _impl = eltwise_prod_relu;
                break;
                case Eltwise_sum:
                    _impl = eltwise_sum_relu;
                break;
                case Eltwise_max:
                    _impl = eltwise_max_relu;
                break;
                default:
                    printf("unknown eltwise type!!\n");
                return SaberUnKownError;
            }
        }
        if(_param->_activation_param._act_type == 10){//prelu
            switch(_param->_eltwise_param._elt_type){
                case Eltwise_prod:
                    _impl = eltwise_prod_prelu;
                break;
                case Eltwise_sum:
                    _impl = eltwise_sum_prelu;
                break;
                case Eltwise_max:
                    _impl = eltwise_max_prelu;
                break;
                default:
                    printf("unknown eltwise type!!\n");
                return SaberUnKownError;
            }
        }
    }

    this->_flag_init = true;
    return SaberSuccess;
}

//template <typename Dtype>
SaberStatus SaberEltwiseAct::dispatch(\
    const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
    std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {

    if (!this->_flag_init) {
        printf("init eltwise first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    const float* din_a = inputs[0]->data();
    const float* din_b = inputs[1]->data();
    float* dout = outputs[0]->mutable_data();

    //int size = outputs[0]->valid_size();
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int channel_size = inputs[0]->width() * inputs[0]->height();
    if(_param->_has_activation && _param->_activation_param._act_type == 2){//relu
        _impl(din_a, din_b, dout, num, channel, channel_size, _param->_eltwise_param._coef, false, nullptr);
        for (int i = 2; i < inputs.size(); ++i) {
            din_a = inputs[i]->data();
            _impl(din_a, dout, dout, num, channel, channel_size, _param->_eltwise_param._coef, false, nullptr);
        }
    }
    if(_param->_has_activation && _param->_activation_param._act_type == 10){//relu
        _impl(din_a, din_b, dout, num, channel, channel_size, _param->_eltwise_param._coef,  \
             _param->_activation_param._prelu_channel_shared, _param->_activation_param._prelu_weights);
        for (int i = 2; i < inputs.size(); ++i) {
            din_a = inputs[i]->data();
            _impl(din_a, dout, dout, num, channel, channel_size, _param->_eltwise_param._coef,  \
             _param->_activation_param._prelu_channel_shared, _param->_activation_param._prelu_weights);
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("eltwise act: %s: time: %f\n", this->_op_name.c_str(), ts);
    OpTimer::add_timer("EltwiseAct", ts);
    OpTimer::add_timer("total", ts);
#endif

    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberEltwiseAct);
} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE