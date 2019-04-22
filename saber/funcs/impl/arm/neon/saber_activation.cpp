
#include "saber/funcs/impl/arm/saber_activation.h"
#include "saber/funcs/impl/arm/neon/impl/neon_mathfun.h"
namespace anakin{
namespace saber {

void act_relu(const float* din, float* dout, int n, int c, int h, int w, \
                const ActivationParam<ARM>* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt = nums_per_thread >> 4;
    int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
    float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        int cnt = neon_loop_cnt;
#ifdef __aarch64__
        for (int num = 0; num < neon_loop_cnt; ++num){
            float32x4_t vr0 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr1 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr2 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr3 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            vr0=vmaxq_f32(vr0,vzero);
            vr1=vmaxq_f32(vr1,vzero);
            vr2=vmaxq_f32(vr2,vzero);
            vr3=vmaxq_f32(vr3,vzero);
            vst1q_f32(ptr_out_thread,vr0);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr1);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr2);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr3);
            ptr_out_thread+=4;
        }

#else
        if (cnt > 0) {
            asm volatile (
                "1:                                     @ loop header\n"
                "vld1.32  {d0-d3}, [%[din]]!            @ load din 0\n"
                "vld1.32  {d4-d7}, [%[din]]!            @ load din 0\n"

                "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
                "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
                "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
                "vmax.f32 q11, q3, %q[vzero]            @ relu\n"

                "vst1.32  {d16-d19}, [%[dout]]!         @ store result, add pointer\n"
                "vst1.32  {d20-d23}, [%[dout]]!         @ store result, add pointer\n"

                "subs %[cnt], #1                        @ loop count minus 1\n"
                "bne    1b                              @ jump to main loop start point\n"
            :[dout] "+r"(ptr_out_thread), [din] "+r"(ptr_in_thread), [cnt] "+r"(cnt)
            :[vzero] "w" (vzero)
            :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
            );
        }
#endif
        for (int j = 0; j < neon_loop_remain; ++j) {
            ptr_out_thread[0] = ptr_in_thread[0] > 0.f? ptr_in_thread[0] : 0.f;
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* out_ptr_remain = dout + threads * nums_per_thread;
    const float* in_ptr_remain = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        out_ptr_remain[0] = in_ptr_remain[0] > 0.f? in_ptr_remain[0] : 0.f;
        in_ptr_remain++;
        out_ptr_remain++;
    }
}

void act_relu_neg(const float* din, float* dout, int n, int c, int h, int w, const ActivationParam<ARM>* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt = nums_per_thread >> 4;
    int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t valpha = vdupq_n_f32(_param->negative_slope);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        int cnt = neon_loop_cnt;
#ifdef __aarch64__
        for (int num = 0; num < neon_loop_cnt; ++num){
        float32x4_t vr0 = vld1q_f32(ptr_in_thread);
        ptr_in_thread += 4;
        float32x4_t vr1 = vld1q_f32(ptr_in_thread);
        ptr_in_thread += 4;
        float32x4_t vr2 = vld1q_f32(ptr_in_thread);
        ptr_in_thread += 4;
        float32x4_t vr3 = vld1q_f32(ptr_in_thread);
        ptr_in_thread += 4;

        uint32x4_t vm0 = vcgeq_f32(vr0, vzero);
        uint32x4_t vm1 = vcgeq_f32(vr1, vzero);
        uint32x4_t vm2 = vcgeq_f32(vr2, vzero);
        uint32x4_t vm3 = vcgeq_f32(vr3, vzero);

        float32x4_t vn0 = vmulq_f32(vr0, valpha);
        float32x4_t vn1 = vmulq_f32(vr1, valpha);
        float32x4_t vn2 = vmulq_f32(vr2, valpha);
        float32x4_t vn3 = vmulq_f32(vr3, valpha);

        float32x4_t vo0 = vbslq_f32(vm0, vr0, vn0);
        float32x4_t vo1 = vbslq_f32(vm1, vr1, vn1);
        float32x4_t vo2 = vbslq_f32(vm2, vr2, vn2);
        float32x4_t vo3 = vbslq_f32(vm3, vr3, vn3);

        vst1q_f32(ptr_out_thread, vo0);
        ptr_out_thread += 4;
        vst1q_f32(ptr_out_thread, vo1);
        ptr_out_thread += 4;
        vst1q_f32(ptr_out_thread, vo2);
        ptr_out_thread += 4;
        vst1q_f32(ptr_out_thread, vo3);
        ptr_out_thread += 4;
    }

#else
        if (cnt > 0) {
            asm volatile (
            "1:                                             @ loop header\n"
                "vld1.32  {d0-d3}, [%[din]]!            @ load din 0\n"
                "vld1.32  {d4-d7}, [%[din]]!            @ load din 0\n"

                "vcge.f32 q8, q0, %q[vzero]             @ get mask\n"
                "vcge.f32 q9, q1, %q[vzero]             @ get mask\n"
                "vcge.f32 q10, q2, %q[vzero]            @ get mask\n"
                "vcge.f32 q11, q3, %q[vzero]            @ get mask\n"

                "vmul.f32   q4, q0, %q[valpha]          @ get neg data\n"
                "vmul.f32   q5, q1, %q[valpha]          @ get neg data\n"
                "vmul.f32   q6, q2, %q[valpha]          @ get neg data\n"
                "vmul.f32   q7, q3, %q[valpha]          @ get neg data\n"

                "vbit   q4, q0, q8                      @ bitsel, insert q0 to q4, if q8 is 1\n"
                "vbit   q5, q1, q9                      @ bitsel, insert q1 to q5, if q9 is 1\n"
                "vbit   q6, q2, q10                     @ bitsel, insert q2 to q6, if q10 is 1\n"
                "vbit   q7, q3, q11                     @ bitsel, insert q3 to q7, if q11 is 1\n"

                "vst1.32  {d8-d11}, [%[dout]]!          @ store result, add pointer\n"
                "vst1.32  {d12-d15}, [%[dout]]!         @ store result, add pointer\n"

                "subs %[cnt], #1                        @ loop count minus 1\n"
                "bne    1b                              @ jump to main loop start point\n"
            :[dout] "+r"(ptr_out_thread), [din] "+r"(ptr_in_thread), [cnt] "+r"(cnt)
            :[vzero] "w" (vzero), [valpha] "w" (valpha)
            :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
            );
        }
#endif
        for (int j = 0; j < neon_loop_remain; ++j) {
            ptr_out_thread[0] = ptr_in_thread[0] > 0.f? ptr_in_thread[0] : ptr_in_thread[0] * _param->negative_slope;
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* out_ptr_remain = dout + threads * nums_per_thread;
    const float* in_ptr_remain = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        out_ptr_remain[0] = in_ptr_remain[0] > 0.f? in_ptr_remain[0] : in_ptr_remain[0] * _param->negative_slope;
        in_ptr_remain++;
        out_ptr_remain++;
    }
}

void clipped_relu(const float* din, float* dout, int n, int c, int h, int w, const ActivationParam<ARM>* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt = nums_per_thread >> 4;
    int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vclip = vdupq_n_f32(_param->coef);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        int cnt = neon_loop_cnt;
#ifdef __aarch64__
        for (int num = 0; num < neon_loop_cnt; ++num){
            float32x4_t vr0 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr1 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr2 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr3 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vt0 = vmaxq_f32(vr0, vzero);
            float32x4_t vt1 = vmaxq_f32(vr1, vzero);
            float32x4_t vt2 = vmaxq_f32(vr2, vzero);
            float32x4_t vt3 = vmaxq_f32(vr3, vzero);

            float32x4_t vo0 = vminq_f32(vt0, vclip);
            float32x4_t vo1 = vminq_f32(vt1, vclip);
            float32x4_t vo2 = vminq_f32(vt2, vclip);
            float32x4_t vo3 = vminq_f32(vt3, vclip);

            vst1q_f32(ptr_out_thread,vo0);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vo1);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vo2);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vo3);
            ptr_out_thread+=4;
        }
#else
        if (cnt > 0) {
            asm volatile (
            "1:                                     @ loop header\n"
                    "vld1.32  {d0-d3}, [%[din]]!            @ load din 0\n"
                    "vld1.32  {d4-d7}, [%[din]]!            @ load din 0\n"

                    "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
                    "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
                    "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
                    "vmax.f32 q11, q3, %q[vzero]            @ relu\n"

                    "vmin.f32 q4, q8, %q[vclip]             @ clip relu\n"
                    "vmin.f32 q5, q9, %q[vclip]             @ clip relu\n"
                    "vmin.f32 q6, q10, %q[vclip]            @ clip relu\n"
                    "vmin.f32 q7, q11, %q[vclip]            @ clip relu\n"

                    "vst1.32  {d8-d11}, [%[dout]]!          @ store result, add pointer\n"
                    "vst1.32  {d12-d15}, [%[dout]]!         @ store result, add pointer\n"

                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    1b                              @ jump to main loop start point\n"
            :[dout] "+r"(ptr_out_thread), [din] "+r"(ptr_in_thread), [cnt] "+r"(cnt)
            :[vzero] "w" (vzero), [vclip] "w" (vclip)
            :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
            );
        }
#endif
        for (int j = 0; j < neon_loop_remain; ++j) {
            ptr_out_thread[0] = ptr_in_thread[0] > 0.f? ptr_in_thread[0] : 0.f;
            ptr_out_thread[0] = ptr_out_thread[0] < _param->coef? ptr_out_thread[0] : _param->coef;
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* out_ptr_remain = dout + threads * nums_per_thread;
    const float* in_ptr_remain = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        out_ptr_remain[0] = in_ptr_remain[0] > 0.f? in_ptr_remain[0] : 0.f;
        out_ptr_remain[0] = out_ptr_remain[0] < _param->coef? out_ptr_remain[0] : _param->coef;
        in_ptr_remain++;
        out_ptr_remain++;
    }
}

void act_prelu(const float* din, float* dout, int num, int channel, int h, int w, const ActivationParam<ARM>* _param, int threads) {
    int csize = w * h;
    int bsize = csize * channel;
    int cnt = csize >> 4;
    int remain = csize & 15;
    float32x4_t vzero = vdupq_n_f32(0.f);
    PreluParam<ARM> prelu_param = _param->prelu_param;
    float* prelu_weights = (float*)(prelu_param.slope->data());
    for (int n = 0; n < num; n++){
        const float* data_in_batch = din + n * bsize;
        float* data_out_batch = dout + n * bsize;
#pragma omp parallel for
        for (int c = 0; c < channel; c++){
            const float* data_in_c = data_in_batch + c * csize;
            float* data_out_c = data_out_batch + c * csize;

            float slope = prelu_param.channel_shared ? prelu_weights[0] : prelu_weights[c];
            float32x4_t vslope = vdupq_n_f32(slope);
#ifdef __aarch64__
            for (int i = 0;i < cnt; ++i){
                float32x4_t vr0 = vld1q_f32(data_in_c);
                float32x4_t vr1 = vld1q_f32(data_in_c + 4);
                float32x4_t vr2 = vld1q_f32(data_in_c + 8);
                float32x4_t vr3 = vld1q_f32(data_in_c + 12);
                uint32x4_t vm0 = vcltq_f32(vr0, vzero);//vr0 <= vzero
                uint32x4_t vm1 = vcltq_f32(vr1, vzero);//vr0 <= vzero
                uint32x4_t vm2 = vcltq_f32(vr2, vzero);//vr0 <= vzero
                uint32x4_t vm3 = vcltq_f32(vr3, vzero);//vr0 <= vzero
                float32x4_t vo0 = vmulq_f32(vr0, vslope);//vr0 * vslope
                float32x4_t vo1 = vmulq_f32(vr1, vslope);//vr0 * vslope
                float32x4_t vo2 = vmulq_f32(vr2, vslope);//vr0 * vslope
                float32x4_t vo3 = vmulq_f32(vr3, vslope);//vr0 * vslope
                float32x4_t vos0 = vbslq_f32(vm0, vo0, vr0);
                float32x4_t vos1 = vbslq_f32(vm1, vo1, vr1);
                float32x4_t vos2 = vbslq_f32(vm2, vo2, vr2);
                float32x4_t vos3 = vbslq_f32(vm3, vo3, vr3);
                vst1q_f32(data_out_c, vos0);
                vst1q_f32(data_out_c + 4, vos1);
                vst1q_f32(data_out_c + 8, vos2);
                vst1q_f32(data_out_c + 12, vos3);
                data_in_c += 16;
                data_out_c += 16;
            }
#else
            int cnt_loop = cnt;
            if (cnt_loop > 0) {
                asm volatile(
                    "vld1.32    {d0-d3}, [%[ptr_in]]!                       @ load input to q0, q1\n"
                    "pld [%[ptr_in]]                                @ preload\n"
                    "pld [%[ptr_in], #64]                           @ preload\n"
                    "pld [%[ptr_in], #128]                          @ preload\n"
                    "pld [%[ptr_in], #192]                          @ preload\n"
                    "1:                                             @main loop\n"
                    "vld1.32    {d4-d7}, [%[ptr_in]]!               @ load input to q2, q3\n"
                    "vclt.f32   q8, q0, %q[vzero]                   @vcle q0 <= vzero\n"
                    "vclt.f32   q9, q1, %q[vzero]                   @vcle q1 <= vzero\n"
                    "vmul.f32  q10, q0, %q[vslope]                  @vmul q0 * vslope\n"
                    "vmul.f32  q11, q1, %q[vslope]                  @vmul q1 * vslope\n"

                    "vclt.f32  q12, q2, %q[vzero]                   @vcle q2 <= vzero\n"
                    "vclt.f32  q13, q3, %q[vzero]                   @vcle q3 <= vzero\n"
                    "vmul.f32  q14, q2, %q[vslope]                  @vmul q2 * vslope\n"
                    "vmul.f32  q15, q3, %q[vslope]                  @vmul q3 * vslope\n"

                    "vbif.32    q10, q0, q8                         @vbit q10, q0, q8\n"
                    "vbif.32    q11, q1, q9                         @vbit q11, q1, q9\n"
                    "vbif.32    q14, q2, q12                        @vbit q14, q2, q12\n"
                    "vbif.32    q15, q3, q13                        @vbit q15, q3, q13\n"

                    "subs       %[cnt], #1                          @subs nn, 1\n"
                    "vld1.32    {d0-d3}, [%[ptr_in]]!               @ load input to q0, q1\n"

                    "vst1.f32   {d20-d23}, [%[dout]]!               @store data\n"
                    "vst1.f32   {d28-d31}, [%[dout]]!               @store data\n"
                    "bne        1b                                  @bne nn\n"
                    "sub    %[ptr_in], #32                          @ ptr-32\n"
                :[ptr_in] "+r" (data_in_c), [cnt] "+r" (cnt_loop), [dout] "+r" (data_out_c)
                :[vzero] "w" (vzero), [vslope] "w" (vslope)
                :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
#endif //__aarch64__
            for (int i = remain; i > 0; i--) {
                *(data_out_c++) = data_in_c[0] > 0.f? data_in_c[0] : data_in_c[0] * slope;
                data_in_c++;
            }
        }
    }
}

void act_sigmoid(const float* din, float* dout, int n, int c, int h, int w, const ActivationParam<ARM>* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt_dim4 = nums_per_thread >> 2;
    int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

    float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        float32x4_t exp_vec = vdupq_n_f32(0.0f);
        float32x4_t recip  = vdupq_n_f32(0.0f);
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        for (int k=0; k<neon_loop_cnt_dim4; ++k ) {
            exp_vec=exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
            exp_vec = vaddq_f32(exp_vec, vdupq_n_f32(1.0f));
            recip = vrecpeq_f32(exp_vec);
            recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
            recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
            vst1q_f32(ptr_out_thread, recip);
            ptr_out_thread+=4;
            ptr_in_thread+=4;
        }
        for (int j = 0; j < neon_loop_remain_dim4; ++j){
            ptr_out_thread[0] = 1.f / ( 1 + expf(-ptr_in_thread[0]));
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* ptr_out = dout + threads * nums_per_thread;
    const float* ptr_in = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] =  1.f / (1 + expf(-ptr_in[0]));
        ptr_in++;
        ptr_out++;
    }
}

// tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
void act_tanh(const float* din, float* dout, int n, int c, int h, int w, const ActivationParam<ARM>* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt_dim4 = nums_per_thread >> 2;
    int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        float32x4_t exp_plus_vec = vdupq_n_f32(0.0f);
        float32x4_t exp_minus_vec = vdupq_n_f32(0.0f);
        float32x4_t exp_sum_vec = vdupq_n_f32(0.0f);
        float32x4_t exp_diff_vec = vdupq_n_f32(0.0f);
        float32x4_t recip  = vdupq_n_f32(0.0f);
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        for (int k=0; k<neon_loop_cnt_dim4; ++k ) {
            exp_plus_vec=exp_ps(vld1q_f32(ptr_in_thread));
            exp_minus_vec=exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
            exp_sum_vec=vaddq_f32(exp_plus_vec,exp_minus_vec);
            exp_diff_vec=vsubq_f32(exp_plus_vec,exp_minus_vec);
            recip = div_ps(exp_diff_vec,exp_sum_vec);
            vst1q_f32(ptr_out_thread, recip);
            ptr_out_thread+=4;
            ptr_in_thread+=4;
        }
        for (int j = 0; j < neon_loop_remain_dim4; ++j){
            ptr_out_thread[0]=(expf(ptr_in_thread[0]) - expf(-ptr_in_thread[0])) / (expf(ptr_in_thread[0]) + expf(-ptr_in_thread[0]));
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* ptr_out = dout + threads * nums_per_thread;
    const float* ptr_in = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] = (expf(ptr_in[0]) - expf(-ptr_in[0])) / (expf(ptr_in[0]) + expf(-ptr_in[0]));
        ptr_in++;
        ptr_out++;
    }
}

//swish: x /(1 + exp(-(b * x)))
void act_swish(const float* din, float* dout, int n, int c, int h, int w, const ActivationParam<ARM>* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt_dim4 = nums_per_thread >> 2;
    int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
    const float beta = _param->coef;
    float32x4_t vbeta = vdupq_n_f32(beta);
    float32x4_t vone = vdupq_n_f32(1.f);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        for (int k = 0; k < neon_loop_cnt_dim4; ++k ) {
            float32x4_t va = vld1q_f32(ptr_in_thread); //x
            float32x4_t vb = vnegq_f32(vld1q_f32(ptr_in_thread)); //-x
            float32x4_t vsum = vmulq_f32(vb, vbeta);
            vsum = exp_ps(vsum);
            float32x4_t vc = vaddq_f32(vone, vsum);
            float32x4_t vrst = div_ps(va, vc);
            vst1q_f32(ptr_out_thread, vrst);
            ptr_out_thread+=4;
            ptr_in_thread+=4;
        }
        for (int j = 0; j < neon_loop_remain_dim4; ++j){
            ptr_out_thread[0] = ptr_in_thread[0] / (1.0 + expf(-ptr_in_thread[0] * beta));
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* ptr_out = dout + threads * nums_per_thread;
    const float* ptr_in = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] = ptr_in[0] / (1.0 + expf(-ptr_in[0] * beta));
        ptr_in++;
        ptr_out++;
    }
}


template<>
SaberStatus SaberActivation<ARM, AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ActivationParam<ARM>& param, Context<ARM>& ctx) {

    switch (param.active) {
        case Active_relu:
            if (fabsf( param.negative_slope) > 1e-6f) {
                _impl = act_relu_neg;
            } else {
                _impl = act_relu;
            }
            break;
        case Active_clipped_relu:
            _impl = clipped_relu;
            break;
        case Active_sigmoid:
            _impl = act_sigmoid;
            break;
        case Active_tanh:
            _impl = act_tanh;
            break;
        case Active_prelu:
            _impl = act_prelu;
            break;
        case Active_swish:
            _impl = act_swish;
            break;
        default:
            return SaberUnImplError;
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberActivation<ARM, AK_FLOAT>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ActivationParam<ARM>& param, Context<ARM>& ctx){
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberActivation<ARM, AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ActivationParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const float* ptr_in = static_cast<const float*>(inputs[0]->data());
    float* ptr_out = static_cast<float*>(outputs[0]->mutable_data());

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int threads = this->_ctx->get_threads();
    _impl(ptr_in, ptr_out, num, channel, height, width, &param, threads);

#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "activation " << this->_op_name.c_str() << " : " << (int)param.active <<", time: " << ts;
    GOPS ops;
    ops.ts = ts;
    ops.ops = 2.f * width * height * channel * num;
    OpTimer::add_timer("activation", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, ARM, AK_INT8);
}
} // namespace anakin
