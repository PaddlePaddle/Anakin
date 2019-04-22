#include "saber/funcs/impl/arm/saber_eltwise_act.h"

namespace anakin{

namespace saber{

template <typename Dtype>
void eltwise_prod_prelu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
    int channel, int channel_size, std::vector<float> coef, bool channel_shared, const float* slop_ptr);

template <typename Dtype>
void eltwise_sum_prelu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, const float* slop_ptr);

template <typename Dtype>
void eltwise_max_prelu(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int num, \
     int channel, int channel_size, std::vector<float> coef, bool channel_shared, const float* slop_ptr);


void eltwise_prod_prelu(const float* din_a, const float* din_b, float* dout, const int num, \
    int channel, int channel_size, std::vector<float> coeff, bool channel_shared, const float* slop_ptr) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int size = channel * channel_size;
    for (int n = 0; n < num; n++){
        const float* dina_ptr = din_a + n * size;
        const float* dinb_ptr = din_b + n * size;
        float* dout_ptr = dout + n * size;
#pragma omp parallel for
        for (int c = 0; c < channel; c++){
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
                :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
                );
            }
#endif //__aarch64__
        int remain_rst = remain;
        for (; remain_rst > 0; remain_rst--) {
                float tmp = *(a_ptr++) * (*(b_ptr++));
                *(out_ptr++) = tmp > 0 ? tmp : tmp * slope_val;
            }
        }
    }
}

void eltwise_sum_prelu(const float* din_a, const float* din_b, float* dout, const int num, \
    int channel, int channel_size, std::vector<float> coeff, bool channel_shared, const float* slop_ptr) {

    int cnt = channel_size >> 4;
    int remain = channel_size & 15;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int size = channel * channel_size;
    for (int n = 0; n < num; n++){
        const float* dina_ptr = din_a + n * size;
        const float* dinb_ptr = din_b + n * size;
        float* dout_ptr = dout + n * size;
#pragma omp parallel for
        for (int c = 0; c < channel; c++){
            const float* a_ptr = dina_ptr + c * channel_size;
            const float* b_ptr = dinb_ptr + c * channel_size;
            float* out_ptr = dout_ptr + c * channel_size;
            float slope_val = channel_shared ? slop_ptr[0] : slop_ptr[c];
            float32x4_t vslope = vdupq_n_f32(slope_val);
#ifdef __aarch64__
            float32x4_t va0 = vld1q_f32(a_ptr);
            float32x4_t vb0 = vld1q_f32(b_ptr);
            float32x4_t va1 = vld1q_f32(a_ptr + 4);
            float32x4_t vb1 = vld1q_f32(b_ptr + 4);
            float32x4_t va2 = vld1q_f32(a_ptr + 8);
            float32x4_t vb2 = vld1q_f32(b_ptr + 8);
            float32x4_t va3 = vld1q_f32(a_ptr + 12);
            float32x4_t vb3 = vld1q_f32(b_ptr + 12);
            for (int i = 0; i < cnt; ++i) {

                float32x4_t vsum0 = vaddq_f32(va0, vb0);
                float32x4_t vsum1 = vaddq_f32(va1, vb1);
                float32x4_t vsum2 = vaddq_f32(va2, vb2);
                b_ptr += 16;
                float32x4_t vsum3 = vaddq_f32(va3, vb3);
                a_ptr += 16;

                uint32x4_t vmask0 = vcltq_f32(vsum0, vzero);//vsum1 <= vzero
                float32x4_t vout0 = vmulq_f32(vsum0, vslope);//vsum1 * vslope
                uint32x4_t vmask1 = vcltq_f32(vsum1, vzero);//vsum2 <= vzero
                float32x4_t vout1 = vmulq_f32(vsum1, vslope);//vsum2 * vslope
                uint32x4_t vmask2 = vcltq_f32(vsum2, vzero);//vsum2 <= vzero
                float32x4_t vout2 = vmulq_f32(vsum2, vslope);//vsum2 * vslope
                uint32x4_t vmask3 = vcltq_f32(vsum3, vzero);//vsum2 <= vzero
                float32x4_t vout3 = vmulq_f32(vsum3, vslope);//vsum2 * vslope

                float32x4_t vout_sel0 = vbslq_f32(vmask0, vout0, vsum0);
                va0 = vld1q_f32(a_ptr);
                float32x4_t vout_sel1 = vbslq_f32(vmask1, vout1, vsum1);
                vb0 = vld1q_f32(b_ptr);
                float32x4_t vout_sel2 = vbslq_f32(vmask2, vout2, vsum2);
                va1 = vld1q_f32(a_ptr + 4);
                float32x4_t vout_sel3 = vbslq_f32(vmask3, vout3, vsum3);
                vb1 = vld1q_f32(b_ptr + 4);

                vst1q_f32(out_ptr, vout_sel0);
                va2 = vld1q_f32(a_ptr + 8);
                vst1q_f32(out_ptr + 4, vout_sel1);
                vb2 = vld1q_f32(b_ptr + 8);
                vst1q_f32(out_ptr + 8, vout_sel2);
                va3 = vld1q_f32(a_ptr + 12);
                vst1q_f32(out_ptr + 12, vout_sel3);
                vb3 = vld1q_f32(b_ptr + 12);

                out_ptr += 16;
            }

#else
            int loop_cnt = cnt;
            if (loop_cnt > 0) {
                asm volatile(
                "pld [%[a_ptr]]                             @ preload data\n"
                        "pld [%[b_ptr]]                         @ preload data\n"
                        "vld1.f32  {d0-d3}, [%[a_ptr]]!         @ load din a\n"
                        "pld [%[a_ptr], #64]                    @ preload data\n"
                        "vld1.f32  {d8-d11},[%[b_ptr]]!         @ load din b\n"
                        "pld [%[b_ptr], #64]                    @ preload data\n"
                        "2:                                     @ main loop start point\n"
                        "vld1.f32  {d4-d7}, [%[a_ptr]]!         @ load din a\n"
                        "vld1.f32 {d12-d15},[%[b_ptr]]!         @ load din b\n"
                        "vadd.f32  q8, q0, q4                   @ q8 = q0 + q4\n"
                        "vadd.f32  q9, q1, q5                   @ q9 = q1 + q5\n"

                        "vadd.f32  q10, q2, q6                  @ q10 = q2 + q6\n"
                        "vadd.f32  q11, q3, q7                  @ q11 = q3 + q7\n"

                        "vclt.f32   q0, q8, %q[vzero]           @vcle q8 <= vzero\n"
                        "vmul.f32   q1, q8, %q[vslope]          @vmul q8 * vslope\n"

                        "vclt.f32   q2, q9, %q[vzero]           @vcle q8 <= vzero\n"
                        "vmul.f32   q3, q9, %q[vslope]          @vmul q8 * vslope\n"

                        "vclt.f32   q4, q10, %q[vzero]          @vcle q8 <= vzero\n"
                        "vmul.f32   q5, q10, %q[vslope]         @vmul q8 * vslope\n"

                        "vclt.f32   q6, q11, %q[vzero]          @vcle q8 <= vzero\n"
                        "vmul.f32   q7, q11, %q[vslope]         @vmul q8 * vslope\n"

                        "vbit.32    q8, q1, q0                  @vbit q0, q1, q0\n"
                        "vbit.32    q9, q3, q2                  @vbit q0, q1, q0\n"
                        "vbit.32    q10, q5, q4                 @vbit q0, q1, q0\n"
                        "vbit.32    q11, q7, q6                 @vbit q0, q1, q0\n"

                        "subs      %[loop_cnt], #1              @ loop --\n"
                        "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                        "vld1.f32  {d0-d3}, [%[a_ptr]]!         @ load din a\n"
                        "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                        "vld1.f32  {d8-d11}, [%[b_ptr]]!        @ load din b\n"
                        "vst1.f32 {d20-d21}, [%[out_ptr]]!      @ store data\n"
                        "vst1.f32 {d22-d23}, [%[out_ptr]]!      @ store data\n"
                        "bne       2b                           @ top_loop \n"
                :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
                [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), [vzero] "+w" (vzero), [vslope] "+w" (vslope)
                :
                :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
                );
            }
#endif //__aarch64__

            for (int i = remain; i > 0; i--) {
                float tmp = *(a_ptr++) + (*(b_ptr++));
                *(out_ptr++) = tmp > 0 ? tmp : tmp * slope_val;
            }
        }
    }
}

void eltwise_max_prelu(const float* din_a, const float* din_b, float* dout, const int num, \
    int channel, int channel_size, std::vector<float> coeff, bool channel_shared, const float* slop_ptr) {

    int cnt = channel_size >> 3;
    int remain = channel_size & 7;
    float32x4_t vzero = vdupq_n_f32(0.f);
    int size = channel * channel_size;
    for (int n = 0; n < num; n++) {
        const float* dina_ptr = din_a + n * size;
        const float* dinb_ptr = din_b + n * size;
        float* dout_ptr = dout + n * size;
#pragma omp parallel for
        for (int c = 0; c <channel; c++){
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
                :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
                );
            }
#endif //__aarch64__

            for (int j = remain; j > 0; j--) {
                float tmp = *a_ptr > *b_ptr ? *a_ptr : *b_ptr;
                a_ptr++;
                b_ptr++;
                *(out_ptr++) = tmp > 0 ? tmp : tmp * slope_val;
            }
        }
    }
}

template <>
SaberStatus SaberEltwiseActive<ARM, AK_FLOAT>::create(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        EltwiseActiveParam<ARM> &param,
        Context<ARM> &ctx) {

    if (param.activation_param.active == Active_prelu){
        switch (param.eltwise_param.operation){
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
                LOG(ERROR) << "ERROR: unknown eltwise type!!";
                return SaberUnKownError;
        }
    } else {
        LOG(ERROR) << "unsupport activation except prelu";
        return SaberUnImplError;
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberEltwiseActive<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        EltwiseActiveParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const void* din_a = inputs[0]->data();
    const void* din_b = inputs[1]->data();
    void* dout = outputs[0]->mutable_data();
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int channel_size = inputs[0]->width() * inputs[0]->height();
    PreluParam<ARM> prelu_param = param.activation_param.prelu_param;
    _impl(din_a, din_b, dout, num, channel, channel_size, param.eltwise_param.coeff,  \
            prelu_param.channel_shared, static_cast<const float*>(prelu_param.slope->data()));
    for (int i = 2; i < inputs.size(); ++i) {
        din_a = inputs[i]->data();
        _impl(din_a, din_b, dout, num, channel, channel_size, param.eltwise_param.coeff,  \
            prelu_param.channel_shared, static_cast<const float*>(prelu_param.slope->data()));
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Eltwise act: " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("EltwiseAct", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberEltwiseActive, EltwiseActiveParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberEltwiseActive, EltwiseActiveParam, ARM, AK_INT8);
} //namespace anakin

} //namespace anakin
