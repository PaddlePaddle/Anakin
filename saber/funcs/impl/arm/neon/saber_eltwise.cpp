#include "saber/funcs/impl/arm/saber_eltwise.h"

namespace anakin{

namespace saber{

void eltwise_prod(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef) {

    float* out_ptr = static_cast<float*>(dout);
    const float* a_ptr = static_cast<const float*>(din_a);
    const float* b_ptr = static_cast<const float*>(din_b);

    int cnt = size >> 3;
    int remain = size & 7;
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vmulq_f32(va0, vb0);
        vst1q_f32(out_ptr, vout1);
        float32x4_t vout2 = vmulq_f32(va1, vb1);
        vst1q_f32(out_ptr + 4, vout2);
        a_ptr += 8;
        b_ptr += 8;
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
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       1b                    @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = *(a_ptr++) * (*(b_ptr++));
    }
}

void eltwise_prod_relu(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef) {

    float* out_ptr = static_cast<float*>(dout);
    const float* a_ptr = static_cast<const float*>(din_a);
    const float* b_ptr = static_cast<const float*>(din_b);
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
        vout1 = vmaxq_f32(vout1, vzero);
        vst1q_f32(out_ptr, vout1);
        float32x4_t vout2 = vmulq_f32(va1, vb1);
        vout2 = vmaxq_f32(vout2, vzero);
        vst1q_f32(out_ptr + 4, vout2);
        a_ptr += 8;
        b_ptr += 8;
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
                "vmax.f32  q8, %q[vzero]                @ relu\n"
                "vmax.f32  q9, %q[vzero]                @ relu\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       1b                    @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        : [vzero] "w" (vzero)
        :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = std::max(*(a_ptr++) * (*(b_ptr++)), 0.f);
    }
}

void eltwise_sum(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef) {
    float* out_ptr = static_cast<float*>(dout);
    const float* a_ptr = static_cast<const float*>(din_a);
    const float* b_ptr = static_cast<const float*>(din_b);

    int cnt = size >> 3;
    int remain = size & 7;
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vaddq_f32(va0, vb0);
        vst1q_f32(out_ptr, vout1);
        float32x4_t vout2 = vaddq_f32(va1, vb1);
        vst1q_f32(out_ptr + 4, vout2);
        a_ptr += 8;
        b_ptr += 8;
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
                "vadd.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vadd.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       1b                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = *(a_ptr++) + (*(b_ptr++));
    }
}

void eltwise_sum_relu(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef) {
    float* out_ptr = static_cast<float*>(dout);
    const float* a_ptr = static_cast<const float*>(din_a);
    const float* b_ptr = static_cast<const float*>(din_b);
    float32x4_t vzero = vdupq_n_f32(0.f);
    int cnt = size >> 3;
    int remain = size & 7;
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vaddq_f32(va0, vb0);
        vout1 = vmaxq_f32(vout1, vzero);
        vst1q_f32(out_ptr, vout1);
        float32x4_t vout2 = vaddq_f32(va1, vb1);
        vout2 = vmaxq_f32(vout2, vzero);
        vst1q_f32(out_ptr + 4, vout2);
        a_ptr += 8;
        b_ptr += 8;
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
                "vadd.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vadd.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "vmax.f32  q8, %q[vzero]                @ relu\n"
                "vmax.f32  q9, %q[vzero]                @ relu\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       1b                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :[vzero] "w" (vzero)
        :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = std::max(*(a_ptr++) + (*(b_ptr++)), 0.f);
    }
}



void eltwise_max(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef) {

    float* out_ptr = static_cast<float*>(dout);
    const float* a_ptr = static_cast<const float*>(din_a);
    const float* b_ptr = static_cast<const float*>(din_b);

    int cnt = size >> 3;
    int remain = size & 7;
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
        "1:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vmax.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vmax.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       1b                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = std::max(*(a_ptr++), *(b_ptr++));
    }
}

void eltwise_max_relu(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef) {

    float* out_ptr = static_cast<float*>(dout);
    const float* a_ptr = static_cast<const float*>(din_a);
    const float* b_ptr = static_cast<const float*>(din_b);

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
        vout1 = vmaxq_f32(vout1, vzero);
        vst1q_f32(out_ptr, vout1);
        float32x4_t vout2 = vmaxq_f32(va1, vb1);
        vout2 = vmaxq_f32(vout2, vzero);
        vst1q_f32(out_ptr + 4, vout2);
        a_ptr += 8;
        b_ptr += 8;
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
                "vmax.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vmax.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "vmax.f32  q8, %q[vzero]                @ relu\n"
                "vmax.f32  q9, %q[vzero]                @ relu\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       1b                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :[vzero] "w" (vzero)
        :"cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = std::max(std::max(*(a_ptr++), *(b_ptr++)), 0.f);
    }
}


template <>
SaberStatus SaberEltwise<ARM, AK_FLOAT>::create(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        EltwiseParam<ARM> &param,
        Context<ARM> &ctx) {

    if (param.activation_param.active == Active_relu){
        switch (param.operation){
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
                LOG(ERROR) << "ERROR: unknown eltwise type!!";
                return SaberUnKownError;
        }
    } else {
        switch (param.operation){
            case Eltwise_prod:
                _impl = eltwise_prod;
                break;
            case Eltwise_sum:
                _impl = eltwise_sum;
                break;
            case Eltwise_max:
                _impl = eltwise_max;
                break;
            default:
                LOG(ERROR) << "ERROR: unknown eltwise type!!";
                return SaberUnKownError;
        }
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberEltwise<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        EltwiseParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const void* din_a = inputs[0]->data();
    const void* din_b = inputs[1]->data();
    void* dout = outputs[0]->mutable_data();
    int size = outputs[0]->valid_size();
    _impl(din_a, din_b, dout, size, param.coeff);
    for (int i = 2; i < inputs.size(); ++i) {
        din_a = inputs[i]->data();
        _impl(din_a, dout, dout, size, param.coeff);
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Eltwise : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("Eltwise", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberEltwise, EltwiseParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberEltwise, EltwiseParam, ARM, AK_INT8);

} //namespace anakin

} //namespace anakin
