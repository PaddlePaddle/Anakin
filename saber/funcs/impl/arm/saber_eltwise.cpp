#include "saber/funcs/impl/arm/saber_eltwise.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{
void eltwise_prod(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

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
        "prod_loop:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vmul.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vmul.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       prod_loop                    @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = *(a_ptr++) * (*(b_ptr++));
    }
}

void eltwise_sum(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

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
        "sum_loop:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vadd.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vadd.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       sum_loop                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = *(a_ptr++) + (*(b_ptr++));
    }
}

void eltwise_sub(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

    int cnt = size >> 3;
    int remain = size & 7;
#ifdef __aarch64__
    for (int i = 0; i < cnt; ++i) {
        float32x4_t va0 = vld1q_f32(a_ptr);
        float32x4_t vb0 = vld1q_f32(b_ptr);
        float32x4_t va1 = vld1q_f32(a_ptr + 4);
        float32x4_t vb1 = vld1q_f32(b_ptr + 4);
        float32x4_t vout1 = vsubq_f32(va0, vb0);
        vst1q_f32(out_ptr, vout1);
        float32x4_t vout2 = vsubq_f32(va1, vb1);
        vst1q_f32(out_ptr + 4, vout2);
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
                "vsub.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vsub.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       sub_loop                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = *(a_ptr++) - (*(b_ptr++));
    }
}
void eltwise_sum_coeff(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

    int cnt = size >> 3;
    int remain = size & 7;
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
        vst1q_f32(out_ptr, vout1);
        float32x4_t vout2 = vmulq_f32(va1, vcoef0);
        vout2 = vmlaq_f32(vout2, vb1, vcoef1);
        vst1q_f32(out_ptr + 4, vout2);
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
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       sum_coeff_loop               @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr), \
            [vcoef0] "+w" (vcoef0), [vcoef1] "+w" (vcoef1)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = *(a_ptr++) * coeff[0] + (*(b_ptr++)) * coeff[1];
    }
}

void eltwise_max(const float* din_a, const float* din_b, float* dout, std::vector<float> coeff, const int size) {

    float* out_ptr = dout;
    const float* a_ptr = din_a;
    const float* b_ptr = din_b;

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
        "max_loop:                                         @ main loop start point\n"
                "vld1.f32  {d0-d1}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d2-d3}, [%[b_ptr]]!         @ load din r1n\n"
                "vld1.f32  {d4-d5}, [%[a_ptr]]!         @ load din r0\n"
                "vld1.f32  {d6-d7}, [%[b_ptr]]!         @ load din r1n\n"
                "vmax.f32  q8, q0, q1                   @ q8 = q0 * q1\n"
                "vmax.f32  q9, q2, q3                   @ q9 = q2 * q3\n"
                "subs      %[loop_cnt], #1              @ loop --\n"
                "vst1.f32 {d16-d17}, [%[out_ptr]]!      @ store data\n"
                "vst1.f32 {d18-d19}, [%[out_ptr]]!      @ store data\n"
                "bne       max_loop                     @ top_loop \n"
        :[loop_cnt] "+r" (loop_cnt), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = std::max(*(a_ptr++), *(b_ptr++));
    }
}


template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberEltwise<ARM, OpDtype, inDtype, outDtype, \
LayOutType_op, LayOutType_in, LayOutType_out>::create(\
    const std::vector<DataTensor_in*>& inputs,\
        std::vector<DataTensor_out*>& outputs,\
        EltwiseParam<OpTensor> &param, \
        Context<ARM> &ctx) {
    this->_ctx = &ctx;
    this->_coeff = param.coeff;
    Shape sh_out_saber = outputs[0]->valid_shape();
    for (int i = 0; i < inputs.size(); i ++){
        Shape sh_in_saber = inputs[i]->valid_shape();
        if (sh_out_saber != sh_in_saber){
                    LOG(INFO) << "input shape is not same with output shape ";
            return SaberInvalidValue;
        }
    }
    switch (param.operation) {
        case Eltwise_prod:
            _impl = eltwise_prod;
            break;
        case Eltwise_sum:
            if (param.coeff[0] == 1 && param.coeff[1] == 1)
                _impl = eltwise_sum;
            else if (param.coeff[0] == 1 && param.coeff[1] == -1)
                _impl = eltwise_sub;
            else
                _impl = eltwise_sum_coeff;
            break;
        case Eltwise_max:
            _impl = eltwise_max;
            break;
        default:
            LOG(ERROR) << "unknown eltwise type!!";
            return SaberUnKownError;
    }
    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberEltwise<ARM, OpDtype, inDtype, outDtype, \
LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
    const std::vector<DataTensor_in*>& inputs, \
    std::vector<DataTensor_out*>& outputs, \
    EltwiseParam<OpTensor> &param) {

    const float* din_a = inputs[0]->data();
    const float* din_b = inputs[1]->data();
    float* dout = outputs[0]->mutable_data();

    int size = outputs[0]->valid_size();

    _impl(din_a, din_b, dout, _coeff, size);
    for (int i = 2; i < inputs.size(); ++i) {
        din_a = inputs[i]->data();
        _impl(din_a, dout, dout, _coeff, size);
    }

    return SaberSuccess;
}

template class SaberEltwise<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE