#include "saber/lite/funcs/saber_eltwise.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

void eltwise_prod(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef);

void eltwise_sum(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef);

void eltwise_max(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef);

void eltwise_prod_int8(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef);

void eltwise_sum_int8(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef);

void eltwise_max_int8(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef);


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


void eltwise_sum_int8(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef){
    //LOGI("use eltwise_sum_int8\n");
    char_t* out_ptr = static_cast<char_t*>(dout);
    const char_t* a_ptr = static_cast<const char_t*>(din_a);
    const char_t* b_ptr = static_cast<const char_t*>(din_b);

    int remain = 0;
    int cnt_32 = size >> 5;
    remain = size & 31;
    int cnt_16 = remain >> 4;
    remain = cnt_16 & 15;
    int cnt_8 = remain >> 3;
    remain = cnt_8 & 7;

    int loop_cnt_32 = cnt_32;
    int loop_cnt_16 = cnt_16;
    int loop_cnt_8 = cnt_8;
#ifdef __aarch64__

    if (loop_cnt_32 > 0) {
        asm volatile(
            "1:\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "ld1 {v1.16b}, [%[b_ptr]], #16\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "ld1 {v3.16b}, [%[b_ptr]], #16\n"
            "add v4.16b, v0.16b, v1.16b\n"
            "add v5.16b, v2.16b, v3.16b\n"
            "subs %[loop_cnt], %[loop_cnt], #1\n"
            "st1 {v4.16b}, [%[out_ptr]], #16\n"
            "st1 {v5.16b}, [%[out_ptr]], #16\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_32), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"v0", "v1", "v2", "v3", "v4", "v5"
        );
    }
    if (loop_cnt_16 > 0) {
        asm volatile(
            "1:\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "ld1 {v1.16b}, [%[b_ptr]], #16\n"
            "add v2.16b, v0.16b, v1.16b\n"
            "subs %[loop_cnt], %[loop_cnt], #1\n"
            "st1 {v2.16b}, [%[out_ptr]], #16\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_16), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"v0", "v1", "v2"
        );
    }
    if (loop_cnt_8 > 0) {
        asm volatile(
            "1:\n"
            "ld1 {v0.8b}, [%[a_ptr]], #8\n"
            "ld1 {v1.8b}, [%[b_ptr]], #8\n"
            "add v2.8b, v0.8b, v1.8b\n"
            "subs %[loop_cnt], %[loop_cnt], #1\n"
            "st1 {v2.8b}, [%[out_ptr]], #8\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_8), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"v0", "v1", "v2"
        );
    }
#else

    if (loop_cnt_32 > 0) {
        asm volatile(
            "1:\n"
            "vld1.8 {d0-d1}, [%[a_ptr]]!\n"
            "vld1.8 {d2-d3}, [%[b_ptr]]!\n"
            "vld1.8 {d4-d5}, [%[a_ptr]]!\n"
            "vld1.8 {d6-d7}, [%[b_ptr]]!\n"
            "vadd.s8 q4, q1, q0\n"
            "vadd.s8 q5, q3, q2\n"
            "subs %[loop_cnt], #1\n"
            "vst1.8 {d8-d9}, [%[out_ptr]]!\n"
            "vst1.8 {d10-d11}, [%[out_ptr]]!\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_32), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2", "q3", "q4", "q5"
        );
    }
    if (loop_cnt_16 > 0) {
        asm volatile(
            "1:\n"
            "vld1.8 {d0-d1}, [%[a_ptr]]!\n"
            "vld1.8 {d2-d3}, [%[b_ptr]]!\n"
            "vadd.s8 q2, q0, q1\n"
            "subs %[loop_cnt], #1\n"
            "vst1.8 {d4-d5}, [%[out_ptr]]!\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_16), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2"
        );
    }
    if (loop_cnt_8 > 0) {
        asm volatile(
            "1:\n"
            "vld1.8 {d0}, [%[a_ptr]]!\n"
            "vld1.8 {d2}, [%[b_ptr]]!\n"
            "vadd.s8 q2, q1, q2\n"
            "subs %[loop_cnt], #1\n"
            "vst1.s8 {d4}, [%[out_ptr]]!\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_8), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = *(a_ptr++) + (*(b_ptr++));
    }
}


void eltwise_max_int8(const void* din_a, const void* din_b, void* dout, const int size, \
    std::vector<float> coef){
    //LOGI("use eltwise_max_int8\n");
    char_t* out_ptr = static_cast<char_t*>(dout);
    const char_t* a_ptr = static_cast<const char_t*>(din_a);
    const char_t* b_ptr = static_cast<const char_t*>(din_b);

    int remain = 0;
    int cnt_32 = size >> 5;
    remain = size & 31;
    int cnt_16 = remain >> 4;
    remain = cnt_16 & 15;
    int cnt_8 = remain >> 3;
    remain = cnt_8 & 7;

    int loop_cnt_32 = cnt_32;
    int loop_cnt_16 = cnt_16;
    int loop_cnt_8 = cnt_8;
#ifdef __aarch64__

    if (loop_cnt_32 > 0) {
        asm volatile(
            "1:\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "ld1 {v1.16b}, [%[b_ptr]], #16\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "ld1 {v3.16b}, [%[b_ptr]], #16\n"
            "smax v4.16b, v0.16b, v1.16b\n"
            "smax v5.16b, v2.16b, v3.16b\n"
            "subs %[loop_cnt], %[loop_cnt], #1\n"
            "st1 {v4.16b}, [%[out_ptr]], #16\n"
            "st1 {v5.16b}, [%[out_ptr]], #16\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_32), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"v0", "v1", "v2", "v3", "v4", "v5"
        );
    }
    if (loop_cnt_16 > 0) {
        asm volatile(
            "1:\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "ld1 {v1.16b}, [%[b_ptr]], #16\n"
            "smax v2.16b, v0.16b, v1.16b\n"
            "subs %[loop_cnt], %[loop_cnt], #1\n"
            "st1 {v2.16b}, [%[out_ptr]], #16\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_16), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"v0", "v1", "v2"
        );
    }
    if (loop_cnt_8 > 0) {
        asm volatile(
            "1:\n"
            "ld1 {v0.8b}, [%[a_ptr]], #8\n"
            "ld1 {v1.8b}, [%[b_ptr]], #8\n"
            "smax v2.8b, v0.8b, v1.8b\n"
            "subs %[loop_cnt], %[loop_cnt], #1\n"
            "st1 {v2.8b}, [%[out_ptr]], #8\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_8), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"v0", "v1", "v2"
        );
    }
#else

    if (loop_cnt_32 > 0) {
        asm volatile(
            "1:\n"
            "vld1.8 {d0-d1}, [%[a_ptr]]!\n"
            "vld1.8 {d2-d3}, [%[b_ptr]]!\n"
            "vld1.8 {d4-d5}, [%[a_ptr]]!\n"
            "vld1.8 {d6-d7}, [%[b_ptr]]!\n"
            "vmax.s8 q4, q1, q0\n"
            "vmax.s8 q5, q3, q2\n"
            "subs %[loop_cnt], #1\n"
            "vst1.8 {d8-d9}, [%[out_ptr]]!\n"
            "vst1.8 {d10-d11}, [%[out_ptr]]!\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_32), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2", "q3", "q4", "q5"
        );
    }
    if (loop_cnt_16 > 0) {
        asm volatile(
            "1:\n"
            "vld1.8 {d0-d1}, [%[a_ptr]]!\n"
            "vld1.8 {d2-d3}, [%[b_ptr]]!\n"
            "vmax.s8 q2, q0, q1\n"
            "subs %[loop_cnt], #1\n"
            "vst1.8 {d4-d5}, [%[out_ptr]]!\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_16), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2"
        );
    }
    if (loop_cnt_8 > 0) {
        asm volatile(
            "1:\n"
            "vld1.8 {d0}, [%[a_ptr]]!\n"
            "vld1.8 {d2}, [%[b_ptr]]!\n"
            "vmax.s8 q2, q1, q2\n"
            "subs %[loop_cnt], #1\n"
            "vst1.s8 {d4}, [%[out_ptr]]!\n"
            "bne 1b\n"
        :[loop_cnt] "+r" (loop_cnt_8), [a_ptr] "+r" (a_ptr), \
            [b_ptr] "+r" (b_ptr), [out_ptr] "+r" (out_ptr)
        :
        :"q0", "q1", "q2"
        );
    }
#endif //__aarch64__

    for (; remain > 0; remain--) {
        *(out_ptr++) = std::max(*(a_ptr++), *(b_ptr++));
    }
}

SaberEltwise::SaberEltwise(ParamBase *param) {
    _param = (EltwiseParam*)param;
    this->_flag_param = true;
}

SaberEltwise::~SaberEltwise() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberEltwise::load_param(ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (EltwiseParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberEltwise::load_param(std::istream &stream, const float *weights) {
    int type;
    int size;
    std::vector<float> coef;

    stream >> type >> size;
    coef.resize(size);
    for (int i = 0; i < size; ++i) {
        stream >> coef[i];
    }
    EltwiseType etype = static_cast<EltwiseType>(type);
    _param = new EltwiseParam(etype, coef);
    this->_flag_create_param = true;
    this->_flag_param  =true;
    return SaberSuccess;
}

SaberStatus SaberEltwise::set_op_precision(DataType ptype) {
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberEltwise::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                               std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_param) {
        LOGE("ERROR: load eltwise param first\n");
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
SaberStatus SaberEltwise::init(\
    const std::vector<Tensor<CPU>*>& inputs,
        std::vector<Tensor<CPU>*>& outputs, Context &ctx) {

    if (!this->_flag_param) {
        LOGE("ERROR: load eltwise param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;
    Shape sh_out_saber = outputs[0]->valid_shape();
    for (int i = 0; i < inputs.size(); i ++){
        Shape sh_in_saber = inputs[i]->valid_shape();
        if (sh_out_saber != sh_in_saber){
            LOGE("ERROR: input shape is not same with output shape\n");
            return SaberInvalidValue;
        }
    }
    DataType op_type = this->get_op_precision();
    switch (op_type){
        case AK_FLOAT:
            _tmp_out.set_dtype(AK_FLOAT);
            _tmp_out.reshape(outputs[0]->valid_shape());
            switch (_param->_elt_type) {
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
                    LOGE("ERROR: unknown eltwise type!!\n");
                    return SaberUnKownError;
            }
            break;
        case AK_INT8:
            _tmp_out.set_dtype(AK_INT8);
            _tmp_out.reshape(outputs[0]->valid_shape());
            switch (_param->_elt_type){
                case Eltwise_sum:
                    _impl = eltwise_sum_int8;
                    break;
                case Eltwise_max:
                    _impl = eltwise_max_int8;
                    break;
                default:
                    LOGE("ERROR: unknown eltwise type!!\n");
                    return SaberUnKownError;
            }
            break;
        default:
            LOGF("ERROR: data type: %d is unsupported now", (int)op_type);
    }
    this->_flag_init = true;
    return SaberSuccess;
}

//template <typename Dtype>
SaberStatus SaberEltwise::dispatch(\
    const std::vector<Tensor<CPU>*>& inputs, \
    std::vector<Tensor<CPU>*>& outputs) {

    if (!this->_flag_init) {
        LOGE("ERROR: init eltwise first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif
    std::vector<void*> din(inputs.size());
    void* dout = nullptr;

    DataType tensor_in_type = inputs[0]->get_dtype();
    DataType op_type = this->get_op_precision();
    if (op_type == AK_INT8 ){
        if (tensor_in_type == AK_FLOAT){
            for (int i = 0; i < inputs.size(); ++i){
                _tmp_in[i].set_dtype(AK_INT8);
                _tmp_in[i].reshape(inputs[i]->valid_shape());
                trans_tensor_fp32_to_int8(*inputs[i], _tmp_in[i], _ctx);
                din[i] = _tmp_in[i].data();
            }
        } else {
            for (int i = 0; i < inputs.size(); ++i){
                din[i] = inputs[i]->data();
            }
        }
    } else if (op_type == AK_FLOAT){
        if (tensor_in_type == AK_INT8){
            for (int i = 0; i < inputs.size(); ++i){
                _tmp_in[i].set_dtype(AK_FLOAT);
                _tmp_in[i].reshape(inputs[i]->valid_shape());
                trans_tensor_int8_to_fp32(*inputs[i], _tmp_in[i], inputs[i]->get_scale()[0], _ctx);
                din[i] = _tmp_in[i].data();
            }
        } else {
            for (int i = 0; i < inputs.size(); ++i){
                din[i] = inputs[i]->data();
            }
        }
    } else {
        LOGE("ERROR: unsupported precision type!!\n");
        return SaberInvalidValue;
    }
    DataType tensor_out_type = outputs[0]->get_dtype();
    if (op_type == AK_INT8 && tensor_out_type == AK_INT8) {
        dout = outputs[0]->mutable_data();
    } else if (op_type == AK_INT8 && tensor_out_type == AK_FLOAT){
        dout = _tmp_out.mutable_data();
    } else if (op_type == AK_FLOAT) {
        dout = outputs[0]->mutable_data();
    } else {
        LOGE("ERROR: unsupported precision type!!\n");
        return SaberInvalidValue;
    }

    if (op_type == AK_FLOAT) {
        //! do nothing
        if (outputs[0]->get_dtype() != AK_FLOAT) {
            LOGE("ERROR: unsupported precision type!!\n");
            return SaberInvalidValue;
        }
    }

    const void* din_a = din[0];
    const void* din_b = din[1];
    int size = outputs[0]->valid_size();

    _impl(din_a, din_b, dout, size, _param->_coef);
    for (int i = 2; i < inputs.size(); ++i) {
        din_a = din[i];
        _impl(din_a, dout, dout, size, _param->_coef);
    }

    if (op_type == AK_INT8) {
        if (tensor_out_type == AK_FLOAT) {
            trans_tensor_int8_to_fp32(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], _ctx);
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    GOPS ops;
    ops.ts = ts;
    // fixme
    ops.ops = 0;
    LOGI("eltwise : %s: time: %f\n", this->_op_name.c_str(), ts);
    OpTimer::add_timer("eltwise", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberEltwise);
} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE
