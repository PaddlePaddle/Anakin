#include "saber/lite/funcs/saber_eltwise.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
void eltwise_prod(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int size, \
    std::vector<float> coef);

template <typename Dtype>
void eltwise_sum(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int size, \
    std::vector<float> coef);

template <typename Dtype>
void eltwise_max(const Dtype* din_a, const Dtype* din_b, Dtype* dout, const int size, \
    std::vector<float> coef);

template <>
void eltwise_prod(const float* din_a, const float* din_b, float* dout, const int size, \
    std::vector<float> coef) {

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

void eltwise_sum(const float* din_a, const float* din_b, float* dout, const int size, \
    std::vector<float> coef) {

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

void eltwise_max(const float* din_a, const float* din_b, float* dout, const int size, \
    std::vector<float> coef) {

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

SaberEltwise::SaberEltwise(const ParamBase *param) {
    _param = (const EltwiseParam*)param;
    this->_flag_param = true;
}

SaberEltwise::~SaberEltwise() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberEltwise::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const EltwiseParam*)param;
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
#if 0
SaberStatus SaberEltwise::load_param(FILE *fp, const float *weights) {

    int type;
    int size;
    std::vector<float> coef;
    fscanf(fp, "%d %d ", &type, &size);
    coef.resize(size);
    for (int i = 0; i < size; ++i) {
        fscanf(fp, "%f ", &coef[i]);
    }
    fscanf(fp, "\n");
    EltwiseType etype = static_cast<EltwiseType>(type);
    _param = new EltwiseParam(etype, coef);
    this->_flag_create_param = true;
    this->_flag_param  =true;
    return SaberSuccess;
}
#endif
SaberStatus SaberEltwise::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
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
SaberStatus SaberEltwise::init(\
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
            printf("unknown eltwise type!!\n");
            return SaberUnKownError;
    }
    this->_flag_init = true;
    return SaberSuccess;
}

//template <typename Dtype>
SaberStatus SaberEltwise::dispatch(\
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

    int size = outputs[0]->valid_size();

    _impl(din_a, din_b, dout, size, _param->_coef);
    for (int i = 2; i < inputs.size(); ++i) {
        din_a = inputs[i]->data();
        _impl(din_a, dout, dout, size, _param->_coef);
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("eltwise time: %f\n", ts);
    OpTimer::add_timer("eltwise", ts);
    OpTimer::add_timer("total", ts);
#endif

    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberEltwise);
} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE