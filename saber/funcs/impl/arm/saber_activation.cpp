#include "saber/funcs/impl/arm/saber_activation.h"

#ifdef USE_ARM_PLACE
namespace anakin{
namespace saber {

template <typename Dtype>
void ker_relu_fwd(const Dtype* din, Dtype* dout, const int threads, \
    const int nums_per_thread, const int dim16, const int dim16_remain, const int remain);
void ker_relu_fwd(const float* din, float* dout, const int threads, \
    const int nums_per_thread, const int dim16, const int dim16_remain, const int remain) {
    float32x4_t vzero = vdupq_n_f32(0.f);
    #pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        int cnt = dim16;
        asm volatile (
                    "relu_loop:                             @ loop header\n"
                    "vld1.32  {d0-d1}, [%[din]]!            @ load din 0\n"
                    "vld1.32  {d2-d3}, [%[din]]!            @ load din 0\n"
                    "vld1.32  {d4-d5}, [%[din]]!            @ load din 0\n"
                    "vld1.32  {d6-d7}, [%[din]]!            @ load din 0\n"
                          
                    "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
                    "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
                    "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
                    "vmax.f32 q11, q3, %q[vzero]            @ relu\n"
                          
                    "vst1.32  {d16-d17}, [%[dout]]!         @ store result, add pointer\n"
                    "pld [%[din]]                           @ preload data\n"
                    "vst1.32  {d18-d19}, [%[dout]]!         @ store result, add pointer\n"
                    "pld [%[din], #128]                     @ preload data\n"
                    "vst1.32  {d20-d21}, [%[dout]]!         @ store result, add pointer\n"
                    "pld [%[din], #256]                     @ preload data\n"
                    "vst1.32  {d22-d23}, [%[dout]]!         @ store result, add pointer\n"
                    "pld [%[din], #384]                     @ preload data\n"
                          
                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    relu_loop                       @ jump to main loop start point\n"
                    :[dout] "+r"(ptr_out_thread), [din] "+r"(ptr_in_thread), [cnt] "+r"(cnt)
                    :[vzero] "w" (vzero)
                    :"q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                );
        for (int j = 0; j < dim16_remain; ++j) {
            ptr_out_thread[0] = ptr_in_thread[0] > 0.f? ptr_in_thread[0] : 0.f;
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* ptr_out = dout+threads * nums_per_thread;
    const float* ptr_in = din+threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] = ptr_in[0] > 0.f? ptr_in[0] : 0.f;
        ptr_in++;
        ptr_out++;
    }
}

template <typename Dtype>
void ker_sigmoid_fwd(const Dtype* din, Dtype* dout, const int threads, \
    const int nums_per_thread, const int dim4, const int dim4_remain, const int remain);
void ker_sigmoid_fwd(const float* din, float* dout, const int threads, \
    const int nums_per_thread, const int dim4, const int dim4_remain, const int remain) {
    
    #pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        float32x4_t exp_vec = vdupq_n_f32(0.0);
        float32x4_t recip  = vdupq_n_f32(0.0f);
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        for (int k=0; k<dim4; ++k ) {
            exp_vec=exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
            exp_vec = vaddq_f32(exp_vec, vdupq_n_f32(1.0));
            recip = vrecpeq_f32(exp_vec);
            recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
            recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
            vst1q_f32(ptr_out_thread, recip);
            ptr_out_thread+=4;
            ptr_in_thread+=4;
        }
        for(int j=0;j<dim4_remain;++j){
            ptr_out_thread[0]=1/(1+exp(-ptr_in_thread[0]));
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* ptr_out = dout+threads * nums_per_thread;
    const float* ptr_in = din+threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] = ptr_in[0] > 0.f? ptr_in[0] : 0.f;
        ptr_in++;
        ptr_out++;
    }
}

template <typename Dtype>
void ker_tanh_fwd(const Dtype* din, Dtype* dout, const int threads, \
    const int nums_per_thread, const int dim4, const int dim4_remain, const int remain);
void ker_tanh_fwd(const float* din, float* dout, const int threads, \
    const int nums_per_thread, const int dim4, const int dim4_remain, const int remain) {
    #pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        float32x4_t exp_plus_vec = vdupq_n_f32(0.0);
        float32x4_t exp_minus_vec = vdupq_n_f32(0.0f);
        float32x4_t exp_sum_vec = vdupq_n_f32(0.0);
        float32x4_t exp_diff_vec = vdupq_n_f32(0.0);
        float32x4_t recip  = vdupq_n_f32(0.0f);
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
        for (int k=0; k<dim4; ++k ) {
            exp_plus_vec=exp_ps(vld1q_f32(ptr_in_thread));
            exp_minus_vec=exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
            exp_sum_vec=vaddq_f32(exp_plus_vec,exp_minus_vec);
            exp_diff_vec=vsubq_f32(exp_plus_vec,exp_minus_vec);
            recip = div_ps(exp_diff_vec,exp_sum_vec);
            vst1q_f32(ptr_out_thread, recip);
            ptr_out_thread+=4;
            ptr_in_thread+=4;
        }
        for(int j=0;j<dim4_remain;++j){
            ptr_out_thread[0]=(exp(ptr_in_thread[0])-exp(-ptr_in_thread[0]))/(exp(ptr_in_thread[0])+exp(-ptr_in_thread[0]));
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* ptr_out = dout+threads * nums_per_thread;
    const float* ptr_in = din+threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] = ptr_in[0] > 0.f? ptr_in[0] : 0.f;
        ptr_in++;
        ptr_out++;
    }
}

template <typename Dtype> 
void ker_prelu_fwd(const Dtype* din, Dtype* dout, const int threads, const int num, const int channel, \
    const int csize, bool channel_shared, const Dtype* slopes);
void ker_prelu_fwd(const float* din, float* dout, const int threads, const int num, const int channel, \
    const int csize, bool channel_shared, const float* slopes) {
    
    for (int n = 0; n < num; n++){
        const float* data_in_batch = din + n * channel * csize;
        float* data_out_batch = dout + n * channel * csize;
#pragma omp parallel for
        for (int c = 0; c < channel; c++){
            const float* data_in_channel = data_in_batch + c * csize;
            float* data_out_channel = data_out_batch + c * csize;
            float slope = channel_shared ? slopes[0] : slopes[c];

            float32x4_t vzero = vdupq_n_f32(0.f);
            float32x4_t vslope = vdupq_n_f32(slope);
            int dim4 = csize >> 2;
            int dim4_remain = csize - (dim4 * 4);
#ifdef __arrch64__
            for (; dim4 > 0; dim4--){
                float32x4_t vr0 = vld1q_f32(data_in_channel);
                uint32x4_t vmask = vcltq_f32(vr0, vzero);//vr0 <= vzero
                float32x4_t vout = vmulq_f32(vr0, vslope);//vr0 * vslope
                float32x4_t vout_sel = vbslq_f32(vmask, vout, vr0);
                vst1q_f32(data_out_channel, vout_sel);
                data_in_channel += 4;
                data_out_channel += 4;
            }
#else
            if (dim4 > 0){
                asm volatile(
                "prelu_loop:                                    @main loop\n"
                        "vld1.f32   {d0-d1}, [%[ptr_in]]!              @load q1\n"
                        "vclt.f32   q1, q0, %q[vzero]                   @vcle q0 <= vzero\n"
                        "vmul.f32   q2, q0, %q[vslope]                  @vmul q0 * vslope\n"
                        "vbit.32    q0, q2, q1                          @vbit q0, q2, q1\n"
                        "subs       %[cnt], #1                          @subs nn, 1\n"
                        "vst1.f32   {d0-d1}, [%[dout]]!                 @store data\n"
                        "bne        prelu_loop                          @bne nn\n"
                :[ptr_in] "+r" (data_in_channel), [cnt] "+r" (dim4), \
                     [dout] "+r" (data_out_channel)
                :[vzero] "w" (vzero), [vslope] "w" (vslope)
                :"q0", "q1", "q2"
                );
            }
#endif //__aarch64__
            for (; dim4_remain > 0; dim4_remain--) {
                if (*data_in_channel < 0){
                    *data_out_channel = *data_in_channel * slope;
                } else {
                    *data_out_channel = *data_in_channel;
                }
                data_in_channel++;
                data_out_channel++;
            }
        }
    }
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberActivation<ARM, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ActivationParam<OpTensor> &param,
        Context<ARM> &ctx)
{
    this->_ctx = &ctx;
    _size = inputs[0]->valid_size();
    this->_ctx->get_mode(_threads);
    _nums_per_thread = _size / _threads;
    _remain = _size - _threads * _nums_per_thread;
    _dim16 = _nums_per_thread >> 4;
    _dim16_remain = _nums_per_thread&15;
    _dim4 = _nums_per_thread>>2;
    _dim4_remain = _nums_per_thread&3;
    _channel = inputs[0]->channel();
    _num = inputs[0]->num();
    return SaberSuccess;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberActivation<ARM, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ActivationParam<OpTensor> &param)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    DataType_out* ptr_out = outputs[0]->mutable_data();
    const DataType_in* ptr_in = inputs[0]->data();
    switch(param.active){
        case Active_relu:
            ker_relu_fwd(ptr_in, ptr_out, _threads, _nums_per_thread, _dim16, _dim16_remain, _remain);
            break;
        case Active_sigmoid:
            ker_sigmoid_fwd(ptr_in, ptr_out, _threads, _nums_per_thread, _dim4, _dim4_remain, _remain);
            break;
        case Active_tanh:
            ker_tanh_fwd(ptr_in, ptr_out, _threads, _nums_per_thread, _dim4, _dim4_remain, _remain);
            break;
        case Active_prelu:
            ker_prelu_fwd(ptr_in, ptr_out, _threads, _num, _channel, _size/_channel, param.prelu_param.channel_shared, param.prelu_param.slope->data());
            break;
    }
    /*
    if (param.active == Active_relu){
        #pragma omp parallel for
        for (int i = 0; i < threads; ++i) {
            const DataType_in* ptr_in_thread = ptr_in + i * nums_per_thread;
            DataType_out* ptr_out_thread = ptr_out + i * nums_per_thread;
            int cnt = dim16;
            asm volatile (
                          "relu_loop:                                     @ loop header\n"
                          "vld1.32  {d0-d1}, [%[din]]!            @ load din 0\n"
                          "vld1.32  {d2-d3}, [%[din]]!            @ load din 0\n"
                          "vld1.32  {d4-d5}, [%[din]]!            @ load din 0\n"
                          "vld1.32  {d6-d7}, [%[din]]!            @ load din 0\n"
                          
                          "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
                          "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
                          "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
                          "vmax.f32 q11, q3, %q[vzero]            @ relu\n"
                          
                          "vst1.32  {d16-d17}, [%[dout]]!         @ store result, add pointer\n"
                          "pld [%[din]]                           @ preload data\n"
                          "vst1.32  {d18-d19}, [%[dout]]!         @ store result, add pointer\n"
                          "pld [%[din], #128]                     @ preload data\n"
                          "vst1.32  {d20-d21}, [%[dout]]!         @ store result, add pointer\n"
                          "pld [%[din], #256]                     @ preload data\n"
                          "vst1.32  {d22-d23}, [%[dout]]!         @ store result, add pointer\n"
                          "pld [%[din], #384]                     @ preload data\n"
                          
                          "subs %[cnt], #1                        @ loop count minus 1\n"
                          "bne    relu_loop                       @ jump to main loop start point\n"
                          :[dout] "+r"(ptr_out_thread), [din] "+r"(ptr_in_thread), [cnt] "+r"(cnt)
                          :[vzero] "w" (vzero)
                          :"q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                          );
            for (int j = 0; j < dim16_remain; ++j) {
                ptr_out_thread[0] = ptr_in_thread[0] > 0.f? ptr_in_thread[0] : 0.f;
                ptr_in_thread++;
                ptr_out_thread++;
            }
        }
        ptr_out = outputs[0]->mutable_data()+threads * nums_per_thread;
        ptr_in = inputs[0]->data()+threads * nums_per_thread;
        for (int j = 0; j < remain; ++j) {
            ptr_out[0] = ptr_in[0] > 0.f? ptr_in[0] : 0.f;
            ptr_in++;
            ptr_out++;
        }
        return SaberSuccess;
    }else if (param.active == Active_sigmoid){
        #pragma omp parallel for
        for (int i = 0; i < threads; ++i) {
            float32x4_t exp_vec = vdupq_n_f32(0.0);
            float32x4_t recip  = vdupq_n_f32(0.0f);
            const DataType_in* ptr_in_thread = ptr_in + i * nums_per_thread;
            DataType_out* ptr_out_thread = ptr_out + i * nums_per_thread;
            for (int k=0; k<dim4; ++k ) {
                exp_vec=exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
                exp_vec = vaddq_f32(exp_vec, vdupq_n_f32(1.0));
                recip = vrecpeq_f32(exp_vec);
                recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
                recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
                vst1q_f32(ptr_out_thread, recip);
                ptr_out_thread+=4;
                ptr_in_thread+=4;
                }
                for(int j=0;j<dim4_remain;++j){
                    ptr_out_thread[0]=1/(1+exp(-ptr_in_thread[0]));
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
        }
            ptr_out = outputs[0]->mutable_data()+threads * nums_per_thread;
            ptr_in = inputs[0]->data()+threads * nums_per_thread;
            for (int j = 0; j < remain; ++j) {
                ptr_out[0]=1/(1+exp(-ptr_in[0]));
                ptr_in++;
                ptr_out++;
            }
            return SaberSuccess;
        }else if (param.active == Active_tanh){
            #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                float32x4_t exp_plus_vec = vdupq_n_f32(0.0);
                float32x4_t exp_minus_vec = vdupq_n_f32(0.0f);
                float32x4_t exp_sum_vec = vdupq_n_f32(0.0);
                float32x4_t exp_diff_vec = vdupq_n_f32(0.0);
                float32x4_t recip  = vdupq_n_f32(0.0f);
                const DataType_in* ptr_in_thread = ptr_in + i * nums_per_thread;
                DataType_out* ptr_out_thread = ptr_out + i * nums_per_thread;
                for (int k=0; k<dim4; ++k ) {
                    exp_plus_vec=exp_ps(vld1q_f32(ptr_in_thread));
                    exp_minus_vec=exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
                    exp_sum_vec=vaddq_f32(exp_plus_vec,exp_minus_vec);
                    exp_diff_vec=vsubq_f32(exp_plus_vec,exp_minus_vec);
                    recip = div_ps(exp_diff_vec,exp_sum_vec);
                    vst1q_f32(ptr_out_thread, recip);
                    ptr_out_thread+=4;
                    ptr_in_thread+=4;
                }
                for(int j=0;j<dim4_remain;++j){
                    ptr_out_thread[0]=(exp(ptr_in_thread[0])-exp(-ptr_in_thread[0]))/(exp(ptr_in_thread[0])+exp(-ptr_in_thread[0]));
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
                ptr_out = outputs[0]->mutable_data()+threads * nums_per_thread;
                ptr_in = inputs[0]->data()+threads * nums_per_thread;
                for (int j = 0; j < remain; ++j) {
                    ptr_out[0]=(exp(ptr_in[0])-exp(-ptr_in[0]))/(exp(ptr_in[0])+exp(-ptr_in[0]));
                    ptr_in++;
                    ptr_out++;
                }
                return SaberSuccess;
            }
        return SaberSuccess;
    */
    return SaberSuccess;
}

template class SaberActivation<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}
} // namespace anakin
#endif // USE_ARM_PLACE
