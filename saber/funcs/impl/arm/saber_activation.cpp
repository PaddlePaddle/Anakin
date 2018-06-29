
#include "saber/funcs/impl/arm/saber_activation.h"
#ifdef USE_ARM_PLACE
namespace anakin{
namespace saber {

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberActivation<ARM, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ActivationParam<OpTensor> &param,
        Context<ARM> &ctx)
{

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = ctx;

    return create(inputs, outputs, param, ctx);
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
    this->_ctx = ctx;
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
    int size = inputs[0]->valid_size();
    int threads = this->_ctx.get_act_ids().size();
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int dim16 = nums_per_thread >> 4;
    int dim16_remain = nums_per_thread&15;
    int dim4=nums_per_thread>>2;
    int dim4_remain=nums_per_thread&3;
    float32x4_t vzero = vdupq_n_f32(0.f);

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
            }return SaberSuccess;
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
                }return SaberSuccess;
            }
        return SaberSuccess;
}

template class SaberActivation<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}
} // namespace anakin
#endif // USE_ARM_PLACE
