#include <arm_neon.h>
#include "saber/funcs/impl/arm/saber_power.h"

#ifdef USE_ARM_PLACE
#include"saber/funcs/impl/arm/impl/neon_mathfun.h"

namespace anakin {

namespace saber {
    template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
    SaberStatus SaberPower<ARM, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::create(\
    const std::vector<DataTensor_in*>& inputs, \
    std::vector<DataTensor_out*>& outputs, \
    PowerParam<OpTensor> &param, Context<ARM> &ctx) {

    this->_ctx = &ctx;
    if (fabsf(param.power - 1.f) < 1e-6f) {
        _do_power = false;
    }
    return SaberSuccess;
}
    template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
    SaberStatus SaberPower<ARM, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::init(\
        const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, \
        PowerParam<OpTensor> &param, Context<ARM> &ctx) {
    
        this->_ctx = &ctx;
        if (fabsf(param.power - 1.f) < 1e-6f) {
            _do_power = false;
        }
        return SaberSuccess;
    }

    template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
    SaberStatus SaberPower<ARM, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
    const std::vector<DataTensor_in *>& inputs,
    std::vector<DataTensor_out *>& outputs, PowerParam<OpTensor> &param) {

        //LOG(ERROR) << "power layer, scale: " << param.scale << ", power" << param.power << ", shift: " << param.shift;


        float scale=param.scale;
        float shift=param.shift;
        float power=param.power;
        float* ptr_out = outputs[0]->mutable_data();
        const float* ptr_in = inputs[0]->data();
        int size = inputs[0]->valid_size();
        int threads=1;
        this->_ctx->get_mode(threads);

        //LOG(ERROR) << "power threads:" << threads;

        int nums_per_thread = size / threads;
        int remain = size - threads * nums_per_thread;
        int neon_loop_cnt = nums_per_thread >> 4;
        int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
        int neon_loop_cnt_dim4 = nums_per_thread >> 2;
        int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
        float32x4_t vscale = vdupq_n_f32(scale);
        float32x4_t vshift=vdupq_n_f32(shift);
        float32x4_t vpower=vdupq_n_f32(power);

        if (!_do_power) {
            const float* din = ptr_in;
            float* dout = ptr_out;
            int i = 0;
            for (i = 0; i < size - 15; i += 16) {
                float32x4_t vin1 = vld1q_f32(din);
                float32x4_t vin2 = vld1q_f32(din + 4);
                float32x4_t vin3 = vld1q_f32(din + 8);
                float32x4_t vin4 = vld1q_f32(din + 12);

                float32x4_t vout1 = vmlaq_f32(vshift, vscale, vin1);
                float32x4_t vout2 = vmlaq_f32(vshift, vscale, vin2);
                float32x4_t vout3 = vmlaq_f32(vshift, vscale, vin3);
                float32x4_t vout4 = vmlaq_f32(vshift, vscale, vin4);

                vst1q_f32(dout, vout1);
                vst1q_f32(dout + 4, vout2);
                vst1q_f32(dout + 8, vout3);
                vst1q_f32(dout + 12, vout4);

                din += 16;
                dout += 16;
            }

            for (; i < size; ++i) {
                ptr_out[i] = shift + scale * ptr_in[i];
            }

            return SaberSuccess;
        }


#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        const float* ptr_in_thread = ptr_in + i * nums_per_thread;
        float* ptr_out_thread = ptr_out + i * nums_per_thread;
        int cnt = neon_loop_cnt;
        for(int num=0;num<neon_loop_cnt;++num){
            float32x4_t vr0 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr1 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr2 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr3 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            vr0=vmulq_f32(vr0,vscale);
            vr1=vmulq_f32(vr1,vscale);
            vr2=vmulq_f32(vr2,vscale);
            vr3=vmulq_f32(vr3,vscale);
            
            
            vr0=vaddq_f32(vr0,vshift);
            vr1=vaddq_f32(vr1,vshift);
            vr2=vaddq_f32(vr2,vshift);
            vr3=vaddq_f32(vr3,vshift);
            
            vr0=pow_ps(vr0,vpower);
            vr1=pow_ps(vr1,vpower);
            vr2=pow_ps(vr2,vpower);
            vr3=pow_ps(vr3,vpower);
            
            vst1q_f32(ptr_out_thread,vr0);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr1);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr2);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr3);
            ptr_out_thread+=4;
        }
        for (int j = 0; j < neon_loop_remain; ++j) {
            ptr_out_thread[0] = std::pow((ptr_in_thread[0]*scale+shift),power);
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    ptr_out = ptr_out+threads * nums_per_thread;
    ptr_in = ptr_in+threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] = std::pow((ptr_in[0]*scale+shift),power);
        ptr_in++;
        ptr_out++;
    }
    return SaberSuccess;
}

template class SaberPower<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE
