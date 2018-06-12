#include "saber/lite/funcs/saber_prelu.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
void prelu_kernel(const Dtype* din, const Dtype* slopes, \
    Dtype* dout, int num, int cin, int csize, bool is_channel_shared);

template <>
void prelu_kernel<float>(const float* din, const float* slopes, \
    float* dout, int num, int cin, int csize, bool is_channel_shared) {

    for (int n = 0; n < num; n++){
        const float* data_in_batch = din + n * cin * csize;
        float* data_out_batch = dout + n * cin * csize;
#pragma omp parallel for
        for (int c = 0; c < cin; c++){
            const float* data_in_channel = data_in_batch + c * csize;
            float* data_out_channel = data_out_batch + c * csize;
            float slope = is_channel_shared ? slopes[0] : slopes[c];

            float32x4_t vzero = vdupq_n_f32(0.f);
            float32x4_t vslope = vdupq_n_f32(slope);
            int cnt = csize >> 2;
            int remain = csize - (cnt * 4);
#ifdef __arrch64__
            for (; cnt > 0; cnt--){
                float32x4_t vr0 = vld1q_f32(data_in_channel);
                uint32x4_t vmask = vcltq_f32(vr0, vzero);//vr0 <= vzero
                float32x4_t vout = vmulq_f32(vr0, vslope);//vr0 * vslope
                float32x4_t vout_sel = vbslq_f32(vmask, vout, vr0);
                vst1q_f32(data_out_channel, vout_sel);
                data_in_channel += 4;
                data_out_channel += 4;
            }
#else
            if (cnt > 0){
                asm volatile(
                "prelu_loop:                                    @main loop\n"
                        "vld1.f32   {d0-d1}, [%[ptr_in]]!              @load q1\n"
                        "vclt.f32   q1, q0, %q[vzero]                   @vcle q0 <= vzero\n"
                        "vmul.f32   q2, q0, %q[vslope]                  @vmul q0 * vslope\n"
                        "vbit.32    q0, q2, q1                          @vbit q0, q2, q1\n"
                        "subs       %[cnt], #1                          @subs nn, 1\n"
                        "vst1.f32   {d0-d1}, [%[dout]]!                 @store data\n"
                        "bne        prelu_loop                          @bne nn\n"
                :[ptr_in] "+r" (data_in_channel), [cnt] "+r" (cnt), \
                     [dout] "+r" (data_out_channel)
                :[vzero] "w" (vzero), [vslope] "w" (vslope)
                :"q0", "q1", "q2"
                );
            }
#endif //__aarch64__
            for (; remain > 0; remain--) {
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

template <typename Dtype>
SaberStatus SaberPrelu<Dtype>::dispatch(\
    const std::vector<Tensor<Dtype>*>& inputs, \
    std::vector<Tensor<Dtype>*>& outputs, \
    PreluParam<Tensor<Dtype>> &param) {

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int width = inputs[0]->width();
    int height = inputs[0]->height();
    const Dtype* din = inputs[0]->data();
    Dtype* dout = outputs[0]->mutable_data();
    const Dtype* ptr_slope = param.slope->data();

    prelu_kernel<float>(din, ptr_slope, dout, num, channel, width * height, param.channel_shared);

    return SaberSuccess;
}

template class SaberPrelu<float>;

} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE