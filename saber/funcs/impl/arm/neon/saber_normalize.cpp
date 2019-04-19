#include "saber/funcs/impl/arm/saber_normalize.h"

namespace anakin{

namespace saber{

void compute_mean(const float* input, Tensor<ARM>& mean, int num, int channel, int height, int width){

    int spatial_size = height * width;
    float* out = static_cast<float*>(mean.mutable_data());

    int cnt = spatial_size / 8;
    for (int n = 0; n < num; ++n){
        const float* in_batch = input + n * channel * spatial_size;
        float* out_batch = out + n * channel;
#pragma omp parallel for
        for (int c = 0; c < channel; ++c){
            const float* in_channel = in_batch + c * spatial_size;
            int i = 0;
            float32x4_t vsum = vdupq_n_f32(0.0f);
            float32x4_t vc = vdupq_n_f32(0.f);
            //! improve float summation precision
            //! https://en.wikipedia.org/wiki/Kahan_summation_algorithm
#ifdef  __aarch64__
            for (; i < cnt; i++){//
                float32x4_t vin1 = vld1q_f32(in_channel);
                float32x4_t vy = vsubq_f32(vin1, vc);
                float32x4_t vt = vaddq_f32(vsum, vy);
                vc = vsubq_f32(vt, vsum);
                vc = vsubq_f32(vc, vy);
                vsum = vt;

                float32x4_t vin2 = vld1q_f32(in_channel + 4);
                vy = vsubq_f32(vin2, vc);
                vt = vaddq_f32(vsum, vy);
                vc = vsubq_f32(vt, vsum);
                vc = vsubq_f32(vc, vy);
                vsum = vt;
                in_channel += 8;
            }
#else
            int loop = cnt;
            if (loop > 0) {
                asm volatile(
                "1:                                     \n"
                "vld1.f32   {d0-d1}, [%[in_channel]]!   \n"
                "vld1.f32   {d2,d3}, [%[in_channel]]!   \n"
                "vsub.f32   q6, q0, %q[c]               \n" // y
                "vadd.f32   q7, %q[vsum], q6            \n" // t
                "vsub.f32   %q[c], q7, %q[vsum]         \n"
                "vsub.f32   %q[c], %q[c], q6            \n"
                "vmov.32    %q[vsum], q7                \n"
                "vsub.f32   q4, q1, %q[c]               \n"
                "vadd.f32   q5, %q[vsum], q4            \n"
                "vsub.f32   %q[c], q5, %q[vsum]         \n"
                "vsub.f32   %q[c], %q[c], q4            \n"
                "vmov.32    %q[vsum], q5                \n"
                "subs       %[loop], #1                 \n"
                "bne        1b                          \n"
                :[in_channel] "+r" (in_channel), [loop] "+r" (loop), [vsum] "+w" (vsum), \
                 [c] "+w" (vc)
                :"r" (in_channel), "r" (num), "w" (vsum)
                : "q0", "q1", "q4", "q5", "q6", "q7"
                );
            }
#endif //__aarch64__
            float32x2_t vsum_tmp = vadd_f32(vget_low_f32(vsum),vget_high_f32(vsum));
            float sum = vget_lane_f32(vsum_tmp,0) + vget_lane_f32(vsum_tmp,1);
            for (i = cnt * 8;i < spatial_size; i++) {
                sum += in_channel[0];
                in_channel++;
            }
            out_batch[c] = sum / (spatial_size * num);
        }
    }

    //add mean in num
    for (int c = 0; c < channel; ++c){
        for (int n = 1; n < num; ++n){
            out[c] += out[n * channel + c];
        }
    }
}


void compute_variance(const float* input, Tensor<ARM>& mean, Tensor<ARM>& variance, \
                        int num, int channel , int height, int width){

     int spatial_size = height * width;
     float* out = static_cast<float*>(variance.mutable_data());
     const float* mean_data = static_cast<const float*>(mean.data());

    int cnt = spatial_size / 8;
    for (int n = 0; n < num; ++n){
        const float* in_batch = input + n * channel * spatial_size;
        float* out_batch = out + n * channel;

#pragma omp parallel for
        for (int c = 0; c < channel; ++c){
            const float* in_channel = in_batch + c * spatial_size;
            int i = 0;
            float mean_val = mean_data[c];
            float32x4_t vsum = vdupq_n_f32(0.0f);
            float32x4_t vc = vdupq_n_f32(0.f);
#ifdef  __aarch64__
            for (; i < cnt; i++){//
                float32x4_t in_data0 = vld1q_f32(in_channel);
                in_data0 = vsubq_f32(in_data0, vdupq_n_f32(mean_val));
                in_data0 = vmulq_f32(in_data0, in_data0);

                float32x4_t in_data1 = vld1q_f32(in_channel + 4);
                in_data1 = vsubq_f32(in_data1, vdupq_n_f32(mean_val));
                in_data1 = vmulq_f32(in_data1, in_data1);

                float32x4_t vy = vsubq_f32(vpaddq_f32(in_data0, in_data1), vc);
                float32x4_t vt = vaddq_f32(vsum, vy);
                vc = vsubq_f32(vt, vsum);
                vc = vsubq_f32(vc, vy);
                vsum = vt;

                in_channel += 8;
            }
#else
            int loop = cnt;
            if (loop > 0) {
                asm volatile(
                "1:                                     \n"
                "vld1.f32   {d0-d1}, [%[in_channel]]!   \n"
                "vdup.f32   q10, %[mean]                \n"
                "vsub.f32   q1, q0, q10                 \n"
                "vmul.f32   q2, q1, q1                  \n"

                "vld1.f32   {d6-d7}, [%[in_channel]]!   \n"
                "vsub.f32   q4, q3, q10                 \n"
                "vmul.f32   q5, q4, q4                  \n"

                "vpadd.f32  d12, d4, d5                 \n"
                "vpadd.f32  d13, d10, d11               \n"
                "vsub.f32   q7, q6, %q[c]               \n" // y
                "vadd.f32   q8, %q[vsum], q7            \n" // t
                "vsub.f32   %q[c], q8, %q[vsum]         \n"
                "vsub.f32   %q[c], %q[c], q7            \n"
                "vmov.32    %q[vsum], q8                \n"
                "subs       %[loop], #1                 \n"
                "bne        1b                          \n"
                :[in_channel] "+r" (in_channel), [loop] "+r" (loop), [vsum] "+w" (vsum), \
                 [mean] "+r" (mean_val), [c] "+w" (vc)
                :"r" (in_channel), "r" (loop), "w" (vsum)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q10"
                );
            }
#endif //__aarch64__
            float32x2_t vsum_tmp = vadd_f32(vget_low_f32(vsum),vget_high_f32(vsum));
            float sum = vget_lane_f32(vsum_tmp, 0) + vget_lane_f32(vsum_tmp, 1);
            float sum_tmp = 0.f;
            for (i = cnt * 8; i < spatial_size; i++) {
                float in_data = in_channel[0];
                in_data = powf(in_data - mean_val, 2);
                sum += in_data;
                in_channel++;
            }
            sum += sum_tmp;
            out_batch[c] = sum / (spatial_size * num);
        }
    }
    //add variance in num
    for (int c = 0; c < channel; ++c){
        for (int n = 1; n < num; ++n){
            out[c] += out[n * channel + c];
        }
    }
}

template <>
SaberStatus SaberNormalize<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        NormalizeParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const float* input = static_cast<const float*>(inputs[0]->mutable_data());
    const float* mean_data = static_cast<const float*>(this->_mean.data());
    const float* variance_data = static_cast<const float*>(this->_variance.data());
    float* output = static_cast<float*>(outputs[0]->mutable_data());
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int spatial_size = width * height;
    int cnt = spatial_size / 8;
    compute_mean(input, this->_mean, num, channel, height, width);
    compute_variance(input, this->_mean, this->_variance, num, channel, height, width);

    for (int n = 0; n < num; ++n){
        const float* input_batch = input + n * spatial_size * channel;
        float* output_batch = output + n * spatial_size * channel;
#pragma omp parallel for
        for (int c = 0; c < channel; ++c){
            const float* input_channel = input_batch + c * spatial_size;
            float* output_channel = output_batch + c * spatial_size;
            int i = 0;
            float mean_val = mean_data[c];
            float std_val = 1.f / sqrt(variance_data[c] + param.eps);
            float32x4_t vmean_val = vdupq_n_f32(mean_val);
            float32x4_t vstd_val = vdupq_n_f32(std_val);
#ifdef __aarch64__
            for (; i < cnt; ++i){
                float32x4_t in_data0 = vld1q_f32(input_channel);
                in_data0 = vsubq_f32(in_data0, vmean_val);
                in_data0 = vmulq_f32(in_data0, vstd_val);
                vst1q_f32(output_channel, in_data0);

                float32x4_t in_data1 = vld1q_f32(input_channel + 4);
                in_data1 = vsubq_f32(in_data1, vmean_val);
                in_data1 = vmulq_f32(in_data1, vstd_val);
                vst1q_f32(output_channel + 4, in_data1);

                input_channel += 8;
                output_channel += 8;
            }
#else
            int loop = cnt;
            if (loop > 0) {
                asm volatile(
                "1:                                       \n"
                "vld1.f32   {d0-d1},[%[in_channel]]!      \n"
                "vsub.f32   q1, q0, %q[mean]              \n"
                "vmul.f32   q2, q1, %q[std]               \n"
                "vst1.32    {d4-d5}, [%[out_channel]]!    \n"

                "vld1.f32   {d6-d7}, [%[in_channel]]!     \n"
                "vsub.f32   q4, q3, %q[mean]              \n"
                "vmul.f32   q5, q4, %q[std]               \n"
                "vst1.32    {d10-d11}, [%[out_channel]]!  \n"
                "subs       %[loop], #1                   \n"
                "bne        1b                            \n"
                :[in_channel] "+r" (input_channel), [out_channel] "+r" (output_channel), [loop] "+r" (loop),\
                 [mean] "+w" (vmean_val), [std] "+w" (vstd_val)
                :"r" (input_channel), "r" (loop)
                : "q0", "q1", "q2", "q3", "q4", "q5"
                );
            }
#endif //__aarch64__
            for (i = cnt * 8; i < spatial_size; ++i){
                float in_data = input_channel[0];
                in_data = (in_data - mean_val) * std_val;
                output_channel[0] = in_data;
                input_channel++;
                output_channel++;
            }

        }
    }
#ifdef ENABLE_OP_TIMER
   this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Normalize : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("Normalize", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberNormalize, NormalizeParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberNormalize, NormalizeParam, ARM, AK_INT8);
//template class SaberNormalize<ARM, AK::FLOAT>;

} //namespace anakin

} //namespace anakin
