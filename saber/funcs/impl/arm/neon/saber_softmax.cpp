#include "saber/funcs/impl/arm/saber_softmax.h"
#include "saber/funcs/impl/arm/neon/impl/neon_mathfun.h"

namespace anakin{

namespace saber{

void softmax_basic(const float* din, float* dout, \
    const int axis_size, const int inner_num, \
    const int outer_num, const int compute_size) {

#pragma omp parallel for
    for (int i = 0; i < compute_size; ++i) {
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        float max_data = din[real_index];
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            max_data = din[real_index] > max_data? din[real_index] : max_data;
        }

        real_index = idx_outer * inner_num + idx_inner;
        //! sub, exp and sum
        dout[real_index] = expf(din[real_index] - max_data);
        float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            dout[real_index] = expf(din[real_index] - max_data);
            sum_data += dout[real_index];
        }

        float sum_inv = 1.f / sum_data;
        real_index = idx_outer * inner_num + idx_inner;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            dout[real_index] *= sum_inv;
            real_index += inner_num;
        }
    }
}

void softmax_arm_lite_channel_in8(const float* din, float* dout, \
    const int axis_size, const int inner_num, \
    const int outer_num, const int compute_size){
    int cmp_cnt = compute_size >> 3;
    int remain = compute_size % 8;
    float32x4_t vone = vdupq_n_f32(1.0f);

#pragma omp parallel for
    for (int c = 0; c < cmp_cnt; ++c) {
        int i = c * 8;
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        //float max_data = din[real_index];
        //! get max axis_size == 4
        const float* din_ptr =  din + real_index;
        const float* din_ptr1 = din_ptr + inner_num;
        const float* din_ptr2 = din_ptr1 + inner_num;
        const float* din_ptr3 = din_ptr2 + inner_num;
        float32x4_t vdata0 = vld1q_f32(din_ptr);
        float32x4_t vdata1 = vld1q_f32(din_ptr1);
        float32x4_t vdata2 = vld1q_f32(din_ptr2);
        float32x4_t vdata3 = vld1q_f32(din_ptr3);

        float32x4_t vdata01 = vld1q_f32(din_ptr + 4);
        float32x4_t vdata11 = vld1q_f32(din_ptr1 + 4);
        float32x4_t vdata21 = vld1q_f32(din_ptr2 + 4);
        float32x4_t vdata31 = vld1q_f32(din_ptr3 + 4);

        float* dout_ptr0 = dout + real_index;
        float* dout_ptr1 = dout_ptr0 + inner_num;
        float32x4_t vmax1 = vmaxq_f32(vdata0, vdata1);
        float32x4_t vmax2 = vmaxq_f32(vdata2, vdata3);
        float32x4_t vmax11 = vmaxq_f32(vdata01, vdata11);
        float32x4_t vmax21 = vmaxq_f32(vdata21, vdata31);
        float* dout_ptr2 = dout_ptr1 + inner_num;
        float* dout_ptr3 = dout_ptr2 + inner_num;
        float32x4_t vmax = vmaxq_f32(vmax1, vmax2);
        float32x4_t vmax_1 = vmaxq_f32(vmax11, vmax21);

        //! sub, exp and sum
        float32x4_t vsum0 = exp_ps(vsubq_f32(vdata0, vmax));
        float32x4_t vsum1 = exp_ps(vsubq_f32(vdata1, vmax));
        float32x4_t vsum2 = exp_ps(vsubq_f32(vdata2, vmax));
        float32x4_t vsum3 = exp_ps(vsubq_f32(vdata3, vmax));

        float32x4_t vsum01 = exp_ps(vsubq_f32(vdata01, vmax_1));
        float32x4_t vsum11 = exp_ps(vsubq_f32(vdata11, vmax_1));
        float32x4_t vsum21 = exp_ps(vsubq_f32(vdata21, vmax_1));
        float32x4_t vsum31 = exp_ps(vsubq_f32(vdata31, vmax_1));

        float32x4_t vsum_1 = vaddq_f32(vsum0, vsum1);
        float32x4_t vsum_2 = vaddq_f32(vsum2, vsum3);
        float32x4_t vsum_11 = vaddq_f32(vsum01, vsum11);
        float32x4_t vsum_21 = vaddq_f32(vsum21, vsum31);

        float32x4_t vsum = vaddq_f32(vsum_1, vsum_2);
        float32x4_t vsum111 = vaddq_f32(vsum_11, vsum_21);

        float32x4_t vinf = div_ps(vone, vsum);
        float32x4_t vinf1 = div_ps(vone, vsum111);

        vsum0 = vmulq_f32(vsum0, vinf);
        vsum1 = vmulq_f32(vsum1, vinf);
        vsum2 = vmulq_f32(vsum2, vinf);
        vsum3 = vmulq_f32(vsum3, vinf);

        vsum01 = vmulq_f32(vsum01, vinf1);
        vsum11 = vmulq_f32(vsum11, vinf1);
        vsum21 = vmulq_f32(vsum21, vinf1);
        vsum31 = vmulq_f32(vsum31, vinf1);

        vst1q_f32(dout_ptr0, vsum0);
        vst1q_f32(dout_ptr1, vsum1);
        vst1q_f32(dout_ptr2, vsum2);
        vst1q_f32(dout_ptr3, vsum3);

        vst1q_f32(dout_ptr0 + 4, vsum01);
        vst1q_f32(dout_ptr1 + 4, vsum11);
        vst1q_f32(dout_ptr2 + 4, vsum21);
        vst1q_f32(dout_ptr3 + 4, vsum31);
    }

    int i = cmp_cnt * 8;

    if (remain > 4){
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;
        //float max_data = din[real_index];
        //! get max axis_size == 4
        const float* din_ptr =  din + real_index;
        const float* din_ptr1 = din_ptr + inner_num;
        const float* din_ptr2 = din_ptr1 + inner_num;
        const float* din_ptr3 = din_ptr2 + inner_num;
        float32x4_t vdata0 = vld1q_f32(din_ptr);
        float32x4_t vdata1 = vld1q_f32(din_ptr1);
        float32x4_t vdata2 = vld1q_f32(din_ptr2);
        float32x4_t vdata3 = vld1q_f32(din_ptr3);

        float* dout_ptr0 = dout + real_index;
        float* dout_ptr1 = dout_ptr0 + inner_num;
        float32x4_t vmax1 = vmaxq_f32(vdata0, vdata1);
        float32x4_t vmax2 = vmaxq_f32(vdata2, vdata3);
        float* dout_ptr2 = dout_ptr1 + inner_num;
        float* dout_ptr3 = dout_ptr2 + inner_num;
        float32x4_t vmax = vmaxq_f32(vmax1, vmax2);

        //! sub, exp and sum
        float32x4_t vsum0 = exp_ps(vsubq_f32(vdata0, vmax));
        float32x4_t vsum1 = exp_ps(vsubq_f32(vdata1, vmax));
        float32x4_t vsum2 = exp_ps(vsubq_f32(vdata2, vmax));
        float32x4_t vsum3 = exp_ps(vsubq_f32(vdata3, vmax));

        float32x4_t vsum_1 = vaddq_f32(vsum0, vsum1);
        float32x4_t vsum_2 = vaddq_f32(vsum2, vsum3);

        float32x4_t vsum = vaddq_f32(vsum_1, vsum_2);

        float32x4_t vone = vdupq_n_f32(1.0f);
        float32x4_t vinf = div_ps(vone, vsum);

        vsum0 = vmulq_f32(vsum0, vinf);
        vsum1 = vmulq_f32(vsum1, vinf);
        vsum2 = vmulq_f32(vsum2, vinf);
        vsum3 = vmulq_f32(vsum3, vinf);

        vst1q_f32(dout_ptr0, vsum0);
        vst1q_f32(dout_ptr1, vsum1);
        vst1q_f32(dout_ptr2, vsum2);
        vst1q_f32(dout_ptr3, vsum3);

        i += 4;
    }
    for (; i < compute_size; i++){
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        float max_data = din[real_index];
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            max_data = din[real_index] > max_data? din[real_index] : max_data;
        }

        real_index = idx_outer * inner_num + idx_inner;
        //! sub, exp and sum
        dout[real_index] = expf(din[real_index] - max_data);
        float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            dout[real_index] = expf(din[real_index] - max_data);
            sum_data += dout[real_index];
        }

        float sum_inv = 1.f / sum_data;
        real_index = idx_outer * inner_num + idx_inner;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            dout[real_index] *= sum_inv;
            real_index += inner_num;
        }
    }
}

void softmax_arm_lite_channel_in4(const float* din, float* dout, \
    const int axis_size, const int inner_num, \
    const int outer_num, const int compute_size){
    int cmp_cnt = compute_size >> 2;
    int remain = compute_size % 4;
    float32x4_t vone = vdupq_n_f32(1.0f);

#pragma omp parallel for
    for (int c = 0; c < cmp_cnt; ++c) {
        int i = c * 4;
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        //float max_data = din[real_index];
        //! get max axis_size == 4
        const float* din_ptr =  din + real_index;
        const float* din_ptr1 = din_ptr + inner_num;
        const float* din_ptr2 = din_ptr1 + inner_num;
        const float* din_ptr3 = din_ptr2 + inner_num;
        float32x4_t vdata0 = vld1q_f32(din_ptr);
        float32x4_t vdata1 = vld1q_f32(din_ptr1);
        float32x4_t vdata2 = vld1q_f32(din_ptr2);
        float32x4_t vdata3 = vld1q_f32(din_ptr3);

        float* dout_ptr0 = dout + real_index;
        float* dout_ptr1 = dout_ptr0 + inner_num;
        float32x4_t vmax1 = vmaxq_f32(vdata0, vdata1);
        float32x4_t vmax2 = vmaxq_f32(vdata2, vdata3);
        float* dout_ptr2 = dout_ptr1 + inner_num;
        float* dout_ptr3 = dout_ptr2 + inner_num;
        float32x4_t vmax = vmaxq_f32(vmax1, vmax2);

        //! sub, exp and sum
        float32x4_t vsum0 = exp_ps(vsubq_f32(vdata0, vmax));
        float32x4_t vsum1 = exp_ps(vsubq_f32(vdata1, vmax));
        float32x4_t vsum2 = exp_ps(vsubq_f32(vdata2, vmax));
        float32x4_t vsum3 = exp_ps(vsubq_f32(vdata3, vmax));

        float32x4_t vsum_1 = vaddq_f32(vsum0, vsum1);
        float32x4_t vsum_2 = vaddq_f32(vsum2, vsum3);

        float32x4_t vsum = vaddq_f32(vsum_1, vsum_2);

        float32x4_t vinf = div_ps(vone, vsum);

        vsum0 = vmulq_f32(vsum0, vinf);
        vsum1 = vmulq_f32(vsum1, vinf);
        vsum2 = vmulq_f32(vsum2, vinf);
        vsum3 = vmulq_f32(vsum3, vinf);

        vst1q_f32(dout_ptr0, vsum0);
        vst1q_f32(dout_ptr1, vsum1);
        vst1q_f32(dout_ptr2, vsum2);
        vst1q_f32(dout_ptr3, vsum3);
    }

    int i = cmp_cnt * 8;
    for (; i < compute_size; i++){
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

      //  printf("real_index: %d, din: %x\n", real_index, din);

        float max_data = din[real_index];
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            max_data = din[real_index] > max_data? din[real_index] : max_data;
        }

        real_index = idx_outer * inner_num + idx_inner;
        //! sub, exp and sum
        dout[real_index] = expf(din[real_index] - max_data);
        float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            dout[real_index] = expf(din[real_index] - max_data);
            sum_data += dout[real_index];
        }

        float sum_inv = 1.f / sum_data;
        real_index = idx_outer * inner_num + idx_inner;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            dout[real_index] *= sum_inv;
            real_index += inner_num;
        }
    }
}

void softmax_arm_lite_in8(const float* din, float* dout, \
    const int axis_size, const int inner_num, \
    const int outer_num, const int compute_size) {

    int cmp_cnt = compute_size >> 3;
#pragma omp parallel for
    for (int c = 0; c < cmp_cnt; ++c) {
        int i = c * 8;
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        //float max_data = din[real_index];
        const float* din_ptr =  din + real_index;
        float32x4_t vmax = vld1q_f32(din_ptr);
        float32x4_t vmax2 = vld1q_f32(din_ptr + 4);
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            din_ptr += inner_num;
            float32x4_t vdata = vld1q_f32(din_ptr);
            float32x4_t vdata2 = vld1q_f32(din_ptr + 4);
            vmax = vmaxq_f32(vmax, vdata);
            vmax2 = vmaxq_f32(vmax2, vdata2);
        }

        //! sub, exp and sum
      //  dout[real_index] = expf(din[real_index] - max_data);
        din_ptr =  din + real_index;
        float* dout_ptr = dout + real_index;
        float32x4_t vdata = vld1q_f32(din_ptr);
        float32x4_t vdata2 = vld1q_f32(din_ptr + 4);
        float32x4_t vsum  = exp_ps(vsubq_f32(vdata, vmax));
        float32x4_t vsum2  = exp_ps(vsubq_f32(vdata2, vmax2));
        din_ptr += inner_num;
        vst1q_f32(dout_ptr, vsum);
        vst1q_f32(dout_ptr + 4, vsum2);
        dout_ptr += inner_num;
        //float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
          //  real_index += inner_num;
            float32x4_t vdata0 = vld1q_f32(din_ptr);
            float32x4_t vdata1 = vld1q_f32(din_ptr + 4);
            vdata0 = exp_ps(vsubq_f32(vdata0, vmax));
            vdata1 = exp_ps(vsubq_f32(vdata1, vmax2));
            din_ptr += inner_num;
            vsum = vaddq_f32(vsum, vdata0);
            vsum2 = vaddq_f32(vsum2, vdata1);
            vst1q_f32(dout_ptr, vdata0);
            vst1q_f32(dout_ptr + 4, vdata1);
            dout_ptr += inner_num;
        }

      //  float sum_inv = 1.f / sum_data;
        float32x4_t vone = vdupq_n_f32(1.0f);
        float32x4_t vinf = div_ps(vone, vsum);
        float32x4_t vinf2 = div_ps(vone, vsum2);
        dout_ptr = dout + real_index;
       //printf("real_index: %d, dout: %x, dout_ptr: %x \n", real_index, dout, dout_ptr);
       // real_index = idx_outer * inner_num + idx_inner;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            float32x4_t vdata0 = vld1q_f32(dout_ptr);
            float32x4_t vdata1 = vld1q_f32(dout_ptr + 4);
            vdata0 = vmulq_f32(vdata0, vinf);
            vdata1 = vmulq_f32(vdata1, vinf2);
            vst1q_f32(dout_ptr, vdata0);
            vst1q_f32(dout_ptr + 4, vdata1);
            dout_ptr += inner_num;
        }
    }

    for (int i = cmp_cnt * 8; i < compute_size; i++){
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

      //  printf("real_index: %d, din: %x\n", real_index, din);

        float max_data = din[real_index];
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            max_data = din[real_index] > max_data? din[real_index] : max_data;
        }

        real_index = idx_outer * inner_num + idx_inner;
        //! sub, exp and sum
        dout[real_index] = expf(din[real_index] - max_data);
        float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            dout[real_index] = expf(din[real_index] - max_data);
            sum_data += dout[real_index];
        }

        float sum_inv = 1.f / sum_data;
        real_index = idx_outer * inner_num + idx_inner;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            dout[real_index] *= sum_inv;
            real_index += inner_num;
        }
    }

}

void softmax_arm_lite_in4(const float* din, float* dout, \
    const int axis_size, const int inner_num, \
    const int outer_num, const int compute_size) {
    int cmp_cnt = compute_size >> 2;
#pragma omp parallel for
    for (int c = 0; c < cmp_cnt; ++c) {
        int i = c * 4;
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        //float max_data = din[real_index];
        const float* din_ptr =  din + real_index;
        float32x4_t vmax = vld1q_f32(din_ptr);
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            din_ptr += inner_num;
            float32x4_t vdata = vld1q_f32(din_ptr);
            vmax = vmaxq_f32(vmax, vdata);
        }
        //! sub, exp and sum
      //  dout[real_index] = expf(din[real_index] - max_data);
        din_ptr =  din + real_index;
        float* dout_ptr = dout + real_index;
        float32x4_t vdata = vld1q_f32(din_ptr);
        float32x4_t vsum  = exp_ps(vsubq_f32(vdata, vmax));
        din_ptr += inner_num;
        vst1q_f32(dout_ptr, vsum);
        dout_ptr += inner_num;
        //float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
          //  real_index += inner_num;
            float32x4_t vdata0 = vld1q_f32(din_ptr);
            vdata0 = exp_ps(vsubq_f32(vdata0, vmax));
            din_ptr += inner_num;
            vsum = vaddq_f32(vsum, vdata0);
            vst1q_f32(dout_ptr, vdata0);
            dout_ptr += inner_num;
        }

      //  float sum_inv = 1.f / sum_data;
        float32x4_t vone = vdupq_n_f32(1.0f);
        float32x4_t vinf = div_ps(vone, vsum);
        dout_ptr = dout + real_index;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            float32x4_t vdata0 = vld1q_f32(dout_ptr);
            vdata0 = vmulq_f32(vdata0, vinf);
            vst1q_f32(dout_ptr, vdata0);
            dout_ptr += inner_num;
        }
    }

    for (int i = cmp_cnt * 4; i < compute_size; i++){
        int idx_inner = i % inner_num;
        int idx_outer = (i / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;

        float max_data = din[real_index];
        //! get max
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            max_data = din[real_index] > max_data? din[real_index] : max_data;
        }

        real_index = idx_outer * inner_num + idx_inner;
        //! sub, exp and sum
        dout[real_index] = expf(din[real_index] - max_data);
        float sum_data = dout[real_index];
        for (int j = 1; j < axis_size; ++j) {
            real_index += inner_num;
            dout[real_index] = expf(din[real_index] - max_data);
            sum_data += dout[real_index];
        }

        float sum_inv = 1.f / sum_data;
        real_index = idx_outer * inner_num + idx_inner;
        //! get softmax result
        for (int j = 0; j < axis_size; ++j) {
            dout[real_index] *= sum_inv;
            real_index += inner_num;
        }
    }

}

//! for inner size == 1
void softmax_inner1(const float* din, float* dout, \
    const int outer_size, const int axis_size) {
#pragma omp parallel for
    for (int i = 0; i < outer_size; ++i) {
        const float* din_ptr = din + i * axis_size;
        float* dout_ptr = dout + i * axis_size;

        const float* din_max_ptr = din_ptr;
        int nn = axis_size >> 2;

        //! get max
        float32x4_t vmax = vld1q_f32(din_max_ptr);
        din_max_ptr += 4;
        int j = 1;
        for (; j < nn; ++j) {
            vmax = vmaxq_f32(vmax, vld1q_f32(din_max_ptr));
            din_max_ptr += 4;
        }
        float32x2_t vhmax = vmax_f32(vget_high_f32(vmax), vget_low_f32(vmax));
        float max_data = std::max(vget_lane_f32(vhmax, 0), vget_lane_f32(vhmax, 1));
        for (j = 4 * j; j < axis_size; ++j) {
            max_data = std::max(max_data, din_max_ptr[0]);
            din_max_ptr++;
        }
        //printf("max data: %.2f\n", max_data);

        //! sub, exp and sum
        const float* din_sum_ptr = din_ptr;
        float* dout_sum_ptr = dout_ptr;
        vmax = vdupq_n_f32(max_data);
        float32x4_t vsub_exp = exp_ps(vsubq_f32(vld1q_f32(din_sum_ptr), vmax));
        float32x4_t vsum = vsub_exp;
        vst1q_f32(dout_sum_ptr, vsub_exp);
        din_sum_ptr += 4;
        dout_sum_ptr += 4;

        j = 1;
        for (; j < nn; ++j) {
            vsub_exp = exp_ps(vsubq_f32(vld1q_f32(din_sum_ptr), vmax));
            vst1q_f32(dout_sum_ptr, vsub_exp);
            vsum = vaddq_f32(vsum, vsub_exp);
            din_sum_ptr += 4;
            dout_sum_ptr += 4;
        }
        float32x2_t vhsum = vadd_f32(vget_high_f32(vsum), vget_low_f32(vsum));
        float sum_data = vget_lane_f32(vhsum, 0) + vget_lane_f32(vhsum, 1);

        for (j = 4 * j; j < axis_size; ++j) {
            dout_sum_ptr[0] = expf(din_sum_ptr[0] - max_data);
            sum_data += dout_sum_ptr[0];
            din_sum_ptr++;
            dout_sum_ptr++;
        }
        //printf("sum data: %.2f\n", sum_data);

        float sum_inv = 1.f / sum_data;
        float* dout_res_ptr = dout_ptr;
        float32x4_t vinv = vdupq_n_f32(sum_inv);
        //! get softmax result
        j = 0;
        for (; j < nn; ++j) {
            float32x4_t vout = vld1q_f32(dout_res_ptr);
            float32x4_t vres= vmulq_f32(vout, vinv);
            vst1q_f32(dout_res_ptr, vres);
            dout_res_ptr += 4;
        }
        for (j = nn * 4; j < axis_size; ++j) {
            dout_ptr[j] *= sum_inv;
        }
    }
}

//! for inner size == 1 aixs_size < 4
void softmax_inner1_s(const float* din, float* dout, \
    const int outer_size, const int axis_size) {
#pragma omp parallel for
    for (int i = 0; i < outer_size; ++i) {
        const float* din_ptr = din + i * axis_size;
        float* dout_ptr = dout + i * axis_size;
        //! get max
        float max_data = din_ptr[0];
        for (int j =1; j < axis_size; ++j) {
            max_data = std::max(max_data, din_ptr[j]);
        }
        //printf("max data: %.2f\n", max_data);

        //! sub, exp and sum
        float sum_data = 0.f;
        for (int j = 0; j < axis_size; ++j) {
            dout_ptr[j] = expf(din_ptr[j] - max_data);
            sum_data += dout_ptr[j];
        }
        //printf("sum data: %.2f\n", sum_data);

        float sum_inv = 1.f / sum_data;
        for (int j = 0; j < axis_size; ++j) {
            dout_ptr[j] *= sum_inv;
        }
    }
}
template <>
SaberStatus SaberSoftmax<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        SoftmaxParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    float* dout = static_cast<float*>(outputs[0]->mutable_data());
    const float* din = static_cast<const float*>(inputs[0]->data());
    if (this->_inner_num == 1) {
        if (_axis_size >= 4){
            softmax_inner1(din, dout, _outer_num, _axis_size);
        }else{
            softmax_inner1_s(din, dout, _outer_num, _axis_size);
        }
    } else {
        int compute_size = inputs[0]->valid_size() / _axis_size;
        // softmax_basic(din, dout, _axis_size, _inner_num, _outer_num, compute_size);
        if (_axis_size == 4 && _inner_num % 8 == 0){
            softmax_arm_lite_channel_in8(din, dout, _axis_size, _inner_num, _outer_num, compute_size);
        }else if (_axis_size == 4 && _inner_num % 4 == 0){
            softmax_arm_lite_channel_in4(din, dout, _axis_size, _inner_num, _outer_num, compute_size);
        }else{
            if (this->_inner_num % 8 == 0){
                softmax_arm_lite_in8(din, dout, _axis_size, _inner_num, _outer_num, compute_size);
            }else if (this->_inner_num % 4 == 0){
                softmax_arm_lite_in4(din, dout, _axis_size, _inner_num, _outer_num, compute_size);
            }else{
                softmax_basic(din, dout, _axis_size, _inner_num, _outer_num, compute_size);
            }
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Softmax : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    float op_macs = 2.f * inputs[0]->valid_size() * 3;
    //fixme
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("Softmax", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, ARM, AK_INT8);

} //namespace anakin

} //namespace anakin
