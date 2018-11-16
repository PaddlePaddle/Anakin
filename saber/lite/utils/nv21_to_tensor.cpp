#include "saber/lite/utils/cv_utils.h"
namespace anakin{

namespace saber{

namespace lite{

void nv21_to_tensor(const unsigned char* nv21, TensorHf& output, int width, int height, float* means, float* scales) {

    LCHECK_EQ(width, output.width(), "ERROR: sizes of two valid shapes must be the same\n");
    LCHECK_EQ(height, output.height(), "ERROR: sizes of two valid shapes must be the same\n");
    LCHECK_EQ(3, output.channel(), "ERROR: sizes of two valid shapes must be the same\n");
    LCHECK_EQ(1, output.num(), "ERROR: sizes of two valid shapes must be the same\n");
    int size = width * height;
    float* ptr0 = static_cast<float*>(output.mutable_data());
    float* ptr1 = static_cast<float*>(output.mutable_data()) + size;
    float* ptr2 = static_cast<float*>(output.mutable_data()) + size * 2;
    float r_means = means[0];
    float g_means = means[1];
    float b_means = means[2];
    float r_scales = scales[0];
    float g_scales = scales[1];
    float b_scales = scales[2];
    const unsigned char* uv_start = nv21 + size;
    int dim8 = width >> 3;
    int remain = width - (dim8 << 3);
    int tile_h = (height + 1) >> 1; //height is even
    float32x4_t _f128 = vdupq_n_f32(128.0);
    float32x4_t rmean = vdupq_n_f32(r_means);
    float32x4_t gmean = vdupq_n_f32(g_means);
    float32x4_t bmean = vdupq_n_f32(b_means);
    float32x4_t rscale = vdupq_n_f32(r_scales);
    float32x4_t gscale = vdupq_n_f32(g_scales);
    float32x4_t bscale = vdupq_n_f32(b_scales);
    for (int h = 0; h < tile_h; h++){
        const unsigned char* nv21_y0 = nv21 + h * 2 * width;
        const unsigned char* nv21_y1 = nv21_y0 + width;
        const unsigned char* nv21_uv = uv_start + h * width;
        float* r_ptr0 = ptr0 + h * 2 * width;
        float* g_ptr0 = ptr1 + h * 2 * width;
        float* b_ptr0 = ptr2 + h * 2 * width;

        float* r_ptr1 = r_ptr0 + width;
        float* g_ptr1 = g_ptr0 + width;
        float* b_ptr1 = b_ptr0 + width;
#pragma omp parallel for
        for (int w = 0; w < dim8; w++){
            uint8x8_t  y0 = vld1_u8(nv21_y0);
            uint8x8_t  v = vld1_u8(nv21_uv);
            uint8x8_t  u = vld1_u8(nv21_uv);//vuvuvuvu
            uint8x8_t  y1 = vld1_u8(nv21_y1);

            uint8x8x2_t vu = vtrn_u8(v, u);//vvvvuuuu

            uint16x8_t y16_0 = vmovl_u8(y0);
            uint16x8_t v16 = vmovl_u8(vu.val[0]);
            uint16x8_t u16 = vmovl_u8(vu.val[1]);
            uint16x8_t y16_1 = vmovl_u8(y1);

            // printf("y16_0: %d, %d, %d, %d \n", vgetq_lane_u16(y16_0, 0), vgetq_lane_u16(y16_0, 1), vgetq_lane_u16(y16_0, 2), vgetq_lane_u16(y16_0, 3));
            // printf("v16: %d, %d, %d, %d \n", vgetq_lane_u16(v16, 0), vgetq_lane_u16(v16, 1), vgetq_lane_u16(v16, 2), vgetq_lane_u16(v16, 3));
            // printf("u16: %d, %d, %d, %d \n", vgetq_lane_u16(u16, 0), vgetq_lane_u16(u16, 1), vgetq_lane_u16(u16, 2), vgetq_lane_u16(u16, 3));

            float32x4_t y0_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(y16_0)));
            float32x4_t y0_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(y16_0)));

            float32x4_t v_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16)));
            float32x4_t v_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v16)));

            float32x4_t u_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16)));
            float32x4_t u_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16)));

            float32x4_t y1_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(y16_1)));
            float32x4_t y1_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(y16_1)));

            // printf("y16_0: %.3f, %.3f, %.3f, %.3f \n", vgetq_lane_f32(y0_low, 0), vgetq_lane_f32(y0_low, 1), vgetq_lane_f32(y0_low, 2), vgetq_lane_f32(y0_low, 3));
            // printf("v16: %.3f, %.3f, %.3f, %.3f \n", vgetq_lane_f32(v_low, 0), vgetq_lane_f32(v_low, 1), vgetq_lane_f32(v_low, 2), vgetq_lane_f32(v_low, 3));
            // printf("u16: %.3f, %.3f, %.3f, %.3f \n", vgetq_lane_f32(u_low, 0), vgetq_lane_f32(u_low, 1), vgetq_lane_f32(u_low, 2), vgetq_lane_f32(u_low, 3));
            //r = y0 + 0.14 * (v-128) - r_val;
            float32x4_t r0_low = vmlaq_f32(y0_low, vsubq_f32(v_low, _f128), vdupq_n_f32(0.14));
            float32x4_t r0_high = vmlaq_f32(y0_high, vsubq_f32(v_high, _f128), vdupq_n_f32(0.14));
             //g= y0 - 0.34 * (u-128) - 0.71 * (v-128)- g_val;
            float32x4_t g0_low = vmlsq_f32(y0_low, vsubq_f32(v_low, _f128), vdupq_n_f32(0.71));
            float32x4_t g0_high = vmlsq_f32(y0_high, vsubq_f32(v_high, _f128), vdupq_n_f32(0.71));
            //*b_ptr0 = y0 + 1.77 * (u-128) - b_val;
            float32x4_t b0_low = vmlaq_f32(y0_low, vsubq_f32(u_low, _f128), vdupq_n_f32(1.77));
            float32x4_t b0_high = vmlaq_f32(y0_high, vsubq_f32(u_high, _f128), vdupq_n_f32(1.77));

            r0_low = vsubq_f32(r0_low, rmean);
            r0_high = vsubq_f32(r0_high, rmean);

            g0_low = vmlsq_f32(g0_low, vsubq_f32(u_low, _f128), vdupq_n_f32(0.34));
            g0_high = vmlsq_f32(g0_high, vsubq_f32(u_high, _f128), vdupq_n_f32(0.34));

            b0_low = vsubq_f32(b0_low, bmean);
            b0_high = vsubq_f32(b0_high, bmean);

            g0_low = vsubq_f32(g0_low, gmean);
            g0_high = vsubq_f32(g0_high, gmean);

            r0_low = vmulq_f32(r0_low, rscale);
            r0_high = vmulq_f32(r0_high, rscale);
            b0_low = vmulq_f32(b0_low, bscale);
            b0_high = vmulq_f32(b0_high, bscale);
            g0_low = vmulq_f32(g0_low, gscale);
            g0_high = vmulq_f32(g0_high, gscale);

            float32x4_t r1_low = vmlaq_f32(y1_low, vsubq_f32(v_low, _f128), vdupq_n_f32(0.14));
            float32x4_t r1_high = vmlaq_f32(y1_high, vsubq_f32(v_high, _f128), vdupq_n_f32(0.14));
            float32x4_t g1_low = vmlsq_f32(y1_low, vsubq_f32(v_low, _f128), vdupq_n_f32(0.71));
            float32x4_t g1_high = vmlsq_f32(y1_high, vsubq_f32(v_high, _f128), vdupq_n_f32(0.71));
            float32x4_t b1_low = vmlaq_f32(y1_low, vsubq_f32(u_low, _f128), vdupq_n_f32(1.77));
            float32x4_t b1_high = vmlaq_f32(y1_high, vsubq_f32(u_high, _f128), vdupq_n_f32(1.77));

            vst1q_f32(r_ptr0, r0_low);
            vst1q_f32(r_ptr0 + 4, r0_high);

            vst1q_f32(g_ptr0, g0_low);
            vst1q_f32(g_ptr0 + 4, g0_high);

            vst1q_f32(b_ptr0, b0_low);
            vst1q_f32(b_ptr0 + 4, b0_high);

            r1_low = vsubq_f32(r1_low, rmean);
            r1_high = vsubq_f32(r1_high, rmean);

            g1_low = vmlsq_f32(g1_low, vsubq_f32(u_low, _f128), vdupq_n_f32(0.34));
            g1_high = vmlsq_f32(g1_high, vsubq_f32(u_high, _f128), vdupq_n_f32(0.34));

            b1_low = vsubq_f32(b1_low, bmean);
            b1_high = vsubq_f32(b1_high, bmean);

            g1_low = vsubq_f32(g1_low, gmean);
            g1_high = vsubq_f32(g1_high, gmean);

            r1_low = vmulq_f32(r1_low, rscale);
            r1_high = vmulq_f32(r1_high, rscale);
            b1_low = vmulq_f32(b1_low, bscale);
            b1_high = vmulq_f32(b1_high, bscale);
            g1_low = vmulq_f32(g1_low, gscale);
            g1_high = vmulq_f32(g1_high, gscale);

            r_ptr0 += 8;
            g_ptr0 += 8;
            b_ptr0 += 8;

            vst1q_f32(r_ptr1, r1_low);
            vst1q_f32(r_ptr1 + 4, r1_high);

            vst1q_f32(g_ptr1, g1_low);
            vst1q_f32(g_ptr1 + 4, g1_high);

            vst1q_f32(b_ptr1, b1_low);
            vst1q_f32(b_ptr1 + 4, b1_high);

            nv21_y0 += 8;
            nv21_y1 += 8;
            nv21_uv += 8;
            r_ptr1 += 8;
            g_ptr1 += 8;
            b_ptr1 += 8;
        }
       // int w = dim8 << 3;
        int u = 0;
        int v = 0;
        for (int i = 0; i < remain; ++i){
            int y0 = nv21_y0[0];
            int y1 = nv21_y1[0];
            if (i % 2 == 0){
                v = nv21_uv[0];
                u = nv21_uv[1];
                nv21_uv += 2;
            }
            *r_ptr0 = (y0 + 0.14 * (v - 128) - r_means) * r_scales;
            *g_ptr0 = (y0 - 0.34 * (u - 128) - 0.71 * (v-128) - g_means) * g_scales;
            *b_ptr0 = (y0 + 1.77 * (u - 128) - b_means) * b_scales;

            *r_ptr1 = (y1 + 0.14 * (v - 128) - r_means) * r_scales;
            *g_ptr1 = (y1 - 0.34 * (u - 128) - 0.71 * (v-128)- g_means) * g_scales;
            *b_ptr1 = (y1 + 1.77 * (u - 128) - b_means) * b_scales;

            nv21_y0++;
            nv21_y1++;

            r_ptr0++;
            g_ptr0++;
            b_ptr0++;

            r_ptr1++;
            g_ptr1++;
            b_ptr1++;
        }
    }

}

} //namespace lite

} //namespace saber

} //namespace anakin

