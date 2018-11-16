#include "saber/lite/utils/cv_utils.h"
namespace anakin{

namespace saber{

namespace lite{

void bgr_to_tensor(const unsigned char* bgr, TensorHf& output, int width, int height, float* means, float* scales) {

    LCHECK_EQ(width, output.width(), "ERROR: sizes of two valid shapes must be the same\n");
    LCHECK_EQ(height, output.height() * 3, "ERROR: sizes of two valid shapes must be the same\n");
    LCHECK_EQ(3, output.channel(), "ERROR: sizes of two valid shapes must be the same\n");
    LCHECK_EQ(1, output.num(), "ERROR: sizes of two valid shapes must be the same\n");

    int size = width * height / 3;
    float* ptr0 = static_cast<float*>(output.mutable_data());
    float r_means = means[0];
    float g_means = means[1];
    float b_means = means[2];
    float r_scales = scales[0];
    float g_scales = scales[1];
    float b_scales = scales[2];

    int dim8 = width >> 3;
    int remain = width - (dim8 << 3);

    float32x4_t vrmean = vdupq_n_f32(r_means);
    float32x4_t vgmean = vdupq_n_f32(g_means);
    float32x4_t vbmean = vdupq_n_f32(b_means);
    float32x4_t vrscale = vdupq_n_f32(r_scales);
    float32x4_t vgscale = vdupq_n_f32(g_scales);
    float32x4_t vbscale = vdupq_n_f32(b_scales);

    for (int i = 0; i < height; i += 3){
        const unsigned char* ptr_b = bgr + (i * 3) * width;
        const unsigned char* ptr_g = ptr_b + width;
        const unsigned char* ptr_r = ptr_g + width;
        float* ptr0_b = ptr0 + (i / 3)* width;
        float* ptr1_g = ptr0_b + size;
        float* ptr2_r = ptr1_g + size;

        for (int j = 0; j < dim8; j++){
            uint8x8_t vb = vld1_u8(ptr_b);
            uint8x8_t vg = vld1_u8(ptr_g);
            uint8x8_t vr = vld1_u8(ptr_r);

            uint16x8_t vb_16 = vmovl_u8(vb);
            uint16x8_t vg_16 = vmovl_u8(vg);
            uint16x8_t vr_16 = vmovl_u8(vr);

            uint32x4_t vb_low_32 = vmovl_u16(vget_low_u16(vb_16));
            uint32x4_t vg_low_32 = vmovl_u16(vget_low_u16(vg_16));
            uint32x4_t vr_low_32 = vmovl_u16(vget_low_u16(vr_16));

            uint32x4_t vb_high_32 = vmovl_u16(vget_high_u16(vb_16));
            uint32x4_t vg_high_32 = vmovl_u16(vget_high_u16(vg_16));
            uint32x4_t vr_high_32 = vmovl_u16(vget_high_u16(vr_16));

            float32x4_t vb_low_f32 = vcvtq_f32_u32(vb_low_32);
            float32x4_t vr_low_f32 = vcvtq_f32_u32(vr_low_32);
            float32x4_t vg_low_f32 = vcvtq_f32_u32(vg_low_32);

            float32x4_t vb_high_f32 = vcvtq_f32_u32(vb_high_32);
            float32x4_t vg_high_f32 = vcvtq_f32_u32(vg_high_32);
            float32x4_t vr_high_f32 = vcvtq_f32_u32(vr_high_32);

            vb_low_f32 = vsubq_f32(vb_low_f32, vbmean);
            vg_low_f32 = vsubq_f32(vg_low_f32, vgmean);
            vr_low_f32 = vsubq_f32(vr_low_f32, vrmean);

            vb_high_f32 = vsubq_f32(vb_high_f32, vbmean);
            vg_high_f32 = vsubq_f32(vg_high_f32, vgmean);
            vr_high_f32 = vsubq_f32(vr_high_f32, vrmean);

            vb_low_f32 = vmulq_f32(vb_low_f32, vbscale);
            vg_low_f32 = vmulq_f32(vg_low_f32, vgscale);
            vr_low_f32 = vmulq_f32(vr_low_f32, vrscale);

            vb_high_f32 = vmulq_f32(vb_high_f32, vbscale);
            vg_high_f32 = vmulq_f32(vg_high_f32, vgscale);
            vr_high_f32 = vmulq_f32(vr_high_f32, vrscale);

            vst1q_f32(ptr0_b, vb_low_f32);
            vst1q_f32(ptr1_g, vg_low_f32);
            vst1q_f32(ptr2_r, vr_low_f32);

            ptr_b += 8;
            ptr_g += 8;
            ptr_r += 8;

            vst1q_f32(ptr0_b + 4, vb_high_f32);
            vst1q_f32(ptr1_g + 4, vg_high_f32);
            vst1q_f32(ptr2_r + 4, vr_high_f32);

            ptr0_b += 8;
            ptr1_g += 8;
            ptr2_r += 8;

        }

        for (int j = 0; j < remain; j++){
            *ptr0_b++ = (*ptr_b - b_means) * b_scales;
            *ptr1_g++ = (*ptr_g - g_means) * g_scales;
            *ptr2_r++ = (*ptr_r - r_means) * r_scales;

            *ptr_b++;
            *ptr_g++;
            *ptr_r++;
        }
    }

}
/*
asm volatile (
            "pld [%[ptr_b]]                         @ preload a, 64byte\n"
            "pld [%[ptr_g]]                         @ preload a, 64byte\n"
            "pld [%[ptr_r]]                         @ preload a, 64byte\n"
            "vld1.8  {d0}, [%[ptr_b]]!    @ load data \n"
            "vld1.8  {d2}, [%[ptr_g]]!    @ load data \n"
            "vld1.8  {d4}, [%[ptr_r]]!    @ load data \n"

            "vmovl.u8 q4, d0                 @ uint8 -> uint16 \n"
            "vmovl.u8 q5, d2                 @ uint8 -> uint16 \n"
            "vmovl.u8 q6, d3                 @ uint8 -> uint16 \n"

            "vmovl.u16 q0, d8                 @ uint8 -> uint16 \n"
            "vmovl.u16 q1, d10                 @ uint8 -> uint16 \n"
            "vmovl.u16 q2, d12                 @ uint8 -> uint16 \n"

            "vmovl.u16 q7, d9                 @ uint16 -> uint32 \n"
            "vmovl.u16 q8, d11                 @ uint16 -> uint32 \n"
            "vmovl.u16 q9, d13                 @ uint16 -> uint32 \n"

            "vcvt.f32.u32 q4, q0               @ uint32 -> float32 \n"
            "vcvt.f32.u32 q5, q1               @ uint32 -> float32 \n"
            "vcvt.f32.u32 q6, q2               @ uint32 -> float32 \n"

            "vcvt.f32.u32 q0, q7               @ uint32 -> float32 \n"
            "vcvt.f32.u32 q1, q8               @ uint32 -> float32 \n"
            "vcvt.f32.u32 q2, q9               @ uint32 -> float32 \n"

            "vsub.f32 q4, q4, %[vbmean]        @ sub \n"
            "vsub.f32 q5, q5, %[vgmean]        @ sub \n"
            "vsub.f32 q6, q6, %[vrmean]        @ sub \n"

            "vsub.f32 q0, q0, %[vbmean]        @ sub \n"
            "vsub.f32 q1, q1, %[vgmean]        @ sub \n"
            "vsub.f32 q2, q2, %[vrmean]        @ sub \n"

            "vmul.f32 q4, q4, %[vbscale]       @ mul \n"
            "vmul.f32 q5, q5, %[vgscale]       @ mul \n"
            "vmul.f32 q6, q6, %[vrscale]       @ mul \n"

            "vmul.f32 q0, q0, %[vbscale]       @ mul \n"
            "vmul.f32 q1, q1, %[vgscale]       @ mul \n"
            "vmul.f32 q2, q2, %[vrscale]       @ mul \n"

            "vst1.32  {d8-d9}, [%[ptr0_b]]!     @ store result, add pointer\n"
            "vst1.32  {d10-d11}, [%[ptr1_g]]!     @ store result, add pointer\n"
            "vst1.32  {d12-d13}, [%[ptr2_r]]!     @ store result, add pointer\n"

            "vst1.32  {d0-d1}, [%[ptr0_b]]!     @ store result, add pointer\n"
            "vst1.32  {d2-d3}, [%[ptr1_g]]!     @ store result, add pointer\n"
            "vst1.32  {d4-d5}, [%[ptr2_r]]!     @ store result, add pointer\n"
            : [ptr_b] "+r"(ptr_b), [ptr_g] "+r"(ptr_g), [ptr_r] "+r"(ptr_r), [ptr0_b] "+r"(ptr0_b), \
              [ptr1_g] "+r"(ptr1_g), [ptr2_r] "+r"(ptr2_r), [vbmean] "+w"(vbmean), [vgmean] "+w"(vgmean), \
              [vrmean] "+r" (vrmean), [vbscale] "+r" (vbscale), [vgscale] "+r" (vgscale), [vrscale] "+r" (vrscale)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"
        );
*/

} //namespace lite

} //namespace saber

} //namespace anakin

