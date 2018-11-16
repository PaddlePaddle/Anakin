#include "saber/lite/funcs/neon/impl/sgemm_prepacked_int8.h"
namespace anakin{
namespace saber{
namespace lite{
//using namespace anakin::saber;
//using namespace anakin::saber::lite;
#ifdef __aarch64__
void prepackA_4x16_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);
void prepackA_trans_4x16_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);
void sgemm_conv_4x16_int8(const char* A_packed, const char* B, int* C, \
    int M, int N, int K, bool transB, Context* ctx);
void sgemm_conv_4x16_bias_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool transB, Context* ctx);
void sgemm_conv_4x16_relu_int8(const char* A_packed, const char* B, int* C, \
    int M, int N, int K, bool transB, Context* ctx);
void sgemm_conv_4x16_bias_relu_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool transB, Context* ctx);
#else //__aarch64__
void prepackA_4x8_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);
void prepackA_trans_4x8_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);
void sgemm_conv_4x8_int8(const char* A_packed, const char* B, int* C, \
    int M, int N, int K, bool transB, Context* ctx);
void sgemm_conv_4x8_bias_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool transB, Context* ctx);
void sgemm_conv_4x8_relu_int8(const char* A_packed, const char* B, int* C, \
    int M, int N, int K, bool transB, Context* ctx);
void sgemm_conv_4x8_bias_relu_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool transB, Context* ctx);
#endif //__aarch64__
/**
 * \brief input data is not transpose
 * for arm-v7a, transform data to block x k x 6 layout
 * for arm-v8a, transform data to block x k x 8 layout
 */
void prepackA_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax, bool is_trans, Context* ctx) {
#ifdef __aarch64__
    if (is_trans) {
        prepackA_trans_4x16_int8(out, in, ldin, m0, mmax, k0, kmax);
    } else {
        prepackA_4x16_int8(out, in, ldin, m0, mmax, k0, kmax);
    }
#else
    if (is_trans) {
        prepackA_trans_4x8_int8(out, in, ldin, m0, mmax, k0, kmax);
    } else {
        prepackA_4x8_int8(out, in, ldin, m0, mmax, k0, kmax);
    }
#endif
}
void sgemm_prepack_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool is_bias, bool is_relu, bool is_transB, Context* ctx) {
    if (is_relu) {
        if (is_bias) {
#ifdef __aarch64__
            sgemm_conv_4x16_bias_relu_int8(A_packed, B, bias, C, M, N, K, is_transB, ctx);
#else
            sgemm_conv_4x8_bias_relu_int8(A_packed, B, bias, C, M, N, K, is_transB, ctx);
#endif //__aarch64__
        } else {
#ifdef __aarch64__
            sgemm_conv_4x16_relu_int8(A_packed, B, C, M, N, K, is_transB, ctx);
#else
            sgemm_conv_4x8_relu_int8(A_packed, B, C, M, N, K, is_transB, ctx);
#endif //__aarch64__
        }
    } else {
        if (is_bias) {
#ifdef __aarch64__
            sgemm_conv_4x16_bias_int8(A_packed, B, bias, C, M, N, K, is_transB, ctx);
#else
            sgemm_conv_4x8_bias_int8(A_packed, B, bias, C, M, N, K, is_transB, ctx);
#endif //__aarch64__
        } else {
#ifdef __aarch64__
            sgemm_conv_4x16_int8(A_packed, B, C, M, N, K, is_transB, ctx);
#else
            sgemm_conv_4x8_int8(A_packed, B, C, M, N, K, is_transB, ctx);
#endif //__aarch64__
        }
    }
}
#ifdef __aarch64__
void prepackA_4x16_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {
    const char *inptr = in;
    int stride = (kmax - k0) * 4;
    int x_len = kmax - k0;
    char zerobuff[x_len];
    memset(zerobuff, 0, sizeof(char) * x_len);
#pragma omp parallel for
    for (int y = m0; y < mmax; y += 4) {
        char* outptr = out + stride * (y - m0) / 4;
        const char *inptr0 = inptr + y * ldin + k0;
        const char *inptr1 = inptr0 + ldin;
        const char *inptr2 = inptr1 + ldin;
        const char *inptr3 = inptr2 + ldin;
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr0], #64]   \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr1], #64]   \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr2], #64]   \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
                "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        :[ptr0] "r"(inptr0),[ptr1] "r"(inptr1),[ptr2] "r"(inptr2),[ptr3] "r"(inptr3)
        :"memory"
        );
        //! cope with row index exceed real size, set to zero buffer
        if ((y + 3) >= mmax) {
            switch ((y + 3) - mmax) {
                case 2:
                    inptr1 = zerobuff;
                case 1:
                    inptr2 = zerobuff;
                case 0:
                    inptr3 = zerobuff;
                default:
                    break;
            }
        }
        int x = x_len;
        for (; x > 15; x -= 16) {
            asm volatile(
            // Load up 8 elements (2 vectors) from each of 8 sources.
            "ldr q0, [%[inptr0]], #16                           \n"/* load r0, 8 int8*/
                    "ldr q1, [%[inptr1]], #16                   \n"/* load r1, 8 int8*/
                    "ldr q2, [%[inptr2]], #16                   \n"/* load r2, 8 int8*/
                    "ldr q3, [%[inptr3]], #16                   \n"/* load r3, 8 int8*/
                    "trn1    v8.16b, v0.16b, v1.16b             \n"/* trans r0, r1, 16x8*/
                    "trn2    v9.16b, v0.16b, v1.16b             \n"/* trans r0, r1, 16x8*/
                    "trn1    v10.16b, v2.16b, v3.16b            \n"/* trans r2, r3, 16x8*/
                    "trn2    v11.16b, v2.16b, v3.16b            \n"/* trans r2, r3, 16x8*/
                    "trn1    v0.8h, v8.8h, v10.8h               \n"/* trans q8, q10, 8x16*/
                    "trn1    v1.8h, v9.8h, v11.8h               \n"/* trans q9, q11, 8x16*/
                    "trn2    v2.8h, v8.8h, v10.8h               \n"/* trans q8, q10, 8x16*/
                    "trn2    v3.8h, v9.8h, v11.8h               \n"/* trans q9, q11, 8x16*/
                    "trn1    v8.4s, v0.4s, v1.4s                \n"/* trans q0, q4, 4x32*/
                    "trn2    v9.4s, v0.4s, v1.4s                \n"/* trans q2, q6, 4x32*/
                    "trn1    v10.4s, v2.4s, v3.4s               \n"/* trans q1, q5, 4x32*/
                    "trn2    v11.4s, v2.4s, v3.4s               \n"/* trans q3, q7, 4x32*/
                    "trn1    v0.2d, v8.2d, v10.2d               \n"/* get result q0*/
                    "trn1    v1.2d, v9.2d, v11.2d               \n"/* get result q1*/
                    "trn2    v2.2d, v8.2d, v10.2d               \n"/* get result q2*/
                    "trn2    v3.2d, v9.2d, v11.2d               \n"/* get result q3*/
                    "stp        q0, q1, [%[outptr]], #32        \n"/* write to output*/
                    "stp        q2, q3, [%[outptr]], #32        \n"/* write to output*/
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
                [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11"
            );
        }
        for(; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
        }
    }
}
void prepackA_trans_4x16_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax){
    const char *inptr = in + k0 * ldin + m0;
    int x_len = mmax - m0;
    int y_len = kmax - k0;
    int x_len_round = 4 * ((x_len + 3) / 4);
    int right_remain = x_len - 4 * (x_len / 4);
    int remain = 4 - right_remain;
    if (right_remain == 0) {
        remain = 0;
    }
    char *outptr_row = out;
    int stride_out = 4 * y_len;
#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {
        const char* ptr0 = inptr + y * ldin;
        const char* ptr1 = ptr0 + ldin;
        const char* ptr2 = ptr1 + ldin;
        const char* ptr3 = ptr2 + ldin;
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr0], #64]   \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr1], #64]   \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr2], #64]   \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
                "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        :[ptr0] "r"(ptr0),[ptr1] "r"(ptr1),[ptr2] "r"(ptr2),[ptr3] "r"(ptr3)
        :"memory"
        );
        char *outptr_row_col = outptr_row + y * 4;
        int i = 0;
        for (; i < x_len - 3; i += 4) {
            memcpy(outptr_row_col, ptr0, 4);
            memcpy(outptr_row_col + 4, ptr1, 4);
            memcpy(outptr_row_col + 8, ptr2, 4);
            memcpy(outptr_row_col + 12, ptr3, 4);
            outptr_row_col += stride_out;
            ptr0 += 4;
            ptr1 += 4;
            ptr2 += 4;
            ptr3 += 4;
        }
        char *ptr_out = outptr_row_col;
        for (int j = 0; j < right_remain; ++j) {
            ptr_out[j] = *(ptr0++);
            ptr_out[j + 4] = *(ptr1++);
            ptr_out[j + 8] = *(ptr2++);
            ptr_out[j + 12] = *(ptr3++);
        }
        for (int j = right_remain; j < remain; ++j) {
            ptr_out[j] = 0;
            ptr_out[j + 4] = 0;
            ptr_out[j + 8] = 0;
            ptr_out[j + 12] = 0;
        }
    }
#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {
        const char* ptr0 = inptr + y * ldin;
        char* outptr_row_col = outptr_row + y * 4;
        int i = 0;
        for (; i < x_len - 3; i += 4) {
            memcpy(outptr_row_col, ptr0, 4);
            outptr_row_col += stride_out;
            ptr0 += 4;
        }
        for (int j = 0; j < right_remain; ++j) {
            outptr_row_col[j] = *(ptr0++);
        }
        for (int j = right_remain; j < remain; ++j) {
            outptr_row_col[j] = 0;
        }
    }
}
#else //__aarch64__
void prepackA_4x8_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {
    int x_len = kmax - k0;
    char zerobuff[x_len];
    memset(zerobuff, 0, sizeof(char) * x_len);
    const char *inptr = in;
    char* outptr = out;
    //! data A is not transposed, transpose A to k * 4
    for (int y = m0; y < mmax; y += 4) {
        const char *inptr0 = inptr + y * ldin + k0;
        const char *inptr1 = inptr0 + ldin;
        const char *inptr2 = inptr1 + ldin;
        const char *inptr3 = inptr2 + ldin;
        int x = x_len;
        //! cope with row index exceed real size, set to zero buffer
        if ((y + 3) >= mmax) {
            switch ((y + 3) - mmax) {
                case 2:
                    inptr1 = zerobuff;
                case 1:
                    inptr2 = zerobuff;
                case 0:
                    inptr3 = zerobuff;
                default:
                    break;
            }
        }
        for (; x > 7; x -= 8) {
            //! zip load 8 elements (2 neon Q registers) from each of 4 rows
            asm volatile (
            "vld1.8  {d0}, [%[inptr0]]!                     @ load r0, d0=r00,r01,r02,r03,r04,r05,r06,r07\n"
                    "vld1.8  {d1}, [%[inptr1]]!             @ load r1, d1=r10,r11,r12,r13,r14,r15,r16,r17\n"
                    "vld1.8  {d2}, [%[inptr2]]!             @ load r2, d2=r20,r21,r22,r23,r24,r25,r26,r27\n"
                    "vld1.8  {d3}, [%[inptr3]]!             @ load r3, d3=r30,r31,r32,r33,r34,r35,r36,r37\n"
                    "vtrn.8  d0, d1                         @ trans data: d0=r00,r10,r02,r12,r04,r14,r06,r16; d1=r01,r11,r03,r13,r05,r15,r07,r17;\n"
                    "vtrn.8  d2, d3                         @ trans data: d2=r20,r30,r22,r32,r24,r34,r26,r36; d3=r21,r31,r23,r33,r25,r35,r27,r37;\n"
                    "vtrn.16  d0, d2                        @ trans data: d0=r00,r10,r20,r30,r04,r14,r24,r34; d2=r02,r12,r22,r32,r06,r16,r26,r36;\n"
                    "vtrn.16  d1, d3                        @ trans data: d1=r01,r11,r21,r31,r05,r15,r25,r35; d3=r03,r13,r23,r33,r07,r17,r27,r37;\n"
                    "vtrn.32  d0, d1                        @ trans data: d0=r00,r10,r20,r30,r01,r11,r21,r31; d1=r02,r12,r22,r32,r03,r13,r23,r33;\n"
                    "vtrn.32  d2, d3                        @ trans data: d2=r04,r14,r24,r34,r05,r15,r25,r35; d3=r06,r16,r26,r36,r07,r17,r27,r37;\n"
                    "vswp   d1, d2                          @ swap d1, d2\n"
                    "vst1.32  {d0-d3},[%[outptr]]!          @ write to result\n"
            : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), \
                [inptr3] "+r" (inptr3), [outptr] "+r" (outptr)
            :
            : "q0", "q1", "memory"
            );
        }
        for (; x > 0; x--) {
            *(outptr++) = *(inptr0++);
            *(outptr++) = *(inptr1++);
            *(outptr++) = *(inptr2++);
            *(outptr++) = *(inptr3++);
        }
    }
}
void prepackA_trans_4x8_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {
    char *outptr = out;
    const char *inptr = in + k0 * ldin + m0;
    int x_len = mmax - m0;
    int y_len = kmax - k0;
    int right_remain = x_len - 4 * (x_len / 4);
    int remain = 4 - right_remain;
    if (right_remain == 0) {
        remain = 0;
    }
    char *outptr_row = outptr;
    int stride_out = 4 * y_len;
#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {
        const char* ptr0 = inptr + y * ldin;
        const char* ptr1 = ptr0 + ldin;
        const char* ptr2 = ptr1 + ldin;
        const char* ptr3 = ptr2 + ldin;
        char *outptr_row_col = outptr_row + y * 4;
        int i = 0;
        for (; i < x_len - 3; i += 4) {
            memcpy(outptr_row_col, ptr0, 4);
            memcpy(outptr_row_col + 4, ptr1, 4);
            memcpy(outptr_row_col + 8, ptr2, 4);
            memcpy(outptr_row_col + 12, ptr3, 4);
            outptr_row_col += stride_out;
            ptr0 += 4;
            ptr1 += 4;
            ptr2 += 4;
            ptr3 += 4;
        }
        char *ptr_out = outptr_row_col;
        for (int j = 0; j < right_remain; ++j) {
            ptr_out[j] = *(ptr0++);
            ptr_out[j + 4] = *(ptr1++);
            ptr_out[j + 8] = *(ptr2++);
            ptr_out[j + 12] = *(ptr3++);
        }
        for (int j = right_remain; j < remain; ++j) {
            ptr_out[j] = 0;
            ptr_out[j + 4] = 0;
            ptr_out[j + 8] = 0;
            ptr_out[j + 12] = 0;
        }
    }
#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {
        const char* ptr0 = inptr + y * ldin;
        char *outptr_row_col = outptr_row + y * 4;
        int i = 0;
        for (; i < x_len - 3; i += 4) {
            memcpy(outptr_row_col, ptr0, 4);
            outptr_row_col += stride_out;
            ptr0 += 4;
        }
        for (int j = 0; j < right_remain; ++j) {
            outptr_row_col[j] = *(ptr0++);
        }
        for (int j = right_remain; j < remain; ++j) {
            outptr_row_col[j] = 0;
        }
    }
}
#endif //__aarch64__
/**
* \brief input data is transpose
* for arm-v7a, transform data to block x k x 8 layout
* for arm-v8a, transform data to block x k x 16 layout
*/
#ifdef __aarch64__
void loadb_int8(char* out, const char* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {
    uint8_t *outptr = reinterpret_cast<uint8_t *>(out);
    const uint8_t *inptr = reinterpret_cast<const uint8_t*>(in) + k0 * ldin + n0;
    const uint8_t mask_buffer[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int x_len = nmax - n0;
    int y_len = kmax - k0;
    unsigned char right_remain = static_cast<unsigned char>(x_len - 16 * (x_len / 16));
    int right_pad = 16 - right_remain;
    const size_t copy_len_remain = sizeof(float) * right_remain;
    const size_t copy_len_pad = sizeof(float) * right_pad;
    const size_t size_ldin = sizeof(float) * ldin;
    uint8_t* outptr_row = outptr;
    int stride_out = 16 * y_len;
    uint8x16_t vzero = vdupq_n_u8(0);
    uint8x16_t vmask = vcltq_u8(vld1q_u8(mask_buffer), vdupq_n_u8(right_remain));
#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {
        const uint8_t *ptr0 = inptr + y * ldin;
        const uint8_t *ptr1 = ptr0 + ldin;
        const uint8_t *ptr2 = ptr1 + ldin;
        const uint8_t *ptr3 = ptr2 + ldin;
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr0], #64]   \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr1], #64]   \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr2], #64]   \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
                "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        :[ptr0] "r"(ptr0),[ptr1] "r"(ptr1),[ptr2] "r"(ptr2),[ptr3] "r"(ptr3)
        :"memory"
        );
        uint8_t *outptr_row_col = outptr_row + y * 16;
        int i = 0;
        for (; i < x_len - 15; i += 16) {
            uint8x16_t vr00 = vld1q_u8(ptr0);
            uint8x16_t vr10 = vld1q_u8(ptr1);
            uint8x16_t vr20 = vld1q_u8(ptr2);
            uint8x16_t vr30 = vld1q_u8(ptr3);
            vst1q_u8(outptr_row_col, vr00);
            vst1q_u8(outptr_row_col + 16, vr10);
            vst1q_u8(outptr_row_col + 32, vr20);
            vst1q_u8(outptr_row_col + 48, vr30);
            ptr0 += 16;
            ptr1 += 16;
            ptr2 += 16;
            ptr3 += 16;
            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {
            uint8x16_t vr00 = vld1q_u8(ptr0);
            uint8x16_t vr10 = vld1q_u8(ptr1);
            uint8x16_t vr20 = vld1q_u8(ptr2);
            uint8x16_t vr30 = vld1q_u8(ptr3);
            uint8x16_t vr00_1 = vbslq_u8(vmask, vr00, vzero);
            uint8x16_t vr10_1 = vbslq_u8(vmask, vr10, vzero);
            uint8x16_t vr20_1 = vbslq_u8(vmask, vr20, vzero);
            uint8x16_t vr30_1 = vbslq_u8(vmask, vr30, vzero);
            vst1q_u8(outptr_row_col, vr00_1);
            vst1q_u8(outptr_row_col + 16, vr10_1);
            vst1q_u8(outptr_row_col + 32, vr20_1);
            vst1q_u8(outptr_row_col + 48, vr30_1);
        }
    }
#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {
        const uint8_t* ptr0 = inptr + y * ldin;
        uint8_t *outptr_row_col = outptr_row + y * 16;
        int i = 0;
        for (; i < x_len - 15; i += 16) {
            uint8x16_t vr00 = vld1q_u8(ptr0);
            vst1q_u8(outptr_row_col, vr00);
            ptr0 += 16;
            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {
            uint8x16_t vr00 = vld1q_u8(ptr0);
            uint8x16_t vr00_1 = vbslq_u8(vmask, vr00, vzero);
            vst1q_u8(outptr_row_col, vr00_1);
        }
    }
}
void loadb_trans_int8(char* out, const char* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {
    int x_len = kmax - k0;
    char *outptr = out;
    const char *inptr = in;
    char zerobuff[x_len];
    memset(zerobuff, 0, x_len * sizeof(char));
    //! data B is not transposed, transpose B to k * 12
    for (int y = n0; y < nmax; y += 16) {
        const char *inptr0 = inptr + y * ldin + k0;
        const char *inptr1 = inptr0 + ldin;
        const char *inptr2 = inptr1 + ldin;
        const char *inptr3 = inptr2 + ldin;
        const char *inptr4 = inptr3 + ldin;
        const char *inptr5 = inptr4 + ldin;
        const char *inptr6 = inptr5 + ldin;
        const char *inptr7 = inptr6 + ldin;
        const char *inptr8 = inptr7 + ldin;
        const char *inptr9 = inptr8 + ldin;
        const char *inptr10 = inptr9 + ldin;
        const char *inptr11 = inptr10 + ldin;
        const char *inptr12 = inptr11 + ldin;
        const char *inptr13 = inptr12 + ldin;
        const char *inptr14 = inptr13 + ldin;
        const char *inptr15 = inptr14 + ldin;
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr0], #64]   \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr1], #64]   \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr2], #64]   \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
                "prfm   pldl1keep, [%[ptr3], #64]   \n"
                "prfm   pldl1keep, [%[ptr4]]        \n"
                "prfm   pldl1keep, [%[ptr4], #64]   \n"
                "prfm   pldl1keep, [%[ptr5]]        \n"
                "prfm   pldl1keep, [%[ptr5], #64]   \n"
                "prfm   pldl1keep, [%[ptr6]]        \n"
                "prfm   pldl1keep, [%[ptr6], #64]   \n"
                "prfm   pldl1keep, [%[ptr7]]        \n"
                "prfm   pldl1keep, [%[ptr7], #64]   \n"
        :
        :[ptr0] "r"(inptr0),[ptr1] "r"(inptr1),[ptr2] "r"(inptr2),[ptr3] "r"(inptr3),\
                [ptr4] "r"(inptr4),[ptr5] "r"(inptr5),[ptr6] "r"(inptr6),[ptr7] "r"(inptr7)
        :"memory"
        );
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr0], #64]   \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr1], #64]   \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr2], #64]   \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
                "prfm   pldl1keep, [%[ptr3], #64]   \n"
                "prfm   pldl1keep, [%[ptr4]]        \n"
                "prfm   pldl1keep, [%[ptr4], #64]   \n"
                "prfm   pldl1keep, [%[ptr5]]        \n"
                "prfm   pldl1keep, [%[ptr5], #64]   \n"
                "prfm   pldl1keep, [%[ptr6]]        \n"
                "prfm   pldl1keep, [%[ptr6], #64]   \n"
                "prfm   pldl1keep, [%[ptr7]]        \n"
                "prfm   pldl1keep, [%[ptr7], #64]   \n"
        :
        :[ptr0] "r"(inptr8),[ptr1] "r"(inptr9),[ptr2] "r"(inptr10),[ptr3] "r"(inptr11),\
                [ptr4] "r"(inptr12),[ptr5] "r"(inptr13),[ptr6] "r"(inptr14),[ptr7] "r"(inptr15)
        :"memory"
        );
        int x = x_len;
        //! cope with row index exceed real size, set to zero buffer
        if ((y + 15) >= nmax) {
            switch ((y + 15) - nmax) {
                case 14:
                    inptr1 = zerobuff;
                case 13:
                    inptr2 = zerobuff;
                case 12:
                    inptr3 = zerobuff;
                case 11:
                    inptr4 = zerobuff;
                case 10:
                    inptr5 = zerobuff;
                case 9:
                    inptr6 = zerobuff;
                case 8:
                    inptr7 = zerobuff;
                case 7:
                    inptr8 = zerobuff;
                case 6:
                    inptr9 = zerobuff;
                case 5:
                    inptr10 = zerobuff;
                case 4:
                    inptr11 = zerobuff;
                case 3:
                    inptr12 = zerobuff;
                case 2:
                    inptr13 = zerobuff;
                case 1:
                    inptr14 = zerobuff;
                case 0:
                    inptr15 = zerobuff;
                default:
                    break;
            }
        }
        for (; x > 15; x -= 16) {
            asm volatile(
            // Load up 8 elements (2 vectors) from each of 8 sources.
            "ldr q0, [%[inptr0]], #16                           \n"/* load r0, 8 int8*/
                    "ldr q1, [%[inptr1]]                        \n"/* load r1, 8 int8*/
                    "ldr q2, [%[inptr2]]                        \n"/* load r2, 8 int8*/
                    "ldr q3, [%[inptr3]]                        \n"/* load r3, 8 int8*/
                    "ldr q4, [%[inptr4]]                        \n"/* load r4, 8 int8*/
                    "ldr q5, [%[inptr5]]                        \n"/* load r5, 8 int8*/
                    "ldr q6, [%[inptr6]]                        \n"/* load r6, 8 int8*/
                    "ldr q7, [%[inptr7]]                        \n"/* load r7, 8 int8*/
                    "ldr q8, [%[inptr8]]                        \n"/* load r7, 8 int8*/
                    "ldr q9, [%[inptr9]]                        \n"/* load r7, 8 int8*/
                    "ldr q10, [%[inptr10]]                      \n"/* load r7, 8 int8*/
                    "ldr q11, [%[inptr11]]                      \n"/* load r7, 8 int8*/
                    "ldr q12, [%[inptr12]]                      \n"/* load r7, 8 int8*/
                    "ldr q13, [%[inptr13]]                      \n"/* load r7, 8 int8*/
                    "ldr q14, [%[inptr14]]                      \n"/* load r7, 8 int8*/
                    "ldr q15, [%[inptr15]]                      \n"/* load r7, 8 int8*/
                    "trn1    v16.16b, v0.16b, v1.16b            \n"/* trans r0, r1, 16x8*/
                    "trn2    v17.16b, v0.16b, v1.16b            \n"/* trans r0, r1, 16x8*/
                    "trn1    v18.16b, v2.16b, v3.16b            \n"/* trans r2, r3, 16x8*/
                    "trn2    v19.16b, v2.16b, v3.16b            \n"/* trans r2, r3, 16x8*/
                    "trn1    v20.16b, v4.16b, v5.16b            \n"/* trans r4, r5, 16x8*/
                    "trn2    v21.16b, v4.16b, v5.16b            \n"/* trans r4, r5, 16x8*/
                    "trn1    v22.16b, v6.16b, v7.16b            \n"/* trans r6, r7, 16x8*/
                    "trn2    v23.16b, v6.16b, v7.16b            \n"/* trans r6, r7, 16x8*/
                    "trn1    v24.16b, v8.16b, v9.16b            \n"/* trans r8, r9, 16x8*/
                    "trn2    v25.16b, v8.16b, v9.16b            \n"/* trans r8, r9, 16x8*/
                    "trn1    v26.16b, v10.16b, v11.16b          \n"/* trans r10, r11, 16x8*/
                    "trn2    v27.16b, v10.16b, v11.16b          \n"/* trans r10, r11, 16x8*/
                    "trn1    v28.16b, v12.16b, v13.16b          \n"/* trans r12, r13, 16x8*/
                    "trn2    v29.16b, v12.16b, v13.16b          \n"/* trans r12, r13, 16x8*/
                    "trn1    v30.16b, v14.16b, v15.16b          \n"/* trans r14, r15, 16x8*/
                    "trn2    v31.16b, v14.16b, v15.16b          \n"/* trans r14, r15, 16x8*/
                    "trn1    v0.8h, v16.8h, v18.8h              \n"/* trans q16, q18, 8x16*/
                    "trn2    v1.8h, v16.8h, v18.8h              \n"/* trans q16, q18, 8x16*/
                    "trn1    v2.8h, v17.8h, v19.8h              \n"/* trans q17, q19, 8x16*/
                    "trn2    v3.8h, v17.8h, v19.8h              \n"/* trans q17, q19, 8x16*/
                    "trn1    v4.8h, v20.8h, v22.8h              \n"/* trans q20, q22, 8x16*/
                    "trn2    v5.8h, v20.8h, v22.8h              \n"/* trans q20, q22, 8x16*/
                    "trn1    v6.8h, v21.8h, v23.8h              \n"/* trans q21, q23, 8x16*/
                    "trn2    v7.8h, v21.8h, v23.8h              \n"/* trans q21, q23, 8x16*/
                    "trn1    v8.8h, v24.8h, v26.8h              \n"/* trans q24, q26, 8x16*/
                    "trn2    v9.8h, v24.8h, v26.8h              \n"/* trans q24, q26, 8x16*/
                    "trn1    v10.8h, v25.8h, v27.8h             \n"/* trans q25, q27, 8x16*/
                    "trn2    v11.8h, v25.8h, v27.8h             \n"/* trans q25, q27, 8x16*/
                    "trn1    v12.8h, v28.8h, v30.8h             \n"/* trans q28, q30, 8x16*/
                    "trn2    v13.8h, v28.8h, v30.8h             \n"/* trans q28, q30, 8x16*/
                    "trn1    v14.8h, v29.8h, v31.8h             \n"/* trans q29, q31, 8x16*/
                    "trn2    v15.8h, v29.8h, v31.8h             \n"/* trans q29, q31, 8x16*/
                    "trn1    v16.4s, v0.4s, v4.4s               \n"/* trans q0, q4, 4x32*/
                    "trn1    v17.4s, v2.4s, v6.4s               \n"/* trans q2, q6, 4x32*/
                    "trn1    v18.4s, v1.4s, v5.4s               \n"/* trans q1, q5, 4x32*/
                    "trn1    v19.4s, v3.4s, v7.4s               \n"/* trans q3, q7, 4x32*/
                    "trn2    v20.4s, v0.4s, v4.4s               \n"/* trans q0, q4, 4x32*/
                    "trn2    v21.4s, v2.4s, v6.4s               \n"/* trans q2, q6, 4x32*/
                    "trn2    v22.4s, v1.4s, v5.4s               \n"/* trans q1, q5, 4x32*/
                    "trn2    v23.4s, v3.4s, v7.4s               \n"/* trans q3, q7, 4x32*/
                    "trn1    v24.4s, v8.4s, v12.4s              \n"/* trans q8, q12, 4x32*/
                    "trn1    v25.4s, v10.4s, v14.4s             \n"/* trans q10, q14, 4x32*/
                    "trn1    v26.4s, v9.4s, v13.4s              \n"/* trans q9, q13, 4x32*/
                    "trn1    v27.4s, v11.4s, v15.4s             \n"/* trans q11, q15, 4x32*/
                    "trn2    v28.4s, v8.4s, v12.4s              \n"/* trans q8, q12, 4x32*/
                    "trn2    v29.4s, v10.4s, v14.4s             \n"/* trans q10, q14, 4x32*/
                    "trn2    v30.4s, v9.4s, v13.4s              \n"/* trans q9, q13, 4x32*/
                    "trn2    v31.4s, v11.4s, v15.4s             \n"/* trans q11, q15, 4x32*/
                    "trn1    v0.2d, v16.2d, v24.2d              \n"/* get result q0*/
                    "trn1    v1.2d, v17.2d, v25.2d              \n"/* get result q1*/
                    "trn1    v2.2d, v18.2d, v26.2d              \n"/* get result q2*/
                    "trn1    v3.2d, v19.2d, v27.2d              \n"/* get result q3*/
                    "trn1    v4.2d, v20.2d, v28.2d              \n"/* get result q4*/
                    "trn1    v5.2d, v21.2d, v29.2d              \n"/* get result q5*/
                    "trn1    v6.2d, v22.2d, v30.2d              \n"/* get result q6*/
                    "trn1    v7.2d, v23.2d, v31.2d              \n"/* get result q7*/
                    "trn2    v8.2d, v16.2d, v24.2d              \n"/* get result q8*/
                    "trn2    v9.2d, v17.2d, v25.2d              \n"/* get result q9*/
                    "trn2    v10.2d, v18.2d, v26.2d             \n"/* get result q10*/
                    "trn2    v11.2d, v19.2d, v27.2d             \n"/* get result q11*/
                    "trn2    v12.2d, v20.2d, v28.2d             \n"/* get result q12*/
                    "trn2    v13.2d, v21.2d, v29.2d             \n"/* get result q13*/
                    "trn2    v14.2d, v22.2d, v30.2d             \n"/* get result q14*/
                    "trn2    v15.2d, v23.2d, v31.2d             \n"/* get result q15*/
                    "stp        q0, q1, [%[outptr]]             \n"/* write to output*/
                    "stp        q2, q3, [%[outptr], #32]        \n"/* write to output*/
                    "stp        q4, q5, [%[outptr], #64]        \n"/* write to output*/
                    "stp        q6, q7, [%[outptr], #96]        \n"/* write to output*/
                    "stp        q8, q9, [%[outptr], #128]       \n"/* write to output*/
                    "stp        q10, q11, [%[outptr], #160]     \n"/* write to output*/
                    "stp        q12, q13, [%[outptr], #192]     \n"/* write to output*/
                    "stp        q14, q15, [%[outptr], #224]     \n"/* write to output*/
            :
            :  [inptr0] "r"(inptr0), [inptr1] "r"(inptr1), [inptr2] "r"(inptr2), [inptr3] "r"(inptr3), \
                [inptr4] "r"(inptr4), [inptr5] "r"(inptr5), [inptr6] "r"(inptr6), [inptr7] "r"(inptr7) \
                , [inptr8] "r"(inptr8), [inptr9] "r"(inptr9), [inptr10] "r"(inptr10), [inptr11] "r"(inptr11) \
                , [inptr12] "r"(inptr12), [inptr13] "r"(inptr13), [inptr14] "r"(inptr14), [inptr15] "r"(inptr15) \
                , [outptr] "r"(outptr)
            :  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                    "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            inptr0 += 16;
            inptr1 += 16;
            inptr2 += 16;
            inptr3 += 16;
            inptr4 += 16;
            inptr5 += 16;
            inptr6 += 16;
            inptr7 += 16;
            inptr8 += 16;
            inptr9 += 16;
            inptr10 += 16;
            inptr11 += 16;
            inptr12 += 16;
            inptr13 += 16;
            inptr14 += 16;
            inptr15 += 16;
            outptr += 256;
        }
        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
            *outptr++ = *inptr6++;
            *outptr++ = *inptr7++;
            *outptr++ = *inptr8++;
            *outptr++ = *inptr9++;
            *outptr++ = *inptr10++;
            *outptr++ = *inptr11++;
            *outptr++ = *inptr12++;
            *outptr++ = *inptr13++;
            *outptr++ = *inptr14++;
            *outptr++ = *inptr15++;
        }
    }
}
#else //__aarch64__
void loadb_int8(char* out, const char* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {
    char *outptr = out;
    const char *inptr = in + k0 * ldin + n0;
    unsigned char mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int x_len = nmax - n0;
    int y_len = kmax - k0;
    unsigned char right_remain = static_cast<unsigned char>(x_len - 8 * (x_len / 8));
    int right_pad = 8 - right_remain;
    char *outptr_row =outptr;
    int stride_out = 8 * y_len;
    uint8x8_t vzero = vdup_n_u8(0);
    uint8x8_t vmask = vclt_u8(vld1_u8(mask_buffer), vdup_n_u8(right_remain));
#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {
        const char* ptr0 = inptr + y * ldin;
        const char* ptr1 = ptr0 + ldin;
        const char* ptr2 = ptr1 + ldin;
        const char* ptr3 = ptr2 + ldin;
        char *outptr_row_col = outptr_row + y * 8;
        int i = 0;
        for (; i < x_len - 7; i += 8) {
            char *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.8 {d0}, [%[ptr0]]!                    @ load r0, 8 elements\n"
                    "vld1.8 {d1}, [%[ptr1]]!            @ load r1, 8 elements\n"
                    "vld1.8 {d2}, [%[ptr2]]!            @ load r2, 8 elements\n"
                    "vld1.8 {d3}, [%[ptr3]]!            @ load r3, 8 elements\n"
                    "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            :
            : "q0", "q1", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {
            char *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.8 {d0}, [%[ptr0]]!                    @ load r0, 8 elements\n"
                    "vld1.8 {d1}, [%[ptr1]]!            @ load r1, 8 elements\n"
                    "vld1.8 {d2}, [%[ptr2]]!            @ load r2, 8 elements\n"
                    "vld1.8 {d3}, [%[ptr3]]!            @ load r3, 8 elements\n"
                    "vbif   d0, %P[vzero], %P[vmask]    @ bit select, pad zero\n"
                    "vbif   d1, %P[vzero], %P[vmask]    @ bit select, pad zero\n"
                    "vbif   d2, %P[vzero], %P[vmask]    @ bit select, pad zero\n"
                    "vbif   d3, %P[vzero], %P[vmask]    @ bit select, pad zero\n"
                    "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            : [vmask] "w" (vmask), [vzero] "w" (vzero)
            : "q0", "q1", "memory"
            );
        }
        //outptr_row += 32;
    }
#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {
        const char* ptr0 = inptr + y * ldin;
        char *outptr_row_col = outptr_row + y * 8;
        int i = 0;
        for (; i < x_len - 7; i += 8) {
            char *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.8 {d0}, [%[ptr0]]!                    @ load r0, 8 elements\n"
                    "vst1.32 {d0}, [%[outptr]]!         @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            :
            : "q0", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {
            char *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.8 {d0}, [%[ptr0]]!                    @ load r0, 8 elements\n"
                    "vbif   d0, %P[vzero], %P[vmask]    @ bit select, pad zero\n"
                    "vst1.32 {d0}, [%[outptr]]!         @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            : [vmask] "w" (vmask), [vzero] "w" (vzero)
            : "q0", "memory"
            );
        }
    }
}
void loadb_trans_int8(char* out, const char* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {
    char *outptr = out;
    const char *inptr = in;
    int x_len = kmax - k0;
    char zerobuff[x_len];
    memset(zerobuff, 0, sizeof(char) * x_len);
    //! data B is not transposed, transpose B to k * 8
    for (int y = n0; y < nmax; y += 8) {
        const char *inptr0 = inptr + y * ldin + k0;
        const char *inptr1 = inptr0 + ldin;
        const char *inptr2 = inptr1 + ldin;
        const char *inptr3 = inptr2 + ldin;
        const char *inptr4 = inptr3 + ldin;
        const char *inptr5 = inptr4 + ldin;
        const char *inptr6 = inptr5 + ldin;
        const char *inptr7 = inptr6 + ldin;
        int x = x_len;
        //! cope with row index exceed real size, set to zero buffer
        if ((y + 7) >= nmax) {
            switch ((y + 7) - nmax) {
                case 6:
                    inptr1 = zerobuff;
                case 5:
                    inptr2 = zerobuff;
                case 4:
                    inptr3 = zerobuff;
                case 3:
                    inptr4 = zerobuff;
                case 2:
                    inptr5 = zerobuff;
                case 1:
                    inptr6 = zerobuff;
                case 0:
                    inptr7 = zerobuff;
                default:
                    break;
            }
        }
        for (; x > 7; x -= 8) {
            //! zip load 8 elements (2 neon Q registers) from each of 8 rows
            asm volatile (
            "vld1.8 {d0}, [%[inptr0]]!                          @ load r0, 8 int8\n"
                    "vld1.8 {d1}, [%[inptr1]]!                  @ load r1, 8 int8\n"
                    "vld1.8 {d2}, [%[inptr2]]!                  @ load r2, 8 int8\n"
                    "vld1.8 {d3}, [%[inptr3]]!                  @ load r3, 8 int8\n"
                    "vld1.8 {d4}, [%[inptr4]]!                  @ load r4, 8 int8\n"
                    "vld1.8 {d5}, [%[inptr5]]!                  @ load r5, 8 int8\n"
                    "vld1.8 {d6}, [%[inptr6]]!                  @ load r6, 8 int8\n"
                    "vld1.8 {d7}, [%[inptr7]]!                  @ load r7, 8 int8\n"
                    "vtrn.8     d0, d1                          @ trans r0, r1, 8x8\n"
                    "vtrn.8     d2, d3                          @ trans r2, r3, 8x8\n"
                    "vtrn.8     d4, d5                          @ trans r4, r5, 8x8\n"
                    "vtrn.8     d6, d7                          @ trans r6, r7, 8x8\n"
                    "vtrn.16    d0, d2                          @ trans d0, d2, 4x16\n"
                    "vtrn.16    d1, d3                          @ trans d1, d3, 4x16\n"
                    "vtrn.16    d4, d6                          @ trans d4, d6, 4x16\n"
                    "vtrn.16    d5, d7                          @ trans d5, d7, 4x16\n"
                    "vtrn.32    d0, d4                          @ trans d0, d4, 2x32\n"
                    "vtrn.32    d1, d5                          @ trans d1, d5, 2x32\n"
                    "vtrn.32    d2, d6                          @ trans d2, d6, 2x32\n"
                    "vtrn.32    d3, d7                          @ trans d3, d7, 2x32\n"
                    "vst1.32    {d0-d3}, [%[outptr]]!           @ write to output\n"
                    "vst1.32    {d4-d7}, [%[outptr]]!           @ write to output\n"
            : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), \
                [inptr3] "+r" (inptr3), [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), \
                [inptr6] "+r" (inptr6), [inptr7] "+r" (inptr7),[outptr] "+r" (outptr)
            :
            : "q0", "q1", "q2", "q3"
            );
        }
        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
            *outptr++ = *inptr6++;
            *outptr++ = *inptr7++;
        }
    }
}
#endif
#ifdef __aarch64__
#define GEMM_4x16_INT8_KERNEL    \
        "cbz    %w[k], 2f\n"                            /* check loop count > 0 */      \
        /*main loop*/       \
        "1:\n"                                          /*main loop*/       \
        /*unroll 0*/        \
        "ldr    q7, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v0.b[0]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[1]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[2]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[3]\n"                      /*duplicate element in A to vector*/        \
        "smull  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smull2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smull  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smull2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smull  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smull2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smull  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smull2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        /*unroll 1*/        \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v0.b[4]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[5]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[6]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[7]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        /*unroll 2*/        \
        "ldr    q7, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v0.b[8]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[9]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[10]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[11]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        /*unroll 3*/        \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v0.b[12]\n"                     /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[13]\n"                     /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[14]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[15]\n"                     /*duplicate element in A to vector*/        \
        "ldr    q1, [%[a_ptr]], #16\n"                  /* load a00,a01 to q0*/     \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        /*unroll 4*/        \
        "ldr    q7, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v1.b[0]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[1]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[2]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[3]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        /*unroll 5*/        \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v1.b[4]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[5]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[6]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[7]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        /*unroll 6*/        \
        "ldr    q7, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v1.b[8]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[9]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[10]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[11]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        /*unroll 7*/        \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v1.b[12]\n"                     /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[13]\n"                     /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[14]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[15]\n"                     /*duplicate element in A to vector*/        \
        "ldr    q0, [%[a_ptr]], #16\n"                  /* load a00,a01 to q0*/     \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        /*loop count - 1*/      \
        "subs   %w[k], %w[k], #1\n"                     /* loop count - 1*/     \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        /*check loop count*/        \
        "bne    1b\n"                                   /*jump to main loop*/       \
        "2:\n"                                          /* process tail*/       \
        "subs       %w[tail], %w[tail], #1\n"           /* tail--*/     \
        "beq        3f\n"                               /*jump to tail = 1*/        \
        /* final unrool 0*/     \
        "ldr    q7, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v0.b[0]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[1]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[2]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[3]\n"                      /*duplicate element in A to vector*/        \
        "smull  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smull2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smull  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smull2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "subs   %w[tail], %w[tail], #1\n"               /* tail--*/     \
        "smull  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smull2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smull  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smull2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "beq        4f\n"                               /*jump to tail = 2*/        \
        /* final unroll 1*/     \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v0.b[4]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[5]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[6]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[7]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "subs   %w[tail], %w[tail], #1\n"               /* tail--*/     \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "beq        5f\n"                               /*jump to tail = 3*/        \
        /* final unroll 2*/     \
        "ldr    q7, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v0.b[8]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[9]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[10]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[11]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "subs   %w[tail], %w[tail], #1\n"               /* tail--*/     \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "beq        6f\n"                               /*jump to tail = 4*/        \
        /* final unroll 3*/     \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v0.b[12]\n"                     /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[13]\n"                     /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[14]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[15]\n"                     /*duplicate element in A to vector*/        \
        "ldr    q1, [%[a_ptr]], #16\n"                  /* load a00,a01 to q0*/     \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "subs   %w[tail], %w[tail], #1\n"               /* tail--*/     \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "beq        7f\n"                               /*jump to tail = 5*/        \
        /* final unroll 4*/     \
        "ldr    q7, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v1.b[0]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[1]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[2]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[3]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "subs   %w[tail], %w[tail], #1\n"               /* tail--*/     \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "beq        8f\n"                               /*jump to tail = 6*/        \
        /* final unroll 5*/     \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v1.b[4]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[5]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[6]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[7]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "subs   %w[tail], %w[tail], #1\n"               /* tail--*/     \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "beq        9f\n"                               /*jump to tail = 7*/        \
        /* final unroll 6*/     \
        "ldr    q7, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v2.16b, v1.b[8]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[9]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[10]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[11]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/
#define GEMM_4x16_INT8_IN   \
        "movi   v16.4s, #0x0\n"                         /* out0 = 0 */      \
        "ldr    q0, [%[a_ptr]], #16\n"                  /* load a00,a01 to q0*/     \
        "movi   v17.4s, #0x0\n"                         /* out1 = 0*/       \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "movi   v18.4s, #0x0\n"                         /* out2 = 0*/       \
        "movi   v19.4s, #0x0\n"                         /* out3 = 0*/       \
        "movi   v20.4s, #0x0\n"                         /* out4 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #64]\n"           /* preload b*/      \
        "movi   v21.4s, #0x0\n"                         /* out5 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #64]\n"           /* preload a*/      \
        "movi   v22.4s, #0x0\n"                         /* out6 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #128]\n"          /* preload b*/      \
        "movi   v23.4s, #0x0\n"                         /* out7 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #128]\n"          /* preload a*/      \
        "movi   v24.4s, #0x0\n"                         /* out8 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #192]\n"          /* preload b*/      \
        "movi   v25.4s, #0x0\n"                         /* out9 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #256]\n"          /* preload b*/      \
        "movi   v26.4s, #0x0\n"                         /* out10 = 0*/      \
        "prfm   pldl1keep, [%[a_ptr], #192]\n"          /* preload a*/      \
        "movi   v27.4s, #0x0\n"                         /* out11 = 0*/      \
        "prfm   pldl1keep, [%[b_ptr], #320]\n"          /* preload b*/      \
        "movi   v28.4s, #0x0\n"                         /* out12 = 0*/      \
        "prfm   pldl1keep, [%[a_ptr], #256]\n"          /* preload a*/      \
        "movi   v29.4s, #0x0\n"                         /* out13 = 0*/      \
        "prfm   pldl1keep, [%[b_ptr], #384]\n"          /* preload b*/      \
        "movi   v30.4s, #0x0\n"                         /* out14 = 0*/      \
        "movi   v31.4s, #0x0\n"                         /* out15 = 0*/
#define GEMM_4x16_INT8_IN_BIAS   \
        "dup    v16.4s, %w[bias0]\n"                    /* out0 = bias0 */      \
        "ldr    q0, [%[a_ptr]], #16\n"                  /* load a00,a01 to q0*/     \
        "dup    v17.4s, %w[bias0]\n"                    /* out1 = bias0 */      \
        "ldr    q6, [%[b_ptr]], #16\n"                  /* load b0, b1 to q6*/      \
        "dup    v18.4s, %w[bias0]\n"                    /* out2 = bias0 */      \
        "dup    v19.4s, %w[bias0]\n"                    /* out3 = bias0 */      \
        "dup    v20.4s, %w[bias1]\n"                    /* out4 = bias1 */      \
        "prfm   pldl1keep, [%[b_ptr], #64]\n"           /* preload b*/      \
        "dup    v21.4s, %w[bias1]\n"                    /* out5 = bias1 */      \
        "prfm   pldl1keep, [%[a_ptr], #64]\n"           /* preload a*/      \
        "dup    v22.4s, %w[bias1]\n"                    /* out6 = bias1 */      \
        "prfm   pldl1keep, [%[b_ptr], #128]\n"          /* preload b*/      \
        "dup    v23.4s, %w[bias1]\n"                    /* out7 = bias1 */      \
        "prfm   pldl1keep, [%[a_ptr], #128]\n"          /* preload a*/      \
        "dup    v24.4s, %w[bias2]\n"                    /* out8 = bias2 */      \
        "prfm   pldl1keep, [%[b_ptr], #192]\n"          /* preload b*/      \
        "dup    v25.4s, %w[bias2]\n"                    /* out9 = bias2 */      \
        "prfm   pldl1keep, [%[b_ptr], #256]\n"          /* preload b*/      \
        "dup    v26.4s, %w[bias2]\n"                    /* out10 = bias2 */      \
        "prfm   pldl1keep, [%[a_ptr], #192]\n"          /* preload a*/      \
        "dup    v27.4s, %w[bias2]\n"                    /* out11 = bias2 */      \
        "prfm   pldl1keep, [%[b_ptr], #320]\n"          /* preload b*/      \
        "dup    v28.4s, %w[bias3]\n"                    /* out12 = bias3 */      \
        "prfm   pldl1keep, [%[a_ptr], #256]\n"          /* preload a*/      \
        "dup    v29.4s, %w[bias3]\n"                    /* out13 = bias3 */      \
        "prfm   pldl1keep, [%[b_ptr], #384]\n"          /* preload b*/      \
        "dup    v30.4s, %w[bias3]\n"                    /* out14 = bias3 */      \
        "dup    v31.4s, %w[bias3]\n"                    /* out15 = bias3 */
#define GEMM_4x16_INT8_OUT  \
        /* final unroll 7*/     \
        "dup    v2.16b, v1.b[12]\n"                     /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[13]\n"                     /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[14]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[15]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b  0f\n"                                       /*jump to end*/     \
        /*final tail = 1*/      \
        "3:\n"                                          /*tail = 1*/        \
        "dup    v2.16b, v0.b[0]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[1]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[2]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[3]\n"                      /*duplicate element in A to vector*/        \
        "smull  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smull2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smull  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smull2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smull  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smull2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smull  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smull2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b      0f\n"                                   /*jump to end*/     \
        "4:\n"                                          /*tail = 2*/        \
        "dup    v2.16b, v0.b[4]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[5]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[6]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[7]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "subs   %w[tail], %w[tail], #1\n"               /* tail--*/     \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b      0f\n"                                   /*jump to end*/     \
        "5:\n"                                          /*tail = 3*/        \
        "dup    v2.16b, v0.b[8]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[9]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[10]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[11]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b      0f\n"                                   /*jump to end*/     \
        "6:\n"                                          /*tail = 4*/        \
        "dup    v2.16b, v0.b[12]\n"                     /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[13]\n"                     /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[14]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[15]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b      0f\n"                                   /*jump to end*/     \
        "7:\n"                                          /*tail = 5*/        \
        "dup    v2.16b, v1.b[0]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[1]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[2]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[3]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b      0f\n"                                   /*jump to end*/     \
        "8:\n"                                          /*tail = 6*/        \
        "dup    v2.16b, v1.b[4]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[5]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[6]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[7]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b      0f\n"                                   /*jump to end*/     \
        "9:\n"                                          /*tail = 7*/        \
        "dup    v2.16b, v1.b[8]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[9]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[10]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[11]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "0:\n"                                          /*end*/     \
        "stp    q30, q31,   [%[c_ptr3]], #32\n"         /*write to memory*/
#define GEMM_4x16_INT8_OUT_RELU  \
        /* final unroll 7*/     \
        "dup    v2.16b, v1.b[12]\n"                     /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[13]\n"                     /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[14]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[15]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "movi   v0.4s, #0\n"                            /* for relu */      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "smax   v16.4s, v16.4s, v0.4s\n"                /*relu*/        \
        "smax   v17.4s, v17.4s, v0.4s\n"                /*relu*/        \
        "smax   v18.4s, v18.4s, v0.4s\n"                /*relu*/        \
        "smax   v19.4s, v19.4s, v0.4s\n"                /*relu*/        \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v20.4s, v20.4s, v0.4s\n"                /*relu*/        \
        "smax   v21.4s, v21.4s, v0.4s\n"                /*relu*/        \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v22.4s, v22.4s, v0.4s\n"                /*relu*/        \
        "smax   v23.4s, v23.4s, v0.4s\n"                /*relu*/        \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v24.4s, v24.4s, v0.4s\n"                /*relu*/        \
        "smax   v25.4s, v25.4s, v0.4s\n"                /*relu*/        \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v26.4s, v26.4s, v0.4s\n"                /*relu*/        \
        "smax   v27.4s, v27.4s, v0.4s\n"                /*relu*/        \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v28.4s, v28.4s, v0.4s\n"                /*relu*/        \
        "smax   v29.4s, v29.4s, v0.4s\n"                /*relu*/        \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v30.4s, v30.4s, v0.4s\n"                /*relu*/        \
        "smax   v31.4s, v31.4s, v0.4s\n"                /*relu*/        \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b  0f\n"                                       /*jump to end*/     \
        /*final tail = 1*/      \
        "3:\n"                                          /*tail = 1*/        \
        "dup    v2.16b, v0.b[0]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[1]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[2]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[3]\n"                      /*duplicate element in A to vector*/        \
        "smull  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smull2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smull  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smull2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smull  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smull2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smull  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smull2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "movi   v0.4s, #0\n"                            /* for relu */      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "smax   v16.4s, v16.4s, v0.4s\n"                /*relu*/        \
        "smax   v17.4s, v17.4s, v0.4s\n"                /*relu*/        \
        "smax   v18.4s, v18.4s, v0.4s\n"                /*relu*/        \
        "smax   v19.4s, v19.4s, v0.4s\n"                /*relu*/        \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v20.4s, v20.4s, v0.4s\n"                /*relu*/        \
        "smax   v21.4s, v21.4s, v0.4s\n"                /*relu*/        \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v22.4s, v22.4s, v0.4s\n"                /*relu*/        \
        "smax   v23.4s, v23.4s, v0.4s\n"                /*relu*/        \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v24.4s, v24.4s, v0.4s\n"                /*relu*/        \
        "smax   v25.4s, v25.4s, v0.4s\n"                /*relu*/        \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v26.4s, v26.4s, v0.4s\n"                /*relu*/        \
        "smax   v27.4s, v27.4s, v0.4s\n"                /*relu*/        \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v28.4s, v28.4s, v0.4s\n"                /*relu*/        \
        "smax   v29.4s, v29.4s, v0.4s\n"                /*relu*/        \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v30.4s, v30.4s, v0.4s\n"                /*relu*/        \
        "smax   v31.4s, v31.4s, v0.4s\n"                /*relu*/        \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b  0f\n"                                       /*jump to end*/     \
        "4:\n"                                          /*tail = 2*/        \
        "dup    v2.16b, v0.b[4]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[5]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[6]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[7]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "subs   %w[tail], %w[tail], #1\n"               /* tail--*/     \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "movi   v0.4s, #0\n"                            /* for relu */      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "smax   v16.4s, v16.4s, v0.4s\n"                /*relu*/        \
        "smax   v17.4s, v17.4s, v0.4s\n"                /*relu*/        \
        "smax   v18.4s, v18.4s, v0.4s\n"                /*relu*/        \
        "smax   v19.4s, v19.4s, v0.4s\n"                /*relu*/        \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v20.4s, v20.4s, v0.4s\n"                /*relu*/        \
        "smax   v21.4s, v21.4s, v0.4s\n"                /*relu*/        \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v22.4s, v22.4s, v0.4s\n"                /*relu*/        \
        "smax   v23.4s, v23.4s, v0.4s\n"                /*relu*/        \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v24.4s, v24.4s, v0.4s\n"                /*relu*/        \
        "smax   v25.4s, v25.4s, v0.4s\n"                /*relu*/        \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v26.4s, v26.4s, v0.4s\n"                /*relu*/        \
        "smax   v27.4s, v27.4s, v0.4s\n"                /*relu*/        \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v28.4s, v28.4s, v0.4s\n"                /*relu*/        \
        "smax   v29.4s, v29.4s, v0.4s\n"                /*relu*/        \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v30.4s, v30.4s, v0.4s\n"                /*relu*/        \
        "smax   v31.4s, v31.4s, v0.4s\n"                /*relu*/        \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b  0f\n"                                       /*jump to end*/     \
        "5:\n"                                          /*tail = 3*/        \
        "dup    v2.16b, v0.b[8]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[9]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[10]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[11]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "movi   v0.4s, #0\n"                            /* for relu */      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "smax   v16.4s, v16.4s, v0.4s\n"                /*relu*/        \
        "smax   v17.4s, v17.4s, v0.4s\n"                /*relu*/        \
        "smax   v18.4s, v18.4s, v0.4s\n"                /*relu*/        \
        "smax   v19.4s, v19.4s, v0.4s\n"                /*relu*/        \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v20.4s, v20.4s, v0.4s\n"                /*relu*/        \
        "smax   v21.4s, v21.4s, v0.4s\n"                /*relu*/        \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v22.4s, v22.4s, v0.4s\n"                /*relu*/        \
        "smax   v23.4s, v23.4s, v0.4s\n"                /*relu*/        \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v24.4s, v24.4s, v0.4s\n"                /*relu*/        \
        "smax   v25.4s, v25.4s, v0.4s\n"                /*relu*/        \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v26.4s, v26.4s, v0.4s\n"                /*relu*/        \
        "smax   v27.4s, v27.4s, v0.4s\n"                /*relu*/        \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v28.4s, v28.4s, v0.4s\n"                /*relu*/        \
        "smax   v29.4s, v29.4s, v0.4s\n"                /*relu*/        \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v30.4s, v30.4s, v0.4s\n"                /*relu*/        \
        "smax   v31.4s, v31.4s, v0.4s\n"                /*relu*/        \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b  0f\n"                                       /*jump to end*/     \
        "6:\n"                                          /*tail = 4*/        \
        "dup    v2.16b, v0.b[12]\n"                     /*duplicate element in A to vector*/        \
        "dup    v3.16b, v0.b[13]\n"                     /*duplicate element in A to vector*/        \
        "dup    v4.16b, v0.b[14]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v0.b[15]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "movi   v0.4s, #0\n"                            /* for relu */      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "smax   v16.4s, v16.4s, v0.4s\n"                /*relu*/        \
        "smax   v17.4s, v17.4s, v0.4s\n"                /*relu*/        \
        "smax   v18.4s, v18.4s, v0.4s\n"                /*relu*/        \
        "smax   v19.4s, v19.4s, v0.4s\n"                /*relu*/        \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v20.4s, v20.4s, v0.4s\n"                /*relu*/        \
        "smax   v21.4s, v21.4s, v0.4s\n"                /*relu*/        \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v22.4s, v22.4s, v0.4s\n"                /*relu*/        \
        "smax   v23.4s, v23.4s, v0.4s\n"                /*relu*/        \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v24.4s, v24.4s, v0.4s\n"                /*relu*/        \
        "smax   v25.4s, v25.4s, v0.4s\n"                /*relu*/        \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v26.4s, v26.4s, v0.4s\n"                /*relu*/        \
        "smax   v27.4s, v27.4s, v0.4s\n"                /*relu*/        \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v28.4s, v28.4s, v0.4s\n"                /*relu*/        \
        "smax   v29.4s, v29.4s, v0.4s\n"                /*relu*/        \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v30.4s, v30.4s, v0.4s\n"                /*relu*/        \
        "smax   v31.4s, v31.4s, v0.4s\n"                /*relu*/        \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b  0f\n"                                       /*jump to end*/     \
        "7:\n"                                          /*tail = 5*/        \
        "dup    v2.16b, v1.b[0]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[1]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[2]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[3]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "movi   v0.4s, #0\n"                            /* for relu */      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "smax   v16.4s, v16.4s, v0.4s\n"                /*relu*/        \
        "smax   v17.4s, v17.4s, v0.4s\n"                /*relu*/        \
        "smax   v18.4s, v18.4s, v0.4s\n"                /*relu*/        \
        "smax   v19.4s, v19.4s, v0.4s\n"                /*relu*/        \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v20.4s, v20.4s, v0.4s\n"                /*relu*/        \
        "smax   v21.4s, v21.4s, v0.4s\n"                /*relu*/        \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v22.4s, v22.4s, v0.4s\n"                /*relu*/        \
        "smax   v23.4s, v23.4s, v0.4s\n"                /*relu*/        \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v24.4s, v24.4s, v0.4s\n"                /*relu*/        \
        "smax   v25.4s, v25.4s, v0.4s\n"                /*relu*/        \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v26.4s, v26.4s, v0.4s\n"                /*relu*/        \
        "smax   v27.4s, v27.4s, v0.4s\n"                /*relu*/        \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v28.4s, v28.4s, v0.4s\n"                /*relu*/        \
        "smax   v29.4s, v29.4s, v0.4s\n"                /*relu*/        \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v30.4s, v30.4s, v0.4s\n"                /*relu*/        \
        "smax   v31.4s, v31.4s, v0.4s\n"                /*relu*/        \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b  0f\n"                                       /*jump to end*/     \
        "8:\n"                                          /*tail = 6*/        \
        "dup    v2.16b, v1.b[4]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[5]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[6]\n"                      /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[7]\n"                      /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v7.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v7.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v7.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v7.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v7.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v7.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v7.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v7.16b\n"                /*q15 = 8x8bit, high*/      \
        "movi   v0.4s, #0\n"                            /* for relu */      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "smax   v16.4s, v16.4s, v0.4s\n"                /*relu*/        \
        "smax   v17.4s, v17.4s, v0.4s\n"                /*relu*/        \
        "smax   v18.4s, v18.4s, v0.4s\n"                /*relu*/        \
        "smax   v19.4s, v19.4s, v0.4s\n"                /*relu*/        \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v20.4s, v20.4s, v0.4s\n"                /*relu*/        \
        "smax   v21.4s, v21.4s, v0.4s\n"                /*relu*/        \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v22.4s, v22.4s, v0.4s\n"                /*relu*/        \
        "smax   v23.4s, v23.4s, v0.4s\n"                /*relu*/        \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v24.4s, v24.4s, v0.4s\n"                /*relu*/        \
        "smax   v25.4s, v25.4s, v0.4s\n"                /*relu*/        \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v26.4s, v26.4s, v0.4s\n"                /*relu*/        \
        "smax   v27.4s, v27.4s, v0.4s\n"                /*relu*/        \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v28.4s, v28.4s, v0.4s\n"                /*relu*/        \
        "smax   v29.4s, v29.4s, v0.4s\n"                /*relu*/        \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v30.4s, v30.4s, v0.4s\n"                /*relu*/        \
        "smax   v31.4s, v31.4s, v0.4s\n"                /*relu*/        \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "b  0f\n"                                       /*jump to end*/     \
        "9:\n"                                          /*tail = 7*/        \
        "dup    v2.16b, v1.b[8]\n"                      /*duplicate element in A to vector*/        \
        "dup    v3.16b, v1.b[9]\n"                      /*duplicate element in A to vector*/        \
        "dup    v4.16b, v1.b[10]\n"                     /*duplicate element in A to vector*/        \
        "dup    v5.16b, v1.b[11]\n"                     /*duplicate element in A to vector*/        \
        "smlal  v8.8h , v2.8b, v6.8b\n"                 /*q8 = 8x8bit, low*/        \
        "smlal2 v9.8h , v2.16b,v6.16b\n"                /*q9 = 8x8bit, high*/       \
        "smlal  v10.8h, v3.8b, v6.8b\n"                 /*q10 = 8x8bit, low*/       \
        "smlal2 v11.8h, v3.16b,v6.16b\n"                /*q11 = 8x8bit, high*/      \
        "smlal  v12.8h, v4.8b, v6.8b\n"                 /*q12 = 8x8bit, low*/       \
        "smlal2 v13.8h, v4.16b,v6.16b\n"                /*q13 = 8x8bit, high*/      \
        "smlal  v14.8h, v5.8b, v6.8b\n"                 /*q14 = 8x8bit, low*/       \
        "smlal2 v15.8h, v5.16b,v6.16b\n"                /*q15 = 8x8bit, high*/      \
        "movi   v0.4s, #0\n"                            /* for relu */      \
        "saddw  v16.4s, v16.4s, v8.4h\n"                /*  0, accumulate to result*/       \
        "saddw2 v17.4s, v17.4s, v8.8h\n"                /*  1, accumulate to result*/       \
        "saddw  v18.4s, v18.4s, v9.4h\n"                /*  2, accumulate to result*/       \
        "saddw2 v19.4s, v19.4s, v9.8h\n"                /*  3, accumulate to result*/       \
        "saddw  v20.4s, v20.4s,v10.4h\n"                /*  4, accumulate to result*/       \
        "saddw2 v21.4s, v21.4s,v10.8h\n"                /*  5, accumulate to result*/       \
        "saddw  v22.4s, v22.4s,v11.4h\n"                /*  6, accumulate to result*/       \
        "saddw2 v23.4s, v23.4s,v11.8h\n"                /*  7, accumulate to result*/       \
        "saddw  v24.4s, v24.4s,v12.4h\n"                /*  8, accumulate to result*/       \
        "saddw2 v25.4s, v25.4s,v12.8h\n"                /*  9, accumulate to result*/       \
        "saddw  v26.4s, v26.4s,v13.4h\n"                /*  10, accumulate to result*/      \
        "saddw2 v27.4s, v27.4s,v13.8h\n"                /*  11, accumulate to result*/      \
        "saddw  v28.4s, v28.4s,v14.4h\n"                /*  12, accumulate to result*/      \
        "saddw2 v29.4s, v29.4s,v14.8h\n"                /*  13, accumulate to result*/      \
        "saddw  v30.4s, v30.4s,v15.4h\n"                /*  14, accumulate to result*/      \
        "saddw2 v31.4s, v31.4s,v15.8h\n"                /*  15, accumulate to result*/      \
        "smax   v16.4s, v16.4s, v0.4s\n"                /*relu*/        \
        "smax   v17.4s, v17.4s, v0.4s\n"                /*relu*/        \
        "smax   v18.4s, v18.4s, v0.4s\n"                /*relu*/        \
        "smax   v19.4s, v19.4s, v0.4s\n"                /*relu*/        \
        "stp    q16, q17,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v20.4s, v20.4s, v0.4s\n"                /*relu*/        \
        "smax   v21.4s, v21.4s, v0.4s\n"                /*relu*/        \
        "stp    q18, q19,   [%[c_ptr0]], #32\n"         /*write to memory*/     \
        "smax   v22.4s, v22.4s, v0.4s\n"                /*relu*/        \
        "smax   v23.4s, v23.4s, v0.4s\n"                /*relu*/        \
        "stp    q20, q21,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v24.4s, v24.4s, v0.4s\n"                /*relu*/        \
        "smax   v25.4s, v25.4s, v0.4s\n"                /*relu*/        \
        "stp    q22, q23,   [%[c_ptr1]], #32\n"         /*write to memory*/     \
        "smax   v26.4s, v26.4s, v0.4s\n"                /*relu*/        \
        "smax   v27.4s, v27.4s, v0.4s\n"                /*relu*/        \
        "stp    q24, q25,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v28.4s, v28.4s, v0.4s\n"                /*relu*/        \
        "smax   v29.4s, v29.4s, v0.4s\n"                /*relu*/        \
        "stp    q26, q27,   [%[c_ptr2]], #32\n"         /*write to memory*/     \
        "smax   v30.4s, v30.4s, v0.4s\n"                /*relu*/        \
        "smax   v31.4s, v31.4s, v0.4s\n"                /*relu*/        \
        "stp    q28, q29,   [%[c_ptr3]], #32\n"         /*write to memory*/     \
        "0:\n"                                          /*end*/     \
        "stp    q30, q31,   [%[c_ptr3]], #32\n"         /*write to memory*/
void sgemm_conv_4x16_int8(const char* A_packed, const char* B, int* C, int M, int N, int K, \
    bool transB, Context* ctx) {
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(char) * K);
    x_block /= 16;
    x_block *= 16;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 15) / 16;
    x_block *= 16;
    // unroll 2 loop
    int tail_pre = (K & 7);
    int k_pre = ((K + 7) / 8) - 1;
    int zerobuf[x_block];
    bool flag_p_remain = false;
    int remain = 0;
    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 15) / 16;
        remain = xmax - x0 - (bblocks - 1) * 16;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        char *b_pannel = static_cast<char *>(workspace);
        //const char* b_pannel = B;
        if (transB) {
            loadb_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb_int8(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }
            int cout0[16];
            int cout1[16];
            int cout2[16];
            int cout3[16];
            int *c_ptr0 = C + y * N + x0;
            int *c_ptr1 = c_ptr0 + N;
            int *c_ptr2 = c_ptr1 + N;
            int *c_ptr3 = c_ptr2 + N;
            int *pout0 = c_ptr0;
            int *pout1 = c_ptr1;
            int *pout2 = c_ptr2;
            int *pout3 = c_ptr3;
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    case 2:
                        c_ptr1 = zerobuf;
                    case 1:
                        c_ptr2 = zerobuf;
                    case 0:
                        c_ptr3 = zerobuf;
                    default:
                        break;
                }
            }
            const char *a_ptr_l = A_packed + y * K;
            const char *b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {
                if (flag_p_remain && (xb == bblocks - 1)) {
                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                }
                const char *a_ptr = a_ptr_l;
                int tail = tail_pre;
                int k = k_pre;
                asm volatile (
                GEMM_4x16_INT8_IN
                GEMM_4x16_INT8_KERNEL
                GEMM_4x16_INT8_OUT
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), [tail] "+r" (tail), \
                  [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
                        [c_ptr3] "+r"(c_ptr3)
                :
                : "v0", "v1","v2","v3","v4","v5","v6","v7","v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                    }
                }
            }
        }
    }
}
void sgemm_conv_4x16_bias_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool transB, Context* ctx) {
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(char) * K);
    x_block /= 16;
    x_block *= 16;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 15) / 16;
    x_block *= 16;
    // unroll 2 loop
    int tail_pre = (K & 7);
    int k_pre = ((K + 7) / 8) - 1;
    int zerobuf[x_block];
    bool flag_p_remain = false;
    int remain = 0;
    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 15) / 16;
        remain = xmax - x0 - (bblocks - 1) * 16;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        char *b_pannel = static_cast<char *>(workspace);
        if (transB) {
            loadb_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb_int8(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }
            int cout0[16];
            int cout1[16];
            int cout2[16];
            int cout3[16];
            int bias_ptr[4] = {0, 0, 0, 0};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }
            int bias0 = bias_ptr[0];
            int bias1 = bias_ptr[1];
            int bias2 = bias_ptr[2];
            int bias3 = bias_ptr[3];
            int *c_ptr0 = C + y * N + x0;
            int *c_ptr1 = c_ptr0 + N;
            int *c_ptr2 = c_ptr1 + N;
            int *c_ptr3 = c_ptr2 + N;
            int *pout0 = c_ptr0;
            int *pout1 = c_ptr1;
            int *pout2 = c_ptr2;
            int *pout3 = c_ptr3;
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    case 2:
                        c_ptr1 = zerobuf;
                    case 1:
                        c_ptr2 = zerobuf;
                    case 0:
                        c_ptr3 = zerobuf;
                    default:
                        break;
                }
            }
            const char *a_ptr_l = A_packed + y * K;
            const char *b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {
                if (flag_p_remain && (xb == bblocks - 1)) {
                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                }
                const char *a_ptr = a_ptr_l;
                int tail = tail_pre;
                int k = k_pre;
                asm volatile (
                GEMM_4x16_INT8_IN_BIAS
                GEMM_4x16_INT8_KERNEL
                GEMM_4x16_INT8_OUT
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), [tail] "+r" (tail), \
                  [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
                        [c_ptr3] "+r"(c_ptr3)
                : [bias0] "r"(bias0), [bias1] "r"(bias1), [bias2] "r"(bias2), [bias3] "r"(bias3)
                : "v0", "v1","v2","v3","v4","v5","v6","v7","v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                    }
                }
            }
        }
    }
}
void sgemm_conv_4x16_relu_int8(const char* A_packed, const char* B, int* C, int M, int N, int K, \
    bool transB, Context* ctx) {
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(char) * K);
    x_block /= 16;
    x_block *= 16;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 15) / 16;
    x_block *= 16;
    // unroll 2 loop
    int tail_pre = (K & 7);
    int k_pre = ((K + 7) / 8) - 1;
    int zerobuf[x_block];
    bool flag_p_remain = false;
    int remain = 0;
    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 15) / 16;
        remain = xmax - x0 - (bblocks - 1) * 16;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        char *b_pannel = static_cast<char *>(workspace);
        //const char* b_pannel = B;
        if (transB) {
            loadb_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb_int8(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }
            int cout0[16];
            int cout1[16];
            int cout2[16];
            int cout3[16];
            int *c_ptr0 = C + y * N + x0;
            int *c_ptr1 = c_ptr0 + N;
            int *c_ptr2 = c_ptr1 + N;
            int *c_ptr3 = c_ptr2 + N;
            int *pout0 = c_ptr0;
            int *pout1 = c_ptr1;
            int *pout2 = c_ptr2;
            int *pout3 = c_ptr3;
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    case 2:
                        c_ptr1 = zerobuf;
                    case 1:
                        c_ptr2 = zerobuf;
                    case 0:
                        c_ptr3 = zerobuf;
                    default:
                        break;
                }
            }
            const char *a_ptr_l = A_packed + y * K;
            const char *b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {
                if (flag_p_remain && (xb == bblocks - 1)) {
                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                }
                const char *a_ptr = a_ptr_l;
                int tail = tail_pre;
                int k = k_pre;
                asm volatile (
                GEMM_4x16_INT8_IN
                GEMM_4x16_INT8_KERNEL
                GEMM_4x16_INT8_OUT_RELU
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), [tail] "+r" (tail), \
                  [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
                        [c_ptr3] "+r"(c_ptr3)
                :
                : "v0", "v1","v2","v3","v4","v5","v6","v7","v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                    }
                }
            }
        }
    }
}
void sgemm_conv_4x16_bias_relu_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool transB, Context* ctx) {
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(char) * K);
    x_block /= 16;
    x_block *= 16;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 15) / 16;
    x_block *= 16;
    // unroll 2 loop
    int tail_pre = (K & 7);
    int k_pre = ((K + 7) / 8) - 1;
    int zerobuf[x_block];
    bool flag_p_remain = false;
    int remain = 0;
    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 15) / 16;
        remain = xmax - x0 - (bblocks - 1) * 16;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        char *b_pannel = static_cast<char *>(workspace);
        //const char* b_pannel = B;
        if (transB) {
            loadb_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb_int8(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }
            int cout0[16];
            int cout1[16];
            int cout2[16];
            int cout3[16];
            int bias_ptr[4] = {0, 0, 0, 0};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }
            int bias0 = bias_ptr[0];
            int bias1 = bias_ptr[1];
            int bias2 = bias_ptr[2];
            int bias3 = bias_ptr[3];
            int *c_ptr0 = C + y * N + x0;
            int *c_ptr1 = c_ptr0 + N;
            int *c_ptr2 = c_ptr1 + N;
            int *c_ptr3 = c_ptr2 + N;
            int *pout0 = c_ptr0;
            int *pout1 = c_ptr1;
            int *pout2 = c_ptr2;
            int *pout3 = c_ptr3;
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    case 2:
                        c_ptr1 = zerobuf;
                    case 1:
                        c_ptr2 = zerobuf;
                    case 0:
                        c_ptr3 = zerobuf;
                    default:
                        break;
                }
            }
            const char *a_ptr_l = A_packed + y * K;
            const char *b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {
                if (flag_p_remain && (xb == bblocks - 1)) {
                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                }
                const char *a_ptr = a_ptr_l;
                int tail = tail_pre;
                int k = k_pre;
                asm volatile (
                GEMM_4x16_INT8_IN_BIAS
                GEMM_4x16_INT8_KERNEL
                GEMM_4x16_INT8_OUT_RELU
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), [tail] "+r" (tail), \
                  [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
                        [c_ptr3] "+r"(c_ptr3)
                : [bias0] "r"(bias0), [bias1] "r"(bias1), [bias2] "r"(bias2), [bias3] "r"(bias3)
                : "v0", "v1","v2","v3","v4","v5","v6","v7","v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                    }
                }
            }
        }
    }
}
#else
#define GEMM_INT8_KERNEL \
        "cmp %[k], #0                           @ check weather k is bigger than 0\n"       \
        "beq 0f                                 @ jump to tail\n"       \
        "1:                                     @ main loop for k\n"        \
        /* Unroll 0*/                                           \
        "vld1.8   {d1}, [%[a_ptr]]!             @ load next a0~a7\n"        \
        "vdup.8     d4, d0[0]                   @ dup a00 to d4\n"      \
        "vdup.8     d5, d0[1]                   @ dup a01 to d5\n"      \
        "vdup.8     d6, d0[2]                   @ dup a02 to d6\n"      \
        "vdup.8     d7, d0[3]                   @ dup a03 to d7\n"      \
        "vld1.8   {d3}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vmull.s8   q4, d2, d4                  @ out0 += b0 * a00\n"       \
        "vmull.s8   q5, d2, d5                  @ out1 += b0 * a01\n"       \
        "vmull.s8   q6, d2, d6                  @ out2 += b0 * a02\n"       \
        "vmull.s8   q7, d2, d7                  @ out3 += b0 * a03\n"       \
        /* Unroll 1*/                                           \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d0[4]                   @ dup a10 to d4\n"      \
        "vdup.8     d5, d0[5]                   @ dup a11 to d5\n"      \
        "vdup.8     d6, d0[6]                   @ dup a12 to d6\n"      \
        "vdup.8     d7, d0[7]                   @ dup a13 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b1 * a10\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b1 * a11\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b1 * a12\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b1 * a13\n"       \
        /* Unroll 2*/                                           \
        "vld1.8   {d3}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d1[0]                   @ dup a20 to d4\n"      \
        "vdup.8     d5, d1[1]                   @ dup a21 to d5\n"      \
        "vdup.8     d6, d1[2]                   @ dup a22 to d6\n"      \
        "vdup.8     d7, d1[3]                   @ dup a23 to d7\n"      \
        "vld1.8   {d0}, [%[a_ptr]]!             @ load next a0~a7\n"        \
        "vmlal.s8   q4, d2, d4                  @ out0 += b2 * a20\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b2 * a21\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b2 * a22\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b2 * a23\n"       \
        /* Unroll 3*/                                           \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d1[4]                   @ dup a30 to d4\n"      \
        "vdup.8     d5, d1[5]                   @ dup a31 to d5\n"      \
        "vdup.8     d6, d1[6]                   @ dup a32 to d6\n"      \
        "vdup.8     d7, d1[7]                   @ dup a33 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b3 * a30\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b3 * a31\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b3 * a32\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b3 * a33\n"       \
        /* Unroll 4*/                                           \
        "vld1.8   {d3}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d0[0]                   @ dup a40 to d4\n"      \
        "vdup.8     d5, d0[1]                   @ dup a41 to d5\n"      \
        "vdup.8     d6, d0[2]                   @ dup a42 to d6\n"      \
        "vdup.8     d7, d0[3]                   @ dup a43 to d7\n"      \
        "vld1.8   {d1}, [%[a_ptr]]!             @ load next a0~a7\n"        \
        "vmlal.s8   q4, d2, d4                  @ out0 += b4 * a40\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b4 * a41\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b4 * a42\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b4 * a43\n"       \
        /* Unroll 5*/                                           \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d0[4]                   @ dup a50 to d4\n"      \
        "vdup.8     d5, d0[5]                   @ dup a51 to d5\n"      \
        "vdup.8     d6, d0[6]                   @ dup a52 to d6\n"      \
        "vdup.8     d7, d0[7]                   @ dup a53 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b5 * a50\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b5 * a51\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b5 * a52\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b5 * a53\n"       \
        /* Unroll 6*/                                           \
        "vld1.8   {d3}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d1[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[3]                   @ dup a63 to d7\n"      \
        "vld1.8   {d0}, [%[a_ptr]]!             @ load next a0~a7\n"        \
        "vmlal.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        /* Unroll 7*/                                           \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d1[4]                   @ dup a70 to d4\n"      \
        "vdup.8     d5, d1[5]                   @ dup a71 to d5\n"      \
        "vdup.8     d6, d1[6]                   @ dup a72 to d6\n"      \
        "vdup.8     d7, d1[7]                   @ dup a73 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b7 * a70\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b7 * a71\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b7 * a72\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b7 * a73\n"       \
        /* Accumulate to final result*/                                         \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "subs       %[k], %[k], #1              @ k--\n"        \
        "bne        1b                          @ jump to main loop\n"      \
        "0:                                     @ process tail\n"       \
        "subs       %[tails], %[tails], #1      @ tail--\n"     \
        "beq        3f                          @ jump to tail = 1\n"       \
        /* final loop, Unroll 0*/                                           \
        "vdup.8     d4, d0[0]                   @ dup a00 to d4\n"      \
        "vdup.8     d5, d0[1]                   @ dup a01 to d5\n"      \
        "vdup.8     d6, d0[2]                   @ dup a02 to d6\n"      \
        "vdup.8     d7, d0[3]                   @ dup a03 to d7\n"      \
        "vld1.8   {d3}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vmull.s8   q4, d2, d4                  @ out0 += b0 * a00\n"       \
        "vmull.s8   q5, d2, d5                  @ out1 += b0 * a01\n"       \
        "vmull.s8   q6, d2, d6                  @ out2 += b0 * a02\n"       \
        "vmull.s8   q7, d2, d7                  @ out3 += b0 * a03\n"       \
        "subs       %[tails], %[tails], #1      @ tail--\n"     \
        "beq        4f                          @ jump to tail==2\n"        \
        /* final loop, Unroll 1 */                                          \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d0[4]                   @ dup a10 to d4\n"      \
        "vdup.8     d5, d0[5]                   @ dup a11 to d5\n"      \
        "vdup.8     d6, d0[6]                   @ dup a12 to d6\n"      \
        "vdup.8     d7, d0[7]                   @ dup a13 to d7\n"      \
        "vld1.8   {d1}, [%[a_ptr]]!             @ load next a0~a7\n"        \
        "vmlal.s8   q4, d3, d4                  @ out0 += b1 * a10\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b1 * a11\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b1 * a12\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b1 * a13\n"       \
        "subs       %[tails], %[tails], #1      @ tail--\n"     \
        "beq        5f                          @ jump to tail==3\n"        \
        /* final loop, Unroll 2 */                                          \
        "vld1.8   {d3}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d1[0]                   @ dup a20 to d4\n"      \
        "vdup.8     d5, d1[1]                   @ dup a21 to d5\n"      \
        "vdup.8     d6, d1[2]                   @ dup a22 to d6\n"      \
        "vdup.8     d7, d1[3]                   @ dup a23 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b2 * a20\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b2 * a21\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b2 * a22\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b2 * a23\n"       \
        "subs       %[tails], %[tails], #1      @ tail--\n"     \
        "beq        6f                          @ jump to tail==4\n"        \
        /* final loop, Unroll 3 */                                          \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d1[4]                   @ dup a30 to d4\n"      \
        "vdup.8     d5, d1[5]                   @ dup a31 to d5\n"      \
        "vdup.8     d6, d1[6]                   @ dup a32 to d6\n"      \
        "vdup.8     d7, d1[7]                   @ dup a33 to d7\n"      \
        "vld1.8   {d0}, [%[a_ptr]]!             @ load next a0~a7\n"        \
        "vmlal.s8   q4, d3, d4                  @ out0 += b3 * a30\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b3 * a31\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b3 * a32\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b3 * a33\n"       \
        "subs       %[tails], %[tails], #1      @ tail--\n"     \
        "beq        7f                          @ jump to tail==5\n"        \
        /* final loop, Unroll 4 */                                          \
        "vld1.8   {d3}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d0[0]                   @ dup a40 to d4\n"      \
        "vdup.8     d5, d0[1]                   @ dup a41 to d5\n"      \
        "vdup.8     d6, d0[2]                   @ dup a42 to d6\n"      \
        "vdup.8     d7, d0[3]                   @ dup a43 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b4 * a40\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b4 * a41\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b4 * a42\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b4 * a43\n"       \
        "subs       %[tails], %[tails], #1      @ tail--\n"     \
        "beq        8f                          @ jump to tail==6\n"        \
        /* final loop, Unroll 5 */                                          \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d0[4]                   @ dup a50 to d4\n"      \
        "vdup.8     d5, d0[5]                   @ dup a51 to d5\n"      \
        "vdup.8     d6, d0[6]                   @ dup a52 to d6\n"      \
        "vdup.8     d7, d0[7]                   @ dup a53 to d7\n"      \
        "vld1.8   {d1}, [%[a_ptr]]!             @ load next a0~a7\n"        \
        "vmlal.s8   q4, d3, d4                  @ out0 += b5 * a50\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b5 * a51\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b5 * a52\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b5 * a53\n"       \
        "subs       %[tails], %[tails], #1      @ tail--\n"     \
        "beq        9f                          @ jump to tail==7\n"        \
        /* final loop, Unroll 6 */                                          \
        "vld1.8   {d3}, [%[b_ptr]]!             @ load next b0~b7\n"        \
        "vdup.8     d4, d1[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[3]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b6 * a63\n"
#define GEMM_INT8_IN    \
        "vld1.8 {d0}, [%[a_ptr]]!               @ load a0~a7\n"             \
        "vmov.i32   q8, #0                      @ out0=0\n"             \
        "pld [%[b_ptr]]                         @ preload b\n"              \
        "vmov.i32   q9, #0                      @ out1=0\n"             \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load b0~b7\n"             \
        "vmov.i32   q10, #0                     @ out2=0\n"             \
        "pld [%[a_ptr]]                         @ preload a, 64byte\n"              \
        "vmov.i32   q11, #0                     @ out3=0\n"             \
        "pld [%[b_ptr], #64]                    @ preload b\n"              \
        "vmov.i32   q12, #0                     @ out4=0\n"             \
        "pld [%[a_ptr], #64]                    @ preload a\n"              \
        "vmov.i32   q13, #0                     @ out5=0\n"             \
        "pld [%[b_ptr], #128]                   @ preload b\n"              \
        "vmov.i32   q14, #0                     @ out6=0\n"             \
        "pld [%[a_ptr], #128]                   @ preload a\n"              \
        "vmov.i32   q15, #0                     @ out7=0\n"             \
        "pld [%[b_ptr], #192]                   @ preload b\n"
#define GEMM_INT8_IN_BIAS    \
        "vld1.8 {d0}, [%[a_ptr]]!               @ load a0~a7\n"             \
        "vdup.32    q8, %[bias0]                @ out0=bias0\n"             \
        "pld [%[b_ptr]]                         @ preload b\n"              \
        "vdup.32    q9, %[bias0]                @ out1=bias0\n"             \
        "vld1.8   {d2}, [%[b_ptr]]!             @ load b0~b7\n"             \
        "vdup.32    q10, %[bias1]               @ out2=bias1\n"             \
        "pld [%[a_ptr]]                         @ preload a, 64byte\n"              \
        "vdup.32    q11, %[bias1]               @ out3=bias1\n"             \
        "pld [%[b_ptr], #64]                    @ preload b\n"              \
        "vdup.32    q12, %[bias2]               @ out4=bias2\n"             \
        "pld [%[a_ptr], #64]                    @ preload a\n"              \
        "vdup.32    q13, %[bias2]               @ out5=bias2\n"             \
        "pld [%[b_ptr], #128]                   @ preload b\n"              \
        "vdup.32    q14, %[bias3]               @ out6=bias3\n"             \
        "pld [%[a_ptr], #128]                   @ preload a\n"              \
        "vdup.32    q15, %[bias3]               @ out7=bias3\n"             \
        "pld [%[b_ptr], #192]                   @ preload b\n"
#define GEMM_INT8_OUT   \
        /* final loop, Unroll 7 */                                                          \
        "vdup.8     d4, d1[4]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[5]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[6]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[7]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b6 * a63\n"       \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f\n"       \
        /* tails==1 final */                                                            \
        "3:                                     @ tail=1\n"     \
        "vdup.8     d4, d0[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d0[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d0[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d0[3]                   @ dup a63 to d7\n"      \
        "vmull.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmull.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmull.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmull.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f\n"       \
        /* tails==2 final*/                                                         \
        "4:                                     @ tail == 2\n"      \
        "vdup.8     d4, d0[4]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d0[5]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d0[6]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d0[7]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b6 * a63\n"       \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==3 final*/                                                         \
        "5:                                     @ tail=3\n"     \
        "vdup.8     d4, d1[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[3]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==4 final*/                                                         \
        "6:                                     @ tail=4\n"     \
        "vdup.8     d4, d1[4]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[5]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[6]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[7]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b6 * a63\n"       \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==5 final*/                                                         \
        "7:                                     @ tail=5\n"     \
        "vdup.8     d4, d0[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d0[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d0[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d0[3]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==6 final*/                                                         \
        "8:                                     @ tail=6\n"     \
        "vdup.8     d4, d0[4]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d0[5]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d0[6]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d0[7]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b6 * a63\n"       \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==7 final*/                                                         \
        "9:                                     @ tail=7\n"     \
        "vdup.8     d4, d1[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[3]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "2:                                     @ end\n"
#define GEMM_INT8_OUT_RELU   \
        /* final loop, Unroll 7 */                                                          \
        "vdup.8     d4, d1[4]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[5]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[6]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[7]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b6 * a63\n"       \
        "vmov.i32   q0, #0                      @ for relu\n"               \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vmax.s32   q8, q0                      @ relu\n" \
        "vmax.s32   q9, q0                      @ relu\n" \
        "vmax.s32   q10, q0                     @ relu\n" \
        "vmax.s32   q11, q0                     @ relu\n" \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vmax.s32   q12, q0                     @ relu\n" \
        "vmax.s32   q13, q0                     @ relu\n" \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vmax.s32   q14, q0                     @ relu\n" \
        "vmax.s32   q15, q0                     @ relu\n" \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f\n"       \
        /* tails==1 final */                                                            \
        "3:                                     @ tail=1\n"     \
        "vdup.8     d4, d0[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d0[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d0[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d0[3]                   @ dup a63 to d7\n"      \
        "vmull.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmull.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmull.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmull.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        "vmov.s32   q0, #0                      @ for relu\n"               \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vmax.s32   q8, q0                      @ relu\n" \
        "vmax.s32   q9, q0                      @ relu\n" \
        "vmax.s32   q10, q0                     @ relu\n" \
        "vmax.s32   q11, q0                     @ relu\n" \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vmax.s32   q12, q0                     @ relu\n" \
        "vmax.s32   q13, q0                     @ relu\n" \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vmax.s32   q14, q0                     @ relu\n" \
        "vmax.s32   q15, q0                     @ relu\n" \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f\n"       \
        /* tails==2 final*/                                                         \
        "4:                                     @ tail == 2\n"      \
        "vdup.8     d4, d0[4]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d0[5]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d0[6]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d0[7]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b6 * a63\n"       \
        "vmov.s32   q0, #0                      @ for relu\n"               \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vmax.s32   q8, q0                      @ relu\n" \
        "vmax.s32   q9, q0                      @ relu\n" \
        "vmax.s32   q10, q0                     @ relu\n" \
        "vmax.s32   q11, q0                     @ relu\n" \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vmax.s32   q12, q0                     @ relu\n" \
        "vmax.s32   q13, q0                     @ relu\n" \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vmax.s32   q14, q0                     @ relu\n" \
        "vmax.s32   q15, q0                     @ relu\n" \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==3 final*/                                                         \
        "5:                                     @ tail=3\n"     \
        "vdup.8     d4, d1[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[3]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        "vmov.s32   q0, #0                      @ for relu\n"               \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vmax.s32   q8, q0                      @ relu\n" \
        "vmax.s32   q9, q0                      @ relu\n" \
        "vmax.s32   q10, q0                     @ relu\n" \
        "vmax.s32   q11, q0                     @ relu\n" \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vmax.s32   q12, q0                     @ relu\n" \
        "vmax.s32   q13, q0                     @ relu\n" \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vmax.s32   q14, q0                     @ relu\n" \
        "vmax.s32   q15, q0                     @ relu\n" \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==4 final*/                                                         \
        "6:                                     @ tail=4\n"     \
        "vdup.8     d4, d1[4]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[5]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[6]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[7]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b6 * a63\n"       \
        "vmov.s32   q0, #0                      @ for relu\n"               \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vmax.s32   q8, q0                      @ relu\n" \
        "vmax.s32   q9, q0                      @ relu\n" \
        "vmax.s32   q10, q0                     @ relu\n" \
        "vmax.s32   q11, q0                     @ relu\n" \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vmax.s32   q12, q0                     @ relu\n" \
        "vmax.s32   q13, q0                     @ relu\n" \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vmax.s32   q14, q0                     @ relu\n" \
        "vmax.s32   q15, q0                     @ relu\n" \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==5 final*/                                                         \
        "7:                                     @ tail=5\n"     \
        "vdup.8     d4, d0[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d0[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d0[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d0[3]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        "vmov.s32   q0, #0                      @ for relu\n"               \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vmax.s32   q8, q0                      @ relu\n" \
        "vmax.s32   q9, q0                      @ relu\n" \
        "vmax.s32   q10, q0                     @ relu\n" \
        "vmax.s32   q11, q0                     @ relu\n" \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vmax.s32   q12, q0                     @ relu\n" \
        "vmax.s32   q13, q0                     @ relu\n" \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vmax.s32   q14, q0                     @ relu\n" \
        "vmax.s32   q15, q0                     @ relu\n" \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==6 final*/                                                         \
        "8:                                     @ tail=6\n"     \
        "vdup.8     d4, d0[4]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d0[5]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d0[6]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d0[7]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d3, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d3, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d3, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d3, d7                  @ out3 += b6 * a63\n"       \
        "vmov.s32   q0, #0                      @ for relu\n"               \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vmax.s32   q8, q0                      @ relu\n" \
        "vmax.s32   q9, q0                      @ relu\n" \
        "vmax.s32   q10, q0                     @ relu\n" \
        "vmax.s32   q11, q0                     @ relu\n" \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vmax.s32   q12, q0                     @ relu\n" \
        "vmax.s32   q13, q0                     @ relu\n" \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vmax.s32   q14, q0                     @ relu\n" \
        "vmax.s32   q15, q0                     @ relu\n" \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "b      2f                              @ jump to end\n"        \
        /* tails==7 final*/                                                         \
        "9:                                     @ tail=7\n"     \
        "vdup.8     d4, d1[0]                   @ dup a60 to d4\n"      \
        "vdup.8     d5, d1[1]                   @ dup a61 to d5\n"      \
        "vdup.8     d6, d1[2]                   @ dup a62 to d6\n"      \
        "vdup.8     d7, d1[3]                   @ dup a63 to d7\n"      \
        "vmlal.s8   q4, d2, d4                  @ out0 += b6 * a60\n"       \
        "vmlal.s8   q5, d2, d5                  @ out1 += b6 * a61\n"       \
        "vmlal.s8   q6, d2, d6                  @ out2 += b6 * a62\n"       \
        "vmlal.s8   q7, d2, d7                  @ out3 += b6 * a63\n"       \
        "vmov.s32   q0, #0                      @ for relu\n"               \
        "vaddw.s16  q8, q8, d8                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q9, q9, d9                  @ accumulate to int32 result\n"     \
        "vaddw.s16  q10, q10, d10               @ accumulate to int32 result\n"     \
        "vaddw.s16  q11, q11, d11               @ accumulate to int32 result\n"     \
        "vaddw.s16  q12, q12, d12               @ accumulate to int32 result\n"     \
        "vaddw.s16  q13, q13, d13               @ accumulate to int32 result\n"     \
        "vaddw.s16  q14, q14, d14               @ accumulate to int32 result\n"     \
        "vaddw.s16  q15, q15, d15               @ accumulate to int32 result\n"     \
        "vmax.s32   q8, q0                      @ relu\n" \
        "vmax.s32   q9, q0                      @ relu\n" \
        "vmax.s32   q10, q0                     @ relu\n" \
        "vmax.s32   q11, q0                     @ relu\n" \
        "vst1.32   {d16-d19}, [%[c_ptr0]]!      @ write 0\n"        \
        "vmax.s32   q12, q0                     @ relu\n" \
        "vmax.s32   q13, q0                     @ relu\n" \
        "vst1.32   {d20-d23}, [%[c_ptr1]]!      @ write 1\n"        \
        "vmax.s32   q14, q0                     @ relu\n" \
        "vmax.s32   q15, q0                     @ relu\n" \
        "vst1.32   {d24-d27}, [%[c_ptr2]]!      @ write 2\n"        \
        "vst1.32   {d28-d31}, [%[c_ptr3]]!      @ write 3\n"        \
        "2:                                     @ end\n"
void sgemm_conv_4x8_int8(const char* A_packed, const char* B, int* C, \
    int M, int N, int K, bool transB, Context* ctx) {
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(char) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;
    int k_pre = ((K + 7) / 8) - 1;
    int tail_pre = (K & 7);
    if (tail_pre == 0) {
        tail_pre = 8;
    }
    int zerobuf[x_block];
    bool flag_p_remain = false;
    int remain = 0;
    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 7) / 8;
        remain = xmax - x0 - (bblocks - 1) * 8;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        char* b_pannel = static_cast<char*>(workspace);
        if (transB) {
            loadb_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb_int8(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }
            int cout0[8];
            int cout1[8];
            int cout2[8];
            int cout3[8];
            int* c_ptr0 = C + y * N + x0;
            int* c_ptr1 = c_ptr0 + N;
            int* c_ptr2 = c_ptr1 + N;
            int* c_ptr3 = c_ptr2 + N;
            int* pout0 = c_ptr0;
            int* pout1 = c_ptr1;
            int* pout2 = c_ptr2;
            int* pout3 = c_ptr3;
            if ((y+3) >= ymax) {
                switch ((y + 3) - ymax) {
                    case 2:
                        c_ptr1 = zerobuf;
                    case 1:
                        c_ptr2 = zerobuf;
                    case 0:
                        c_ptr3 = zerobuf;
                    default:
                        break;
                }
            }
            const char* a_ptr_l = A_packed + y * K;
            const char* b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {
                if (flag_p_remain && (xb == bblocks - 1)) {
                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                }
                const char* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                /* sgemm kernel*/
                GEMM_INT8_IN
                GEMM_INT8_KERNEL
                GEMM_INT8_OUT
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [k] "+r" (k), [tails] "+r" (tails)
                :
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                    }
                }
            }
        }
    }
}
void sgemm_conv_4x8_bias_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool transB, Context* ctx) {
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(char) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;
    int k_pre = ((K + 7) / 8) - 1;
    int tail_pre = (K & 7);
    if (tail_pre == 0) {
        tail_pre = 8;
    }
    int zerobuf[x_block];
    bool flag_p_remain = false;
    int remain = 0;
    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 7) / 8;
        remain = xmax - x0 - (bblocks - 1) * 8;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        char* b_pannel = static_cast<char*>(workspace);
        if (transB) {
            loadb_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb_int8(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }
            int cout0[8];
            int cout1[8];
            int cout2[8];
            int cout3[8];
            int bias_ptr[4] = {0, 0, 0, 0};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }
            int bias0 = bias_ptr[0];
            int bias1 = bias_ptr[1];
            int bias2 = bias_ptr[2];
            int bias3 = bias_ptr[3];
            int* c_ptr0 = C + y * N + x0;
            int* c_ptr1 = c_ptr0 + N;
            int* c_ptr2 = c_ptr1 + N;
            int* c_ptr3 = c_ptr2 + N;
            int* pout0 = c_ptr0;
            int* pout1 = c_ptr1;
            int* pout2 = c_ptr2;
            int* pout3 = c_ptr3;
            if ((y+3) >= ymax) {
                switch ((y + 3) - ymax) {
                    case 2:
                        c_ptr1 = zerobuf;
                    case 1:
                        c_ptr2 = zerobuf;
                    case 0:
                        c_ptr3 = zerobuf;
                    default:
                        break;
                }
            }
            const char* a_ptr_l = A_packed + y * K;
            const char* b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {
                if (flag_p_remain && (xb == bblocks - 1)) {
                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                }
                const char* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                /* sgemm kernel*/
                GEMM_INT8_IN_BIAS
                GEMM_INT8_KERNEL
                GEMM_INT8_OUT
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [k] "+r" (k), [tails] "+r" (tails)
                : [bias0] "r"(bias0), [bias1] "r"(bias1), [bias2] "r"(bias2), [bias3] "r"(bias3)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                    }
                }
            }
        }
    }
}
void sgemm_conv_4x8_relu_int8(const char* A_packed, const char* B, int* C, \
    int M, int N, int K, bool transB, Context* ctx) {
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(char) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;
    int k_pre = ((K + 7) / 8) - 1;
    int tail_pre = (K & 7);
    if (tail_pre == 0) {
        tail_pre = 8;
    }
    int zerobuf[x_block];
    bool flag_p_remain = false;
    int remain = 0;
    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 7) / 8;
        remain = xmax - x0 - (bblocks - 1) * 8;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        char* b_pannel = static_cast<char*>(workspace);
        if (transB) {
            loadb_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb_int8(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }
            int cout0[8];
            int cout1[8];
            int cout2[8];
            int cout3[8];
            int* c_ptr0 = C + y * N + x0;
            int* c_ptr1 = c_ptr0 + N;
            int* c_ptr2 = c_ptr1 + N;
            int* c_ptr3 = c_ptr2 + N;
            int* pout0 = c_ptr0;
            int* pout1 = c_ptr1;
            int* pout2 = c_ptr2;
            int* pout3 = c_ptr3;
            if ((y+3) >= ymax) {
                switch ((y + 3) - ymax) {
                    case 2:
                        c_ptr1 = zerobuf;
                    case 1:
                        c_ptr2 = zerobuf;
                    case 0:
                        c_ptr3 = zerobuf;
                    default:
                        break;
                }
            }
            const char* a_ptr_l = A_packed + y * K;
            const char* b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {
                if (flag_p_remain && (xb == bblocks - 1)) {
                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                }
                const char* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                /* sgemm kernel*/
                GEMM_INT8_IN
                GEMM_INT8_KERNEL
                GEMM_INT8_OUT_RELU
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [k] "+r" (k), [tails] "+r" (tails)
                :
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                    }
                }
            }
        }
    }
}
void sgemm_conv_4x8_bias_relu_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool transB, Context* ctx) {
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(char) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;
    int k_pre = ((K + 7) / 8) - 1;
    int tail_pre = (K & 7);
    if (tail_pre == 0) {
        tail_pre = 8;
    }
    int zerobuf[x_block];
    bool flag_p_remain = false;
    int remain = 0;
    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 7) / 8;
        remain = xmax - x0 - (bblocks - 1) * 8;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        char* b_pannel = static_cast<char*>(workspace);
        if (transB) {
            loadb_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb_int8(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }
            int cout0[8];
            int cout1[8];
            int cout2[8];
            int cout3[8];
            int bias_ptr[4] = {0, 0, 0, 0};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }
            int bias0 = bias_ptr[0];
            int bias1 = bias_ptr[1];
            int bias2 = bias_ptr[2];
            int bias3 = bias_ptr[3];
            int* c_ptr0 = C + y * N + x0;
            int* c_ptr1 = c_ptr0 + N;
            int* c_ptr2 = c_ptr1 + N;
            int* c_ptr3 = c_ptr2 + N;
            int* pout0 = c_ptr0;
            int* pout1 = c_ptr1;
            int* pout2 = c_ptr2;
            int* pout3 = c_ptr3;
            if ((y+3) >= ymax) {
                switch ((y + 3) - ymax) {
                    case 2:
                        c_ptr1 = zerobuf;
                    case 1:
                        c_ptr2 = zerobuf;
                    case 0:
                        c_ptr3 = zerobuf;
                    default:
                        break;
                }
            }
            const char* a_ptr_l = A_packed + y * K;
            const char* b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {
                if (flag_p_remain && (xb == bblocks - 1)) {
                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                }
                const char* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                /* sgemm kernel*/
                GEMM_INT8_IN_BIAS
                GEMM_INT8_KERNEL
                GEMM_INT8_OUT_RELU
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [k] "+r" (k), [tails] "+r" (tails)
                : [bias0] "r"(bias0), [bias1] "r"(bias1), [bias2] "r"(bias2), [bias3] "r"(bias3)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                    }
                }
            }
        }
    }
}
#endif
} //lite
} //saber
} //anakin
