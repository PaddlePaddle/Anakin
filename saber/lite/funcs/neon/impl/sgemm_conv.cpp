#include "saber/lite/funcs/neon/impl/sgemm_conv.h"

namespace anakin{

namespace saber{

namespace lite{

#ifdef __aarch64__

void prepackA_8x12(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);

void prepackA_trans_8x12(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);

void sgemm_conv_8x12(const float* A_packed, const float* B, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_8x12_bias(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_8x12_relu(const float* A_packed, const float* B, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_8x12_bias_relu(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

#else //__aarch64__
//for A72
void prepackA_6x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);

void prepackA_trans_6x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);
//for A73
void prepackA_4x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);
void prepackA_trans_4x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);

//for A72, 6x8
void sgemm_conv_6x8(const float* A_packed, const float* B, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_6x8_bias(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_6x8_relu(const float* A_packed, const float* B, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_6x8_bias_relu(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

//for A73, 4x8
void sgemm_conv_4x8(const float* A_packed, const float* B, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_4x8_bias(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_4x8_relu(const float* A_packed, const float* B, float* C, \
    int M, int N, int K, bool transB, Context* ctx);

void sgemm_conv_4x8_bias_relu(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx);
#endif //__aarch64__

/**
 * \brief input data is not transpose
 * for arm-v7a, transform data to block x k x 6 layout
 * for arm-v8a, transform data to block x k x 8 layout
 */
void prepackA(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax, bool is_trans, Context* ctx) {
#ifdef __aarch64__

    if (is_trans) {
        prepackA_trans_8x12(out, in, ldin, m0, mmax, k0, kmax);
    } else {
        prepackA_8x12(out, in, ldin, m0, mmax, k0, kmax);
    }
#else
    if (ctx->get_arch() == A73) {
        if (is_trans) {
            prepackA_trans_4x8(out, in, ldin, m0, mmax, k0, kmax);
        } else {
            prepackA_4x8(out, in, ldin, m0, mmax, k0, kmax);
        }
    } else {
        if (is_trans) {
            prepackA_trans_6x8(out, in, ldin, m0, mmax, k0, kmax);
        } else {
            prepackA_6x8(out, in, ldin, m0, mmax, k0, kmax);
        }
    }
#endif
}

void sgemm_prepack(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, \
    bool is_bias, bool is_relu, bool is_transB, Context* ctx) {
    if (is_bias) {
        if (is_relu) {
#ifdef __aarch64__
            sgemm_conv_8x12_bias_relu(A_packed, B, bias, C, M, N, K, is_transB, ctx);
#else
            if (ctx->get_arch() == A73) {
                sgemm_conv_4x8_bias_relu(A_packed, B, bias, C, M, N, K, is_transB, ctx);
            } else {
                sgemm_conv_6x8_bias_relu(A_packed, B, bias, C, M, N, K, is_transB, ctx);
            }
#endif //__aarch64__
        } else {
#ifdef __aarch64__
            sgemm_conv_8x12_bias(A_packed, B, bias, C, M, N, K, is_transB, ctx);
#else
            if (ctx->get_arch() == A73) {
                sgemm_conv_4x8_bias(A_packed, B, bias, C, M, N, K, is_transB, ctx);
            } else {
                sgemm_conv_6x8_bias(A_packed, B, bias, C, M, N, K, is_transB, ctx);
            }
#endif //__aarch64__
        }
    } else {
        if (is_relu) {
#ifdef __aarch64__
            sgemm_conv_8x12_relu(A_packed, B, C, M, N, K, is_transB, ctx);
#else
            if (ctx->get_arch() == A73) {
                sgemm_conv_4x8_relu(A_packed, B, C, M, N, K, is_transB, ctx);
            } else {
                sgemm_conv_6x8_relu(A_packed, B, C, M, N, K, is_transB, ctx);
            }
#endif //__aarch64__
        } else {
#ifdef __aarch64__
            sgemm_conv_8x12(A_packed, B, C, M, N, K, is_transB, ctx);
#else
            if (ctx->get_arch() == A73) {
                sgemm_conv_4x8(A_packed, B, C, M, N, K, is_transB, ctx);
            } else {
                sgemm_conv_6x8(A_packed, B, C, M, N, K, is_transB, ctx);
            }
#endif //__aarch64__
        }
    }
}

#ifdef __aarch64__
void prepackA_8x12(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {

    int x_len = kmax - k0;
    uint32_t zerobuff[x_len];
    memset(zerobuff, 0, sizeof(uint32_t) * x_len);

    uint32_t *dout = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);

    int stride = x_len * 8;
#pragma omp parallel for
    for (int y = m0; y < mmax; y += 8) {
        uint32_t* outptr = dout + stride * (y - m0) / 8;

        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;
        const uint32_t *inptr6 = inptr5 + ldin;
        const uint32_t *inptr7 = inptr6 + ldin;

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

        int x = x_len;
        //! cope with row index exceed real size, set to zero buffer
        if ((y + 7) >= mmax) {
            switch ((y + 7) - mmax) {
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
            asm volatile(
            // Load up 8 elements (2 vectors) from each of 8 sources.
            "LDP        q0, q1, [%[inptr0]], #32\n" // q0=A0A1A2A3
                    "LDP        q2, q3, [%[inptr1]], #32\n" // q2=B0B1B2B3
                    "LDP        q4, q5, [%[inptr2]], #32\n" // q4=C0C1C2C3
                    "ZIP1       v16.4s, v0.4s, v4.4s\n"     // q16=A0C0A1C1
                    "prfm   pldl1keep, [%[inptr0], #128] \n"
                    "LDP        q6, q7, [%[inptr3]], #32\n" // q6=D0D1D2D3
                    "ZIP1       v17.4s, v2.4s, v6.4s\n"     // q17=B0D0B1D1
                    "LDP        q8, q9, [%[inptr4]], #32\n"
                    "LDP        q10, q11, [%[inptr5]], #32\n"
                    "LDP        q12, q13, [%[inptr6]], #32\n"
                    "ZIP1       v18.4s, v8.4s, v12.4s\n"
                    "prfm   pldl1keep, [%[inptr1], #128]\n"
                    "LDP        q14, q15, [%[inptr7]], #32\n"
                    "ZIP1       v19.4s, v10.4s, v14.4s\n"

                    "ZIP1       v20.4s, v16.4s, v17.4s\n" // q20=A0B0C0D0
                    "prfm   pldl1keep, [%[inptr2], #128]\n"
                    "ZIP1       v21.4s, v18.4s, v19.4s\n"
                    "ZIP2       v22.4s, v16.4s, v17.4s\n"
                    "ZIP2       v23.4s, v18.4s, v19.4s\n"

                    "ZIP2       v16.4s, v0.4s, v4.4s\n"
                    "prfm   pldl1keep, [%[inptr3], #128]\n"
                    "ZIP2       v17.4s, v2.4s, v6.4s\n"
                    "STP        q20, q21, [%[outptr]], #32\n" // Write back the first element of each source

                    "ZIP2       v18.4s, v8.4s, v12.4s\n"
                    "ZIP2       v19.4s, v10.4s, v14.4s\n"
                    "STP        q22, q23, [%[outptr]], #32\n" // Write back the second element of each source

                    "ZIP1       v20.4s, v16.4s, v17.4s\n"
                    "prfm   pldl1keep, [%[inptr4], #128]\n"
                    "ZIP1       v21.4s, v18.4s, v19.4s\n"
                    "ZIP2       v22.4s, v16.4s, v17.4s\n"
                    "ZIP2       v23.4s, v18.4s, v19.4s\n"

                    "ZIP1       v16.4s, v1.4s, v5.4s\n"
                    "prfm   pldl1keep, [%[inptr5], #128]\n"
                    "ZIP1       v17.4s, v3.4s, v7.4s\n"
                    "STP        q20, q21, [%[outptr]], #32\n" // Third element

                    "ZIP1       v18.4s, v9.4s, v13.4s\n"
                    "ZIP1       v19.4s, v11.4s, v15.4s\n"
                    "STP        q22, q23, [%[outptr]], #32\n" // Fourth element

                    "ZIP1       v20.4s, v16.4s, v17.4s\n"
                    "ZIP1       v21.4s, v18.4s, v19.4s\n"
                    "ZIP2       v22.4s, v16.4s, v17.4s\n"
                    "prfm   pldl1keep, [%[inptr6], #128]\n"
                    "ZIP2       v23.4s, v18.4s, v19.4s\n"

                    "ZIP2       v16.4s, v1.4s, v5.4s\n"
                    "ZIP2       v17.4s, v3.4s, v7.4s\n"
                    "STP        q20, q21, [%[outptr]], #32\n" // Fifth element

                    "ZIP2       v18.4s, v9.4s, v13.4s\n"
                    "prfm   pldl1keep, [%[inptr7], #128]\n"
                    "ZIP2       v19.4s, v11.4s, v15.4s\n"
                    "STP        q22, q23, [%[outptr]], #32\n" // Sixth element

                    "ZIP1       v20.4s, v16.4s, v17.4s\n"
                    "ZIP1       v21.4s, v18.4s, v19.4s\n"
                    "STP        q20, q21, [%[outptr]], #32\n" // Seventh element

                    "ZIP2       v22.4s, v16.4s, v17.4s\n"
                    "ZIP2       v23.4s, v18.4s, v19.4s\n"
                    "STP        q22, q23, [%[outptr]], #32\n" // Eighth element
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
            [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5), [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                    "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        }

        for(; x > 0; x--) {
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

void prepackA_trans_8x12(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax){

    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + m0;

    uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int x_len = mmax - m0;
    int y_len = kmax - k0;
    int right_remain = x_len - 8 * (x_len / 8);
    int right_pad = 8 - right_remain;
    if (right_remain == 0) {
        right_pad = 0;
    }

    uint32_t *outptr_row = outptr;
    int stride_out = 8 * y_len;

    uint32x4_t vzero = vdupq_n_u32(0);
    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {

        const uint32_t* ptr0 = inptr + y * ldin;
        const uint32_t* ptr1 = ptr0 + ldin;
        const uint32_t* ptr2 = ptr1 + ldin;
        const uint32_t* ptr3 = ptr2 + ldin;

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

        uint32_t *outptr_row_col = outptr_row + y * 8;
        int i = 0;
        for (; i < x_len - 7; i += 8) {
            uint32x4_t vr00 = vld1q_u32(ptr0);
            uint32x4_t vr01 = vld1q_u32(ptr0 + 4);

            uint32x4_t vr10 = vld1q_u32(ptr1);
            uint32x4_t vr11 = vld1q_u32(ptr1 + 4);

            vst1q_u32(outptr_row_col, vr00);
            vst1q_u32(outptr_row_col + 4, vr01);

            uint32x4_t vr20 = vld1q_u32(ptr2);
            uint32x4_t vr21 = vld1q_u32(ptr2 + 4);

            vst1q_u32(outptr_row_col + 8, vr10);
            vst1q_u32(outptr_row_col + 12, vr11);

            uint32x4_t vr30 = vld1q_u32(ptr3);
            uint32x4_t vr31 = vld1q_u32(ptr3 + 4);

            vst1q_u32(outptr_row_col + 16, vr20);
            vst1q_u32(outptr_row_col + 20, vr21);

            vst1q_u32(outptr_row_col + 24, vr30);
            vst1q_u32(outptr_row_col + 28, vr31);

            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;

            outptr_row_col += stride_out;

        }
        if (right_remain > 0) {
            uint32x4_t vr00 = vld1q_u32(ptr0);
            uint32x4_t vr01 = vld1q_u32(ptr0 + 4);

            uint32x4_t vr10 = vld1q_u32(ptr1);
            uint32x4_t vr11 = vld1q_u32(ptr1 + 4);

            uint32x4_t vr00_1 = vbslq_u32(vmask1, vr00, vzero);
            uint32x4_t vr01_1 = vbslq_u32(vmask2, vr01, vzero);

            uint32x4_t vr20 = vld1q_u32(ptr2);
            uint32x4_t vr21 = vld1q_u32(ptr2 + 4);

            vst1q_u32(outptr_row_col, vr00_1);
            vst1q_u32(outptr_row_col + 4, vr01_1);

            uint32x4_t vr10_1 = vbslq_u32(vmask1, vr10, vzero);
            uint32x4_t vr11_1 = vbslq_u32(vmask2, vr11, vzero);

            uint32x4_t vr30 = vld1q_u32(ptr3);
            uint32x4_t vr31 = vld1q_u32(ptr3 + 4);

            vst1q_u32(outptr_row_col + 8, vr10_1);
            vst1q_u32(outptr_row_col + 12, vr11_1);

            uint32x4_t vr20_1 = vbslq_u32(vmask1, vr20, vzero);
            uint32x4_t vr21_1 = vbslq_u32(vmask2, vr21, vzero);


            uint32x4_t vr30_1 = vbslq_u32(vmask1, vr30, vzero);
            uint32x4_t vr31_1 = vbslq_u32(vmask2, vr31, vzero);

            vst1q_u32(outptr_row_col + 16, vr20_1);
            vst1q_u32(outptr_row_col + 20, vr21_1);
            vst1q_u32(outptr_row_col + 24, vr30_1);
            vst1q_u32(outptr_row_col + 28, vr31_1);
        }
    }

#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {

        const uint32_t* ptr0 = inptr + y * ldin;
        uint32_t *outptr_row_col = outptr_row + y * 8;
        int i = 0;
        for (; i < x_len - 7; i += 8) {

            uint32x4_t vr0 = vld1q_u32(ptr0);
            uint32x4_t vr1 = vld1q_u32(ptr0 + 4);
            vst1q_u32(outptr_row_col, vr0);
            vst1q_u32(outptr_row_col + 4, vr1);

            ptr0 += 8;

            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {
            uint32x4_t vr0 = vld1q_u32(ptr0);
            uint32x4_t vr1 = vld1q_u32(ptr0 + 4);

            uint32x4_t vr0_1 = vbslq_u32(vmask1, vr0, vzero);
            uint32x4_t vr1_1 = vbslq_u32(vmask2, vr1, vzero);

            vst1q_u32(outptr_row_col, vr0_1);
            vst1q_u32(outptr_row_col + 4, vr1_1);
        }
    }
}

#else //__aarch64__
void prepackA_6x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {

    int x_len = kmax - k0;
    uint32_t zerobuff[x_len];
    memset(zerobuff, 0, sizeof(uint32_t) * x_len);

    uint32_t *dout = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);

    uint32_t* outptr = dout;

    //! data A is not transposed, transpose A to k * 6
    for (int y = m0; y < mmax; y += 6) {
        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;

        int x = x_len;
        //! cope with row index exceed real size, set to zero buffer
        if ((y + 5) >= mmax) {
            switch ((y + 5) - mmax) {
                case 4:
                    inptr1 = zerobuff;
                case 3:
                    inptr2 = zerobuff;
                case 2:
                    inptr3 = zerobuff;
                case 1:
                    inptr4 = zerobuff;
                case 0:
                    inptr5 = zerobuff;
                default:
                    break;
            }
        }

        for (; x > 7; x -= 8) {

            //! zip load 8 elements (2 neon Q registers) from each of 6 rows
            asm volatile (
            "vld4.32  {d0-d3}, [%[inptr0]]!   @ zip load r0, q0,q1=r00,r04,r01,r05,r02,r06,r03,r07\n"
                    "vld4.32  {d4-d7}, [%[inptr1]]!   @ zip load r1, q2,q3=r10,r14,r11,r15,r12,r16,r13,r17\n"
                    "vld4.32  {d8-d11}, [%[inptr2]]!  @ zip load r2, q4,q5=r20,r24,r21,r25,r22,r26,r23,r27\n"
                    "vld4.32  {d12-d15}, [%[inptr3]]! @ zip load r3, q6,q7=r30,r34,r31,r35,r32,r36,r33,r37\n"
                    "vld4.32  {d16-d19}, [%[inptr4]]! @ zip load r4, q8,q9=r40,r44,r41,r45,r42,r46,r43,r47\n"
                    "vld4.32  {d20-d23}, [%[inptr5]]! @ zip load r5, q10,q11=r50,r54,r51,r55,r52,r56,r53,r57\n"

                    "vtrn.32  q0, q2                  @ trans data: q0=r00,r10,r01,r11; q2=r04,r14,r05,r15\n"
                    "vtrn.32  q4, q6                  @ trans data: q4=r20,r30,r21,r31; q6=r24,r34,r25,r35\n"
                    "vtrn.32  q8, q10                 @ trans data: q8=r40,r50,r41,r51; q10=r44,r54,r45,r55\n"

                    "vswp     d1, d8                  @ swap d1, d8, q0=r00,r10,r20,r30; q4=r01,r11,r21,r31\n"
                    "vst1.32  {d0-d1},  [%[outptr]]!  @ write q0:r00,r10,r20,r30\n"
                    "vst1.32  {d16},    [%[outptr]]!  @ write d16(q8,low),r40,r50\n"
                    "vst1.32  {d8-d9},  [%[outptr]]!  @ write q4:r01,r11,r21,r31\n"
                    "vst1.32  {d17},    [%[outptr]]!  @ write d16(q8,high),r41,r51\n"

                    "vtrn.32  q1, q3                  @ trans data: q1=r02,r12,r03,r13; q3=r06,r16,r07,r17\n"
                    "vtrn.32  q5, q7                  @ trans data: q5=r22,r32,r23,r33; q7=r26,r36,r27,r37\n"
                    "vtrn.32  q9, q11                 @ trans data: q9=r42,r52,r43,r53; q11=r46,r56,r47,r57\n"

                    "vswp     d3, d10                 @ swap d3, d10, q1=r02,r12,r22,r32; q5=r03,r13,r23,r33\n"
                    "vst1.32  {d2-d3},  [%[outptr]]!  @ write q1:r02,r12,r22,r32\n"
                    "vst1.32  {d18},    [%[outptr]]!  @ write d18(q9,low),r42,r52\n"
                    "vst1.32  {d10-d11},[%[outptr]]!  @ write q5:r03,r13,r23,r33\n"
                    "vst1.32  {d19},    [%[outptr]]!  @ write d19(q9,high),r43,r53\n"

                    "vswp     d5, d12                 @ swap d5, d12,q2=r04,r14,r24,r34; q6=r05,r15,r25,r35\n"
                    "vst1.32  {d4-d5},  [%[outptr]]!  @ write q2:r04,r14,r24,r34\n"
                    "vst1.32  {d20},    [%[outptr]]!  @ write d20(q10,low),r44,r54\n"
                    "vst1.32  {d12-d13},[%[outptr]]!  @ write q6:r05,r15,r25,r35\n"
                    "vst1.32  {d21},    [%[outptr]]!  @ write d21(q10,high),r45,r55\n"

                    "vswp     d7, d14                 @ swap d7, d14, q3=r06,r16,r26,r36; q7=r07,r17,r27,r37\n"
                    "vst1.32  {d6-d7},  [%[outptr]]!  @ write q3:r06,r16,r26,r36\n"
                    "vst1.32  {d22},    [%[outptr]]!  @ write d22(q11,low),r46,r56\n"
                    "vst1.32  {d14-d15},[%[outptr]]!  @ write q7:r07,r17,r27,r37\n"
                    "vst1.32  {d23},    [%[outptr]]!  @ write d23(q11,high),r47,r57\n"
            : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), \
                [inptr3] "+r" (inptr3), [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), \
                [outptr] "+r" (outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
            );
        }

        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
        }
    }
}

void prepackA_trans_6x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {

    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + m0;

    uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int x_len = mmax - m0;
    int y_len = kmax - k0;
    int right_remain = x_len - 6 * (x_len / 6);
    int right_pad = 6 - right_remain;
    if (right_remain == 0) {
        right_pad = 0;
    }

    uint32_t *outptr_row = outptr;
    int stride_out = 6 * y_len;

    uint32x4_t vzero = vdupq_n_u32(0);
    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {

        const uint32_t* ptr0 = inptr + y * ldin;
        const uint32_t* ptr1 = ptr0 + ldin;
        const uint32_t* ptr2 = ptr1 + ldin;
        const uint32_t* ptr3 = ptr2 + ldin;

        uint32_t *outptr_row_col = outptr_row + y * 6;
        int i = 0;
        for (; i < x_len - 5; i += 6) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
                    "vld1.32 {d4-d6}, [%[ptr1]]!        @ load r1, 6 elements\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"

                    "vld1.32 {d0-d2}, [%[ptr2]]!        @ load r2, 6 elements\n"
                    "vld1.32 {d4-d6}, [%[ptr3]]!        @ load r3, 6 elements\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            :
            : "q0", "q1", "q2", "q3", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_pad > 0) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
                    "vld1.32 {d4-d6}, [%[ptr1]]!        @ load r1, 6 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d2, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d6, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"

                    "vld1.32 {d0-d2}, [%[ptr2]]!        @ load r2, 8 elements\n"
                    "vld1.32 {d4-d6}, [%[ptr3]]!        @ load r3, 8 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d2, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d6, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            : [vmask1] "w" (vmask1), [vmask2] "w" (vmask2), \
              [vzero] "w" (vzero)
            : "q0", "q1", "q2", "q3", "memory"
            );
        }
    }

#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {

        const uint32_t* ptr0 = inptr + y * ldin;
        uint32_t *outptr_row_col = outptr_row + y * 6;
        int i = 0;
        for (; i < x_len - 5; i += 6) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            :
            : "q0", "q1", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_pad > 0) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d2, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            : [vmask1] "w" (vmask1), [vmask2] "w" (vmask2), \
              [vzero] "w" (vzero)
            : "q0", "q1", "memory"
            );
        }
    }
}

void prepackA_4x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {

    int x_len = kmax - k0;
    uint32_t zerobuff[x_len];
    memset(zerobuff, 0, sizeof(uint32_t) * x_len);

    uint32_t *dout = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);

    uint32_t* outptr = dout;
    //! data A is not transposed, transpose A to k * 4
    for (int y = m0; y < mmax; y += 4) {
        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;

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
            "vld4.32  {d0-d3}, [%[inptr0]]!   @ zip load r0, q0,q1=r00,r04,r01,r05,r02,r06,r03,r07\n"
                    "vld4.32  {d4-d7}, [%[inptr1]]!   @ zip load r1, q2,q3=r10,r14,r11,r15,r12,r16,r13,r17\n"
                    "vld4.32  {d8-d11}, [%[inptr2]]!  @ zip load r2, q4,q5=r20,r24,r21,r25,r22,r26,r23,r27\n"
                    "vld4.32  {d12-d15}, [%[inptr3]]! @ zip load r3, q6,q7=r30,r34,r31,r35,r32,r36,r33,r37\n"

                    "vtrn.32  q0, q2                  @ trans data: q0=r00,r10,r01,r11; q2=r04,r14,r05,r15\n"
                    "vtrn.32  q4, q6                  @ trans data: q4=r20,r30,r21,r31; q6=r24,r34,r25,r35\n"

                    "vswp     d1, d8                  @ swap d1, d8, q0=r00,r10,r20,r30; q4=r01,r11,r21,r31\n"
                    "vst1.32  {d0-d1},  [%[outptr]]!  @ write q0:r00,r10,r20,r30\n"
                    "vst1.32  {d8-d9},  [%[outptr]]!  @ write q4:r01,r11,r21,r31\n"

                    "vtrn.32  q1, q3                  @ trans data: q1=r02,r12,r03,r13; q3=r06,r16,r07,r17\n"
                    "vtrn.32  q5, q7                  @ trans data: q5=r22,r32,r23,r33; q7=r26,r36,r27,r37\n"

                    "vswp     d3, d10                 @ swap d3, d10, q1=r02,r12,r22,r32; q5=r03,r13,r23,r33\n"
                    "vst1.32  {d2-d3},  [%[outptr]]!  @ write q1:r02,r12,r22,r32\n"
                    "vst1.32  {d10-d11},[%[outptr]]!  @ write q5:r03,r13,r23,r33\n"

                    "vswp     d5, d12                 @ swap d5, d12,q2=r04,r14,r24,r34; q6=r05,r15,r25,r35\n"
                    "vst1.32  {d4-d5},  [%[outptr]]!  @ write q2:r04,r14,r24,r34\n"
                    "vst1.32  {d12-d13},[%[outptr]]!  @ write q6:r05,r15,r25,r35\n"

                    "vswp     d7, d14                 @ swap d7, d14, q3=r06,r16,r26,r36; q7=r07,r17,r27,r37\n"
                    "vst1.32  {d6-d7},  [%[outptr]]!  @ write q3:r06,r16,r26,r36\n"
                    "vst1.32  {d14-d15},[%[outptr]]!  @ write q7:r07,r17,r27,r37\n"
            : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), \
                [inptr3] "+r" (inptr3), [outptr] "+r" (outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
            );
        }

        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
        }
    }
}

void prepackA_trans_4x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {

    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + m0;

    uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int x_len = mmax - m0;
    int y_len = kmax - k0;
    int right_remain = x_len - 4 * (x_len / 4);
    int right_pad = 4 - right_remain;
    if (right_remain == 0) {
        right_pad = 0;
    }

    uint32_t *outptr_row = outptr;
    int stride_out = 4 * y_len;

    uint32x4_t vzero = vdupq_n_u32(0);
    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
    //uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {

        const uint32_t* ptr0 = inptr + y * ldin;
        const uint32_t* ptr1 = ptr0 + ldin;
        const uint32_t* ptr2 = ptr1 + ldin;
        const uint32_t* ptr3 = ptr2 + ldin;

        uint32_t *outptr_row_col = outptr_row + y * 4;
        int i = 0;
        for (; i < x_len - 3; i += 4) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d1}, [%[ptr0]]!                @ load r0, 4 elements\n"
                    "vld1.32 {d2-d3}, [%[ptr1]]!        @ load r1, 4 elements\n"
                    "vst1.32 {d0-d1}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d2-d3}, [%[outptr]]!      @ write to output ptr\n"

                    "vld1.32 {d4-d5}, [%[ptr2]]!        @ load r2, 4 elements\n"
                    "vld1.32 {d6-d7}, [%[ptr3]]!        @ load r3, 4 elements\n"
                    "vst1.32 {d4-d5}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d6-d7}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            :
            : "q0", "q1", "q2", "q3", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_pad > 0) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d1}, [%[ptr0]]!                @ load r0, 4 elements\n"
                    "vld1.32 {d2-d3}, [%[ptr1]]!        @ load r1, 4 elements\n"
                    "vld1.32 {d4-d5}, [%[ptr2]]!        @ load r2, 4 elements\n"
                    "vld1.32 {d6-d7}, [%[ptr3]]!        @ load r3, 4 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   q1, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   q3, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d1}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d2-d3}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d5}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d6-d7}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            : [vmask1] "w" (vmask1), [vzero] "w" (vzero)
            : "q0", "q1", "q2", "q3", "memory"
            );
        }
    }

#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {

        const uint32_t* ptr0 = inptr + y * ldin;
        uint32_t *outptr_row_col = outptr_row + y * 4;
        int i = 0;
        for (; i < x_len - 3; i += 4) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d1}, [%[ptr0]]!                @ load r0, 4 elements\n"
                    "vst1.32 {d0-d1}, [%[outptr]]!      @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            :
            : "q0", "q1", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_pad > 0) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d1}, [%[ptr0]]!                @ load r0, 4 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d1}, [%[outptr]]!      @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            : [vmask1] "w" (vmask1), [vzero] "w" (vzero)
            : "q0", "q1", "memory"
            );
        }
    }
}

#endif //__aarch64__


/**
* \brief input data is transpose
* for arm-v7a, transform data to block x k x 8 layout
* for arm-v8a, transform data to block x k x 12 layout
*/
#ifdef __aarch64__
void loadb(float* out, const float* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {
    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + n0;
    uint32_t mask_buffer[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int x_len = nmax - n0;
    int y_len = kmax - k0;
    int right_remain = x_len - 12 * (x_len / 12);
    int right_pad = 12 - right_remain;
    const size_t copy_len_remain = sizeof(float) * right_remain;
    const size_t copy_len_pad = sizeof(float) * right_pad;
    const size_t size_ldin = sizeof(float) * ldin;

    uint32_t *outptr_row = outptr;
    int stride_out = 12 * y_len;

    uint32x4_t vzero = vdupq_n_u32(0);
    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));
    uint32x4_t vmask3 = vcltq_u32(vld1q_u32(mask_buffer + 8), vdupq_n_u32(right_remain));

#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {

        const uint32_t *ptr0 = inptr + y * ldin;
        const uint32_t *ptr1 = ptr0 + ldin;
        const uint32_t *ptr2 = ptr1 + ldin;
        const uint32_t *ptr3 = ptr2 + ldin;
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

        uint32_t *outptr_row_col = outptr_row + y * 12;

        int i = 0;
        for (; i < x_len - 11; i += 12) {

            uint32x4_t vr00 = vld1q_u32(ptr0);
            uint32x4_t vr01 = vld1q_u32(ptr0 + 4);
            uint32x4_t vr02 = vld1q_u32(ptr0 + 8);

            uint32x4_t vr10 = vld1q_u32(ptr1);
            uint32x4_t vr11 = vld1q_u32(ptr1 + 4);
            uint32x4_t vr12 = vld1q_u32(ptr1 + 8);

            vst1q_u32(outptr_row_col, vr00);
            vst1q_u32(outptr_row_col + 4, vr01);
            vst1q_u32(outptr_row_col + 8, vr02);

            uint32x4_t vr20 = vld1q_u32(ptr2);
            uint32x4_t vr21 = vld1q_u32(ptr2 + 4);
            uint32x4_t vr22 = vld1q_u32(ptr2 + 8);

            vst1q_u32(outptr_row_col + 12, vr10);
            vst1q_u32(outptr_row_col + 16, vr11);
            vst1q_u32(outptr_row_col + 20, vr12);

            uint32x4_t vr30 = vld1q_u32(ptr3);
            uint32x4_t vr31 = vld1q_u32(ptr3 + 4);
            uint32x4_t vr32 = vld1q_u32(ptr3 + 8);

            vst1q_u32(outptr_row_col + 24, vr20);
            vst1q_u32(outptr_row_col + 28, vr21);
            vst1q_u32(outptr_row_col + 32, vr22);

            vst1q_u32(outptr_row_col + 36, vr30);
            vst1q_u32(outptr_row_col + 40, vr31);
            vst1q_u32(outptr_row_col + 44, vr32);

            ptr0 += 12;
            ptr1 += 12;
            ptr2 += 12;
            ptr3 += 12;

            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {

            uint32x4_t vr00 = vld1q_u32(ptr0);
            uint32x4_t vr01 = vld1q_u32(ptr0 + 4);
            uint32x4_t vr02 = vld1q_u32(ptr0 + 8);

            uint32x4_t vr10 = vld1q_u32(ptr1);
            uint32x4_t vr11 = vld1q_u32(ptr1 + 4);
            uint32x4_t vr12 = vld1q_u32(ptr1 + 8);


            uint32x4_t vr00_1 = vbslq_u32(vmask1, vr00, vzero);
            uint32x4_t vr01_1 = vbslq_u32(vmask2, vr01, vzero);
            uint32x4_t vr02_1 = vbslq_u32(vmask3, vr02, vzero);

            uint32x4_t vr20 = vld1q_u32(ptr2);
            uint32x4_t vr21 = vld1q_u32(ptr2 + 4);
            uint32x4_t vr22 = vld1q_u32(ptr2 + 8);

            vst1q_u32(outptr_row_col, vr00_1);
            vst1q_u32(outptr_row_col + 4, vr01_1);
            vst1q_u32(outptr_row_col + 8, vr02_1);


            uint32x4_t vr10_1 = vbslq_u32(vmask1, vr10, vzero);
            uint32x4_t vr11_1 = vbslq_u32(vmask2, vr11, vzero);
            uint32x4_t vr12_1 = vbslq_u32(vmask3, vr12, vzero);


            uint32x4_t vr30 = vld1q_u32(ptr3);
            uint32x4_t vr31 = vld1q_u32(ptr3 + 4);
            uint32x4_t vr32 = vld1q_u32(ptr3 + 8);

            vst1q_u32(outptr_row_col + 12, vr10_1);
            vst1q_u32(outptr_row_col + 16, vr11_1);
            vst1q_u32(outptr_row_col + 20, vr12_1);


            uint32x4_t vr20_1 = vbslq_u32(vmask1, vr20, vzero);
            uint32x4_t vr21_1 = vbslq_u32(vmask2, vr21, vzero);
            uint32x4_t vr22_1 = vbslq_u32(vmask3, vr22, vzero);

            uint32x4_t vr30_1 = vbslq_u32(vmask1, vr30, vzero);
            uint32x4_t vr31_1 = vbslq_u32(vmask2, vr31, vzero);
            uint32x4_t vr32_1 = vbslq_u32(vmask3, vr32, vzero);

            vst1q_u32(outptr_row_col + 24, vr20_1);
            vst1q_u32(outptr_row_col + 28, vr21_1);
            vst1q_u32(outptr_row_col + 32, vr22_1);

            vst1q_u32(outptr_row_col + 36, vr30_1);
            vst1q_u32(outptr_row_col + 40, vr31_1);
            vst1q_u32(outptr_row_col + 44, vr32_1);
        }
    }

#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {

        const uint32_t* ptr0 = inptr + y * ldin;
        uint32_t *outptr_row_col = outptr_row + y * 12;

        int i = 0;
        for (; i < x_len - 11; i += 12) {

            uint32x4_t vr0 = vld1q_u32(ptr0);
            uint32x4_t vr1 = vld1q_u32(ptr0 + 4);
            uint32x4_t vr2 = vld1q_u32(ptr0 + 8);
            vst1q_u32(outptr_row_col, vr0);
            vst1q_u32(outptr_row_col + 4, vr1);
            vst1q_u32(outptr_row_col + 8, vr2);

            ptr0 += 12;

            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {

            uint32x4_t vr0 = vld1q_u32(ptr0);
            uint32x4_t vr1 = vld1q_u32(ptr0 + 4);
            uint32x4_t vr2 = vld1q_u32(ptr0 + 8);

            uint32x4_t vr0_1 = vbslq_u32(vmask1, vr0, vzero);
            uint32x4_t vr1_1 = vbslq_u32(vmask2, vr1, vzero);
            uint32x4_t vr2_1 = vbslq_u32(vmask3, vr2, vzero);

            vst1q_u32(outptr_row_col, vr0_1);
            vst1q_u32(outptr_row_col + 4, vr1_1);
            vst1q_u32(outptr_row_col + 8, vr2_1);
        }
    }
}

void loadb_trans(float* out, const float* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {

    int x_len = kmax - k0;
    uint32_t zerobuff[x_len];
    memset(zerobuff, 0, sizeof(uint32_t) * x_len);
    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);

    //! data B is not transposed, transpose B to k * 12
    for (int y = n0; y < nmax; y += 12) {
        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;
        const uint32_t *inptr6 = inptr5 + ldin;
        const uint32_t *inptr7 = inptr6 + ldin;
        const uint32_t *inptr8 = inptr7 + ldin;
        const uint32_t *inptr9 = inptr8 + ldin;
        const uint32_t *inptr10 = inptr9 + ldin;
        const uint32_t *inptr11 = inptr10 + ldin;

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
                "prfm   pldl1keep, [%[ptr8]]        \n"
                "prfm   pldl1keep, [%[ptr8], #64]   \n"
                "prfm   pldl1keep, [%[ptr9]]        \n"
                "prfm   pldl1keep, [%[ptr9], #64]   \n"
                "prfm   pldl1keep, [%[ptr10]]        \n"
                "prfm   pldl1keep, [%[ptr10], #64]   \n"
                "prfm   pldl1keep, [%[ptr11]]        \n"
                "prfm   pldl1keep, [%[ptr11], #64]   \n"
        :
        :[ptr0] "r"(inptr0),[ptr1] "r"(inptr1),[ptr2] "r"(inptr2),[ptr3] "r"(inptr3), \
                [ptr4] "r"(inptr4),[ptr5] "r"(inptr5),[ptr6] "r"(inptr6),[ptr7] "r"(inptr7), \
                [ptr8] "r"(inptr8),[ptr9] "r"(inptr9),[ptr10] "r"(inptr10),[ptr11] "r"(inptr11)
        :"memory"
        );

        int x = x_len;

        //! cope with row index exceed real size, set to zero buffer
        if ((y + 11) >= nmax) {
            switch ((y + 11) - nmax) {
                case 10:
                    inptr1 = zerobuff;
                case 9:
                    inptr2 = zerobuff;
                case 8:
                    inptr3 = zerobuff;
                case 7:
                    inptr4 = zerobuff;
                case 6:
                    inptr5 = zerobuff;
                case 5:
                    inptr6 = zerobuff;
                case 4:
                    inptr7 = zerobuff;
                case 3:
                    inptr8 = zerobuff;
                case 2:
                    inptr9 = zerobuff;
                case 1:
                    inptr10 = zerobuff;
                case 0:
                    inptr11 = zerobuff;
                default:
                    break;
            }
        }
        for (; x > 7; x -= 8) {
            asm volatile (
            // Load up 12 elements (3 vectors) from each of 8 sources.
            "LDP        q0, q1, [%[inptr0]], #32\n" // q0=A0A1A2A3
                    "LDP        q2, q3, [%[inptr1]], #32\n" // q2=B0B1B2B3
                    "LDP        q4, q5, [%[inptr2]], #32\n" // q4=C0C1C2C3
                    "ZIP1       v16.4s, v0.4s, v4.4s\n"     // q16=A0C0A1C1
                    "prfm   pldl1keep, [%[inptr0], #128] \n"
                    "LDP        q6, q7, [%[inptr3]], #32\n" // q6=D0D1D2D3
                    "ZIP1       v17.4s, v2.4s, v6.4s\n"     // q17=B0D0B1D1
                    "LDP        q8, q9, [%[inptr4]], #32\n"
                    "LDP        q10, q11, [%[inptr5]], #32\n"
                    "LDP        q12, q13, [%[inptr6]], #32\n"
                    "ZIP1       v18.4s, v8.4s, v12.4s\n"
                    "prfm   pldl1keep, [%[inptr1], #128]\n"
                    "LDP        q14, q15, [%[inptr7]], #32\n"
                    "ZIP1       v19.4s, v10.4s, v14.4s\n"

                    "ZIP1       v20.4s, v16.4s, v17.4s\n" // q20=A0B0C0D0
                    "prfm   pldl1keep, [%[inptr2], #128]\n"
                    "ZIP1       v21.4s, v18.4s, v19.4s\n"
                    "ZIP2       v22.4s, v16.4s, v17.4s\n"
                    "ZIP2       v23.4s, v18.4s, v19.4s\n"

                    "LDP        q24, q25, [%[inptr8]], #32\n" // q24=A0A1A2A3
                    "LDP        q26, q27, [%[inptr9]], #32\n" // q26=B0B1B2B3
                    "LDP        q28, q29, [%[inptr10]], #32\n" // q28=C0C1C2C3
                    "LDP        q30, q31, [%[inptr11]], #32\n" // q30=D0D1D2D3
                    "prfm   pldl1keep, [%[inptr3], #128]\n"
                    "prfm   pldl1keep, [%[inptr4], #128]\n"
                    "ZIP1       v16.4s, v24.4s, v28.4s\n"     // q16=A0C0A1C1
                    "ZIP1       v17.4s, v26.4s, v30.4s\n"     // q17=B0D0B1D1
                    "STP        q20, q21, [%[outptr]], #32\n" // Write back the first element of each source
                    "ZIP1       v18.4s, v16.4s, v17.4s\n"    // q20=A0B0C0D0
                    "ZIP2       v19.4s, v16.4s, v17.4s\n"    // q20=A0B0C0D0

                    "ZIP2       v16.4s, v0.4s, v4.4s\n"
                    "prfm   pldl1keep, [%[inptr5], #128]\n"
                    "ZIP2       v17.4s, v2.4s, v6.4s\n"
                    "STR       q18, [%[outptr]], #16\n" // Write back the second element of each source

                    "STP        q22, q23, [%[outptr]], #32\n" // Write back the second element of each source
                    "ZIP2       v18.4s, v8.4s, v12.4s\n"
                    "prfm   pldl1keep, [%[inptr6], #128]\n"
                    "STR        q19, [%[outptr]], #16\n" // Write back the second element of each source
                    "ZIP2       v19.4s, v10.4s, v14.4s\n"

                    "ZIP1       v20.4s, v16.4s, v17.4s\n"
                    "prfm   pldl1keep, [%[inptr7], #128]\n"
                    "ZIP1       v21.4s, v18.4s, v19.4s\n"
                    "ZIP2       v22.4s, v16.4s, v17.4s\n"
                    "ZIP2       v23.4s, v18.4s, v19.4s\n"

                    "ZIP2       v16.4s, v24.4s, v28.4s\n"     // q16=A0C0A1C1
                    "ZIP2       v17.4s, v26.4s, v30.4s\n"     // q17=B0D0B1D1
                    "prfm   pldl1keep, [%[inptr8], #128]\n"
                    "STP        q20, q21, [%[outptr]], #32\n" // Third element
                    "ZIP1       v18.4s, v16.4s, v17.4s\n"
                    "ZIP2       v19.4s, v16.4s, v17.4s\n"

                    "ZIP1       v16.4s, v1.4s, v5.4s\n"
                    "prfm   pldl1keep, [%[inptr9], #128]\n"
                    "ZIP1       v17.4s, v3.4s, v7.4s\n"
                    "STR       q18, [%[outptr]], #16\n" // Write back the second element of each source

                    "STP        q22, q23, [%[outptr]], #32\n" // Fourth element
                    "ZIP1       v18.4s, v9.4s, v13.4s\n"
                    "prfm   pldl1keep, [%[inptr10], #128]\n"
                    "STR        q19, [%[outptr]], #16\n" // Write back the second element of each source
                    "ZIP1       v19.4s, v11.4s, v15.4s\n"

                    "ZIP1       v20.4s, v16.4s, v17.4s\n"
                    "ZIP1       v21.4s, v18.4s, v19.4s\n"
                    "ZIP2       v22.4s, v16.4s, v17.4s\n"
                    "prfm   pldl1keep, [%[inptr11], #128]\n"
                    "ZIP2       v23.4s, v18.4s, v19.4s\n"

                    "ZIP1       v16.4s, v25.4s, v29.4s\n"
                    "ZIP1       v17.4s, v27.4s, v31.4s\n"
                    "STP        q20, q21, [%[outptr]], #32\n" // Fifth element
                    "ZIP1       v18.4s, v16.4s, v17.4s\n"
                    "ZIP2       v19.4s, v16.4s, v17.4s\n"

                    "ZIP2       v16.4s, v1.4s, v5.4s\n"
                    "ZIP2       v17.4s, v3.4s, v7.4s\n"
                    "STR       q18, [%[outptr]], #16\n"

                    "STP        q22, q23, [%[outptr]], #32\n" // Sixth element
                    "ZIP2       v18.4s, v9.4s, v13.4s\n"
                    "STR       q19, [%[outptr]], #16\n" // Sixth element

                    "ZIP2       v19.4s, v11.4s, v15.4s\n"
                    "ZIP1       v20.4s, v16.4s, v17.4s\n"
                    "ZIP1       v21.4s, v18.4s, v19.4s\n"

                    "ZIP2       v22.4s, v16.4s, v17.4s\n"
                    "ZIP2       v23.4s, v18.4s, v19.4s\n"

                    "ZIP2       v16.4s, v25.4s, v29.4s\n"
                    "ZIP2       v17.4s, v27.4s, v31.4s\n"
                    "STP        q20, q21, [%[outptr]], #32\n" // Seventh element

                    "ZIP1       v18.4s, v16.4s, v17.4s\n"
                    "ZIP2       v19.4s, v16.4s, v17.4s\n"
                    "STR       q18, [%[outptr]], #16\n"
                    "STP        q22, q23, [%[outptr]], #32\n" // Eighth element
                    "STR       q19, [%[outptr]], #16\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
             [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5), [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), \
             [inptr8] "+r"(inptr8), [inptr9] "+r"(inptr9), [inptr10] "+r"(inptr10), [inptr11] "+r"(inptr11), \
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                    "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
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
            *outptr++ = *inptr8++;
            *outptr++ = *inptr9++;
            *outptr++ = *inptr10++;
            *outptr++ = *inptr11++;
        }
    }
}

#else //__aarch64__
void loadb(float* out, const float* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {

    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + n0;
    uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int x_len = nmax - n0;
    int y_len = kmax - k0;
    int right_remain = x_len - 8 * (x_len / 8);
    int right_pad = 8 - right_remain;
    const size_t copy_len_remain = sizeof(float) * right_remain;
    const size_t copy_len_pad = sizeof(float) * right_pad;
    const size_t size_ldin = sizeof(float) * ldin;

    uint32_t *outptr_row =outptr;
    int stride_out = 8 * y_len;

    uint32x4_t vzero = vdupq_n_u32(0);
    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {

        const uint32_t* ptr0 = inptr + y * ldin;
        const uint32_t* ptr1 = ptr0 + ldin;
        const uint32_t* ptr2 = ptr1 + ldin;
        const uint32_t* ptr3 = ptr2 + ldin;
        uint32_t *outptr_row_col = outptr_row + y * 8;
        int i = 0;
        for (; i < x_len - 7; i += 8) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d3}, [%[ptr0]]!        @ load r0, 8 elements\n"
                    "vld1.32 {d4-d7}, [%[ptr1]]!        @ load r1, 8 elements\n"
                    "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"

                    "vld1.32 {d0-d3}, [%[ptr2]]!        @ load r2, 8 elements\n"
                    "vld1.32 {d4-d7}, [%[ptr3]]!        @ load r3, 8 elements\n"
                    "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            :
            : "q0", "q1", "q2", "q3", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d3}, [%[ptr0]]!        @ load r0, 8 elements\n"
                    "vld1.32 {d4-d7}, [%[ptr1]]!        @ load r1, 8 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   q1, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
                    //"vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
                    "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   q3, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"

                    "vld1.32 {d0-d3}, [%[ptr2]]!        @ load r2, 8 elements\n"
                    "vld1.32 {d4-d7}, [%[ptr3]]!        @ load r3, 8 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   q1, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
                    //"vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
                    "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   q3, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            : [vmask1] "w" (vmask1), [vmask2] "w" (vmask2), \
              [vzero] "w" (vzero)
            : "q0", "q1", "q2", "q3", "memory"
            );
        }
    }
#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {

        const uint32_t* ptr0 = inptr + y * ldin;
        uint32_t *outptr_row_col = outptr_row + y * 8;
        int i = 0;
        for (; i < x_len - 7; i += 8) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d3}, [%[ptr0]]!        @ load r0, 8 elements\n"
                    "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            :
            : "q0", "q1", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_remain > 0) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d3}, [%[ptr0]]!        @ load r0, 8 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   q1, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            : [vmask1] "w" (vmask1), [vmask2] "w" (vmask2), \
              [vzero] "w" (vzero)
            : "q0", "q1", "memory"
            );
        }
    }
}

void loadb_trans(float* out, const float* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {

    int x_len = kmax - k0;
    uint32_t zerobuff[x_len];
    memset(zerobuff, 0, sizeof(uint32_t) * x_len);

    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);
    //! data B is not transposed, transpose B to k * 8
    for (int y = n0; y < nmax; y += 8) {
        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;
        const uint32_t *inptr6 = inptr5 + ldin;
        const uint32_t *inptr7 = inptr6 + ldin;

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
            "vld4.32  {d0-d3}, [%[inptr0]]!   @ zip load r0, q0,q1=r00,r04,r01,r05,r02,r06,r03,r07\n"
                    "vld4.32  {d4-d7}, [%[inptr1]]!   @ zip load r1, q2,q3=r10,r14,r11,r15,r12,r16,r13,r17\n"
                    "vtrn.32  q0, q2                  @ trans data: q0=r00,r10,r01,r11; q2=r04,r14,r05,r15\n"
                    "vst1.32  {d0},    [%[outptr]]!   @ write d0(q0,low),r00,r10\n"

                    "vld4.32  {d8-d11}, [%[inptr2]]!  @ zip load r2, q4,q5=r20,r24,r21,r25,r22,r26,r23,r27\n"
                    "vld4.32  {d12-d15}, [%[inptr3]]! @ zip load r3, q6,q7=r30,r34,r31,r35,r32,r36,r33,r37\n"
                    "vtrn.32  q4, q6                  @ trans data: q4=r20,r30,r21,r31; q6=r24,r34,r25,r35\n"
                    "vst1.32  {d8},    [%[outptr]]!   @ write d8(q4,low),r20,r30\n"

                    "vld4.32  {d16-d19}, [%[inptr4]]! @ zip load r4, q8,q9=r40,r44,r41,r45,r42,r46,r43,r47\n"
                    "vld4.32  {d20-d23}, [%[inptr5]]! @ zip load r5, q10,q11=r50,r54,r51,r55,r52,r56,r53,r57\n"
                    "vtrn.32  q8, q10                 @ trans data: q8=r40,r50,r41,r51; q10=r44,r54,r45,r55\n"
                    "vst1.32  {d16},    [%[outptr]]!  @ write d16(q8,low),r40,r50\n"

                    "vld4.32  {d24-d27}, [%[inptr6]]! @ zip load r6, q12,q13=r60,r64,r61,r65,r62,r66,r63,r67\n"
                    "vld4.32  {d28-d31}, [%[inptr7]]! @ zip load r7, q14,q15=r70,r74,r71,r75,r72,r76,r73,r77\n"
                    "vtrn.32  q12, q14                @ trans data:q12=r60,r70,r61,r71; q14=r64,r74,r65,r75\n"
                    "vst1.32  {d24},    [%[outptr]]!  @ write d24(q8,low),r60,r70\n"

                    //"pld      [%[inptr0], #128]       @ preload r0 data to cache, fill pipeline\n"
                    "vst1.32  {d1},     [%[outptr]]!  @ write d1(q0,high),r01,r11\n"
                    "vst1.32  {d9},     [%[outptr]]!  @ write d9(q4,high),r21,r31\n"
                    "vst1.32  {d17},    [%[outptr]]!  @ write d17(q8,high),r41,r51\n"
                    "vst1.32  {d25},    [%[outptr]]!  @ write d25(q12,high),r61,r71\n"

                    "vtrn.32  q1, q3                  @ trans data: q1=r02,r12,r03,r13; q3=r06,r16,r07,r17\n"
                    "vst1.32  {d2},     [%[outptr]]!  @ write d2(q1,low),r02,r12\n"
                    "vtrn.32  q5, q7                  @ trans data: q5=r22,r32,r23,r33; q7=r26,r36,r27,r37\n"
                    "vst1.32  {d10},    [%[outptr]]!  @ write d10(q5,low),r22,r32\n"
                    "vtrn.32  q9, q11                 @ trans data: q9=r42,r52,r43,r53; q11=r46,r56,r47,r57\n"
                    "vst1.32  {d18},    [%[outptr]]!  @ write d18(q9,low),r42,r52\n"
                    "vtrn.32  q13, q15                @ trans data:q13=r62,r72,r63,r73; q15=r66,r76,r67,r77\n"
                    "vst1.32  {d26},    [%[outptr]]!  @ write d18(q9,low),r62,r72\n"

                    //"pld      [%[inptr1], #128]       @ preload r1 data to cache, fill pipeline\n"
                    "vst1.32  {d3},     [%[outptr]]!  @ write d3(q1,high),r03,r13\n"
                    "vst1.32  {d11},    [%[outptr]]!  @ write d11(q5,high),r23,r33\n"
                    "vst1.32  {d19},    [%[outptr]]!  @ write d19(q9,high),r43,r53\n"
                    "vst1.32  {d27},    [%[outptr]]!  @ write d27(q13,high),r63,r73\n"

                    //"pld      [%[inptr2], #128]       @ preload r2 data to cache, fill pipeline\n"
                    "vst1.32  {d4},     [%[outptr]]!  @ write d4(q2,low),r04,r14\n"
                    "vst1.32  {d12},    [%[outptr]]!  @ write d12(q6,low),r24,r34\n"
                    "vst1.32  {d20},    [%[outptr]]!  @ write d20(q10,low),r44,r54\n"
                    "vst1.32  {d28},    [%[outptr]]!  @ write d28(q14,low),r64,r74\n"

                    //"pld      [%[inptr3], #128]       @ preload r3 data to cache, fill pipeline\n"
                    "vst1.32  {d5},     [%[outptr]]!  @ write d5(q2,high),r05,r15\n"
                    "vst1.32  {d13},    [%[outptr]]!  @ write d13(q6,high),r25,r35\n"
                    "vst1.32  {d21},    [%[outptr]]!  @ write d21(q10,high),r45,r55\n"
                    "vst1.32  {d29},    [%[outptr]]!  @ write d29(q14,high),r65,r75\n"

                    //"pld      [%[inptr4], #128]       @ preload r4 data to cache, fill pipeline\n"
                    "vst1.32  {d6},     [%[outptr]]!  @ write d6(q3,low),r06,r16\n"
                    "vst1.32  {d14},    [%[outptr]]!  @ write d14(q7,low),r26,r36\n"
                    "vst1.32  {d22},    [%[outptr]]!  @ write d22(q11,low),r46,r56\n"
                    "vst1.32  {d30},    [%[outptr]]!  @ write d30(q15,low),r66,r76\n"

                    //"pld      [%[inptr5], #128]       @ preload r5 data to cache, fill pipeline\n"
                    "vst1.32  {d7},     [%[outptr]]!  @ write d7(q3,high),r07,r17\n"
                    "vst1.32  {d15},    [%[outptr]]!  @ write d15(q7,high),r27,r37\n"
                    "vst1.32  {d23},    [%[outptr]]!  @ write d23(q11,high),r47,r57\n"
                    "vst1.32  {d31},    [%[outptr]]!  @ write d31(q15,high),r67,r77\n"
            : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), \
                [inptr3] "+r" (inptr3), [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), \
                [inptr6] "+r" (inptr6), [inptr7] "+r" (inptr7),[outptr] "+r" (outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
                    "q13", "q14", "q15"
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

#endif //__aarch64__

#ifdef __aarch64__

#define GEMM_KERNEL8x12 \
        "cbz	%w[k], 2f\n"                            /* check loop count > 0 */              \
        /* main loop */                                                                         \
        /* unrool 0*/                                                                           \
        "1:\n"                                          /* main loop */                         \
        "fmla 	v8.4s ,  v4.4s,  v0.s[0]\n"             /* out0 = b0 * a00[0], b0 = q4 */       \
        "fmla  	v9.4s ,  v4.4s,  v0.s[1]\n"             /* out1 = b0 * a00[1], b0 = q4 */       \
        "ldp	q6, q7, [%[b_ptr]], #32\n"              /* load b2, b0 to q6, q7       */       \
        "fmla	v10.4s,  v4.4s,  v0.s[2]\n"             /* out2 = b0 * a00[2], b0 = q4 */       \
        "fmla	v11.4s,  v4.4s,  v0.s[3]\n"             /* out3 = b0 * a00[3], b0 = q4 */       \
        "ldp	q2, q3, [%[a_ptr]], #32\n"              /* load a10, a11 to q3, q4     */       \
        "fmla 	v12.4s,  v4.4s,  v1.s[0]\n"             /* out4 = b0 * a01[0], b0 = q4 */       \
        "fmla	v13.4s,  v4.4s,  v1.s[1]\n"             /* out5 = b0 * a01[1], b0 = q4 */       \
        "fmla	v14.4s,  v4.4s,  v1.s[2]\n"             /* out6 = b0 * a01[2], b0 = q4 */       \
        "fmla	v15.4s,  v4.4s,  v1.s[3]\n"             /* out7 = b0 * a01[3], b0 = q4 */       \
                                                                                                \
        "fmla	v16.4s,  v5.4s,  v0.s[0]\n"             /* out8 = b1 * a00[0], b1 = q5 */       \
        "fmla	v17.4s,  v5.4s,  v0.s[1]\n"             /* out9 = b1 * a00[1], b1 = q5 */       \
        "fmla	v18.4s,  v5.4s,  v0.s[2]\n"             /* out10 = b1 * a00[2], b1 = q5*/       \
        "fmla	v19.4s,  v5.4s,  v0.s[3]\n"             /* out11 = b1 * a00[3], b1 = q5*/       \
        "fmla	v20.4s,  v5.4s,  v1.s[0]\n"             /* out12 = b1 * a01[0], b1 = q5*/       \
        "fmla	v21.4s,  v5.4s,  v1.s[1]\n"             /* out13 = b1 * a01[1], b1 = q5*/       \
        "fmla	v22.4s,  v5.4s,  v1.s[2]\n"             /* out14 = b1 * a01[2], b1 = q5*/       \
        "fmla	v23.4s,  v5.4s,  v1.s[3]\n"             /* out15 = b1 * a01[3], b1 = q5*/       \
                                                                                                \
        "ldp	q4, q5, [%[b_ptr]], #32\n"              /* load b1, b2 to q4, q5       */       \
                                                                                                \
        "fmla	v24.4s,  v6.4s,  v0.s[0]\n"             /* out16 = b2 * a00[0], b2 = q6*/       \
        "fmla	v25.4s,  v6.4s,  v0.s[1]\n"             /* out17 = b2 * a00[1], b2 = q6*/       \
        "prfm   pldl1keep, [%[b_ptr], #384]\n"                                              \
        "fmla	v26.4s,  v6.4s,  v0.s[2]\n"             /* out18 = b2 * a00[2], b2 = q6*/       \
        "fmla	v27.4s,  v6.4s,  v0.s[3]\n"             /* out19 = b2 * a00[3], b2 = q6*/       \
        "fmla	v28.4s,  v6.4s,  v1.s[0]\n"             /* out20 = b2 * a00[0], b2 = q6*/       \
        "fmla	v29.4s,  v6.4s,  v1.s[1]\n"             /* out21 = b2 * a00[1], b2 = q6*/       \
        "fmla	v30.4s,  v6.4s,  v1.s[2]\n"             /* out22 = b2 * a00[2], b2 = q6*/       \
        "fmla	v31.4s,  v6.4s,  v1.s[3]\n"             /* out23 = b2 * a00[3], b2 = q6*/       \
                                                                                                \
        "ldp	q0, q1, [%[a_ptr]], #32\n"              /* load a00, a01 to q0, q1     */       \
                                                                                                \
        /* unrool 1 */                                                                          \
        "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n"             /* out0 = b0 * a10[0], b0 = q7 */       \
        "fmla	v9.4s ,  v7.4s,  v2.s[1]\n"             /* out1 = b0 * a10[1], b0 = q7 */       \
        "fmla	v10.4s,  v7.4s,  v2.s[2]\n"             /* out2 = b0 * a10[2], b0 = q7 */       \
        "prfm   pldl1keep, [%[a_ptr], #256]\n"                                              \
        "fmla	v11.4s,  v7.4s,  v2.s[3]\n"             /* out3 = b0 * a10[3], b0 = q7 */       \
        "fmla 	v12.4s,  v7.4s,  v3.s[0]\n"             /* out4 = b0 * a11[0], b0 = q7 */       \
        "fmla   v13.4s,  v7.4s,  v3.s[1]\n"             /* out5 = b0 * a11[1], b0 = q7 */       \
        "fmla	v14.4s,  v7.4s,  v3.s[2]\n"             /* out6 = b0 * a11[2], b0 = q7 */       \
        "fmla	v15.4s,  v7.4s,  v3.s[3]\n"             /* out7 = b0 * a11[3], b0 = q7 */       \
                                                                                                \
        "ldp	q6, q7, [%[b_ptr]], #32\n"              /* load b0, b1 to q6, q7       */       \
                                                                                                \
        "fmla	v16.4s,  v4.4s,  v2.s[0]\n"             /* out8 = b0 * a10[0], b1 = q4 */       \
        "fmla	v17.4s,  v4.4s,  v2.s[1]\n"             /* out9 = b0 * a10[1], b1 = q4 */       \
        "fmla	v18.4s,  v4.4s,  v2.s[2]\n"             /* out10 = b1 * a10[2], b1 = q4*/       \
        "fmla	v19.4s,  v4.4s,  v2.s[3]\n"             /* out11 = b1 * a10[3], b1 = q4*/       \
        "fmla	v20.4s,  v4.4s,  v3.s[0]\n"             /* out12 = b1 * a10[0], b1 = q4*/       \
        "fmla	v21.4s,  v4.4s,  v3.s[1]\n"             /* out13 = b1 * a10[1], b1 = q4*/       \
        "fmla	v22.4s,  v4.4s,  v3.s[2]\n"             /* out14 = b1 * a10[2], b1 = q4*/       \
        "fmla	v23.4s,  v4.4s,  v3.s[3]\n"             /* out15 = b1 * a10[3], b1 = q4*/       \
                                                                                                \
        "fmla	v24.4s,  v5.4s,  v2.s[0]\n"             /* out16 = b2 * a10[0], b2 = q5*/       \
        "fmla	v25.4s,  v5.4s,  v2.s[1]\n"             /* out17 = b2 * a10[0], b2 = q5*/       \
        "fmla	v26.4s,  v5.4s,  v2.s[2]\n"             /* out18 = b2 * a10[0], b2 = q5*/       \
        "fmla	v27.4s,  v5.4s,  v2.s[3]\n"             /* out19 = b2 * a10[0], b2 = q5*/       \
        "fmla	v28.4s,  v5.4s,  v3.s[0]\n"             /* out20 = b2 * a10[0], b2 = q5*/       \
        "fmla	v29.4s,  v5.4s,  v3.s[1]\n"             /* out21 = b2 * a10[0], b2 = q5*/       \
        "fmla	v30.4s,  v5.4s,  v3.s[2]\n"             /* out22 = b2 * a10[0], b2 = q5*/       \
        "fmla	v31.4s,  v5.4s,  v3.s[3]\n"             /* out23 = b2 * a10[0], b2 = q5*/       \
        "ldp	q4, q5, [%[b_ptr]], #32\n"              /* load b2, b0 to q4, q5       */       \
        /* unrool 2*/       \
        "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n"             /* out0 = b0 * a00[0], b0 = q6 */       \
        "fmla  	v9.4s ,  v6.4s,  v0.s[1]\n"             /* out1 = b0 * a00[1], b0 = q6 */       \
        "ldp	q2, q3, [%[a_ptr]], #32\n"              /* load a10, a11 to q3, q4*/            \
        "fmla	v10.4s,  v6.4s,  v0.s[2]\n"             /* out2 = b0 * a00[2], b0 = q6*/        \
        "fmla	v11.4s,  v6.4s,  v0.s[3]\n"             /* out3 = b0 * a00[3], b0 = q6*/        \
        "fmla 	v12.4s,  v6.4s,  v1.s[0]\n"             /* out4 = b0 * a01[0], b0 = q6*/        \
        "fmla	v13.4s,  v6.4s,  v1.s[1]\n"             /* out5 = b0 * a01[1], b0 = q6*/        \
        "fmla	v14.4s,  v6.4s,  v1.s[2]\n"             /* out6 = b0 * a01[2], b0 = q6*/        \
        "fmla	v15.4s,  v6.4s,  v1.s[3]\n"             /* out7 = b0 * a01[3], b0 = q6*/        \
        "fmla	v16.4s,  v7.4s,  v0.s[0]\n"             /* out8 = b1 * a00[0], b1 = q7*/        \
        "fmla	v17.4s,  v7.4s,  v0.s[1]\n"             /* out9 = b1 * a00[1], b1 = q7*/        \
        "prfm   pldl1keep, [%[b_ptr], #384]\n"                                              \
        "fmla	v18.4s,  v7.4s,  v0.s[2]\n"             /* out10 = b1 * a00[2], b1 = q7*/       \
        "fmla	v19.4s,  v7.4s,  v0.s[3]\n"             /* out11 = b1 * a00[3], b1 = q7*/       \
        "fmla	v20.4s,  v7.4s,  v1.s[0]\n"             /* out12 = b1 * a01[0], b1 = q7*/       \
        "fmla	v21.4s,  v7.4s,  v1.s[1]\n"             /* out13 = b1 * a01[1], b1 = q7*/       \
        "fmla	v22.4s,  v7.4s,  v1.s[2]\n"             /* out14 = b1 * a01[2], b1 = q7*/       \
        "fmla	v23.4s,  v7.4s,  v1.s[3]\n"             /* out15 = b1 * a01[3], b1 = q7*/       \
                                                                                                \
        "ldp	q6, q7, [%[b_ptr]], #32\n"              /* load b1, b2 to q6, q7*/              \
                                                                                                \
        "fmla	v24.4s,  v4.4s,  v0.s[0]\n"             /* out16 = b2 * a00[0], b2 = q4*/       \
        "fmla	v25.4s,  v4.4s,  v0.s[1]\n"             /* out17 = b2 * a00[1], b2 = q4*/       \
        "fmla	v26.4s,  v4.4s,  v0.s[2]\n"             /* out18 = b2 * a00[2], b2 = q4*/       \
        "fmla	v27.4s,  v4.4s,  v0.s[3]\n"             /* out19 = b2 * a00[3], b2 = q4*/       \
        "fmla	v28.4s,  v4.4s,  v1.s[0]\n"             /* out20 = b2 * a00[0], b2 = q4*/       \
        "fmla	v29.4s,  v4.4s,  v1.s[1]\n"             /* out21 = b2 * a00[1], b2 = q4*/       \
        "fmla	v30.4s,  v4.4s,  v1.s[2]\n"             /* out22 = b2 * a00[2], b2 = q4*/       \
        "fmla	v31.4s,  v4.4s,  v1.s[3]\n"             /* out23 = b2 * a00[3], b2 = q4*/       \
        "ldp	q0, q1, [%[a_ptr]], #32\n"              /* load a00, a01 to q0, q1*/            \
        /* unrool 3*/       \
        "fmla 	v8.4s ,  v5.4s,  v2.s[0]\n"             /* out0 = b0 * a10[0], b0 = q5*/       \
        "fmla	v9.4s ,  v5.4s,  v2.s[1]\n"             /* out1 = b0 * a10[1], b0 = q5*/       \
        "fmla	v10.4s,  v5.4s,  v2.s[2]\n"             /* out2 = b0 * a10[2], b0 = q5*/       \
        "fmla	v11.4s,  v5.4s,  v2.s[3]\n"             /* out3 = b0 * a10[3], b0 = q5*/       \
        "fmla 	v12.4s,  v5.4s,  v3.s[0]\n"             /* out4 = b0 * a11[0], b0 = q5*/       \
        "fmla   v13.4s,  v5.4s,  v3.s[1]\n"             /* out5 = b0 * a11[1], b0 = q5*/       \
        "fmla	v14.4s,  v5.4s,  v3.s[2]\n"             /* out6 = b0 * a11[2], b0 = q5*/       \
        "fmla	v15.4s,  v5.4s,  v3.s[3]\n"             /* out7 = b0 * a11[3], b0 = q5*/       \
        "ldp	q4, q5, [%[b_ptr]], #32\n"              /* load b0, b1 to q4, q5*/             \
        "fmla	v16.4s,  v6.4s,  v2.s[0]\n"             /* out8 = b0 * a10[0], b1 = q6*/       \
        "fmla	v17.4s,  v6.4s,  v2.s[1]\n"             /* out9 = b0 * a10[1], b1 = q6*/       \
        "prfm   pldl1keep, [%[a_ptr], #256]\n"                                             \
        "fmla	v18.4s,  v6.4s,  v2.s[2]\n"             /* out10 = b1 * a10[2], b1 = q6*/      \
        "fmla	v19.4s,  v6.4s,  v2.s[3]\n"             /* out11 = b1 * a10[3], b1 = q6*/      \
        "fmla	v20.4s,  v6.4s,  v3.s[0]\n"             /* out12 = b1 * a10[0], b1 = q6*/      \
        "fmla	v21.4s,  v6.4s,  v3.s[1]\n"             /* out13 = b1 * a10[1], b1 = q6*/      \
        "fmla	v22.4s,  v6.4s,  v3.s[2]\n"             /* out14 = b1 * a10[2], b1 = q6*/      \
        "prfm   pldl1keep, [%[b_ptr], #384]\n"                                             \
        "fmla	v23.4s,  v6.4s,  v3.s[3]\n"             /* out15 = b1 * a10[3], b1 = q6*/      \
        "fmla	v24.4s,  v7.4s,  v2.s[0]\n"             /* out16 = b2 * a10[0], b2 = q7*/      \
        "fmla	v25.4s,  v7.4s,  v2.s[1]\n"             /* out17 = b2 * a10[0], b2 = q7*/      \
        "fmla	v26.4s,  v7.4s,  v2.s[2]\n"             /* out18 = b2 * a10[0], b2 = q7*/      \
        "fmla	v27.4s,  v7.4s,  v2.s[3]\n"             /* out19 = b2 * a10[0], b2 = q7*/      \
        "fmla	v28.4s,  v7.4s,  v3.s[0]\n"             /* out20 = b2 * a10[0], b2 = q7*/      \
        "fmla	v29.4s,  v7.4s,  v3.s[1]\n"             /* out21 = b2 * a10[0], b2 = q7*/      \
        "subs	%w[k], %w[k], #1\n"                     /* loop count - 1*/                    \
        "fmla	v30.4s,  v7.4s,  v3.s[2]\n"             /* out22 = b2 * a10[0], b2 = q7*/      \
        "fmla	v31.4s,  v7.4s,  v3.s[3]\n"             /* out23 = b2 * a10[0], b2 = q7*/      \
        "bne	1b\n"                                                                          \
        /* Target to use when K is 1 or 2 (i.e. zero iterations of main loop)*/\
        "2:\n"                                          /* process tail*/       \
        "subs		%w[tail], %w[tail], #1\n"           /* tail--*/       \
        "beq		3f\n"                               /*jump to tail = 1*/       \
        /* final unrool 0*/       \
        /* unrool 0, tail > 1*/       \
        "fmla 	v8.4s ,  v4.4s,  v0.s[0]\n"             /* out0 = b0 * a00[0], b0 = q4*/       \
        "fmla  	v9.4s ,  v4.4s,  v0.s[1]\n"             /* out1 = b0 * a00[1], b0 = q4*/       \
        "ldp	q6, q7, [%[b_ptr]], #32\n"              /* load b2, b0 to q6, q7*/       \
        "fmla	v10.4s,  v4.4s,  v0.s[2]\n"             /* out2 = b0 * a00[2], b0 = q4*/       \
        "fmla	v11.4s,  v4.4s,  v0.s[3]\n"             /* out3 = b0 * a00[3], b0 = q4*/       \
        "ldp	q2, q3, [%[a_ptr]], #32\n"              /* load a10, a11 to q2, q3*/       \
        "fmla 	v12.4s,  v4.4s,  v1.s[0]\n"             /* out4 = b0 * a01[0], b0 = q4*/       \
        "fmla	v13.4s,  v4.4s,  v1.s[1]\n"             /* out5 = b0 * a01[1], b0 = q4*/       \
        "fmla	v14.4s,  v4.4s,  v1.s[2]\n"             /* out6 = b0 * a01[2], b0 = q4*/       \
        "fmla	v15.4s,  v4.4s,  v1.s[3]\n"             /* out7 = b0 * a01[3], b0 = q4*/       \
        "subs	%w[tail], %w[tail], #1\n"               /* tail--*/       \
        "fmla	v16.4s,  v5.4s,  v0.s[0]\n"             /* out8 = b1 * a00[0], b1 = q5*/       \
        "fmla	v17.4s,  v5.4s,  v0.s[1]\n"             /* out9 = b1 * a00[1], b1 = q5*/       \
        "fmla	v18.4s,  v5.4s,  v0.s[2]\n"             /* out10 = b1 * a00[2], b1 = q5*/       \
        "fmla	v19.4s,  v5.4s,  v0.s[3]\n"             /* out11 = b1 * a00[3], b1 = q5*/       \
        "fmla	v20.4s,  v5.4s,  v1.s[0]\n"             /* out12 = b1 * a01[0], b1 = q5*/       \
        "fmla	v21.4s,  v5.4s,  v1.s[1]\n"             /* out13 = b1 * a01[1], b1 = q5*/       \
        "fmla	v22.4s,  v5.4s,  v1.s[2]\n"             /* out14 = b1 * a01[2], b1 = q5*/       \
        "fmla	v23.4s,  v5.4s,  v1.s[3]\n"             /* out15 = b1 * a01[3], b1 = q5*/       \
        "ldp	q4, q5, [%[b_ptr]], #32\n"              /* load b1, b2 to q4, q5*/       \
        "fmla	v24.4s,  v6.4s,  v0.s[0]\n"             /* out16 = b2 * a00[0], b2 = q6*/       \
        "fmla	v25.4s,  v6.4s,  v0.s[1]\n"             /* out17 = b2 * a00[1], b2 = q6*/       \
        "fmla	v26.4s,  v6.4s,  v0.s[2]\n"             /* out18 = b2 * a00[2], b2 = q6*/       \
        "fmla	v27.4s,  v6.4s,  v0.s[3]\n"             /* out19 = b2 * a00[3], b2 = q6*/       \
        "fmla	v28.4s,  v6.4s,  v1.s[0]\n"             /* out20 = b2 * a00[0], b2 = q6*/       \
        "fmla	v29.4s,  v6.4s,  v1.s[1]\n"             /* out21 = b2 * a00[1], b2 = q6*/       \
        "fmla	v30.4s,  v6.4s,  v1.s[2]\n"             /* out22 = b2 * a00[2], b2 = q6*/       \
        "fmla	v31.4s,  v6.4s,  v1.s[3]\n"             /* out23 = b2 * a00[3], b2 = q6*/       \
        "beq		4f\n"                               /*jump to tail = 2*/       \
        /* unrool 1, tail > 2*/       \
        "ldp	q0, q1, [%[a_ptr]], #32\n"              /* load a00, a01 to q0, q1*/       \
        "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n"             /* out0 = b0 * a10[0], b0 = q7*/       \
        "fmla	v9.4s ,  v7.4s,  v2.s[1]\n"             /* out1 = b0 * a10[1], b0 = q7*/       \
        "fmla	v10.4s,  v7.4s,  v2.s[2]\n"             /* out2 = b0 * a10[2], b0 = q7*/       \
        "fmla	v11.4s,  v7.4s,  v2.s[3]\n"             /* out3 = b0 * a10[3], b0 = q7*/       \
        "fmla 	v12.4s,  v7.4s,  v3.s[0]\n"             /* out4 = b0 * a11[0], b0 = q7*/       \
        "fmla   v13.4s,  v7.4s,  v3.s[1]\n"             /* out5 = b0 * a11[1], b0 = q7*/       \
        "fmla	v14.4s,  v7.4s,  v3.s[2]\n"             /* out6 = b0 * a11[2], b0 = q7*/       \
        "fmla	v15.4s,  v7.4s,  v3.s[3]\n"             /* out7 = b0 * a11[3], b0 = q7*/       \
        "ldp	q6, q7, [%[b_ptr]], #32\n"              /* load b0, b1 to q6, q7*/       \
        "fmla	v16.4s,  v4.4s,  v2.s[0]\n"             /* out8 = b0 * a10[0], b1 = q4*/       \
        "fmla	v17.4s,  v4.4s,  v2.s[1]\n"             /* out9 = b0 * a10[1], b1 = q4*/       \
        "fmla	v18.4s,  v4.4s,  v2.s[2]\n"             /* out10 = b1 * a10[2], b1 = q4*/       \
        "fmla	v19.4s,  v4.4s,  v2.s[3]\n"             /* out11 = b1 * a10[3], b1 = q4*/       \
        "fmla	v20.4s,  v4.4s,  v3.s[0]\n"             /* out12 = b1 * a10[0], b1 = q4*/       \
        "fmla	v21.4s,  v4.4s,  v3.s[1]\n"             /* out13 = b1 * a10[1], b1 = q4*/       \
        "fmla	v22.4s,  v4.4s,  v3.s[2]\n"             /* out14 = b1 * a10[2], b1 = q4*/       \
        "fmla	v23.4s,  v4.4s,  v3.s[3]\n"             /* out15 = b1 * a10[3], b1 = q4*/       \
        "subs	%w[tail], %w[tail], #1\n"               /* tail--*/       \
        "fmla	v24.4s,  v5.4s,  v2.s[0]\n"             /* out16 = b2 * a10[0], b2 = q5*/       \
        "fmla	v25.4s,  v5.4s,  v2.s[1]\n"             /* out17 = b2 * a10[0], b2 = q5*/       \
        "fmla	v26.4s,  v5.4s,  v2.s[2]\n"             /* out18 = b2 * a10[0], b2 = q5*/       \
        "fmla	v27.4s,  v5.4s,  v2.s[3]\n"             /* out19 = b2 * a10[0], b2 = q5*/       \
        "fmla	v28.4s,  v5.4s,  v3.s[0]\n"             /* out20 = b2 * a10[0], b2 = q5*/       \
        "fmla	v29.4s,  v5.4s,  v3.s[1]\n"             /* out21 = b2 * a10[0], b2 = q5*/       \
        "fmla	v30.4s,  v5.4s,  v3.s[2]\n"             /* out22 = b2 * a10[0], b2 = q5*/       \
        "fmla	v31.4s,  v5.4s,  v3.s[3]\n"             /* out23 = b2 * a10[0], b2 = q5*/       \
        "beq		5f\n"                               /*jump to tail = 3*/       \
        /* unrool 2, tail = 4*/       \
        "ldp	q4, q5, [%[b_ptr]], #32\n"              /* load b2, b0 to q4, q5*/       \
        "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n"             /* out0 = b0 * a00[0], b0 = q6*/       \
        "fmla  	v9.4s ,  v6.4s,  v0.s[1]\n"             /* out1 = b0 * a00[1], b0 = q6*/       \
        "ldp	q2, q3, [%[a_ptr]], #32\n"              /* load a10, a11 to q3, q4*/       \
        "fmla	v10.4s,  v6.4s,  v0.s[2]\n"             /* out2 = b0 * a00[2], b0 = q6*/       \
        "fmla	v11.4s,  v6.4s,  v0.s[3]\n"             /* out3 = b0 * a00[3], b0 = q6*/       \
        "fmla 	v12.4s,  v6.4s,  v1.s[0]\n"             /* out4 = b0 * a01[0], b0 = q6*/       \
        "fmla	v13.4s,  v6.4s,  v1.s[1]\n"             /* out5 = b0 * a01[1], b0 = q6*/       \
        "fmla	v14.4s,  v6.4s,  v1.s[2]\n"             /* out6 = b0 * a01[2], b0 = q6*/       \
        "fmla	v15.4s,  v6.4s,  v1.s[3]\n"             /* out7 = b0 * a01[3], b0 = q6*/       \
        "fmla	v16.4s,  v7.4s,  v0.s[0]\n"             /* out8 = b1 * a00[0], b1 = q7*/       \
        "fmla	v17.4s,  v7.4s,  v0.s[1]\n"             /* out9 = b1 * a00[1], b1 = q7*/       \
        "fmla	v18.4s,  v7.4s,  v0.s[2]\n"             /* out10 = b1 * a00[2], b1 = q7*/       \
        "fmla	v19.4s,  v7.4s,  v0.s[3]\n"             /* out11 = b1 * a00[3], b1 = q7*/       \
        "fmla	v20.4s,  v7.4s,  v1.s[0]\n"             /* out12 = b1 * a01[0], b1 = q7*/       \
        "fmla	v21.4s,  v7.4s,  v1.s[1]\n"             /* out13 = b1 * a01[1], b1 = q7*/       \
        "fmla	v22.4s,  v7.4s,  v1.s[2]\n"             /* out14 = b1 * a01[2], b1 = q7*/       \
        "fmla	v23.4s,  v7.4s,  v1.s[3]\n"             /* out15 = b1 * a01[3], b1 = q7*/       \
        "ldp	q6, q7, [%[b_ptr]], #32\n"              /* load b1, b2 to q6, q7*/       \
        "fmla	v24.4s,  v4.4s,  v0.s[0]\n"             /* out16 = b2 * a00[0], b2 = q4*/       \
        "fmla	v25.4s,  v4.4s,  v0.s[1]\n"             /* out17 = b2 * a00[1], b2 = q4*/       \
        "fmla	v26.4s,  v4.4s,  v0.s[2]\n"             /* out18 = b2 * a00[2], b2 = q4*/       \
        "fmla	v27.4s,  v4.4s,  v0.s[3]\n"             /* out19 = b2 * a00[3], b2 = q4*/       \
        "fmla	v28.4s,  v4.4s,  v1.s[0]\n"             /* out20 = b2 * a00[0], b2 = q4*/       \
        "fmla	v29.4s,  v4.4s,  v1.s[1]\n"             /* out21 = b2 * a00[1], b2 = q4*/       \
        "fmla	v30.4s,  v4.4s,  v1.s[2]\n"             /* out22 = b2 * a00[2], b2 = q4*/       \
        "fmla	v31.4s,  v4.4s,  v1.s[3]\n"             /* out23 = b2 * a00[3], b2 = q4*/

#define GEMM_IN \
        "movi	v8.4s, #0x0\n"                          /* out0 = 0 */       \
        "ldp	q0, q1, [%[a_ptr]], #32\n"              /* load a00,a01 to q0, q1*/       \
        "movi	v9.4s, #0x0\n"                          /* out1 = 0*/       \
        "ldp	q4, q5, [%[b_ptr]], #32\n"              /* load b0, b1 to q4, q5*/       \
        "movi	v10.4s, #0x0\n"                         /* out2 = 0*/       \
        "movi	v11.4s, #0x0\n"                         /* out3 = 0*/       \
        "movi	v12.4s, #0x0\n"                         /* out4 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #64]\n"           /* preload b*/       \
        "movi	v13.4s, #0x0\n"                         /* out5 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #64]\n"           /* preload a*/       \
        "movi	v14.4s, #0x0\n"                         /* out6 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #128]\n"          /* preload b*/       \
        "movi	v15.4s, #0x0\n"                         /* out7 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #128]\n"          /* preload a*/       \
        "movi	v16.4s, #0x0\n"                         /* out8 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #192]\n"          /* preload b*/       \
        "movi	v17.4s, #0x0\n"                         /* out9 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #256]\n"          /* preload b*/       \
        "movi	v18.4s, #0x0\n"                         /* out10 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #192]\n"          /* preload a*/       \
        "movi	v19.4s, #0x0\n"                         /* out11 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #320]\n"          /* preload b*/       \
        "movi	v20.4s, #0x0\n"                         /* out12 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #256]\n"          /* preload a*/       \
        "movi	v21.4s, #0x0\n"                         /* out13 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #384]\n"          /* preload b*/       \
        "movi	v22.4s, #0x0\n"                         /* out14 = 0*/       \
        "movi	v23.4s, #0x0\n"                         /* out15 = 0*/       \
        "movi	v24.4s, #0x0\n"                         /* out16 = 0*/       \
        "movi	v25.4s, #0x0\n"                         /* out17 = 0*/       \
        "movi	v26.4s, #0x0\n"                         /* out18 = 0*/       \
        "movi	v27.4s, #0x0\n"                         /* out19 = 0*/       \
        "movi	v28.4s, #0x0\n"                         /* out20 = 0*/       \
        "movi	v29.4s, #0x0\n"                         /* out21 = 0*/       \
        "movi	v30.4s, #0x0\n"                         /* out22 = 0*/       \
        "movi	v31.4s, #0x0\n"                         /* out23 = 0*/

#define GEMM_IN_BIAS \
        "ldp	q2, q3, [%[bias_ptr]]\n"                /* load bias to q2, q3*/       \
        "ldp	q0, q1, [%[a_ptr]], #32\n"              /* load a00,a01 to q0, q1*/       \
        "ldp	q4, q5, [%[b_ptr]], #32\n"              /* load b0, b1 to q4, q5*/       \
        "dup	v8.4s,  v2.s[0]\n"                      /* out0 = 0 */       \
        "dup	v9.4s,  v2.s[1]\n"                      /* out1 = 0*/       \
        "dup	v10.4s, v2.s[2]\n"                         /* out2 = 0*/       \
        "dup	v11.4s, v2.s[3]\n"                         /* out3 = 0*/       \
        "dup	v12.4s, v3.s[0]\n"                         /* out4 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #64]\n"           /* preload b*/       \
        "dup	v13.4s, v3.s[1]\n"                         /* out5 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #64]\n"           /* preload a*/       \
        "dup	v14.4s, v3.s[2]\n"                         /* out6 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #128]\n"          /* preload b*/       \
        "dup	v15.4s, v3.s[3]\n"                         /* out7 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #128]\n"          /* preload a*/       \
        "dup	v16.4s, v2.s[0]\n"                         /* out8 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #192]\n"          /* preload b*/       \
        "dup	v17.4s, v2.s[1]\n"                         /* out9 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #256]\n"          /* preload b*/       \
        "dup	v18.4s, v2.s[2]\n"                         /* out10 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #192]\n"          /* preload a*/       \
        "dup	v19.4s, v2.s[3]\n"                         /* out11 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #320]\n"          /* preload b*/       \
        "dup	v20.4s, v3.s[0]\n"                         /* out12 = 0*/       \
        "prfm   pldl1keep, [%[a_ptr], #256]\n"          /* preload a*/       \
        "dup	v21.4s, v3.s[1]\n"                         /* out13 = 0*/       \
        "prfm   pldl1keep, [%[b_ptr], #384]\n"          /* preload b*/       \
        "dup	v22.4s, v3.s[2]\n"                         /* out14 = 0*/       \
        "dup	v23.4s, v3.s[3]\n"                         /* out15 = 0*/       \
        "dup	v24.4s, v2.s[0]\n"                         /* out16 = 0*/       \
        "dup	v25.4s, v2.s[1]\n"                         /* out17 = 0*/       \
        "dup	v26.4s, v2.s[2]\n"                         /* out18 = 0*/       \
        "dup	v27.4s, v2.s[3]\n"                         /* out19 = 0*/       \
        "dup	v28.4s, v3.s[0]\n"                         /* out20 = 0*/       \
        "dup	v29.4s, v3.s[1]\n"                         /* out21 = 0*/       \
        "dup	v30.4s, v3.s[2]\n"                         /* out22 = 0*/       \
        "dup	v31.4s, v3.s[3]\n"                         /* out23 = 0*/

#define GEMM_OUT \
        /* unrool 3, tail = 4*/     \
        "fmla 	v8.4s ,  v5.4s,  v2.s[0]\n"             /* out0 = b0 * a10[0], b0 = q5*/       \
        "fmla	v16.4s,  v6.4s,  v2.s[0]\n"             /* out8 = b0 * a10[0], b1 = q6*/       \
        "fmla	v24.4s,  v7.4s,  v2.s[0]\n"             /* out16 = b2 * a10[0], b2 = q7*/       \
        "fmla	v9.4s ,  v5.4s,  v2.s[1]\n"             /* out1 = b0 * a10[1], b0 = q5*/       \
        "str    q8, [%[c_ptr0]], #16\n"                 /* write out0 to r00*/       \
        "fmla	v17.4s,  v6.4s,  v2.s[1]\n"             /* out9 = b1 * a10[1], b1 = q6*/       \
        "str    q16, [%[c_ptr0]], #16\n"                /* write out8 to r01*/       \
        "fmla	v25.4s,  v7.4s,  v2.s[1]\n"             /* out17 = b2 * a10[0], b2 = q7*/       \
        "str    q24, [%[c_ptr0]], #16\n"                /* write out 16 to r02*/       \
        "fmla	v10.4s,  v5.4s,  v2.s[2]\n"             /* out2 = b0 * a10[2], b0 = q5*/       \
        "str    q9, [%[c_ptr1]], #16\n"                 /* write out1 to r10*/       \
        "fmla	v18.4s,  v6.4s,  v2.s[2]\n"             /* out10 = b1 * a10[2], b1 = q6*/       \
        "str    q17, [%[c_ptr1]], #16\n"                /* write out9 to r11*/       \
        "fmla	v26.4s,  v7.4s,  v2.s[2]\n"             /* out18 = b2 * a10[0], b2 = q7*/       \
        "str    q25, [%[c_ptr1]], #16\n"                /* write out17 to r12*/       \
        "fmla	v11.4s,  v5.4s,  v2.s[3]\n"             /* out3 = b0 * a10[3], b0 = q5*/       \
        "str    q10, [%[c_ptr2]], #16\n"                /* write out2 to r20*/       \
        "fmla	v19.4s,  v6.4s,  v2.s[3]\n"             /* out11 = b1 * a10[3], b1 = q6*/       \
        "str    q18, [%[c_ptr2]], #16\n"                /* write out10 to r21*/       \
        "fmla	v27.4s,  v7.4s,  v2.s[3]\n"             /* out19 = b2 * a10[0], b2 = q7*/       \
        "str    q26, [%[c_ptr2]], #16\n"                /* write out18 to r22*/       \
        "fmla 	v12.4s,  v5.4s,  v3.s[0]\n"             /* out4 = b0 * a11[0], b0 = q5*/       \
        "str    q11, [%[c_ptr3]], #16\n"                /* write out3 to r30*/       \
        "fmla	v20.4s,  v6.4s,  v3.s[0]\n"             /* out12 = b1 * a10[0], b1 = q6*/       \
        "str    q19, [%[c_ptr3]], #16\n"                /* write out11 to r31*/       \
        "fmla	v28.4s,  v7.4s,  v3.s[0]\n"             /* out20 = b2 * a10[0], b2 = q7*/       \
        "str    q27, [%[c_ptr3]], #16\n"                /* write out19 to r32*/       \
        "fmla   v13.4s,  v5.4s,  v3.s[1]\n"             /* out5 = b0 * a11[1], b0 = q5*/       \
        "str    q12, [%[c_ptr4]], #16\n"                /* write out4 to r40*/       \
        "fmla	v21.4s,  v6.4s,  v3.s[1]\n"             /* out13 = b1 * a10[1], b1 = q6*/       \
        "str    q20, [%[c_ptr4]], #16\n"                /* write out12 to r41*/       \
        "fmla	v29.4s,  v7.4s,  v3.s[1]\n"             /* out21 = b2 * a10[0], b2 = q7*/       \
        "str    q28, [%[c_ptr4]], #16\n"                /* write out20 to r42*/       \
        "fmla	v14.4s,  v5.4s,  v3.s[2]\n"             /* out6 = b0 * a11[2], b0 = q5*/       \
        "str    q13, [%[c_ptr5]], #16\n"                /* write out5 to r50*/       \
        "fmla	v22.4s,  v6.4s,  v3.s[2]\n"             /* out14 = b1 * a10[2], b1 = q6*/       \
        "str    q21, [%[c_ptr5]], #16\n"                /* write out13 to r51*/       \
        "fmla	v30.4s,  v7.4s,  v3.s[2]\n"             /* out22 = b2 * a10[0], b2 = q7*/       \
        "str    q29, [%[c_ptr5]], #16\n"                /* write out21 to r52*/       \
        "fmla	v15.4s,  v5.4s,  v3.s[3]\n"             /* out7 = b0 * a11[3], b0 = q5*/       \
        "str    q14, [%[c_ptr6]], #16\n"                /* write out6 to r60*/       \
        "fmla	v23.4s,  v6.4s,  v3.s[3]\n"             /* out15 = b1 * a10[3], b1 = q6*/       \
        "str    q22, [%[c_ptr6]], #16\n"                /* write out14 to r61*/       \
        "fmla	v31.4s,  v7.4s,  v3.s[3]\n"             /* out23 = b2 * a10[0], b2 = q7*/       \
        "str    q30, [%[c_ptr6]], #16\n"                /* write out22 to r62*/       \
        "b		11f\n"       \
        /* tails==1 final tail*/       \
        "3: \n"                                         /* tail=1*/       \
        "ldr	q6, [%[b_ptr]], #16\n"                  /* load b2 to q6*/       \
        "fmla 	v8.4s ,  v4.4s,  v0.s[0]\n"             /* out0 = b0 * a10[0], b0 = q4*/       \
        "fmla	v16.4s,  v5.4s,  v0.s[0]\n"             /* out8 = b0 * a10[0], b1 = q5*/       \
        "fmla	v24.4s,  v6.4s,  v0.s[0]\n"             /* out16 = b2 * a10[0], b2 = q6*/       \
        "fmla	v9.4s ,  v4.4s,  v0.s[1]\n"             /* out1 = b0 * a10[1], b0 = q4*/       \
        "str    q8, [%[c_ptr0]], #16\n"                 /* write out0 to r00*/       \
        "fmla	v17.4s,  v5.4s,  v0.s[1]\n"             /* out9 = b1 * a10[1], b1 = q5*/       \
        "str    q16, [%[c_ptr0]], #16\n"                /* write out8 to r01*/       \
        "fmla	v25.4s,  v6.4s,  v0.s[1]\n"             /* out17 = b2 * a10[0], b2 = q6*/       \
        "str    q24, [%[c_ptr0]], #16\n"                /* write out 16 to r02*/       \
        "fmla	v10.4s,  v4.4s,  v0.s[2]\n"             /* out2 = b0 * a10[2], b0 = q4*/       \
        "str    q9, [%[c_ptr1]], #16\n"                 /* write out1 to r10*/       \
        "fmla	v18.4s,  v5.4s,  v0.s[2]\n"             /* out10 = b1 * a10[2], b1 = q5*/       \
        "str    q17, [%[c_ptr1]], #16\n"                /* write out9 to r11*/       \
        "fmla	v26.4s,  v6.4s,  v0.s[2]\n"             /* out18 = b2 * a10[0], b2 = q6*/       \
        "str    q25, [%[c_ptr1]], #16\n"                /* write out17 to r12*/       \
        "fmla	v11.4s,  v4.4s,  v0.s[3]\n"             /* out3 = b0 * a10[3], b0 = q4*/       \
        "str    q10, [%[c_ptr2]], #16\n"                /* write out2 to r20*/       \
        "fmla	v19.4s,  v5.4s,  v0.s[3]\n"             /* out11 = b1 * a10[3], b1 = q5*/       \
        "str    q18, [%[c_ptr2]], #16\n"                /* write out10 to r21*/       \
        "fmla	v27.4s,  v6.4s,  v0.s[3]\n"             /* out19 = b2 * a10[0], b2 = q6*/       \
        "str    q26, [%[c_ptr2]], #16\n"                /* write out18 to r22*/       \
        "fmla 	v12.4s,  v4.4s,  v1.s[0]\n"             /* out4 = b0 * a11[0], b0 = q4*/       \
        "str    q11, [%[c_ptr3]], #16\n"                /* write out3 to r30*/       \
        "fmla	v20.4s,  v5.4s,  v1.s[0]\n"             /* out12 = b1 * a10[0], b1 = q5*/       \
        "str    q19, [%[c_ptr3]], #16\n"                /* write out11 to r31*/       \
        "fmla	v28.4s,  v6.4s,  v1.s[0]\n"             /* out20 = b2 * a10[0], b2 = q6*/       \
        "str    q27, [%[c_ptr3]], #16\n"                /* write out19 to r32*/       \
        "fmla   v13.4s,  v4.4s,  v1.s[1]\n"             /* out5 = b0 * a11[1], b0 = q4*/       \
        "str    q12, [%[c_ptr4]], #16\n"                /* write out4 to r40*/       \
        "fmla	v21.4s,  v5.4s,  v1.s[1]\n"             /* out13 = b1 * a10[1], b1 = q5*/       \
        "str    q20, [%[c_ptr4]], #16\n"                /* write out12 to r41*/       \
        "fmla	v29.4s,  v6.4s,  v1.s[1]\n"             /* out21 = b2 * a10[0], b2 = q6*/       \
        "str    q28, [%[c_ptr4]], #16\n"                /* write out20 to r42*/       \
        "fmla	v14.4s,  v4.4s,  v1.s[2]\n"             /* out6 = b0 * a11[2], b0 = q4*/       \
        "str    q13, [%[c_ptr5]], #16\n"                /* write out5 to r50*/       \
        "fmla	v22.4s,  v5.4s,  v1.s[2]\n"             /* out14 = b1 * a10[2], b1 = q5*/       \
        "str    q21, [%[c_ptr5]], #16\n"                /* write out13 to r51*/       \
        "fmla	v30.4s,  v6.4s,  v1.s[2]\n"             /* out22 = b2 * a10[0], b2 = q6*/       \
        "str    q29, [%[c_ptr5]], #16\n"                /* write out21 to r52*/       \
        "fmla	v15.4s,  v4.4s,  v1.s[3]\n"             /* out7 = b0 * a11[3], b0 = q4*/       \
        "str    q14, [%[c_ptr6]], #16\n"                /* write out6 to r60*/       \
        "fmla	v23.4s,  v5.4s,  v1.s[3]\n"             /* out15 = b1 * a10[3], b1 = q5*/       \
        "str    q22, [%[c_ptr6]], #16\n"                /* write out14 to r61*/       \
        "fmla	v31.4s,  v6.4s,  v1.s[3]\n"             /* out23 = b2 * a10[0], b2 = q6*/       \
        "str    q30, [%[c_ptr6]], #16\n"                /* write out22 to r62*/       \
        "b		11f\n"                                  /* jump to end*/       \
        /* tails==2 final tail*/       \
        "4:\n"                                          /* tail = 2*/       \
        "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n"             /* out0 = b0 * a10[0], b0 = q7*/       \
        "fmla	v16.4s,  v4.4s,  v2.s[0]\n"             /* out8 = b0 * a10[0], b1 = q4*/       \
        "fmla	v24.4s,  v5.4s,  v2.s[0]\n"             /* out16 = b2 * a10[0], b2 = q5*/       \
        "fmla	v9.4s ,  v7.4s,  v2.s[1]\n"             /* out1 = b0 * a10[1], b0 = q7*/       \
        "str    q8, [%[c_ptr0]], #16\n"                 /* write out0 to r00*/       \
        "fmla	v17.4s,  v4.4s,  v2.s[1]\n"             /* out9 = b0 * a10[1], b1 = q4*/       \
        "str    q16, [%[c_ptr0]], #16\n"                /* write out8 to r01*/       \
        "fmla	v25.4s,  v5.4s,  v2.s[1]\n"             /* out17 = b2 * a10[0], b2 = q5*/       \
        "str    q24, [%[c_ptr0]], #16\n"                /* write out 16 to r02*/       \
        "fmla	v10.4s,  v7.4s,  v2.s[2]\n"             /* out2 = b0 * a10[2], b0 = q7*/       \
        "str    q9, [%[c_ptr1]], #16\n"                 /* write out1 to r10*/       \
        "fmla	v18.4s,  v4.4s,  v2.s[2]\n"             /* out10 = b1 * a10[2], b1 = q4*/       \
        "str    q17, [%[c_ptr1]], #16\n"                /* write out9 to r11*/       \
        "fmla	v26.4s,  v5.4s,  v2.s[2]\n"             /* out18 = b2 * a10[0], b2 = q5*/       \
        "str    q25, [%[c_ptr1]], #16\n"                /* write out17 to r12*/       \
        "fmla	v11.4s,  v7.4s,  v2.s[3]\n"             /* out3 = b0 * a10[3], b0 = q7*/       \
        "str    q10, [%[c_ptr2]], #16\n"                /* write out2 to r20*/       \
        "fmla	v19.4s,  v4.4s,  v2.s[3]\n"             /* out11 = b1 * a10[3], b1 = q4*/       \
        "str    q18, [%[c_ptr2]], #16\n"                /* write out10 to r21*/       \
        "fmla	v27.4s,  v5.4s,  v2.s[3]\n"             /* out19 = b2 * a10[0], b2 = q5*/       \
        "str    q26, [%[c_ptr2]], #16\n"                /* write out18 to r22*/       \
        "fmla 	v12.4s,  v7.4s,  v3.s[0]\n"             /* out4 = b0 * a11[0], b0 = q7*/       \
        "str    q11, [%[c_ptr3]], #16\n"                /* write out3 to r30*/       \
        "fmla	v20.4s,  v4.4s,  v3.s[0]\n"             /* out12 = b1 * a10[0], b1 = q4*/       \
        "str    q19, [%[c_ptr3]], #16\n"                /* write out11 to r31*/       \
        "fmla	v28.4s,  v5.4s,  v3.s[0]\n"             /* out20 = b2 * a10[0], b2 = q5*/       \
        "str    q27, [%[c_ptr3]], #16\n"                /* write out19 to r32*/       \
        "fmla   v13.4s,  v7.4s,  v3.s[1]\n"             /* out5 = b0 * a11[1], b0 = q7*/       \
        "str    q12, [%[c_ptr4]], #16\n"                /* write out4 to r40*/       \
        "fmla	v21.4s,  v4.4s,  v3.s[1]\n"             /* out13 = b1 * a10[1], b1 = q4*/       \
        "str    q20, [%[c_ptr4]], #16\n"                /* write out12 to r41*/       \
        "fmla	v29.4s,  v5.4s,  v3.s[1]\n"             /* out21 = b2 * a10[0], b2 = q5*/       \
        "str    q28, [%[c_ptr4]], #16\n"                /* write out20 to r42*/       \
        "fmla	v14.4s,  v7.4s,  v3.s[2]\n"             /* out6 = b0 * a11[2], b0 = q7*/       \
        "str    q13, [%[c_ptr5]], #16\n"                /* write out5 to r50*/       \
        "fmla	v22.4s,  v4.4s,  v3.s[2]\n"             /* out14 = b1 * a10[2], b1 = q4*/       \
        "str    q21, [%[c_ptr5]], #16\n"                /* write out13 to r51*/       \
        "fmla	v30.4s,  v5.4s,  v3.s[2]\n"             /* out22 = b2 * a10[0], b2 = q5*/       \
        "str    q29, [%[c_ptr5]], #16\n"                /* write out21 to r52*/       \
        "fmla	v15.4s,  v7.4s,  v3.s[3]\n"             /* out7 = b0 * a11[3], b0 = q7*/       \
        "str    q14, [%[c_ptr6]], #16\n"                /* write out6 to r60*/       \
        "fmla	v23.4s,  v4.4s,  v3.s[3]\n"             /* out15 = b1 * a10[3], b1 = q4*/       \
        "str    q22, [%[c_ptr6]], #16\n"                /* write out14 to r61*/       \
        "fmla	v31.4s,  v5.4s,  v3.s[3]\n"             /* out23 = b2 * a10[0], b2 = q5*/       \
        "str    q30, [%[c_ptr6]], #16\n"                /* write out22 to r62*/       \
        "b		11f\n"                                  /* jump to end*/       \
        /* tails==3 final tail*/       \
        "5:\n"                                          /* tail = 3*/       \
        "ldr	q4, [%[b_ptr]], #16\n"                  /* load b2, b0 to q4*/       \
        "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n"             /* out0 = b0 * a00[0], b0 = q6*/       \
        "fmla	v16.4s,  v7.4s,  v0.s[0]\n"             /* out8 = b1 * a00[0], b1 = q7*/       \
        "fmla	v24.4s,  v4.4s,  v0.s[0]\n"             /* out16 = b2 * a00[0], b2 = q4*/       \
        "fmla  	v9.4s ,  v6.4s,  v0.s[1]\n"             /* out1 = b0 * a00[1], b0 = q6*/       \
        "str    q8, [%[c_ptr0]], #16\n"                 /* write out0 to r00*/       \
        "fmla	v17.4s,  v7.4s,  v0.s[1]\n"             /* out9 = b1 * a00[1], b1 = q7*/       \
        "str    q16, [%[c_ptr0]], #16\n"                /* write out8 to r01*/       \
        "fmla	v25.4s,  v4.4s,  v0.s[1]\n"             /* out17 = b2 * a00[1], b2 = q4*/       \
        "str    q24, [%[c_ptr0]], #16\n"                /* write out 16 to r02*/       \
        "fmla	v10.4s,  v6.4s,  v0.s[2]\n"             /* out2 = b0 * a00[2], b0 = q6*/       \
        "str    q9, [%[c_ptr1]], #16\n"                 /* write out1 to r10*/       \
        "fmla	v18.4s,  v7.4s,  v0.s[2]\n"             /* out10 = b1 * a00[2], b1 = q7*/       \
        "str    q17, [%[c_ptr1]], #16\n"                /* write out9 to r11*/       \
        "fmla	v26.4s,  v4.4s,  v0.s[2]\n"             /* out18 = b2 * a00[2], b2 = q4*/       \
        "str    q25, [%[c_ptr1]], #16\n"                /* write out17 to r12*/       \
        "fmla	v11.4s,  v6.4s,  v0.s[3]\n"             /* out3 = b0 * a00[3], b0 = q6*/       \
        "str    q10, [%[c_ptr2]], #16\n"                /* write out2 to r20*/       \
        "fmla	v19.4s,  v7.4s,  v0.s[3]\n"             /* out11 = b1 * a00[3], b1 = q7*/       \
        "str    q18, [%[c_ptr2]], #16\n"                /* write out10 to r21*/       \
        "fmla	v27.4s,  v4.4s,  v0.s[3]\n"             /* out19 = b2 * a00[3], b2 = q4*/       \
        "str    q26, [%[c_ptr2]], #16\n"                /* write out18 to r22*/       \
        "fmla 	v12.4s,  v6.4s,  v1.s[0]\n"             /* out4 = b0 * a01[0], b0 = q6*/       \
        "str    q11, [%[c_ptr3]], #16\n"                /* write out3 to r30*/       \
        "fmla	v20.4s,  v7.4s,  v1.s[0]\n"             /* out12 = b1 * a01[0], b1 = q7*/       \
        "str    q19, [%[c_ptr3]], #16\n"                /* write out11 to r31*/       \
        "fmla	v28.4s,  v4.4s,  v1.s[0]\n"             /* out20 = b2 * a00[0], b2 = q4*/       \
        "str    q27, [%[c_ptr3]], #16\n"                /* write out19 to r32*/       \
        "fmla	v13.4s,  v6.4s,  v1.s[1]\n"             /* out5 = b0 * a01[1], b0 = q6*/       \
        "str    q12, [%[c_ptr4]], #16\n"                /* write out4 to r40*/       \
        "fmla	v21.4s,  v7.4s,  v1.s[1]\n"             /* out13 = b1 * a01[1], b1 = q7*/       \
        "str    q20, [%[c_ptr4]], #16\n"                /* write out12 to r41*/       \
        "fmla	v29.4s,  v4.4s,  v1.s[1]\n"             /* out21 = b2 * a00[1], b2 = q4*/       \
        "str    q28, [%[c_ptr4]], #16\n"                /* write out20 to r42*/       \
        "fmla	v14.4s,  v6.4s,  v1.s[2]\n"             /* out6 = b0 * a01[2], b0 = q6*/       \
        "str    q13, [%[c_ptr5]], #16\n"                /* write out5 to r50*/       \
        "fmla	v22.4s,  v7.4s,  v1.s[2]\n"             /* out14 = b1 * a01[2], b1 = q7*/       \
        "str    q21, [%[c_ptr5]], #16\n"                /* write out13 to r51*/       \
        "fmla	v30.4s,  v4.4s,  v1.s[2]\n"             /* out22 = b2 * a00[2], b2 = q4*/       \
        "str    q29, [%[c_ptr5]], #16\n"                /* write out21 to r52*/       \
        "fmla	v15.4s,  v6.4s,  v1.s[3]\n"             /* out7 = b0 * a01[3], b0 = q6*/       \
        "str    q14, [%[c_ptr6]], #16\n"                /* write out6 to r60*/       \
        "fmla	v23.4s,  v7.4s,  v1.s[3]\n"             /* out15 = b1 * a01[3], b1 = q7*/       \
        "str    q22, [%[c_ptr6]], #16\n"                /* write out14 to r61*/       \
        "fmla	v31.4s,  v4.4s,  v1.s[3]\n"             /* out23 = b2 * a00[3], b2 = q4*/       \
        "str    q30, [%[c_ptr6]], #16\n"                /* write out22 to r62*/       \
        "11: \n"                                        /* end write*/       \
        "str    q15, [%[c_ptr7]], #16\n"                /* write out7 to r70*/       \
        "str    q23, [%[c_ptr7]], #16\n"                /* write out15 to r71*/       \
        "str    q31, [%[c_ptr7]], #16\n"                /* write out23 to r72*/

#define GEMM_OUT_RELU \
        /* unrool 3, tail = 4*/     \
        "fmla 	v8.4s ,  v5.4s,  v2.s[0]\n"             /* out0 = b0 * a10[0], b0 = q5*/       \
        "fmla	v16.4s,  v6.4s,  v2.s[0]\n"             /* out8 = b0 * a10[0], b1 = q6*/       \
        "fmla	v24.4s,  v7.4s,  v2.s[0]\n"             /* out16 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v9.4s ,  v5.4s,  v2.s[1]\n"             /* out1 = b0 * a10[1], b0 = q5*/       \
        "fmla	v17.4s,  v6.4s,  v2.s[1]\n"             /* out9 = b1 * a10[1], b1 = q6*/       \
        "fmla	v25.4s,  v7.4s,  v2.s[1]\n"             /* out17 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v10.4s,  v5.4s,  v2.s[2]\n"             /* out2 = b0 * a10[2], b0 = q5*/       \
        "fmla	v18.4s,  v6.4s,  v2.s[2]\n"             /* out10 = b1 * a10[2], b1 = q6*/       \
        "fmla	v26.4s,  v7.4s,  v2.s[2]\n"             /* out18 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v11.4s,  v5.4s,  v2.s[3]\n"             /* out3 = b0 * a10[3], b0 = q5*/       \
        "fmla	v19.4s,  v6.4s,  v2.s[3]\n"             /* out11 = b1 * a10[3], b1 = q6*/       \
        "fmla	v27.4s,  v7.4s,  v2.s[3]\n"             /* out19 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla 	v12.4s,  v5.4s,  v3.s[0]\n"             /* out4 = b0 * a11[0], b0 = q5*/       \
        "fmla	v20.4s,  v6.4s,  v3.s[0]\n"             /* out12 = b1 * a10[0], b1 = q6*/       \
        "fmla	v28.4s,  v7.4s,  v3.s[0]\n"             /* out20 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla   v13.4s,  v5.4s,  v3.s[1]\n"             /* out5 = b0 * a11[1], b0 = q5*/       \
        "fmla	v21.4s,  v6.4s,  v3.s[1]\n"             /* out13 = b1 * a10[1], b1 = q6*/       \
        "fmla	v29.4s,  v7.4s,  v3.s[1]\n"             /* out21 = b2 * a10[0], b2 = q7*/       \
        \
        "movi   v1.4s, #0\n"                            /* for relu*/                           \
        "fmax   v8.4s, v8.4s, v1.4s\n"                  /* relu*/                               \
        "fmax   v16.4s,v16.4s,v1.4s\n"                  /* relu*/                               \
        "fmax   v24.4s,v24.4s,v1.4s\n"                  /* relu*/                               \
        \
        "fmla	v14.4s,  v5.4s,  v3.s[2]\n"             /* out6 = b0 * a11[2], b0 = q5*/       \
        "fmla	v22.4s,  v6.4s,  v3.s[2]\n"             /* out14 = b1 * a10[2], b1 = q6*/       \
        "fmla	v30.4s,  v7.4s,  v3.s[2]\n"             /* out22 = b2 * a10[0], b2 = q7*/       \
        \
        "fmax   v9.4s, v9.4s, v1.4s\n"                  /* relu*/                               \
        "fmax   v17.4s,v17.4s,v1.4s\n"                  /* relu*/                               \
        "fmax   v25.4s,v25.4s,v1.4s\n"                  /* relu*/                               \
        \
        "fmla	v15.4s,  v5.4s,  v3.s[3]\n"             /* out7 = b0 * a11[3], b0 = q5*/       \
        "fmla	v23.4s,  v6.4s,  v3.s[3]\n"             /* out15 = b1 * a10[3], b1 = q6*/       \
        "fmla	v31.4s,  v7.4s,  v3.s[3]\n"             /* out23 = b2 * a10[0], b2 = q7*/       \
        \
        "str    q8, [%[c_ptr0]], #16\n"                 /* write out0 to r00*/       \
        "fmax   v10.4s, v10.4s, v1.4s\n"                /* relu*/                               \
        "str    q16, [%[c_ptr0]], #16\n"                /* write out8 to r01*/       \
        "fmax   v18.4s, v18.4s, v1.4s\n"                /* relu*/                               \
        "str    q24, [%[c_ptr0]], #16\n"                /* write out 16 to r02*/       \
        "fmax   v26.4s, v26.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q9, [%[c_ptr1]], #16\n"                 /* write out1 to r10*/       \
        "fmax   v11.4s, v11.4s, v1.4s\n"                /* relu*/                               \
        "str    q17, [%[c_ptr1]], #16\n"                /* write out9 to r11*/       \
        "fmax   v19.4s, v19.4s, v1.4s\n"                /* relu*/                               \
        "str    q25, [%[c_ptr1]], #16\n"                /* write out17 to r12*/       \
        "fmax   v27.4s, v27.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q10, [%[c_ptr2]], #16\n"                /* write out2 to r20*/       \
        "fmax   v12.4s, v12.4s, v1.4s\n"                /* relu*/                               \
        "str    q18, [%[c_ptr2]], #16\n"                /* write out10 to r21*/       \
        "fmax   v20.4s, v20.4s, v1.4s\n"                /* relu*/                               \
        "str    q26, [%[c_ptr2]], #16\n"                /* write out18 to r22*/       \
        "fmax   v28.4s, v28.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q11, [%[c_ptr3]], #16\n"                /* write out3 to r30*/       \
        "fmax   v13.4s, v13.4s, v1.4s\n"                /* relu*/                               \
        "str    q19, [%[c_ptr3]], #16\n"                /* write out11 to r31*/       \
        "fmax   v21.4s, v21.4s, v1.4s\n"                /* relu*/                               \
        "str    q27, [%[c_ptr3]], #16\n"                /* write out19 to r32*/       \
        "fmax   v29.4s, v29.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q12, [%[c_ptr4]], #16\n"                /* write out4 to r40*/       \
        "fmax   v14.4s, v14.4s, v1.4s\n"                /* relu*/                               \
        "str    q20, [%[c_ptr4]], #16\n"                /* write out12 to r41*/       \
        "fmax   v22.4s, v22.4s, v1.4s\n"                /* relu*/                               \
        "str    q28, [%[c_ptr4]], #16\n"                /* write out20 to r42*/       \
        "fmax   v30.4s, v30.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q13, [%[c_ptr5]], #16\n"                /* write out5 to r50*/       \
        "fmax   v15.4s, v15.4s, v1.4s\n"                /* relu*/                               \
        "str    q21, [%[c_ptr5]], #16\n"                /* write out13 to r51*/       \
        "fmax   v23.4s, v23.4s, v1.4s\n"                /* relu*/                               \
        "str    q29, [%[c_ptr5]], #16\n"                /* write out21 to r52*/       \
        "fmax   v31.4s, v31.4s, v1.4s\n"                /* relu*/                               \
        "str    q14, [%[c_ptr6]], #16\n"                /* write out6 to r60*/       \
        "str    q22, [%[c_ptr6]], #16\n"                /* write out14 to r61*/       \
        "str    q30, [%[c_ptr6]], #16\n"                /* write out22 to r62*/       \
        "b		11f\n"       \
        /* tails==1 final tail*/       \
        "3: \n"                                         /* tail=1*/       \
        "ldr	q6, [%[b_ptr]], #16\n"                  /* load b2 to q6*/       \
        "fmla 	v8.4s ,  v4.4s,  v0.s[0]\n"             /* out0 = b0 * a10[0], b0 = q5*/       \
        "fmla	v16.4s,  v5.4s,  v0.s[0]\n"             /* out8 = b0 * a10[0], b1 = q6*/       \
        "fmla	v24.4s,  v6.4s,  v0.s[0]\n"             /* out16 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v9.4s ,  v4.4s,  v0.s[1]\n"             /* out1 = b0 * a10[1], b0 = q5*/       \
        "fmla	v17.4s,  v5.4s,  v0.s[1]\n"             /* out9 = b1 * a10[1], b1 = q6*/       \
        "fmla	v25.4s,  v6.4s,  v0.s[1]\n"             /* out17 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v10.4s,  v4.4s,  v0.s[2]\n"             /* out2 = b0 * a10[2], b0 = q5*/       \
        "fmla	v18.4s,  v5.4s,  v0.s[2]\n"             /* out10 = b1 * a10[2], b1 = q6*/       \
        "fmla	v26.4s,  v6.4s,  v0.s[2]\n"             /* out18 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v11.4s,  v4.4s,  v0.s[3]\n"             /* out3 = b0 * a10[3], b0 = q5*/       \
        "fmla	v19.4s,  v5.4s,  v0.s[3]\n"             /* out11 = b1 * a10[3], b1 = q6*/       \
        "fmla	v27.4s,  v6.4s,  v0.s[3]\n"             /* out19 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla 	v12.4s,  v4.4s,  v1.s[0]\n"             /* out4 = b0 * a11[0], b0 = q5*/       \
        "fmla	v20.4s,  v5.4s,  v1.s[0]\n"             /* out12 = b1 * a10[0], b1 = q6*/       \
        "fmla	v28.4s,  v6.4s,  v1.s[0]\n"             /* out20 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla   v13.4s,  v4.4s,  v1.s[1]\n"             /* out5 = b0 * a11[1], b0 = q5*/       \
        "fmla	v21.4s,  v5.4s,  v1.s[1]\n"             /* out13 = b1 * a10[1], b1 = q6*/       \
        "fmla	v29.4s,  v6.4s,  v1.s[1]\n"             /* out21 = b2 * a10[0], b2 = q7*/       \
        \
        "movi   v2.4s, #0\n"                            /* for relu*/                           \
        "fmax   v8.4s, v8.4s, v2.4s\n"                  /* relu*/                               \
        "fmax   v16.4s,v16.4s,v2.4s\n"                  /* relu*/                               \
        "fmax   v24.4s,v24.4s,v2.4s\n"                  /* relu*/                               \
        \
        "fmla	v14.4s,  v4.4s,  v1.s[2]\n"             /* out6 = b0 * a11[2], b0 = q5*/       \
        "fmla	v22.4s,  v5.4s,  v1.s[2]\n"             /* out14 = b1 * a10[2], b1 = q6*/       \
        "fmla	v30.4s,  v6.4s,  v1.s[2]\n"             /* out22 = b2 * a10[0], b2 = q7*/       \
        \
        "fmax   v9.4s, v9.4s, v2.4s\n"                  /* relu*/                               \
        "fmax   v17.4s,v17.4s,v2.4s\n"                  /* relu*/                               \
        "fmax   v25.4s,v25.4s,v2.4s\n"                  /* relu*/                               \
        \
        "fmla	v15.4s,  v4.4s,  v1.s[3]\n"             /* out7 = b0 * a11[3], b0 = q5*/       \
        "fmla	v23.4s,  v5.4s,  v1.s[3]\n"             /* out15 = b1 * a10[3], b1 = q6*/       \
        "fmla	v31.4s,  v6.4s,  v1.s[3]\n"             /* out23 = b2 * a10[0], b2 = q7*/       \
        \
        "str    q8, [%[c_ptr0]], #16\n"                 /* write out0 to r00*/       \
        "fmax   v10.4s, v10.4s, v2.4s\n"                /* relu*/                               \
        "str    q16, [%[c_ptr0]], #16\n"                /* write out8 to r01*/       \
        "fmax   v18.4s, v18.4s, v2.4s\n"                /* relu*/                               \
        "str    q24, [%[c_ptr0]], #16\n"                /* write out 16 to r02*/       \
        "fmax   v26.4s, v26.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q9, [%[c_ptr1]], #16\n"                 /* write out1 to r10*/       \
        "fmax   v11.4s, v11.4s, v2.4s\n"                /* relu*/                               \
        "str    q17, [%[c_ptr1]], #16\n"                /* write out9 to r11*/       \
        "fmax   v19.4s, v19.4s, v2.4s\n"                /* relu*/                               \
        "str    q25, [%[c_ptr1]], #16\n"                /* write out17 to r12*/       \
        "fmax   v27.4s, v27.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q10, [%[c_ptr2]], #16\n"                /* write out2 to r20*/       \
        "fmax   v12.4s, v12.4s, v2.4s\n"                /* relu*/                               \
        "str    q18, [%[c_ptr2]], #16\n"                /* write out10 to r21*/       \
        "fmax   v20.4s, v20.4s, v2.4s\n"                /* relu*/                               \
        "str    q26, [%[c_ptr2]], #16\n"                /* write out18 to r22*/       \
        "fmax   v28.4s, v28.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q11, [%[c_ptr3]], #16\n"                /* write out3 to r30*/       \
        "fmax   v13.4s, v13.4s, v2.4s\n"                /* relu*/                               \
        "str    q19, [%[c_ptr3]], #16\n"                /* write out11 to r31*/       \
        "fmax   v21.4s, v21.4s, v2.4s\n"                /* relu*/                               \
        "str    q27, [%[c_ptr3]], #16\n"                /* write out19 to r32*/       \
        "fmax   v29.4s, v29.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q12, [%[c_ptr4]], #16\n"                /* write out4 to r40*/       \
        "fmax   v14.4s, v14.4s, v2.4s\n"                /* relu*/                               \
        "str    q20, [%[c_ptr4]], #16\n"                /* write out12 to r41*/       \
        "fmax   v22.4s, v22.4s, v2.4s\n"                /* relu*/                               \
        "str    q28, [%[c_ptr4]], #16\n"                /* write out20 to r42*/       \
        "fmax   v30.4s, v30.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q13, [%[c_ptr5]], #16\n"                /* write out5 to r50*/       \
        "fmax   v15.4s, v15.4s, v2.4s\n"                /* relu*/                               \
        "str    q21, [%[c_ptr5]], #16\n"                /* write out13 to r51*/       \
        "fmax   v23.4s, v23.4s, v2.4s\n"                /* relu*/                               \
        "str    q29, [%[c_ptr5]], #16\n"                /* write out21 to r52*/       \
        "fmax   v31.4s, v31.4s, v2.4s\n"                /* relu*/                               \
        "str    q14, [%[c_ptr6]], #16\n"                /* write out6 to r60*/       \
        "str    q22, [%[c_ptr6]], #16\n"                /* write out14 to r61*/       \
        "str    q30, [%[c_ptr6]], #16\n"                /* write out22 to r62*/       \
        "b		11f\n"       \
        /* tails==2 final tail*/       \
        "4:\n"                                          /* tail = 2*/       \
        "fmla 	v8.4s ,  v7.4s,  v2.s[0]\n"             /* out0 = b0 * a10[0], b0 = q5*/       \
        "fmla	v16.4s,  v4.4s,  v2.s[0]\n"             /* out8 = b0 * a10[0], b1 = q6*/       \
        "fmla	v24.4s,  v5.4s,  v2.s[0]\n"             /* out16 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v9.4s ,  v7.4s,  v2.s[1]\n"             /* out1 = b0 * a10[1], b0 = q5*/       \
        "fmla	v17.4s,  v4.4s,  v2.s[1]\n"             /* out9 = b1 * a10[1], b1 = q6*/       \
        "fmla	v25.4s,  v5.4s,  v2.s[1]\n"             /* out17 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v10.4s,  v7.4s,  v2.s[2]\n"             /* out2 = b0 * a10[2], b0 = q5*/       \
        "fmla	v18.4s,  v4.4s,  v2.s[2]\n"             /* out10 = b1 * a10[2], b1 = q6*/       \
        "fmla	v26.4s,  v5.4s,  v2.s[2]\n"             /* out18 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v11.4s,  v7.4s,  v2.s[3]\n"             /* out3 = b0 * a10[3], b0 = q5*/       \
        "fmla	v19.4s,  v4.4s,  v2.s[3]\n"             /* out11 = b1 * a10[3], b1 = q6*/       \
        "fmla	v27.4s,  v5.4s,  v2.s[3]\n"             /* out19 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla 	v12.4s,  v7.4s,  v3.s[0]\n"             /* out4 = b0 * a11[0], b0 = q5*/       \
        "fmla	v20.4s,  v4.4s,  v3.s[0]\n"             /* out12 = b1 * a10[0], b1 = q6*/       \
        "fmla	v28.4s,  v5.4s,  v3.s[0]\n"             /* out20 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla   v13.4s,  v7.4s,  v3.s[1]\n"             /* out5 = b0 * a11[1], b0 = q5*/       \
        "fmla	v21.4s,  v4.4s,  v3.s[1]\n"             /* out13 = b1 * a10[1], b1 = q6*/       \
        "fmla	v29.4s,  v5.4s,  v3.s[1]\n"             /* out21 = b2 * a10[0], b2 = q7*/       \
        \
        "movi   v1.4s, #0\n"                            /* for relu*/                           \
        "fmax   v8.4s, v8.4s, v1.4s\n"                  /* relu*/                               \
        "fmax   v16.4s,v16.4s,v1.4s\n"                  /* relu*/                               \
        "fmax   v24.4s,v24.4s,v1.4s\n"                  /* relu*/                               \
        \
        "fmla	v14.4s,  v7.4s,  v3.s[2]\n"             /* out6 = b0 * a11[2], b0 = q5*/       \
        "fmla	v22.4s,  v4.4s,  v3.s[2]\n"             /* out14 = b1 * a10[2], b1 = q6*/       \
        "fmla	v30.4s,  v5.4s,  v3.s[2]\n"             /* out22 = b2 * a10[0], b2 = q7*/       \
        \
        "fmax   v9.4s, v9.4s, v1.4s\n"                  /* relu*/                               \
        "fmax   v17.4s,v17.4s,v1.4s\n"                  /* relu*/                               \
        "fmax   v25.4s,v25.4s,v1.4s\n"                  /* relu*/                               \
        \
        "fmla	v15.4s,  v7.4s,  v3.s[3]\n"             /* out7 = b0 * a11[3], b0 = q5*/       \
        "fmla	v23.4s,  v4.4s,  v3.s[3]\n"             /* out15 = b1 * a10[3], b1 = q6*/       \
        "fmla	v31.4s,  v5.4s,  v3.s[3]\n"             /* out23 = b2 * a10[0], b2 = q7*/       \
        \
        "str    q8, [%[c_ptr0]], #16\n"                 /* write out0 to r00*/       \
        "fmax   v10.4s, v10.4s, v1.4s\n"                /* relu*/                               \
        "str    q16, [%[c_ptr0]], #16\n"                /* write out8 to r01*/       \
        "fmax   v18.4s, v18.4s, v1.4s\n"                /* relu*/                               \
        "str    q24, [%[c_ptr0]], #16\n"                /* write out 16 to r02*/       \
        "fmax   v26.4s, v26.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q9, [%[c_ptr1]], #16\n"                 /* write out1 to r10*/       \
        "fmax   v11.4s, v11.4s, v1.4s\n"                /* relu*/                               \
        "str    q17, [%[c_ptr1]], #16\n"                /* write out9 to r11*/       \
        "fmax   v19.4s, v19.4s, v1.4s\n"                /* relu*/                               \
        "str    q25, [%[c_ptr1]], #16\n"                /* write out17 to r12*/       \
        "fmax   v27.4s, v27.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q10, [%[c_ptr2]], #16\n"                /* write out2 to r20*/       \
        "fmax   v12.4s, v12.4s, v1.4s\n"                /* relu*/                               \
        "str    q18, [%[c_ptr2]], #16\n"                /* write out10 to r21*/       \
        "fmax   v20.4s, v20.4s, v1.4s\n"                /* relu*/                               \
        "str    q26, [%[c_ptr2]], #16\n"                /* write out18 to r22*/       \
        "fmax   v28.4s, v28.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q11, [%[c_ptr3]], #16\n"                /* write out3 to r30*/       \
        "fmax   v13.4s, v13.4s, v1.4s\n"                /* relu*/                               \
        "str    q19, [%[c_ptr3]], #16\n"                /* write out11 to r31*/       \
        "fmax   v21.4s, v21.4s, v1.4s\n"                /* relu*/                               \
        "str    q27, [%[c_ptr3]], #16\n"                /* write out19 to r32*/       \
        "fmax   v29.4s, v29.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q12, [%[c_ptr4]], #16\n"                /* write out4 to r40*/       \
        "fmax   v14.4s, v14.4s, v1.4s\n"                /* relu*/                               \
        "str    q20, [%[c_ptr4]], #16\n"                /* write out12 to r41*/       \
        "fmax   v22.4s, v22.4s, v1.4s\n"                /* relu*/                               \
        "str    q28, [%[c_ptr4]], #16\n"                /* write out20 to r42*/       \
        "fmax   v30.4s, v30.4s, v1.4s\n"                /* relu*/                               \
        \
        "str    q13, [%[c_ptr5]], #16\n"                /* write out5 to r50*/       \
        "fmax   v15.4s, v15.4s, v1.4s\n"                /* relu*/                               \
        "str    q21, [%[c_ptr5]], #16\n"                /* write out13 to r51*/       \
        "fmax   v23.4s, v23.4s, v1.4s\n"                /* relu*/                               \
        "str    q29, [%[c_ptr5]], #16\n"                /* write out21 to r52*/       \
        "fmax   v31.4s, v31.4s, v1.4s\n"                /* relu*/                               \
        "str    q14, [%[c_ptr6]], #16\n"                /* write out6 to r60*/       \
        "str    q22, [%[c_ptr6]], #16\n"                /* write out14 to r61*/       \
        "str    q30, [%[c_ptr6]], #16\n"                /* write out22 to r62*/       \
        "b		11f\n"       \
        /* tails==3 final tail*/       \
        "5:\n"                                          /* tail = 3*/       \
        "ldr	q4, [%[b_ptr]], #16\n"                  /* load b2, b0 to q4*/       \
        "fmla 	v8.4s ,  v6.4s,  v0.s[0]\n"             /* out0 = b0 * a10[0], b0 = q5*/       \
        "fmla	v16.4s,  v7.4s,  v0.s[0]\n"             /* out8 = b0 * a10[0], b1 = q6*/       \
        "fmla	v24.4s,  v4.4s,  v0.s[0]\n"             /* out16 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v9.4s ,  v6.4s,  v0.s[1]\n"             /* out1 = b0 * a10[1], b0 = q5*/       \
        "fmla	v17.4s,  v7.4s,  v0.s[1]\n"             /* out9 = b1 * a10[1], b1 = q6*/       \
        "fmla	v25.4s,  v4.4s,  v0.s[1]\n"             /* out17 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v10.4s,  v6.4s,  v0.s[2]\n"             /* out2 = b0 * a10[2], b0 = q5*/       \
        "fmla	v18.4s,  v7.4s,  v0.s[2]\n"             /* out10 = b1 * a10[2], b1 = q6*/       \
        "fmla	v26.4s,  v4.4s,  v0.s[2]\n"             /* out18 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla	v11.4s,  v6.4s,  v0.s[3]\n"             /* out3 = b0 * a10[3], b0 = q5*/       \
        "fmla	v19.4s,  v7.4s,  v0.s[3]\n"             /* out11 = b1 * a10[3], b1 = q6*/       \
        "fmla	v27.4s,  v4.4s,  v0.s[3]\n"             /* out19 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla 	v12.4s,  v6.4s,  v1.s[0]\n"             /* out4 = b0 * a11[0], b0 = q5*/       \
        "fmla	v20.4s,  v7.4s,  v1.s[0]\n"             /* out12 = b1 * a10[0], b1 = q6*/       \
        "fmla	v28.4s,  v4.4s,  v1.s[0]\n"             /* out20 = b2 * a10[0], b2 = q7*/       \
        \
        "fmla   v13.4s,  v6.4s,  v1.s[1]\n"             /* out5 = b0 * a11[1], b0 = q5*/       \
        "fmla	v21.4s,  v7.4s,  v1.s[1]\n"             /* out13 = b1 * a10[1], b1 = q6*/       \
        "fmla	v29.4s,  v4.4s,  v1.s[1]\n"             /* out21 = b2 * a10[0], b2 = q7*/       \
        \
        "movi   v2.4s, #0\n"                            /* for relu*/                           \
        "fmax   v8.4s, v8.4s, v2.4s\n"                  /* relu*/                               \
        "fmax   v16.4s,v16.4s,v2.4s\n"                  /* relu*/                               \
        "fmax   v24.4s,v24.4s,v2.4s\n"                  /* relu*/                               \
        \
        "fmla	v14.4s,  v6.4s,  v1.s[2]\n"             /* out6 = b0 * a11[2], b0 = q5*/       \
        "fmla	v22.4s,  v7.4s,  v1.s[2]\n"             /* out14 = b1 * a10[2], b1 = q6*/       \
        "fmla	v30.4s,  v4.4s,  v1.s[2]\n"             /* out22 = b2 * a10[0], b2 = q7*/       \
        \
        "fmax   v9.4s, v9.4s, v2.4s\n"                  /* relu*/                               \
        "fmax   v17.4s,v17.4s,v2.4s\n"                  /* relu*/                               \
        "fmax   v25.4s,v25.4s,v2.4s\n"                  /* relu*/                               \
        \
        "fmla	v15.4s,  v6.4s,  v1.s[3]\n"             /* out7 = b0 * a11[3], b0 = q5*/       \
        "fmla	v23.4s,  v7.4s,  v1.s[3]\n"             /* out15 = b1 * a10[3], b1 = q6*/       \
        "fmla	v31.4s,  v4.4s,  v1.s[3]\n"             /* out23 = b2 * a10[0], b2 = q7*/       \
        \
        "str    q8, [%[c_ptr0]], #16\n"                 /* write out0 to r00*/       \
        "fmax   v10.4s, v10.4s, v2.4s\n"                /* relu*/                               \
        "str    q16, [%[c_ptr0]], #16\n"                /* write out8 to r01*/       \
        "fmax   v18.4s, v18.4s, v2.4s\n"                /* relu*/                               \
        "str    q24, [%[c_ptr0]], #16\n"                /* write out 16 to r02*/       \
        "fmax   v26.4s, v26.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q9, [%[c_ptr1]], #16\n"                 /* write out1 to r10*/       \
        "fmax   v11.4s, v11.4s, v2.4s\n"                /* relu*/                               \
        "str    q17, [%[c_ptr1]], #16\n"                /* write out9 to r11*/       \
        "fmax   v19.4s, v19.4s, v2.4s\n"                /* relu*/                               \
        "str    q25, [%[c_ptr1]], #16\n"                /* write out17 to r12*/       \
        "fmax   v27.4s, v27.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q10, [%[c_ptr2]], #16\n"                /* write out2 to r20*/       \
        "fmax   v12.4s, v12.4s, v2.4s\n"                /* relu*/                               \
        "str    q18, [%[c_ptr2]], #16\n"                /* write out10 to r21*/       \
        "fmax   v20.4s, v20.4s, v2.4s\n"                /* relu*/                               \
        "str    q26, [%[c_ptr2]], #16\n"                /* write out18 to r22*/       \
        "fmax   v28.4s, v28.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q11, [%[c_ptr3]], #16\n"                /* write out3 to r30*/       \
        "fmax   v13.4s, v13.4s, v2.4s\n"                /* relu*/                               \
        "str    q19, [%[c_ptr3]], #16\n"                /* write out11 to r31*/       \
        "fmax   v21.4s, v21.4s, v2.4s\n"                /* relu*/                               \
        "str    q27, [%[c_ptr3]], #16\n"                /* write out19 to r32*/       \
        "fmax   v29.4s, v29.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q12, [%[c_ptr4]], #16\n"                /* write out4 to r40*/       \
        "fmax   v14.4s, v14.4s, v2.4s\n"                /* relu*/                               \
        "str    q20, [%[c_ptr4]], #16\n"                /* write out12 to r41*/       \
        "fmax   v22.4s, v22.4s, v2.4s\n"                /* relu*/                               \
        "str    q28, [%[c_ptr4]], #16\n"                /* write out20 to r42*/       \
        "fmax   v30.4s, v30.4s, v2.4s\n"                /* relu*/                               \
        \
        "str    q13, [%[c_ptr5]], #16\n"                /* write out5 to r50*/       \
        "fmax   v15.4s, v15.4s, v2.4s\n"                /* relu*/                               \
        "str    q21, [%[c_ptr5]], #16\n"                /* write out13 to r51*/       \
        "fmax   v23.4s, v23.4s, v2.4s\n"                /* relu*/                               \
        "str    q29, [%[c_ptr5]], #16\n"                /* write out21 to r52*/       \
        "fmax   v31.4s, v31.4s, v2.4s\n"                /* relu*/                               \
        "str    q14, [%[c_ptr6]], #16\n"                /* write out6 to r60*/       \
        "str    q22, [%[c_ptr6]], #16\n"                /* write out14 to r61*/       \
        "str    q30, [%[c_ptr6]], #16\n"                /* write out22 to r62*/       \
        "11: \n"                                        /* end write*/       \
        "str    q15, [%[c_ptr7]], #16\n"                /* write out7 to r70*/       \
        "str    q23, [%[c_ptr7]], #16\n"                /* write out15 to r71*/       \
        "str    q31, [%[c_ptr7]], #16\n"                /* write out23 to r72*/

void sgemm_conv_8x12(const float* A_packed, const float* B, float* C, int M, int N, int K, bool transB, Context* ctx) {

    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 12;
    x_block *= 12;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 11) / 12;
    x_block *= 12;

    // unroll 2 loop
    int tail_pre = (K & 3);
    int k_pre = ((K + 3) / 4) - 1;

    float zerobuf[x_block];

    bool flag_p_remain = false;
    int remain = 0;

    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 11) / 12;
        remain = xmax - x0 - (bblocks - 1) * 12;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        float *b_pannel = static_cast<float *>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 8) {
            unsigned int ymax = y + 8;
            if (ymax > M) {
                ymax = M;
            }

            float cout0[12];
            float cout1[12];
            float cout2[12];
            float cout3[12];
            float cout4[12];
            float cout5[12];
            float cout6[12];
            float cout7[12];

            float *c_ptr0 = C + y * N + x0;
            float *c_ptr1 = c_ptr0 + N;
            float *c_ptr2 = c_ptr1 + N;
            float *c_ptr3 = c_ptr2 + N;
            float *c_ptr4 = c_ptr3 + N;
            float *c_ptr5 = c_ptr4 + N;
            float *c_ptr6 = c_ptr5 + N;
            float *c_ptr7 = c_ptr6 + N;

            float *pout0 = c_ptr0;
            float *pout1 = c_ptr1;
            float *pout2 = c_ptr2;
            float *pout3 = c_ptr3;
            float *pout4 = c_ptr4;
            float *pout5 = c_ptr5;
            float *pout6 = c_ptr6;
            float *pout7 = c_ptr7;


            if ((y + 7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        c_ptr1 = zerobuf;
                    case 5:
                        c_ptr2 = zerobuf;
                    case 4:
                        c_ptr3 = zerobuf;
                    case 3:
                        c_ptr4 = zerobuf;
                    case 2:
                        c_ptr5 = zerobuf;
                    case 1:
                        c_ptr6 = zerobuf;
                    case 0:
                        c_ptr7 = zerobuf;
                    default:
                        break;
                }
            }

            const float *a_ptr_l = A_packed + y * K;
            const float *b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {

                if (flag_p_remain && (xb == bblocks - 1)) {

                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    pout4 = c_ptr4;
                    pout5 = c_ptr5;
                    pout6 = c_ptr6;
                    pout7 = c_ptr7;

                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                    c_ptr4 = cout4;
                    c_ptr5 = cout5;
                    c_ptr6 = cout6;
                    c_ptr7 = cout7;
                }
                const float *a_ptr = a_ptr_l;
                int tail = tail_pre;
                int k = k_pre;

                asm volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                        GEMM_IN
                        GEMM_KERNEL8x12
                        GEMM_OUT

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), [tail] "+r" (tail), \
                  [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
                        [c_ptr3] "+r"(c_ptr3), [c_ptr4] "+r"(c_ptr4), [c_ptr5] "+r"(c_ptr5), \
                        [c_ptr6] "+r"(c_ptr6), [c_ptr7] "+r"(c_ptr7)
                :
                : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                        *pout4++ = cout4[i];
                        *pout5++ = cout5[i];
                        *pout6++ = cout6[i];
                        *pout7++ = cout7[i];
                    }
                }
            }
        }
    }
}

void sgemm_conv_8x12_relu(const float* A_packed, const float* B, float* C, int M, int N, int K, bool transB, Context* ctx) {

    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 12;
    x_block *= 12;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 11) / 12;
    x_block *= 12;

    // unroll 2 loop
    int tail_pre = (K & 3);
    int k_pre = ((K + 3) / 4) - 1;

    float zerobuf[x_block];

    bool flag_p_remain = false;
    int remain = 0;

    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 11) / 12;
        remain = xmax - x0 - (bblocks - 1) * 12;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        float *b_pannel = static_cast<float *>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 8) {
            unsigned int ymax = y + 8;
            if (ymax > M) {
                ymax = M;
            }

            float cout0[12];
            float cout1[12];
            float cout2[12];
            float cout3[12];
            float cout4[12];
            float cout5[12];
            float cout6[12];
            float cout7[12];

            float *c_ptr0 = C + y * N + x0;
            float *c_ptr1 = c_ptr0 + N;
            float *c_ptr2 = c_ptr1 + N;
            float *c_ptr3 = c_ptr2 + N;
            float *c_ptr4 = c_ptr3 + N;
            float *c_ptr5 = c_ptr4 + N;
            float *c_ptr6 = c_ptr5 + N;
            float *c_ptr7 = c_ptr6 + N;

            float *pout0 = c_ptr0;
            float *pout1 = c_ptr1;
            float *pout2 = c_ptr2;
            float *pout3 = c_ptr3;
            float *pout4 = c_ptr4;
            float *pout5 = c_ptr5;
            float *pout6 = c_ptr6;
            float *pout7 = c_ptr7;


            if ((y + 7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        c_ptr1 = zerobuf;
                    case 5:
                        c_ptr2 = zerobuf;
                    case 4:
                        c_ptr3 = zerobuf;
                    case 3:
                        c_ptr4 = zerobuf;
                    case 2:
                        c_ptr5 = zerobuf;
                    case 1:
                        c_ptr6 = zerobuf;
                    case 0:
                        c_ptr7 = zerobuf;
                    default:
                        break;
                }
            }

            const float *a_ptr_l = A_packed + y * K;
            const float *b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {

                if (flag_p_remain && (xb == bblocks - 1)) {

                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    pout4 = c_ptr4;
                    pout5 = c_ptr5;
                    pout6 = c_ptr6;
                    pout7 = c_ptr7;

                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                    c_ptr4 = cout4;
                    c_ptr5 = cout5;
                    c_ptr6 = cout6;
                    c_ptr7 = cout7;
                }
                const float *a_ptr = a_ptr_l;
                int tail = tail_pre;
                int k = k_pre;

                asm volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                        GEMM_IN
                        GEMM_KERNEL8x12
                        GEMM_OUT_RELU

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), [tail] "+r" (tail), \
                  [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
                        [c_ptr3] "+r"(c_ptr3), [c_ptr4] "+r"(c_ptr4), [c_ptr5] "+r"(c_ptr5), \
                        [c_ptr6] "+r"(c_ptr6), [c_ptr7] "+r"(c_ptr7)
                :
                : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                        *pout4++ = cout4[i];
                        *pout5++ = cout5[i];
                        *pout6++ = cout6[i];
                        *pout7++ = cout7[i];
                    }
                }
            }
        }
    }
}

void sgemm_conv_8x12_bias(const float* A_packed, const float* B, const float* bias, \
    float* C, int M, int N, int K, bool transB, Context* ctx) {

    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 12;
    x_block *= 12;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 11) / 12;
    x_block *= 12;

    // unroll 2 loop
    int tail_pre = (K & 3);
    int k_pre = ((K + 3) / 4) - 1;

    float zerobuf[x_block];

    bool flag_p_remain = false;
    int remain = 0;

    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 11) / 12;
        remain = xmax - x0 - (bblocks - 1) * 12;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        float *b_pannel = static_cast<float *>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 8) {
            unsigned int ymax = y + 8;
            if (ymax > M) {
                ymax = M;
            }

            float bias_ptr[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }

            float cout0[12];
            float cout1[12];
            float cout2[12];
            float cout3[12];
            float cout4[12];
            float cout5[12];
            float cout6[12];
            float cout7[12];

            float *c_ptr0 = C + y * N + x0;
            float *c_ptr1 = c_ptr0 + N;
            float *c_ptr2 = c_ptr1 + N;
            float *c_ptr3 = c_ptr2 + N;
            float *c_ptr4 = c_ptr3 + N;
            float *c_ptr5 = c_ptr4 + N;
            float *c_ptr6 = c_ptr5 + N;
            float *c_ptr7 = c_ptr6 + N;

            float *pout0 = c_ptr0;
            float *pout1 = c_ptr1;
            float *pout2 = c_ptr2;
            float *pout3 = c_ptr3;
            float *pout4 = c_ptr4;
            float *pout5 = c_ptr5;
            float *pout6 = c_ptr6;
            float *pout7 = c_ptr7;


            if ((y + 7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        c_ptr1 = zerobuf;
                    case 5:
                        c_ptr2 = zerobuf;
                    case 4:
                        c_ptr3 = zerobuf;
                    case 3:
                        c_ptr4 = zerobuf;
                    case 2:
                        c_ptr5 = zerobuf;
                    case 1:
                        c_ptr6 = zerobuf;
                    case 0:
                        c_ptr7 = zerobuf;
                    default:
                        break;
                }
            }

            const float *a_ptr_l = A_packed + y * K;
            const float *b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {

                if (flag_p_remain && (xb == bblocks - 1)) {

                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    pout4 = c_ptr4;
                    pout5 = c_ptr5;
                    pout6 = c_ptr6;
                    pout7 = c_ptr7;

                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                    c_ptr4 = cout4;
                    c_ptr5 = cout5;
                    c_ptr6 = cout6;
                    c_ptr7 = cout7;
                }
                const float *a_ptr = a_ptr_l;
                int tail = tail_pre;
                int k = k_pre;

                asm volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                        GEMM_IN_BIAS
                        GEMM_KERNEL8x12
                        GEMM_OUT

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), [tail] "+r" (tail), \
                  [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
                        [c_ptr3] "+r"(c_ptr3), [c_ptr4] "+r"(c_ptr4), [c_ptr5] "+r"(c_ptr5), \
                        [c_ptr6] "+r"(c_ptr6), [c_ptr7] "+r"(c_ptr7)
                : [bias_ptr] "r" (bias_ptr)
                : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                        *pout4++ = cout4[i];
                        *pout5++ = cout5[i];
                        *pout6++ = cout6[i];
                        *pout7++ = cout7[i];
                    }
                }
            }
        }
    }
}

void sgemm_conv_8x12_bias_relu(const float* A_packed, const float* B, const float* bias, \
    float* C, int M, int N, int K, bool transB, Context* ctx) {

    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 12;
    x_block *= 12;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 11) / 12;
    x_block *= 12;

    // unroll 2 loop
    int tail_pre = (K & 3);
    int k_pre = ((K + 3) / 4) - 1;

    float zerobuf[x_block];

    bool flag_p_remain = false;
    int remain = 0;

    //! apanel is pre_compute outside gemm
    for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
        unsigned int xmax = x0 + x_block;
        if (xmax > N) {
            xmax = N;
        }
        int bblocks = (xmax - x0 + 11) / 12;
        remain = xmax - x0 - (bblocks - 1) * 12;
        if (remain > 0) {
            flag_p_remain = true;
        }
        //! load bpanel
        float *b_pannel = static_cast<float *>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 8) {
            unsigned int ymax = y + 8;
            if (ymax > M) {
                ymax = M;
            }

            float bias_ptr[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }

            float cout0[12];
            float cout1[12];
            float cout2[12];
            float cout3[12];
            float cout4[12];
            float cout5[12];
            float cout6[12];
            float cout7[12];

            float *c_ptr0 = C + y * N + x0;
            float *c_ptr1 = c_ptr0 + N;
            float *c_ptr2 = c_ptr1 + N;
            float *c_ptr3 = c_ptr2 + N;
            float *c_ptr4 = c_ptr3 + N;
            float *c_ptr5 = c_ptr4 + N;
            float *c_ptr6 = c_ptr5 + N;
            float *c_ptr7 = c_ptr6 + N;

            float *pout0 = c_ptr0;
            float *pout1 = c_ptr1;
            float *pout2 = c_ptr2;
            float *pout3 = c_ptr3;
            float *pout4 = c_ptr4;
            float *pout5 = c_ptr5;
            float *pout6 = c_ptr6;
            float *pout7 = c_ptr7;


            if ((y + 7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        c_ptr1 = zerobuf;
                    case 5:
                        c_ptr2 = zerobuf;
                    case 4:
                        c_ptr3 = zerobuf;
                    case 3:
                        c_ptr4 = zerobuf;
                    case 2:
                        c_ptr5 = zerobuf;
                    case 1:
                        c_ptr6 = zerobuf;
                    case 0:
                        c_ptr7 = zerobuf;
                    default:
                        break;
                }
            }

            const float *a_ptr_l = A_packed + y * K;
            const float *b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {

                if (flag_p_remain && (xb == bblocks - 1)) {

                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    pout4 = c_ptr4;
                    pout5 = c_ptr5;
                    pout6 = c_ptr6;
                    pout7 = c_ptr7;

                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                    c_ptr4 = cout4;
                    c_ptr5 = cout5;
                    c_ptr6 = cout6;
                    c_ptr7 = cout7;
                }
                const float *a_ptr = a_ptr_l;
                int tail = tail_pre;
                int k = k_pre;

                asm volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                        GEMM_IN_BIAS
                        GEMM_KERNEL8x12
                        GEMM_OUT_RELU

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k), [tail] "+r" (tail), \
                  [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1), [c_ptr2] "+r"(c_ptr2), \
                        [c_ptr3] "+r"(c_ptr3), [c_ptr4] "+r"(c_ptr4), [c_ptr5] "+r"(c_ptr5), \
                        [c_ptr6] "+r"(c_ptr6), [c_ptr7] "+r"(c_ptr7)
                : [bias_ptr] "r" (bias_ptr)
                : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                        *pout4++ = cout4[i];
                        *pout5++ = cout5[i];
                        *pout6++ = cout6[i];
                        *pout7++ = cout7[i];
                    }
                }
            }
        }
    }
}

#else //__aarch64__

#define SGEMM_KERNEL6x8 \
                        "cmp %[k], #0                           @ check weather k is bigger than 0\n"   \
                        "beq 0f                                 @ jump to tail\n"       \
                                                                                        \
                        "1:                                     @ main loop for k\n"    \
                        /* Unroll 0*/                                                   \
                        "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a4, a5, and next a0, a1\n"       \
                        "vmla.f32	q4, q2, d0[0]               @ out0 += b1 * a0\n"    \
                        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"            \
                        "vmla.f32	q5, q2, d0[1]               @ out1 += b1 * a1\n"    \
                        "vmla.f32	q6, q2, d1[0]               @ out2 += b1 * a2\n"    \
                        "vmla.f32	q7, q2, d1[1]               @ out3 += b1 * a3\n"    \
                        "vmla.f32	q8, q2, d2[0]               @ out4 += b1 * a4\n"    \
                        "vmla.f32	q9, q2, d2[1]               @ out5 += b1 * a5\n"    \
                                                                                        \
                        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"            \
                        "vmla.f32	q10, q3, d0[0]              @ out6 += b2 * a0\n"    \
                        "vmla.f32	q11, q3, d0[1]              @ out7 += b2 * a1\n"    \
                        "vmla.f32	q12, q3, d1[0]              @ out8 += b2 * a2\n"    \
                        "vmla.f32	q13, q3, d1[1]              @ out9 += b2 * a3\n"    \
                        "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a2~a5\n"         \
                        "vmla.f32	q14, q3, d2[0]              @ out10 += b2 * a4\n"   \
                        "vmla.f32	q15, q3, d2[1]              @ out11 += b2 * a5\n"   \
                        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"            \
                                                                                        \
                        /* Unroll 1 */                                                  \
                        "vmla.f32	q4, q2, d3[0]               @ out0 += b1 * a0\n"    \
                        "vmla.f32	q5, q2, d3[1]               @ out1 += b1 * a1\n"    \
                        /*"pld [%[a_ptr], #64]                    @ preload a\n"*/      \
                        "vmla.f32	q6, q2, d0[0]               @ out2 += b1 * a2\n"    \
                        "vmla.f32	q7, q2, d0[1]               @ out3 += b1 * a3\n"    \
                        /*"pld [%[b_ptr], #192]\n"*/                                    \
                        "vmla.f32	q8, q2, d1[0]               @ out4 += b1 * a4\n"    \
                        "vmla.f32	q9, q2, d1[1]               @ out5 += b1 * a5\n"    \
                        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"            \
                                                                                        \
                        "vmla.f32	q10, q3, d3[0]              @ out6 += b2 * a0\n"    \
                        "vmla.f32	q11, q3, d3[1]              @ out7 += b2 * a1\n"    \
                        "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a0~a3\n"         \
                        "vmla.f32	q12, q3, d0[0]              @ out8 += b2 * a2\n"    \
                        "vmla.f32	q13, q3, d0[1]              @ out9 += b2 * a3\n"    \
                        "vmla.f32	q14, q3, d1[0]              @ out10 += b2 * a4\n"   \
                        "vmla.f32	q15, q3, d1[1]              @ out11 += b2 * a5\n"   \
                        "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a4, a5, a0, a1\n"\
                                                                                        \
                        /* Unroll 2 */                                                  \
                        "vmla.f32	q4, q2, d2[0]               @ out0 += b1 * a0\n"    \
                        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"            \
                        "vmla.f32	q5, q2, d2[1]               @ out1 += b1 * a1\n"    \
                        "vmla.f32	q6, q2, d3[0]               @ out2 += b1 * a2\n"    \
                        "vmla.f32	q7, q2, d3[1]               @ out3 += b1 * a3\n"    \
                        /*"pld [%[a_ptr], #240]                   @ preload\n"*/        \
                        "vmla.f32	q8, q2, d0[0]               @ out4 += b1 * a4\n"    \
                        "vmla.f32	q9, q2, d0[1]               @ out5 += b1 * a5\n"    \
                        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"            \
                                                                                        \
                        "vmla.f32	q10, q3, d2[0]              @ out6 += b2 * a0\n"    \
                        "vmla.f32	q11, q3, d2[1]              @ out7 += b2 * a1\n"    \
                        /*"pld [%[b_ptr], #208]     \n"*/                               \
                        "vmla.f32	q12, q3, d3[0]              @ out8 += b2 * a2\n"    \
                        "vmla.f32	q13, q3, d3[1]              @ out9 += b2 * a3\n"    \
                        "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a2~a5\n"         \
                        "vmla.f32	q14, q3, d0[0]              @ out10 += b2 * a4\n"   \
                        "vmla.f32	q15, q3, d0[1]              @ out11 += b2 * a5\n"   \
                        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"            \
                                                                                        \
                        /* Unroll 3 */                                                  \
                        "vmla.f32	q4, q2, d1[0]               @ out0 += b1 * a0\n"    \
                        "vmla.f32	q5, q2, d1[1]               @ out1 += b1 * a1\n"    \
                        "vmla.f32	q6, q2, d2[0]               @ out2 += b1 * a2\n"    \
                        "vmla.f32	q7, q2, d2[1]               @ out3 += b1 * a3\n"    \
                        "vmla.f32	q8, q2, d3[0]               @ out4 += b1 * a4\n"    \
                        "vmla.f32	q9, q2, d3[1]               @ out5 += b1 * a5\n"    \
                        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"            \
                                                                                        \
                        "vmla.f32	q10, q3, d1[0]              @ out6 += b2 * a0\n"    \
                        "vmla.f32	q11, q3, d1[1]              @ out7 += b2 * a1\n"    \
                        "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a0~a3\n"         \
                        "vmla.f32	q12, q3, d2[0]              @ out8 += b2 * a2\n"    \
                        "vmla.f32	q13, q3, d2[1]              @ out9 += b2 * a3\n"    \
                        "subs		%[k], %[k], #1              @ k--\n"                \
                        "vmla.f32	q14, q3, d3[0]              @ out10 += b2 * a4\n"   \
                        "vmla.f32	q15, q3, d3[1]              @ out11 += b2 * a5\n"   \
                        "bne		1b                          @ jump to main loop\n"  \
                                                                                        \
                        "0:                                     @ process tail\n"       \
                        "subs		%[tails], %[tails], #1      @ tail--\n"             \
                        "beq		3f                          @ jump to tail = 1\n"   \
                                                                                        \
                        /* Unroll 0*/                                                   \
                        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"            \
                        "vmla.f32	q4, q2, d0[0]               @ out0 += b1 * a0\n"    \
                        "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a4,5, a0, a1\n"  \
                        "vmla.f32	q5, q2, d0[1]               @ out1 += b1 * a1\n"    \
                        "vmla.f32	q6, q2, d1[0]               @ out2 += b1 * a2\n"    \
                        "vmla.f32	q7, q2, d1[1]               @ out3 += b1 * a3\n"    \
                        "vmla.f32	q8, q2, d2[0]               @ out4 += b1 * a4\n"    \
                        "subs		%[tails], %[tails], #1      @ tail--\n"             \
                        "vmla.f32	q9, q2, d2[1]               @ out5 += b1 * a5\n"    \
                        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"            \
                                                                                        \
                        "vmla.f32	q10, q3, d0[0]              @ out6 += b2 * a0\n"    \
                        "vmla.f32	q11, q3, d0[1]              @ out7 += b2 * a1\n"    \
                        "vmla.f32	q12, q3, d1[0]              @ out8 += b2 * a2\n"    \
                        "vmla.f32	q13, q3, d1[1]              @ out9 += b2 * a3\n"    \
                        "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a2~a5\n"         \
                        "vmla.f32	q14, q3, d2[0]              @ out10 += b2 * a4\n"   \
                        "vmla.f32	q15, q3, d2[1]              @ out11 += b2 * a5\n"   \
                        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"            \
                        "beq		4f                          @ jump to tail==2\n"    \
                                                                                        \
                        /* Unroll 1*/                                                   \
                        "vmla.f32	q4, q2, d3[0]               @ out0 += b1 * a0\n"    \
                        "vmla.f32	q5, q2, d3[1]               @ out1 += b1 * a1\n"    \
                        "subs		%[tails], %[tails], #1      @ tail--\n"             \
                        "vmla.f32	q6, q2, d0[0]               @ out2 += b1 * a2\n"    \
                        "vmla.f32	q7, q2, d0[1]               @ out3 += b1 * a3\n"    \
                        "vmla.f32	q8, q2, d1[0]               @ out4 += b1 * a4\n"    \
                        "vmla.f32	q9, q2, d1[1]               @ out5 += b1 * a5\n"    \
                        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"            \
                                                                                        \
                        "vmla.f32	q10, q3, d3[0]              @ out6 += b2 * a0\n"    \
                        "vmla.f32	q11, q3, d3[1]              @ out7 += b2 * a1\n"    \
                        "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a0~a3\n"         \
                        "vmla.f32	q12, q3, d0[0]              @ out8 += b2 * a2\n"    \
                        "vmla.f32	q13, q3, d0[1]              @ out9 += b2 * a3\n"    \
                        "vmla.f32	q14, q3, d1[0]              @ out10 += b2 * a4\n"   \
                        "vmla.f32	q15, q3, d1[1]              @ out11 += b2 * a5\n"   \
                        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"            \
                        "beq		5f                          @ jump to tail==3\n"    \
                                                                                        \
                        /* Unroll 2 */                                                  \
                        "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a4,a5, a0,a1\n"  \
                        "vmla.f32	q4, q2, d2[0]               @ out0 += b1 * a0\n"    \
                        "vmla.f32	q5, q2, d2[1]               @ out1 += b1 * a1\n"    \
                        "vmla.f32	q6, q2, d3[0]               @ out2 += b1 * a2\n"    \
                        "vmla.f32	q7, q2, d3[1]               @ out3 += b1 * a3\n"    \
                        "vmla.f32	q8, q2, d0[0]               @ out4 += b1 * a4\n"    \
                        "vmla.f32	q9, q2, d0[1]               @ out5 += b1 * a5\n"    \
                        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"            \
                                                                                        \
                        "vmla.f32	q10, q3, d2[0]              @ out6 += b2 * a0\n"    \
                        "vmla.f32	q11, q3, d2[1]              @ out7 += b2 * a1\n"    \
                        "vmla.f32	q12, q3, d3[0]              @ out8 += b2 * a2\n"    \
                        "vmla.f32	q13, q3, d3[1]              @ out9 += b2 * a3\n"    \
                        "vld1.32	{d2-d3}, [%[a_ptr] :64]!    @ load a2~a5\n"         \
                        "vmla.f32	q14, q3, d0[0]              @ out10 += b2 * a4\n"   \
                        "vmla.f32	q15, q3, d0[1]              @ out11 += b2 * a5\n"   \
                        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"

#define SGEMM_KERNEL_4x8 \
        "cmp %[k], #0                           @ check weather k is bigger than 0\n"   \
        "beq 0f                                 @ jump to tail\n"                       \
                                                                                        \
        "1:                                     @ main loop for k\n"                    \
        /* Unroll 0*/                                                                   \
        "vld1.32  {d12-d15}, [%[b_ptr] :128]!   @ load next b1, b2\n"                   \
        "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"                    \
        "vld1.32	{d4-d7}, [%[a_ptr] :128]!   @ load next 2xa0~a3\n"                  \
        "vmla.f32	q9, q4, d0[1]               @ out1 += b1 * a1\n"                    \
        "vmla.f32	q10, q4, d1[0]              @ out2 += b1 * a2\n"                    \
        "vmla.f32	q11, q4, d1[1]              @ out3 += b1 * a3\n"                    \
        "vmla.f32	q12, q5, d0[0]              @ out4 += b2 * a0\n"                    \
        "vmla.f32	q13, q5, d0[1]              @ out5 += b2 * a1\n"                    \
        "vmla.f32	q14, q5, d1[0]              @ out6 += b2 * a2\n"                    \
        "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"                    \
        "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"                   \
                                                                                        \
        /* Unroll 1 */                                                                  \
        "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"                    \
        "pld [%[b_ptr], #64]                    @ preload b\n"                          \
        "vmla.f32	q9, q6, d2[1]               @ out1 += b1 * a1\n"                    \
        "vmla.f32	q10, q6, d3[0]              @ out2 += b1 * a2\n"                    \
        "vmla.f32	q11, q6, d3[1]              @ out3 += b1 * a3\n"                    \
        "vmla.f32	q12, q7, d2[0]              @ out6 += b2 * a0\n"                    \
        "vmla.f32	q13, q7, d2[1]              @ out7 += b2 * a1\n"                    \
        "vmla.f32	q14, q7, d3[0]              @ out8 += b2 * a2\n"                    \
        "vmla.f32	q15, q7, d3[1]              @ out9 += b2 * a3\n"                    \
        "vld1.32	{d12-d15}, [%[b_ptr] :128]! @ load next b1,b2\n"                    \
                                                                                        \
        /* Unroll 2 */                                                                  \
        "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"                    \
        "vld1.32	{d0-d3}, [%[a_ptr] :128]!   @ load next a0~a3\n"                    \
        "vmla.f32	q9, q4, d4[1]               @ out1 += b1 * a1\n"                    \
        "vmla.f32	q10, q4, d5[0]              @ out2 += b1 * a2\n"                    \
        "vmla.f32	q11, q4, d5[1]              @ out3 += b1 * a3\n"                    \
        "vmla.f32	q12, q5, d4[0]              @ out4 += b2 * a0\n"                    \
        "vmla.f32	q13, q5, d4[1]              @ out5 += b2 * a1\n"                    \
        "vmla.f32	q14, q5, d5[0]              @ out6 += b2 * a2\n"                    \
        "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"                    \
        "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"                   \
                                                                                        \
        /* Unroll 3 */                                                                  \
        "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"                    \
        "pld [%[a_ptr], #64]                    @ preload a\n"                          \
        "vmla.f32	q9, q6, d6[1]               @ out1 += b1 * a1\n"                    \
        "vmla.f32	q10, q6, d7[0]              @ out2 += b1 * a2\n"                    \
        "vmla.f32	q11, q6, d7[1]              @ out3 += b1 * a3\n"                    \
        "vmla.f32	q12, q7, d6[0]              @ out4 += b2 * a0\n"                    \
        "vmla.f32	q13, q7, d6[1]              @ out5 += b2 * a1\n"                    \
        "vmla.f32	q14, q7, d7[0]              @ out6 += b2 * a2\n"                    \
        "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"                    \
        "subs		%[k], %[k], #1              @ k--\n"                                \
        "bne		1b                          @ jump to main loop\n"                  \
                                                                                        \
        "0:                                     @ process tail\n"                       \
        "subs		%[tails], %[tails], #1      @ tail--\n"                             \
        "beq		3f                          @ jump to tail = 1\n"                   \
                                                                                        \
        /* Unroll 0*/                                                                   \
        "vld1.32  {d12-d15}, [%[b_ptr] :128]!   @ load next b1, b2\n"                   \
        "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"                    \
        "vmla.f32	q9, q4, d0[1]               @ out1 += b1 * a1\n"                    \
        "subs		%[tails], %[tails], #1      @ tail--\n"                             \
        "vmla.f32	q10, q4, d1[0]              @ out2 += b1 * a2\n"                    \
        "vmla.f32	q11, q4, d1[1]              @ out3 += b1 * a3\n"                    \
        "vmla.f32	q12, q5, d0[0]              @ out4 += b2 * a0\n"                    \
        "vmla.f32	q13, q5, d0[1]              @ out5 += b2 * a1\n"                    \
        "vmla.f32	q14, q5, d1[0]              @ out6 += b2 * a2\n"                    \
        "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"                    \
        "beq		4f                          @ jump to tail==2\n"                    \
                                                                                        \
        /* Unroll 1 */                                                                  \
        "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"                   \
        "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"                    \
        "vld1.32	{d4-d7}, [%[a_ptr] :128]!   @ load next 2xa0~a3\n"                  \
        "vmla.f32	q9, q6, d2[1]               @ out1 += b1 * a1\n"                    \
        "subs		%[tails], %[tails], #1      @ tail--\n"                             \
        "vmla.f32	q10, q6, d3[0]              @ out2 += b1 * a2\n"                    \
        "vmla.f32	q11, q6, d3[1]              @ out3 += b1 * a3\n"                    \
        "vmla.f32	q12, q7, d2[0]              @ out6 += b2 * a0\n"                    \
        "vmla.f32	q13, q7, d2[1]              @ out7 += b2 * a1\n"                    \
        "vmla.f32	q14, q7, d3[0]              @ out8 += b2 * a2\n"                    \
        "vmla.f32	q15, q7, d3[1]              @ out9 += b2 * a3\n"                    \
        "beq		5f                          @ jump to tail==3\n"                    \
                                                                                        \
        /* Unroll 2 */                                                                  \
        "vld1.32	{d12-d15}, [%[b_ptr] :128]! @ load next b1,b2\n"                    \
        "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"                    \
        "vmla.f32	q9, q4, d4[1]               @ out1 += b1 * a1\n"                    \
        "vmla.f32	q10, q4, d5[0]              @ out2 += b1 * a2\n"                    \
        "vmla.f32	q11, q4, d5[1]              @ out3 += b1 * a3\n"                    \
        "vmla.f32	q12, q5, d4[0]              @ out4 += b2 * a0\n"                    \
        "vmla.f32	q13, q5, d4[1]              @ out5 += b2 * a1\n"                    \
        "vmla.f32	q14, q5, d5[0]              @ out6 += b2 * a2\n"                    \
        "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"

#define SGEMM_CONV_IN_6x8 \
        "vld1.32	{d0-d1}, [%[a_ptr] :64]!            @ load a0~a3\n"                 \
        "vmov.i32	q4, #0                      @ out0=0\n"                             \
        "vmov.i32	q5, #0                      @ out1=0\n"                             \
        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"                            \
        "vmov.i32	q6, #0                      @ out2=0\n"                             \
        "pld [%[a_ptr]]                         @ preload a, 64byte\n"                  \
        "vmov.i32	q7, #0                      @ out3=0\n"                             \
        "pld [%[b_ptr]]                         @ preload b\n"                          \
        "vmov.i32	q8, #0                      @ out4=0\n"                             \
        "pld [%[a_ptr], #64]                    @ preload a\n"                          \
        "vmov.i32	q9, #0                      @ out5=0\n"                             \
        "pld [%[b_ptr], #64]                    @ preload b\n"                          \
        "vmov.i32	q10, #0                     @ out6=0\n"                             \
        "pld [%[a_ptr], #128]                   @ preload a\n"                          \
        "vmov.i32	q11, #0                     @ out7=0\n"                             \
        "pld [%[b_ptr], #128]                   @ preload b\n"                          \
        "vmov.i32	q12, #0                     @ out8=0\n"                             \
        "pld [%[a_ptr], #192]                   @ preload a\n"                          \
        "vmov.i32	q13, #0                     @ out9=0\n"                             \
        "pld [%[b_ptr], #192]                   @ preload b\n"                          \
        "vmov.i32	q14, #0                     @ out10=0\n"                            \
        "vmov.i32	q15, #0                     @ out11=0\n"

#define SGEMM_CONV_IN_6x8_BIAS \
        /* with bias */                                                                 \
        "vld1.32	{d2-d4}, [%[bias_ptr]]      @ load bias 6 elements\n"                             \
        "vld1.32	{d0-d1}, [%[a_ptr] :64]!    @ load a0~a3\n"                             \
        "vdup.i32	q4, d2[0]                   @ out0=0\n"                             \
        "vdup.i32	q5, d2[1]                   @ out1=0\n"                             \
        "vdup.i32	q6, d3[0]                   @ out2=0\n"                             \
        "pld [%[a_ptr]]                         @ preload a, 64byte\n"                             \
        "vdup.i32	q7, d3[1]                   @ out3=0\n"                             \
        "pld [%[b_ptr]]                         @ preload b\n"                             \
        "vdup.i32	q8, d4[0]                   @ out4=0\n"                             \
        "pld [%[a_ptr], #64]                    @ preload a\n"                             \
        "vdup.i32	q9, d4[1]                   @ out5=0\n"                             \
        "pld [%[b_ptr], #64]                    @ preload b\n"                             \
        "vdup.i32	q14,d4[0]                   @ out10=0\n"                             \
        "vdup.i32	q15,d4[1]                   @ out11=0\n"                             \
        "vld1.32	{d4-d5}, [%[b_ptr] :128]!   @ load b1\n"                             \
        "vdup.i32	q10,d2[0]                   @ out6=0\n"                             \
        "pld [%[a_ptr], #128]                   @ preload a\n"                             \
        "vdup.i32	q11,d2[1]                   @ out7=0\n"                             \
        "pld [%[b_ptr], #128]                   @ preload b\n"                             \
        "vdup.i32	q12,d3[0]                   @ out8=0\n"                             \
        "pld [%[a_ptr], #192]                   @ preload a\n"                             \
        "vdup.i32	q13,d3[1]                   @ out9=0\n"                             \
        "pld [%[b_ptr], #192]                   @ preload b\n"

#define SGEMM_CONV_OUT_6x8 \
        /* Unroll 3*/                                                                  \
        "vmla.f32	q4, q2, d1[0]               @ out0 += b1 * a0\n"                   \
        "vmla.f32	q10, q3, d1[0]              @ out6 += b2 * a0\n"                   \
        "vst1.32   {d8-d9}, [%[c_ptr0]]!   @ write 00\n"                   \
        "vmla.f32	q5, q2, d1[1]               @ out1 += b1 * a1\n"                   \
        "vst1.32   {d20-d21}, [%[c_ptr0]]! @ write 01\n"                   \
        "vmla.f32	q11, q3, d1[1]              @ out7 += b2 * a1\n"                   \
        "vst1.32   {d10-d11}, [%[c_ptr1]]! @ write 10\n"                   \
        "vmla.f32	q6, q2, d2[0]               @ out2 += b1 * a2\n"                   \
        "vst1.32   {d22-d23}, [%[c_ptr1]]! @ write 11\n"                   \
        "vmla.f32	q12, q3, d2[0]              @ out8 += b2 * a2\n"                   \
        "vst1.32   {d12-d13}, [%[c_ptr2]]! @ write 20\n"                   \
        "vmla.f32	q7, q2, d2[1]               @ out3 += b1 * a3\n"                   \
        "vst1.32   {d24-d25}, [%[c_ptr2]]! @ write 21\n"                   \
        "vmla.f32	q13, q3, d2[1]              @ out9 += b2 * a3\n"                   \
        "vst1.32   {d14-d15}, [%[c_ptr3]]! @ write 30\n"                   \
        "vmla.f32	q8, q2, d3[0]               @ out4 += b1 * a4\n"                   \
        "vst1.32   {d26-d27}, [%[c_ptr3]]! @ write 31\n"                   \
        "vmla.f32	q14, q3, d3[0]              @ out10 += b2 * a4\n"                   \
        "vst1.32   {d16-d17}, [%[c_ptr4]]! @ write 40\n"                   \
        "vmla.f32	q9, q2, d3[1]               @ out5 += b1 * a5\n"                   \
        "vst1.32   {d28-d29}, [%[c_ptr4]]! @ write 41\n"                   \
        "vmla.f32	q15, q3, d3[1]              @ out11 += b2 * a5\n"                   \
        "vst1.32   {d18-d19}, [%[c_ptr5]]! @ write 50\n"                   \
        "b		2f\n"                   \
        /* tails==1 final tail*/                                                \
        "3:                                     @ tail=1\n"                   \
        "vmla.f32	q4, q2, d0[0]               @ out0 += b1 * a0\n"                   \
        "vld1.32	{d2}, [%[a_ptr] :64]!       @ load a4,a5\n"                   \
        "vmla.f32	q5, q2, d0[1]               @ out1 += b1 * a1\n"                   \
        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"                   \
        "vmla.f32	q6, q2, d1[0]               @ out2 += b1 * a2\n"                   \
        "vst1.32   {d8-d9}, [%[c_ptr0]]!   @ write 00\n"                   \
        "vmla.f32	q10, q3, d0[0]              @ out6 += b2 * a0\n"                   \
        "vst1.32   {d20-d21}, [%[c_ptr0]]! @ write 01\n"                   \
        "vmla.f32	q11, q3, d0[1]              @ out7 += b2 * a1\n"                   \
        "vst1.32   {d10-d11}, [%[c_ptr1]]! @ write 10\n"                   \
        "vmla.f32	q12, q3, d1[0]              @ out8 += b2 * a2\n"                   \
        "vst1.32   {d22-d23}, [%[c_ptr1]]! @ write 11\n"                   \
        "vmla.f32	q7, q2, d1[1]               @ out3 += b1 * a3\n"                   \
        "vst1.32   {d12-d13}, [%[c_ptr2]]! @ write 20\n"                   \
        "vmla.f32	q13, q3, d1[1]              @ out9 += b2 * a3\n"                   \
        "vst1.32   {d24-d25}, [%[c_ptr2]]! @ write 21\n"                   \
        "vmla.f32	q8, q2, d2[0]               @ out4 += b1 * a4\n"                   \
        "vst1.32   {d14-d15}, [%[c_ptr3]]! @ write 30\n"                   \
        "vmla.f32	q14, q3, d2[0]              @ out10 += b2 * a4\n"                   \
        "vst1.32   {d26-d27}, [%[c_ptr3]]! @ write 31\n"                   \
        "vmla.f32	q9, q2, d2[1]               @ out5 += b1 * a5\n"                   \
        "vst1.32   {d16-d17}, [%[c_ptr4]]! @ write 40\n"                   \
        "vmla.f32	q15, q3, d2[1]              @ out11 += b2 * a5\n"                   \
        "vst1.32   {d28-d29}, [%[c_ptr4]]! @ write 41\n"                   \
        "vst1.32   {d18-d19}, [%[c_ptr5]]! @ write 50\n"                   \
        "b		2f                              @ jump to end\n"                   \
        /* tails==2 final tail*/                                            \
        "4:                                     @ tail == 2\n"                   \
        "vmla.f32	q4, q2, d3[0]               @ out0 += b1 * a0\n"                   \
        "vmla.f32	q10, q3, d3[0]              @ out6 += b2 * a0\n"                   \
        "vst1.32   {d8-d9}, [%[c_ptr0]]!   @ write 00\n"                   \
        "vmla.f32	q5, q2, d3[1]               @ out1 += b1 * a1\n"                   \
        "vst1.32   {d20-d21}, [%[c_ptr0]]! @ write 01\n"                   \
        "vmla.f32	q11, q3, d3[1]              @ out7 += b2 * a1\n"                   \
        "vst1.32   {d10-d11}, [%[c_ptr1]]! @ write 10\n"                   \
        "vmla.f32	q6, q2, d0[0]               @ out2 += b1 * a2\n"                   \
        "vst1.32   {d22-d23}, [%[c_ptr1]]! @ write 11\n"                   \
        "vmla.f32	q12, q3, d0[0]              @ out8 += b2 * a2\n"                   \
        "vst1.32   {d12-d13}, [%[c_ptr2]]! @ write 20\n"                   \
        "vmla.f32	q7, q2, d0[1]               @ out3 += b1 * a3\n"                   \
        "vst1.32   {d24-d25}, [%[c_ptr2]]! @ write 21\n"                   \
        "vmla.f32	q13, q3, d0[1]              @ out9 += b2 * a3\n"                   \
        "vst1.32   {d14-d15}, [%[c_ptr3]]! @ write 30\n"                   \
        "vmla.f32	q8, q2, d1[0]               @ out4 += b1 * a4\n"                   \
        "vst1.32   {d26-d27}, [%[c_ptr3]]! @ write 31\n"                   \
        "vmla.f32	q14, q3, d1[0]              @ out10 += b2 * a4\n"                   \
        "vst1.32   {d16-d17}, [%[c_ptr4]]! @ write 40\n"                   \
        "vmla.f32	q9, q2, d1[1]               @ out5 += b1 * a5\n"                   \
        "vst1.32   {d28-d29}, [%[c_ptr4]]! @ write 41\n"                   \
        "vmla.f32	q15, q3, d1[1]              @ out11 += b2 * a5\n"                   \
        "vst1.32   {d18-d19}, [%[c_ptr5]]! @ write 50\n"                   \
        "b		2f                              @ jump to end\n"                   \
        /* tails==3 final tail*/                                                        \
        "5:                                     @ tail=3\n"                   \
        "vmla.f32	q4, q2, d2[0]               @ out0 += b1 * a0\n"                   \
        "vld1.32	{d0}, [%[a_ptr] :64]!       @ load a4,a5\n"                   \
        "vmla.f32	q5, q2, d2[1]               @ out1 += b1 * a1\n"                   \
        "vmla.f32	q6, q2, d3[0]               @ out2 += b1 * a2\n"                   \
        "vst1.32   {d8-d9}, [%[c_ptr0]]!   @ write 00\n"                   \
        "vmla.f32	q10, q3, d2[0]              @ out6 += b2 * a0\n"                   \
        "vst1.32   {d20-d21}, [%[c_ptr0]]! @ write 01\n"                   \
        "vmla.f32	q11, q3, d2[1]              @ out7 += b2 * a1\n"                   \
        "vst1.32   {d10-d11}, [%[c_ptr1]]! @ write 10\n"                   \
        "vmla.f32	q12, q3, d3[0]              @ out8 += b2 * a2\n"                   \
        "vst1.32   {d22-d23}, [%[c_ptr1]]! @ write 11\n"                   \
        "vmla.f32	q7, q2, d3[1]               @ out3 += b1 * a3\n"                   \
        "vst1.32   {d12-d13}, [%[c_ptr2]]! @ write 20\n"                   \
        "vmla.f32	q13, q3, d3[1]              @ out9 += b2 * a3\n"                   \
        "vst1.32   {d24-d25}, [%[c_ptr2]]! @ write 21\n"                   \
        "vmla.f32	q8, q2, d0[0]               @ out4 += b1 * a4\n"                   \
        "vst1.32   {d14-d15}, [%[c_ptr3]]! @ write 30\n"                   \
        "vmla.f32	q14, q3, d0[0]              @ out10 += b2 * a4\n"                   \
        "vst1.32   {d26-d27}, [%[c_ptr3]]! @ write 31\n"                   \
        "vmla.f32	q9, q2, d0[1]               @ out5 += b1 * a5\n"                   \
        "vst1.32   {d16-d17}, [%[c_ptr4]]! @ write 40\n"                   \
        "vmla.f32	q15, q3, d0[1]              @ out11 += b2 * a5\n"                   \
        "vst1.32   {d28-d29}, [%[c_ptr4]]! @ write 41\n"                   \
        "vst1.32   {d18-d19}, [%[c_ptr5]]! @ write 50\n"                   \
        "2:                                     @ end\n"                   \
        "vst1.32   {d30-d31}, [%[c_ptr5]]! @ write 51\n"

#define SGEMM_CONV_OUT_6x8_RELU \
        /* Unroll 3*/                                                       \
        /* with relu*/                                                          \
        "vmla.f32	q4, q2, d1[0]               @ out0 += b1 * a0\n"                    \
        "vmla.f32	q10, q3, d1[0]              @ out6 += b2 * a0\n"                    \
        "vmla.f32	q5, q2, d1[1]               @ out1 += b1 * a1\n"                    \
        "vmla.f32	q11, q3, d1[1]              @ out7 += b2 * a1\n"                    \
        "vmov.u32    q0, #0                     @ for relu\n"                    \
        "vmla.f32	q6, q2, d2[0]               @ out2 += b1 * a2\n"                    \
        "vmax.f32   q4, q4, q0                  @ relu\n"                    \
        "vmax.f32   q10, q10, q0                @ relu\n"                    \
        "vmla.f32	q7, q2, d2[1]               @ out3 += b1 * a3\n"                    \
        "vmax.f32   q5, q5, q0                  @ relu\n"                    \
        "vst1.32   {d8-d9}, [%[c_ptr0]]!        @ write 00\n"                    \
        "vmla.f32	q8, q2, d3[0]               @ out4 += b1 * a4\n"                    \
        "vst1.32   {d20-d21}, [%[c_ptr0]]!      @ write 01\n"                    \
        "vmax.f32   q11, q11, q0                @ relu\n"                    \
        "vst1.32   {d10-d11}, [%[c_ptr1]]!      @ write 10\n"                    \
        "vmla.f32	q9, q2, d3[1]               @ out5  += b1 * a5\n"                    \
        "vst1.32   {d22-d23}, [%[c_ptr1]]!      @ write 11\n"                    \
        "vmax.f32   q6, q6, q0                  @ relu\n"                    \
        "vmla.f32	q12, q3, d2[0]              @ out8 += b2 * a2\n"                    \
        "vmax.f32   q7, q7, q0                  @ relu\n"                    \
        "vst1.32   {d12-d13}, [%[c_ptr2]]!      @ write 20\n"                    \
        "vmla.f32	q13, q3, d2[1]              @ out9 += b2 * a3\n"                    \
        "vst1.32   {d14-d15}, [%[c_ptr3]]!      @ write 30\n"                    \
        "vmax.f32   q8, q8, q0                  @ relu\n"                    \
        "vmla.f32	q14, q3, d3[0]              @ out10 += b2 * a4\n"                    \
        "vmax.f32   q9, q9, q0                  @ relu\n"                    \
        "vst1.32   {d16-d17}, [%[c_ptr4]]!      @ write 40\n"                    \
        "vmla.f32	q15, q3, d3[1]              @ out11 += b2 * a5\n"                    \
        "vmax.f32   q12, q12, q0                @ relu\n"                    \
        "vst1.32   {d18-d19}, [%[c_ptr5]]!      @ write 50\n"                    \
        "vmax.f32   q13, q13, q0                @ relu\n"                    \
        "vst1.32   {d24-d25}, [%[c_ptr2]]!      @ write 21\n"                    \
        "vmax.f32   q14, q14, q0                @ relu\n"                    \
        "vst1.32   {d26-d27}, [%[c_ptr3]]!      @ write 31\n"                    \
        "vmax.f32   q15, q15, q0                @ relu\n"                    \
        "vst1.32   {d28-d29}, [%[c_ptr4]]!      @ write 41\n"                    \
        "b		2f\n"                    \
        /* tails==1 final tail*/                                                \
        "3:                                     @ tail=1\n"                    \
        "vmla.f32	q4, q2, d0[0]               @ out0 += b1 * a0\n"                    \
        "vld1.32	{d2}, [%[a_ptr] :64]!       @ load a4,a5\n"                    \
        "vmla.f32	q5, q2, d0[1]               @ out1 += b1 * a1\n"                    \
        "vld1.32	{d6-d7}, [%[b_ptr] :128]!   @ load b2\n"                    \
        "vmla.f32	q6, q2, d1[0]               @ out2 += b1 * a2\n"                    \
        "vmla.f32	q10, q3, d0[0]              @ out6 += b2 * a0\n"                    \
        "vmla.f32	q11, q3, d0[1]              @ out7 += b2 * a1\n"                    \
        "vmla.f32	q12, q3, d1[0]              @ out8 += b2 * a2\n"                    \
        "vmla.f32	q7, q2, d1[1]               @ out3 += b1 * a3\n"                    \
        "vmla.f32	q13, q3, d1[1]              @ out9 += b2 * a3\n"                    \
        "vmla.f32	q8, q2, d2[0]               @ out4 += b1 * a4\n"                    \
        "vmla.f32	q14, q3, d2[0]              @ out10 += b2 * a4\n"                    \
        "vmla.f32	q9, q2, d2[1]               @ out5 += b1 * a5\n"                    \
        "vmla.f32	q15, q3, d2[1]              @ out11 += b2 * a5\n"                    \
        "vmov.u32    q0, #0                      @ for relu\n"                    \
        "vmax.f32   q4, q4, q0                  @ relu\n"                    \
        "vmax.f32   q10, q10, q0                @ relu\n"                    \
        "vmax.f32   q5, q5, q0                  @ relu\n"                    \
        "vmax.f32   q11, q11, q0                @ relu\n"                    \
        "vst1.32   {d8-d9}, [%[c_ptr0]]!        @ write 00\n"                    \
        "vmax.f32   q6, q6, q0                  @ relu\n"                    \
        "vst1.32   {d20-d21}, [%[c_ptr0]]!      @ write 01\n"                    \
        "vmax.f32   q12, q12, q0                @ relu\n"                    \
        "vst1.32   {d10-d11}, [%[c_ptr1]]!      @ write 10\n"                    \
        "vmax.f32   q7, q7, q0                  @ relu\n"                    \
        "vst1.32   {d22-d23}, [%[c_ptr1]]!      @ write 11\n"                    \
        "vmax.f32   q13, q13, q0                @ relu\n"                    \
        "vst1.32   {d12-d13}, [%[c_ptr2]]!      @ write 20\n"                    \
        "vmax.f32   q8, q8, q0                  @ relu\n"                    \
        "vst1.32   {d24-d25}, [%[c_ptr2]]!      @ write 21\n"                    \
        "vmax.f32   q14, q14, q0                @ relu\n"                    \
        "vst1.32   {d14-d15}, [%[c_ptr3]]!      @ write 30\n"                    \
        "vmax.f32   q9, q9, q0                  @ relu\n"                    \
        "vst1.32   {d26-d27}, [%[c_ptr3]]!      @ write 31\n"                    \
        "vmax.f32   q15, q15, q0                @ relu\n"                    \
        "vst1.32   {d16-d17}, [%[c_ptr4]]!      @ write 40\n"                    \
        "vst1.32   {d28-d29}, [%[c_ptr4]]!      @ write 41\n"                    \
        "vst1.32   {d18-d19}, [%[c_ptr5]]!      @ write 50\n"                    \
        "b		2f                              @ jump to end\n"                    \
        /* tails==2 final tail*/                                                \
        "4:                                     @ tail == 2\n"                    \
        "vmla.f32	q4, q2, d3[0]               @ out0 += b1 * a0\n"                    \
        "vmla.f32	q10, q3, d3[0]              @ out6 += b2 * a0\n"                    \
        "vmla.f32	q5, q2, d3[1]               @ out1 += b1 * a1\n"                    \
        "vmla.f32	q11, q3, d3[1]              @ out7 += b2 * a1\n"                    \
        "vmla.f32	q6, q2, d0[0]               @ out2 += b1 * a2\n"                    \
        "vmla.f32	q12, q3, d0[0]              @ out8 += b2 * a2\n"                    \
        "vmla.f32	q7, q2, d0[1]               @ out3 += b1 * a3\n"                    \
        "vmla.f32	q13, q3, d0[1]              @ out9 += b2 * a3\n"                    \
        "vmla.f32	q8, q2, d1[0]               @ out4 += b1 * a4\n"                    \
        "vmla.f32	q14, q3, d1[0]              @ out10 += b2 * a4\n"                    \
        "vmla.f32	q9, q2, d1[1]               @ out5 += b1 * a5\n"                    \
        "vmla.f32	q15, q3, d1[1]              @ out11 += b2 * a5\n"                    \
        "vmov.u32    q0, #0                     @ for relu\n"                    \
        "vmax.f32   q4, q4, q0                  @ relu\n"                    \
        "vmax.f32   q10, q10, q0                @ relu\n"                    \
        "vmax.f32   q5, q5, q0                  @ relu\n"                    \
        "vmax.f32   q11, q11, q0                @ relu\n"                    \
        "vst1.32   {d8-d9}, [%[c_ptr0]]!        @ write 00\n"                    \
        "vmax.f32   q6, q6, q0                  @ relu\n"                    \
        "vst1.32   {d20-d21}, [%[c_ptr0]]!      @ write 01\n"                    \
        "vmax.f32   q12, q12, q0                @ relu\n"                    \
        "vst1.32   {d10-d11}, [%[c_ptr1]]!      @ write 10\n"                    \
        "vmax.f32   q7, q7, q0                  @ relu\n"                    \
        "vst1.32   {d22-d23}, [%[c_ptr1]]!      @ write 11\n"                    \
        "vmax.f32   q13, q13, q0                @ relu\n"                    \
        "vst1.32   {d12-d13}, [%[c_ptr2]]!      @ write 20\n"                    \
        "vmax.f32   q8, q8, q0                  @ relu\n"                    \
        "vst1.32   {d24-d25}, [%[c_ptr2]]!      @ write 21\n"                    \
        "vmax.f32   q14, q14, q0                @ relu\n"                    \
        "vst1.32   {d14-d15}, [%[c_ptr3]]!      @ write 30\n"                    \
        "vmax.f32   q9, q9, q0                  @ relu\n"                    \
        "vst1.32   {d26-d27}, [%[c_ptr3]]!      @ write 31\n"                    \
        "vmax.f32   q15, q15, q0                @ relu\n"                    \
        "vst1.32   {d16-d17}, [%[c_ptr4]]!      @ write 40\n"                    \
        "vst1.32   {d28-d29}, [%[c_ptr4]]!      @ write 41\n"                    \
        "vst1.32   {d18-d19}, [%[c_ptr5]]!      @ write 50\n"                    \
        "b		2f                              @ jump to end\n"                    \
        /* tails==3 final tail*/                                                      \
        "5:                                     @ tail=3\n"                    \
        "vmla.f32	q4, q2, d2[0]               @ out0 += b1 * a0\n"                    \
        "vld1.32	{d0}, [%[a_ptr] :64]!       @ load a4,a5\n"                    \
        "vmla.f32	q5, q2, d2[1]               @ out1 += b1 * a1\n"                    \
        "vmla.f32	q6, q2, d3[0]               @ out2 += b1 * a2\n"                    \
        "vmla.f32	q10, q3, d2[0]              @ out6 += b2 * a0\n"                    \
        "vmla.f32	q11, q3, d2[1]              @ out7 += b2 * a1\n"                    \
        "vmla.f32	q12, q3, d3[0]              @ out8 += b2 * a2\n"                    \
        "vmla.f32	q7, q2, d3[1]               @ out3 += b1 * a3\n"                    \
        "vmla.f32	q13, q3, d3[1]              @ out9 += b2 * a3\n"                    \
        "vmla.f32	q8, q2, d0[0]               @ out4 += b1 * a4\n"                    \
        "vmla.f32	q14, q3, d0[0]              @ out10 += b2 * a4\n"                    \
        "vmla.f32	q9, q2, d0[1]               @ out5 += b1 * a5\n"                    \
        "vmla.f32	q15, q3, d0[1]              @ out11 += b2 * a5\n"                    \
        "vmov.u32    q0, #0                     @ for relu\n"                    \
        "vmax.f32   q4, q4, q0                  @ relu\n"                    \
        "vmax.f32   q10, q10, q0                @ relu\n"                    \
        "vmax.f32   q5, q5, q0                  @ relu\n"                    \
        "vmax.f32   q11, q11, q0                @ relu\n"                    \
        "vst1.32   {d8-d9}, [%[c_ptr0]]!        @ write 00\n"                    \
        "vmax.f32   q6, q6, q0                  @ relu\n"                    \
        "vst1.32   {d20-d21}, [%[c_ptr0]]!      @ write 01\n"                    \
        "vmax.f32   q12, q12, q0                @ relu\n"                    \
        "vst1.32   {d10-d11}, [%[c_ptr1]]!      @ write 10\n"                    \
        "vmax.f32   q7, q7, q0                  @ relu\n"                    \
        "vst1.32   {d22-d23}, [%[c_ptr1]]!      @ write 11\n"                    \
        "vmax.f32   q13, q13, q0                @ relu\n"                    \
        "vst1.32   {d12-d13}, [%[c_ptr2]]!      @ write 20\n"                    \
        "vmax.f32   q8, q8, q0                  @ relu\n"                    \
        "vst1.32   {d24-d25}, [%[c_ptr2]]!      @ write 21\n"                    \
        "vmax.f32   q14, q14, q0                @ relu\n"                    \
        "vst1.32   {d14-d15}, [%[c_ptr3]]!      @ write 30\n"                    \
        "vmax.f32   q9, q9, q0                  @ relu\n"                    \
        "vst1.32   {d26-d27}, [%[c_ptr3]]!      @ write 31\n"                    \
        "vmax.f32   q15, q15, q0                @ relu\n"                    \
        "vst1.32   {d16-d17}, [%[c_ptr4]]!      @ write 40\n"                    \
        "vst1.32   {d28-d29}, [%[c_ptr4]]!      @ write 41\n"                    \
        "vst1.32   {d18-d19}, [%[c_ptr5]]!      @ write 50\n"                    \
        "2:                                     @ end\n"                    \
        "vst1.32   {d30-d31}, [%[c_ptr5]]!      @ write 51\n"

#define SGEMM_CONV_4x8_IN \
        "vld1.32	{d0-d3}, [%[a_ptr] :128]!   @ load a0~a3\n"                          \
        "vmov.i32	q8, #0                      @ out0=0\n"                          \
        "vmov.i32	q9, #0                      @ out1=0\n"                          \
        "vld1.32   {d8-d11}, [%[b_ptr] :128]!   @ load b1\n"                          \
        "vmov.i32	q10, #0                     @ out2=0\n"                          \
        "pld [%[a_ptr]]                         @ preload a, 64byte\n"                          \
        "vmov.i32	q11, #0                     @ out3=0\n"                          \
        "pld [%[b_ptr]]                         @ preload b\n"                          \
        "vmov.i32	q12, #0                     @ out4=0\n"                          \
        "pld [%[a_ptr], #64]                    @ preload a\n"                          \
        "vmov.i32	q13, #0                     @ out5=0\n"                          \
        "pld [%[b_ptr], #64]                    @ preload b\n"                          \
        "vmov.i32	q14, #0                     @ out6=0\n"                          \
        "pld [%[a_ptr], #128]                   @ preload a\n"                          \
        "vmov.i32	q15, #0                     @ out7=0\n"                          \
        "pld [%[b_ptr], #128]                   @ preload b\n"

#define SGEMM_CONV_4x8_IN_BIAS \
        "vld1.32	{d0-d3}, [%[a_ptr] :128]!   @ load a0~a3\n"                       \
        "vdup.32    q8, %[bias0]                @ add bias to out0, row0\n"                       \
        "vdup.32    q9, %[bias1]                @ add bias to out1, row1\n"                       \
        "vld1.32   {d8-d11}, [%[b_ptr] :128]!   @ load b1\n"                       \
        "vdup.32    q10, %[bias2]               @ add bias to out2, row2\n"                       \
        "pld [%[a_ptr]]                         @ preload a, 64byte\n"                       \
        "vdup.32    q11, %[bias3]               @ add bias to out3, row3\n"                       \
        "pld [%[b_ptr]]                         @ preload b\n"                       \
        "vdup.32    q12, %[bias0]               @ add bias to out4, row0\n"                       \
        "pld [%[a_ptr], #64]                    @ preload a\n"                       \
        "vdup.32    q13, %[bias1]               @ add bias to out5, row1\n"                       \
        "pld [%[b_ptr], #64]                    @ preload b\n"                       \
        "vdup.32    q14, %[bias2]               @ add bias to out6, row2\n"                       \
        "pld [%[a_ptr], #128]                   @ preload a\n"                       \
        "vdup.32    q15, %[bias3]               @ add bias to out7, row3\n"                       \
        "pld [%[b_ptr], #128]                   @ preload b\n"


#define SGEMM_CONV_4x8_OUT \
        /* Unroll 3 */ \
       "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"                            \
        "vmla.f32	q12, q7, d6[0]              @ out4 += b2 * a0\n"                            \
        "vmla.f32	q9, q6, d6[1]               @ out1 += b1 * a1\n"                            \
        "vmla.f32	q13, q7, d6[1]              @ out5 += b2 * a1\n"                            \
        "vmla.f32	q10, q6, d7[0]              @ out2 += b1 * a2\n"                            \
        "vmla.f32	q14, q7, d7[0]              @ out6 += b2 * a2\n"                            \
        "vmla.f32	q11, q6, d7[1]              @ out3 += b1 * a3\n"                            \
        "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"                            \
        "vst1.32   {d16-d17}, [%[c_ptr0]]!      @ write 00\n"                            \
        "vst1.32   {d24-d25}, [%[c_ptr0]]!      @ write 01\n"                            \
        "vst1.32   {d18-d19}, [%[c_ptr1]]!      @ write 10\n"                            \
        "vst1.32   {d26-d27}, [%[c_ptr1]]!      @ write 11\n"                            \
        "vst1.32   {d20-d21}, [%[c_ptr2]]!      @ write 20\n"                            \
        "vst1.32   {d28-d29}, [%[c_ptr2]]!      @ write 21\n"                            \
        "vst1.32   {d22-d23}, [%[c_ptr3]]!      @ write 30\n"                            \
        "b		2f\n"                            \
        /* tails==1 final tail */ \
        "3:                                     @ tail=1\n"                            \
        "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"                            \
        "vmla.f32	q12, q5, d0[0]              @ out4 += b2 * a0\n"                            \
        "vmla.f32	q9, q4, d0[1]               @ out1 += b1 * a1\n"                            \
        "vmla.f32	q13, q5, d0[1]              @ out5 += b2 * a1\n"                            \
        "vmla.f32	q10, q4, d1[0]              @ out2 += b1 * a2\n"                            \
        "vmla.f32	q14, q5, d1[0]              @ out6 += b2 * a2\n"                            \
        "vmla.f32	q11, q4, d1[1]              @ out3 += b1 * a3\n"                            \
        "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"                            \
        "vst1.32   {d16-d17}, [%[c_ptr0]]!      @ write 00\n"                            \
        "vst1.32   {d24-d25}, [%[c_ptr0]]!      @ write 01\n"                            \
        "vst1.32   {d18-d19}, [%[c_ptr1]]!      @ write 10\n"                            \
        "vst1.32   {d26-d27}, [%[c_ptr1]]!      @ write 11\n"                            \
        "vst1.32   {d20-d21}, [%[c_ptr2]]!      @ write 20\n"                            \
        "vst1.32   {d28-d29}, [%[c_ptr2]]!      @ write 21\n"                            \
        /*aptr - 16 */  \
        "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"                            \
        "vst1.32   {d22-d23}, [%[c_ptr3]]! @ write 30\n"                            \
        "b		2f                              @ jump to end\n"                            \
        /* tails==2 final tail*/ \
        "4:                                     @ tail == 2\n"                            \
        "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"                            \
        "vmla.f32	q12, q7, d2[0]              @ out4 += b2 * a0\n"                            \
        "vmla.f32	q9, q6, d2[1]               @ out1 += b1 * a1\n"                            \
        "vmla.f32	q13, q7, d2[1]              @ out5 += b2 * a1\n"                            \
        "vmla.f32	q10, q6, d3[0]              @ out2 += b1 * a2\n"                            \
        "vmla.f32	q14, q7, d3[0]              @ out6 += b2 * a2\n"                            \
        "vmla.f32	q11, q6, d3[1]              @ out3 += b1 * a3\n"                            \
        "vmla.f32	q15, q7, d3[1]              @ out7 += b2 * a3\n"                            \
        "vst1.32   {d16-d17}, [%[c_ptr0]]!      @ write 00\n"                            \
        "vst1.32   {d24-d25}, [%[c_ptr0]]!      @ write 01\n"                            \
        "vst1.32   {d18-d19}, [%[c_ptr1]]!      @ write 10\n"                            \
        "vst1.32   {d26-d27}, [%[c_ptr1]]!      @ write 11\n"                            \
        "vst1.32   {d20-d21}, [%[c_ptr2]]!      @ write 20\n"                            \
        "vst1.32   {d28-d29}, [%[c_ptr2]]!      @ write 21\n"                            \
        "vst1.32   {d22-d23}, [%[c_ptr3]]!      @ write 30\n"                            \
        "b		2f                              @ jump to end\n"                            \
        /* tails==3 final tail*/ \
        "5:                                     @ tail=3\n"                            \
        "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"                            \
        "vmla.f32	q12, q5, d4[0]              @ out4 += b2 * a0\n"                            \
        "vmla.f32	q9, q4, d4[1]               @ out1 += b1 * a1\n"                            \
        "vmla.f32	q13, q5, d4[1]              @ out5 += b2 * a1\n"                            \
        "vmla.f32	q10, q4, d5[0]              @ out2 += b1 * a2\n"                            \
        "vmla.f32	q14, q5, d5[0]              @ out6 += b2 * a2\n"                            \
        "vmla.f32	q11, q4, d5[1]              @ out3 += b1 * a3\n"                            \
        "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"                            \
        "vst1.32   {d16-d17}, [%[c_ptr0]]!      @ write 00\n"                            \
        "vst1.32   {d24-d25}, [%[c_ptr0]]!      @ write 01\n"                            \
        "vst1.32   {d18-d19}, [%[c_ptr1]]!      @ write 10\n"                            \
        "vst1.32   {d26-d27}, [%[c_ptr1]]!      @ write 11\n"                            \
        "vst1.32   {d20-d21}, [%[c_ptr2]]!      @ write 20\n"                            \
        "vst1.32   {d28-d29}, [%[c_ptr2]]!      @ write 21\n"                            \
        /*aptr - 16*/ \
        "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"                            \
        "vst1.32   {d22-d23}, [%[c_ptr3]]!      @ write 30\n"                            \
        "2:                                     @ end\n"                            \
        "vst1.32   {d30-d31}, [%[c_ptr3]]!      @ write 31\n"

#define SGEMM_CONV_4x8_OUT_RELU \
        /* with relu*/ \
        /* Unroll 3*/ \
        "vmov.u32   q4, #0                      @ for relu\n"                      \
        "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"                      \
        "vmla.f32	q12, q7, d6[0]              @ out4 += b2 * a0\n"                      \
        "vmla.f32	q9, q6, d6[1]               @ out1 += b1 * a1\n"                      \
        "vmla.f32	q13, q7, d6[1]              @ out5 += b2 * a1\n"                      \
        "vmla.f32	q10, q6, d7[0]              @ out2 += b1 * a2\n"                      \
        "vmla.f32	q14, q7, d7[0]              @ out6 += b2 * a2\n"                      \
        "vmla.f32	q11, q6, d7[1]              @ out3 += b1 * a3\n"                      \
        "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"                      \
        "vmax.f32   q8, q8, q4                  @ relu\n"                      \
        "vmax.f32   q12, q12, q4                @ relu\n"                      \
        "vmax.f32   q9, q9, q4                  @ relu\n"                      \
        "vmax.f32   q13, q13, q4                @ relu\n"                      \
        "vmax.f32   q10, q10, q4                @ relu\n"                      \
        "vmax.f32   q14, q14, q4                @ relu\n"                      \
        "vmax.f32   q11, q11, q4                @ relu\n"                      \
        "vmax.f32   q15, q15, q4                @ relu\n"                      \
        "vst1.32   {d16-d17}, [%[c_ptr0]]!      @ write 00\n"                      \
        "vst1.32   {d24-d25}, [%[c_ptr0]]!      @ write 01\n"                      \
        "vst1.32   {d18-d19}, [%[c_ptr1]]!      @ write 10\n"                      \
        "vst1.32   {d26-d27}, [%[c_ptr1]]!      @ write 11\n"                      \
        "vst1.32   {d20-d21}, [%[c_ptr2]]!      @ write 20\n"                      \
        "vst1.32   {d28-d29}, [%[c_ptr2]]!      @ write 21\n"                      \
        "vst1.32   {d22-d23}, [%[c_ptr3]]!      @ write 30\n"                      \
        "b		2f\n"                      \
        /* tails==1 final tail*/ \
        "3:                                     @ tail=1\n"                      \
        "vmov.u32   q6, #0                      @ for relu\n"                      \
        "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"                      \
        "vmla.f32	q12, q5, d0[0]              @ out4 += b2 * a0\n"                      \
        "vmla.f32	q9, q4, d0[1]               @ out1 += b1 * a1\n"                      \
        "vmla.f32	q13, q5, d0[1]              @ out5 += b2 * a1\n"                      \
        "vmla.f32	q10, q4, d1[0]              @ out2 += b1 * a2\n"                      \
        "vmla.f32	q14, q5, d1[0]              @ out6 += b2 * a2\n"                      \
        "vmla.f32	q11, q4, d1[1]              @ out3 += b1 * a3\n"                      \
        "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"                      \
        "vmax.f32   q8, q8, q6                  @ relu\n"                      \
        "vmax.f32   q12, q12, q6                @ relu\n"                      \
        "vmax.f32   q9, q9, q6                  @ relu\n"                      \
        "vmax.f32   q13, q13, q6                @ relu\n"                      \
        "vmax.f32   q10, q10, q6                @ relu\n"                      \
        "vmax.f32   q14, q14, q6                @ relu\n"                      \
        "vmax.f32   q11, q11, q6                @ relu\n"                      \
        "vmax.f32   q15, q15, q6                @ relu\n"                      \
        "vst1.32   {d16-d17}, [%[c_ptr0]]!      @ write 00\n"                      \
        "vst1.32   {d24-d25}, [%[c_ptr0]]!      @ write 01\n"                      \
        "vst1.32   {d18-d19}, [%[c_ptr1]]!      @ write 10\n"                      \
        "vst1.32   {d26-d27}, [%[c_ptr1]]!      @ write 11\n"                      \
        "vst1.32   {d20-d21}, [%[c_ptr2]]!      @ write 20\n"                      \
        "vst1.32   {d28-d29}, [%[c_ptr2]]!      @ write 21\n"                      \
        /*aptr - 16*/ \
        "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"                      \
        "vst1.32   {d22-d23}, [%[c_ptr3]]! @ write 30\n"                      \
        "b		2f                              @ jump to end\n"                      \
        /* tails==2 final tail*/ \
        "4:                                     @ tail == 2\n"                      \
        "vmov.u32   q4, #0                      @ for relu\n"                      \
        "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"                      \
        "vmla.f32	q12, q7, d2[0]              @ out4 += b2 * a0\n"                      \
        "vmla.f32	q9, q6, d2[1]               @ out1 += b1 * a1\n"                      \
        "vmla.f32	q13, q7, d2[1]              @ out5 += b2 * a1\n"                      \
        "vmla.f32	q10, q6, d3[0]              @ out2 += b1 * a2\n"                      \
        "vmla.f32	q14, q7, d3[0]              @ out6 += b2 * a2\n"                      \
        "vmla.f32	q11, q6, d3[1]              @ out3 += b1 * a3\n"                      \
        "vmla.f32	q15, q7, d3[1]              @ out7 += b2 * a3\n"                      \
        "vmax.f32   q8, q8, q4                  @ relu\n"                      \
        "vmax.f32   q12, q12, q4                @ relu\n"                      \
        "vmax.f32   q9, q9, q4                  @ relu\n"                      \
        "vmax.f32   q13, q13, q4                @ relu\n"                      \
        "vmax.f32   q10, q10, q4                @ relu\n"                      \
        "vmax.f32   q14, q14, q4                @ relu\n"                      \
        "vmax.f32   q11, q11, q4                @ relu\n"                      \
        "vmax.f32   q15, q15, q4                @ relu\n"                      \
        "vst1.32   {d16-d17}, [%[c_ptr0]]!      @ write 00\n"                      \
        "vst1.32   {d24-d25}, [%[c_ptr0]]!      @ write 01\n"                      \
        "vst1.32   {d18-d19}, [%[c_ptr1]]!      @ write 10\n"                      \
        "vst1.32   {d26-d27}, [%[c_ptr1]]!      @ write 11\n"                      \
        "vst1.32   {d20-d21}, [%[c_ptr2]]!      @ write 20\n"                      \
        "vst1.32   {d28-d29}, [%[c_ptr2]]!      @ write 21\n"                      \
        "vst1.32   {d22-d23}, [%[c_ptr3]]!      @ write 30\n"                      \
        "b		2f                              @ jump to end\n"                      \
        /* tails==3 final tail */ \
        "5:                                     @ tail=3\n"                      \
        "vmov.u32   q6, #0                      @ for relu\n"                      \
        "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"                      \
        "vmla.f32	q12, q5, d4[0]              @ out4 += b2 * a0\n"                      \
        "vmla.f32	q9, q4, d4[1]               @ out1 += b1 * a1\n"                      \
        "vmla.f32	q13, q5, d4[1]              @ out5 += b2 * a1\n"                      \
        "vmla.f32	q10, q4, d5[0]              @ out2 += b1 * a2\n"                      \
        "vmla.f32	q14, q5, d5[0]              @ out6 += b2 * a2\n"                      \
        "vmla.f32	q11, q4, d5[1]              @ out3 += b1 * a3\n"                      \
        "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"                      \
        "vmax.f32   q8, q8, q6                  @ relu\n"                      \
        "vmax.f32   q12, q12, q6                @ relu\n"                      \
        "vmax.f32   q9, q9, q6                  @ relu\n"                      \
        "vmax.f32   q13, q13, q6                @ relu\n"                      \
        "vmax.f32   q10, q10, q6                @ relu\n"                      \
        "vmax.f32   q14, q14, q6                @ relu\n"                      \
        "vmax.f32   q11, q11, q6                @ relu\n"                      \
        "vmax.f32   q15, q15, q6                @ relu\n"                      \
        "vst1.32   {d16-d17}, [%[c_ptr0]]!      @ write 00\n"                      \
        "vst1.32   {d24-d25}, [%[c_ptr0]]!      @ write 01\n"                      \
        "vst1.32   {d18-d19}, [%[c_ptr1]]!      @ write 10\n"                      \
        "vst1.32   {d26-d27}, [%[c_ptr1]]!      @ write 11\n"                      \
        "vst1.32   {d20-d21}, [%[c_ptr2]]!      @ write 20\n"                      \
        "vst1.32   {d28-d29}, [%[c_ptr2]]!      @ write 21\n"                      \
        /*aptr - 16*/ \
        "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"                      \
        "vst1.32   {d22-d23}, [%[c_ptr3]]!      @ write 30\n"                      \
        "2:                                     @ end\n"                      \
        "vst1.32   {d30-d31}, [%[c_ptr3]]!      @ write 31\n"                      \

/**
 * \brief gemm with ablock = 6, bblock = 8, output 6x8
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @param threads
 * @param workspace
 */
void sgemm_conv_6x8(const float* A_packed, const float* B, float* C, int M, int N, int K, bool transB, Context* ctx) {
    //L2 cache 2M
    // output width = 8
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }

    float zerobuf[x_block];

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
        float* b_pannel = static_cast<float*>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 6) {
            unsigned int ymax = y + 6;
            if (ymax > M) {
                ymax = M;
            }
            float* c_ptr0 = C + y * N + x0;
            float* c_ptr1 = c_ptr0 + N;
            float* c_ptr2 = c_ptr1 + N;
            float* c_ptr3 = c_ptr2 + N;
            float* c_ptr4 = c_ptr3 + N;
            float* c_ptr5 = c_ptr4 + N;

            float* pout0 = c_ptr0;
            float* pout1 = c_ptr1;
            float* pout2 = c_ptr2;
            float* pout3 = c_ptr3;
            float* pout4 = c_ptr4;
            float* pout5 = c_ptr5;

            float cout0[8];
            float cout1[8];
            float cout2[8];
            float cout3[8];
            float cout4[8];
            float cout5[8];

            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        c_ptr1 = zerobuf;
                    case 3:
                        c_ptr2 = zerobuf;
                    case 2:
                        c_ptr3 = zerobuf;
                    case 1:
                        c_ptr4 = zerobuf;
                    case 0:
                        c_ptr5 = zerobuf;
                    default:
                        break;
                }
            }

            const float* a_ptr_l = A_packed + y * K;
            const float* b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {

                if (flag_p_remain && (xb == bblocks - 1)) {

                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    pout4 = c_ptr4;
                    pout5 = c_ptr5;

                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                    c_ptr4 = cout4;
                    c_ptr5 = cout5;
                }
                const float* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                //sgemm 6x8
                        SGEMM_CONV_IN_6x8
                        SGEMM_KERNEL6x8
                        SGEMM_CONV_OUT_6x8
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [c_ptr4] "+r" (c_ptr4), [c_ptr5] "+r" (c_ptr5), \
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
                        *pout4++ = cout4[i];
                        *pout5++ = cout5[i];
                    }
                }
            }
        }
    }
}

void sgemm_conv_6x8_relu(const float* A_packed, const float* B, float* C, int M, int N, int K, bool transB, Context* ctx) {
    //L2 cache 2M
    // output width = 8
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }

    float zerobuf[x_block];

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
        float* b_pannel = static_cast<float*>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 6) {
            unsigned int ymax = y + 6;
            if (ymax > M) {
                ymax = M;
            }

            float cout0[8];
            float cout1[8];
            float cout2[8];
            float cout3[8];
            float cout4[8];
            float cout5[8];

            float* c_ptr0 = C + y * N + x0;
            float* c_ptr1 = c_ptr0 + N;
            float* c_ptr2 = c_ptr1 + N;
            float* c_ptr3 = c_ptr2 + N;
            float* c_ptr4 = c_ptr3 + N;
            float* c_ptr5 = c_ptr4 + N;

            float* pout0 = c_ptr0;
            float* pout1 = c_ptr1;
            float* pout2 = c_ptr2;
            float* pout3 = c_ptr3;
            float* pout4 = c_ptr4;
            float* pout5 = c_ptr5;

            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        c_ptr1 = zerobuf;
                    case 3:
                        c_ptr2 = zerobuf;
                    case 2:
                        c_ptr3 = zerobuf;
                    case 1:
                        c_ptr4 = zerobuf;
                    case 0:
                        c_ptr5 = zerobuf;
                    default:
                        break;
                }
            }

            const float* a_ptr_l = A_packed + y * K;
            const float* b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {

                if (flag_p_remain && (xb == bblocks - 1)) {

                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    pout4 = c_ptr4;
                    pout5 = c_ptr5;

                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                    c_ptr4 = cout4;
                    c_ptr5 = cout5;
                }
                const float* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                //sgemm 6x8
                        SGEMM_CONV_IN_6x8
                        SGEMM_KERNEL6x8
                        SGEMM_CONV_OUT_6x8_RELU
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [c_ptr4] "+r" (c_ptr4), [c_ptr5] "+r" (c_ptr5), \
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
                        *pout4++ = cout4[i];
                        *pout5++ = cout5[i];
                    }
                }
            }
        }
    }
}

void sgemm_conv_6x8_bias(const float* A_packed, const float* B, const float* bias, float* C, int M, int N, int K, bool transB, Context* ctx) {
    //L2 cache 2M
    // output width = 8
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }

    float zerobuf[x_block];

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
        float* b_pannel = static_cast<float*>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 6) {
            unsigned int ymax = y + 6;
            if (ymax > M) {
                ymax = M;
            }

            float bias_ptr[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }

            float cout0[8];
            float cout1[8];
            float cout2[8];
            float cout3[8];
            float cout4[8];
            float cout5[8];

            float* c_ptr0 = C + y * N + x0;
            float* c_ptr1 = c_ptr0 + N;
            float* c_ptr2 = c_ptr1 + N;
            float* c_ptr3 = c_ptr2 + N;
            float* c_ptr4 = c_ptr3 + N;
            float* c_ptr5 = c_ptr4 + N;

            float* pout0 = c_ptr0;
            float* pout1 = c_ptr1;
            float* pout2 = c_ptr2;
            float* pout3 = c_ptr3;
            float* pout4 = c_ptr4;
            float* pout5 = c_ptr5;

            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        c_ptr1 = zerobuf;
                    case 3:
                        c_ptr2 = zerobuf;
                    case 2:
                        c_ptr3 = zerobuf;
                    case 1:
                        c_ptr4 = zerobuf;
                    case 0:
                        c_ptr5 = zerobuf;
                    default:
                        break;
                }
            }

            const float* a_ptr_l = A_packed + y * K;
            const float* b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {

                if (flag_p_remain && (xb == bblocks - 1)) {

                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    pout4 = c_ptr4;
                    pout5 = c_ptr5;

                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                    c_ptr4 = cout4;
                    c_ptr5 = cout5;
                }
                const float* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                //sgemm 6x8
                        SGEMM_CONV_IN_6x8_BIAS
                        SGEMM_KERNEL6x8
                        SGEMM_CONV_OUT_6x8
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [c_ptr4] "+r" (c_ptr4), [c_ptr5] "+r" (c_ptr5), \
                    [k] "+r" (k), [tails] "+r" (tails)
                : [bias_ptr] "r" (bias_ptr)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc"
                );

                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                        *pout4++ = cout4[i];
                        *pout5++ = cout5[i];
                    }
                }
            }
        }
    }
}

void sgemm_conv_6x8_bias_relu(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx) {
    //L2 cache 2M
    // output width = 8
    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();
    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }

    float zerobuf[x_block];

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
        float* b_pannel = static_cast<float*>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 6) {
            unsigned int ymax = y + 6;
            if (ymax > M) {
                ymax = M;
            }

            float bias_ptr[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }

            float cout0[8];
            float cout1[8];
            float cout2[8];
            float cout3[8];
            float cout4[8];
            float cout5[8];

            float* c_ptr0 = C + y * N + x0;
            float* c_ptr1 = c_ptr0 + N;
            float* c_ptr2 = c_ptr1 + N;
            float* c_ptr3 = c_ptr2 + N;
            float* c_ptr4 = c_ptr3 + N;
            float* c_ptr5 = c_ptr4 + N;

            float* pout0 = c_ptr0;
            float* pout1 = c_ptr1;
            float* pout2 = c_ptr2;
            float* pout3 = c_ptr3;
            float* pout4 = c_ptr4;
            float* pout5 = c_ptr5;

            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        c_ptr1 = zerobuf;
                    case 3:
                        c_ptr2 = zerobuf;
                    case 2:
                        c_ptr3 = zerobuf;
                    case 1:
                        c_ptr4 = zerobuf;
                    case 0:
                        c_ptr5 = zerobuf;
                    default:
                        break;
                }
            }

            const float* a_ptr_l = A_packed + y * K;
            const float* b_ptr = b_pannel;
            for (int xb = 0; xb < bblocks; xb++) {

                if (flag_p_remain && (xb == bblocks - 1)) {

                    pout0 = c_ptr0;
                    pout1 = c_ptr1;
                    pout2 = c_ptr2;
                    pout3 = c_ptr3;
                    pout4 = c_ptr4;
                    pout5 = c_ptr5;

                    c_ptr0 = cout0;
                    c_ptr1 = cout1;
                    c_ptr2 = cout2;
                    c_ptr3 = cout3;
                    c_ptr4 = cout4;
                    c_ptr5 = cout5;
                }
                const float* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                // sgemm 6x8
                        SGEMM_CONV_IN_6x8_BIAS
                        SGEMM_KERNEL6x8
                        SGEMM_CONV_OUT_6x8_RELU
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [c_ptr4] "+r" (c_ptr4), [c_ptr5] "+r" (c_ptr5), \
                    [k] "+r" (k), [tails] "+r" (tails)
                : [bias_ptr] "r" (bias_ptr)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc"
                );

                if (flag_p_remain && (xb == bblocks - 1)) {
                    for (int i = 0; i < remain; ++i) {
                        *pout0++ = cout0[i];
                        *pout1++ = cout1[i];
                        *pout2++ = cout2[i];
                        *pout3++ = cout3[i];
                        *pout4++ = cout4[i];
                        *pout5++ = cout5[i];
                    }
                }
            }
        }
    }
}

void sgemm_conv_4x8(const float* A_packed, const float* B, float* C, \
    int M, int N, int K, bool transB, Context* ctx) {

    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();

    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }

    float zerobuf[x_block];

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
        float* b_pannel = static_cast<float*>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }

            float cout0[8];
            float cout1[8];
            float cout2[8];
            float cout3[8];

            float* c_ptr0 = C + y * N + x0;
            float* c_ptr1 = c_ptr0 + N;
            float* c_ptr2 = c_ptr1 + N;
            float* c_ptr3 = c_ptr2 + N;

            float* pout0 = c_ptr0;
            float* pout1 = c_ptr1;
            float* pout2 = c_ptr2;
            float* pout3 = c_ptr3;

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

            const float* a_ptr_l = A_packed + y * K;
            const float* b_ptr = b_pannel;
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
                const float* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                        /* sgemm kernel*/
                        SGEMM_CONV_4x8_IN
                        SGEMM_KERNEL_4x8
                        SGEMM_CONV_4x8_OUT
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

void sgemm_conv_4x8_relu(const float* A_packed, const float* B, float* C, \
    int M, int N, int K, bool transB, Context* ctx) {

    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();

    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }

    float zerobuf[x_block];

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
        float* b_pannel = static_cast<float*>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }

            float cout0[8];
            float cout1[8];
            float cout2[8];
            float cout3[8];

            float* c_ptr0 = C + y * N + x0;
            float* c_ptr1 = c_ptr0 + N;
            float* c_ptr2 = c_ptr1 + N;
            float* c_ptr3 = c_ptr2 + N;

            float* pout0 = c_ptr0;
            float* pout1 = c_ptr1;
            float* pout2 = c_ptr2;
            float* pout3 = c_ptr3;

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

            const float* a_ptr_l = A_packed + y * K;
            const float* b_ptr = b_pannel;
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
                const float* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                // sgemm 4x8
                        SGEMM_CONV_4x8_IN
                        SGEMM_KERNEL_4x8
                        SGEMM_CONV_4x8_OUT_RELU
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

void sgemm_conv_4x8_bias(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx) {

    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();

    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }

    float zerobuf[x_block];

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
        float* b_pannel = static_cast<float*>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }

            float cout0[8];
            float cout1[8];
            float cout2[8];
            float cout3[8];

            float bias_ptr[4] = {0.f, 0.f, 0.f, 0.f};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }
            float bias0 = bias_ptr[0];
            float bias1 = bias_ptr[1];
            float bias2 = bias_ptr[2];
            float bias3 = bias_ptr[3];

            float* c_ptr0 = C + y * N + x0;
            float* c_ptr1 = c_ptr0 + N;
            float* c_ptr2 = c_ptr1 + N;
            float* c_ptr3 = c_ptr2 + N;

            float* pout0 = c_ptr0;
            float* pout1 = c_ptr1;
            float* pout2 = c_ptr2;
            float* pout3 = c_ptr3;

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

            const float* a_ptr_l = A_packed + y * K;
            const float* b_ptr = b_pannel;
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
                const float* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                // sgemm 4x8
                        SGEMM_CONV_4x8_IN_BIAS
                        SGEMM_KERNEL_4x8
                        SGEMM_CONV_4x8_OUT
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [k] "+r" (k), [tails] "+r" (tails)
                : [bias0] "r" (bias0), [bias1] "r" (bias1), [bias2] "r" (bias2), [bias3] "r" (bias3)
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

void sgemm_conv_4x8_bias_relu(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, bool transB, Context* ctx) {

    size_t l2_cache = ctx->l2_cache_size() > 0? ctx->l2_cache_size() : 512 * 1024;
    void* workspace = ctx->get_work_space();
    int threads = ctx->get_threads();

    int x_block = l2_cache / (sizeof(float) * K);
    x_block /= 8;
    x_block *= 8;
    int x_num = (N + (x_block - 1)) / x_block;
    x_block = (N + x_num - 1) / x_num;
    x_block = (x_block + 7) / 8;
    x_block *= 8;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }

    float zerobuf[x_block];

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
        float* b_pannel = static_cast<float*>(workspace);
        if (transB) {
            loadb_trans(b_pannel, B, K, 0, K, x0, xmax);
        } else {
            loadb(b_pannel, B, N, 0, K, x0, xmax);
        }
#pragma omp parallel for num_threads(threads)
        for (unsigned int y = 0; y < M; y += 4) {
            unsigned int ymax = y + 4;
            if (ymax > M) {
                ymax = M;
            }

            float cout0[8];
            float cout1[8];
            float cout2[8];
            float cout3[8];

            float bias_ptr[4] = {0.f, 0.f, 0.f, 0.f};
            for (int j = y; j < ymax; ++j) {
                bias_ptr[j - y] = bias[j];
            }
            float bias0 = bias_ptr[0];
            float bias1 = bias_ptr[1];
            float bias2 = bias_ptr[2];
            float bias3 = bias_ptr[3];

            float* c_ptr0 = C + y * N + x0;
            float* c_ptr1 = c_ptr0 + N;
            float* c_ptr2 = c_ptr1 + N;
            float* c_ptr3 = c_ptr2 + N;

            float* pout0 = c_ptr0;
            float* pout1 = c_ptr1;
            float* pout2 = c_ptr2;
            float* pout3 = c_ptr3;

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

            const float* a_ptr_l = A_packed + y * K;
            const float* b_ptr = b_pannel;
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
                const float* a_ptr = a_ptr_l;
                int tails = tail_pre;
                int k = k_pre;
                asm volatile (
                // sgemm 4x8
                        SGEMM_CONV_4x8_IN_BIAS
                        SGEMM_KERNEL_4x8
                        SGEMM_CONV_4x8_OUT_RELU
                : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr0] "+r" (c_ptr0), \
                    [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3), \
                    [k] "+r" (k), [tails] "+r" (tails)
                : [bias0] "r" (bias0), [bias1] "r" (bias1), [bias2] "r" (bias2), [bias3] "r" (bias3)
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

#endif //__aarch64__
} //lite

} //saber

} //anakin



