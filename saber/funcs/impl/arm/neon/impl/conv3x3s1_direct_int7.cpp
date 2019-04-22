#include "saber/funcs/impl/arm/neon/impl/conv_arm_impl.h"
#include "saber/funcs/impl/arm/neon/impl/conv_block_utils.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

void conv_3x3s1_direct_int7(const int8_t* din, int32_t* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const int8_t* weights, const int32_t* bias, \
                          ConvParam<ARM>& param, Context<ARM>& ctx, DataType out_type, const float* scale){
    //! 3x3s1 convolution, implemented by direct algorithm
    // //! pad is done implicit
    // printf("conv_3x3s1_direct_int8 \n");
    const int hin_r_block = 4;
    const int hout_c_block = 8;
    const int hout_r_block = 2;

    int stride_w = param.stride_w;
    int pad_w = param.pad_w;
    int pad_h = param.pad_h;
    bool flag_relu = false;
    bool flag_bias = param.bias()->size() > 0;
    if (param.activation_param.has_active){
        if (param.activation_param.active == Active_relu || fabs(param.activation_param.negative_slope) > 1e-6f){
            flag_relu = true;
        }
    }

    int wout_round = ((wout + 3) / 4) * 4;
    int win_round = wout_round * stride_w + 4;

    int threads = ctx.get_threads();

    int* tmp_work_space = static_cast<int*>(ctx.get_work_space());
    int* ptr_zero = tmp_work_space;
    memset(ptr_zero, 0, sizeof(int) * win_round);
    int* ptr_write = ptr_zero + win_round;

    int in_len = win_round * chin;
    int pre_in_size = hin_r_block * in_len;
    int pre_out_size = hout_c_block * hout_r_block * wout_round;

    signed char* pre_din = reinterpret_cast<signed char*>(ptr_write + wout_round);

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    int ws = -pad_w;
    int we = ws + win_round;
    int w_loop = wout_round / 4;

    int size_out = wout_round * hout_c_block;

    // printf("win_round: %d, wout_round: %d, ws: %d, we: %d\n", win_round, wout_round, ws, we);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        signed char* dout_batch = reinterpret_cast<signed char*>(dout) + n * chout * size_out_channel* type_length(out_type);
        //int *dout_batch = static_cast<int*>(dout) + n * chout * size_out_channel;

        for (int h = 0; h < hout; h += 2) {
            int hs = h - pad_h;
            int he = hs + 4;
            // printf("hs: %d, he: %d, chin: %d, hin: %d, win: %d \n", hs, he, chin, hin, win);
            prepack_input_nxw(din_batch, pre_din, 0, chin, hs, he, ws, we, chin, win, hin, (signed char*)ptr_zero);

#pragma omp parallel for num_threads(threads)
            for (int c = 0; c < chout; c += 8) {

#ifdef USE_OPENMP
                int* pre_out = reinterpret_cast<int*>(pre_din + (pre_in_size + 3) / 4 * 4) + omp_get_thread_num() * pre_out_size;
#else
                int* pre_out = reinterpret_cast<int*>(pre_din + (pre_in_size + 3) / 4 * 4);
#endif
                // printf("ptr_zero_int: %x, ptr_zero: %x, ptr_write: %x, pre_din: %x, pre_out: %x \n", ptr_zero_int, ptr_zero, ptr_write, pre_din, pre_out);
                const signed char* inr0 = pre_din;
                const signed char* inr1 = inr0 + in_len;
                const signed char* inr2 = inr1 + in_len;
                const signed char* inr3 = inr2 + in_len;

                const signed char* wc0 = weights + c * w_stride;

                const int* bias_ptr = ptr_zero;
                if (flag_bias) {
                    bias_ptr = bias + c;
                }
                //hout_r_block * wout_round * hout_c_block
                fill_packed_bias_nxmw_int8(bias_ptr, pre_out, hout_c_block, hout_r_block, wout_round);

                for (int i = 0; i < chin; ++i) {
                    const signed char* r0 = inr0;
                    const signed char* r1 = inr1;
                    const signed char* r2 = inr2;
                    const signed char* r3 = inr3;

                    int* ptr_out0 = pre_out;
                    int* ptr_out1 = pre_out + size_out;
                    int cnt = w_loop;
#ifdef __aarch64__
                    int8x8_t v0 = vld1_s8(wc0);      //w0
                    int8x8_t v1 = vld1_s8(wc0 + 8);  //w1
                    int8x8_t v2 = vld1_s8(wc0 + 16); //w2,

                    int8x8_t v3 = vld1_s8(wc0 + 24); //w3
                    int8x8_t v4 = vld1_s8(wc0 + 32); //w4
                    int8x8_t v5 = vld1_s8(wc0 + 40); //w5

                    int8x8_t v6 = vld1_s8(wc0 + 48); //w6
                    int8x8_t v7 = vld1_s8(wc0 + 56); //w7
                    int8x8_t v8 = vld1_s8(wc0 + 64); //w8

                    asm volatile(
                    "1:                                          \n"         /* main loop*/
                      "ld1    {v0.8b}, [%[r0]]                       \n"        /* load a00-a015 to q0*/
                      "ldp    q13, q14, [%[ptr_out0]]             \n"         /* load outr00-outr80*/
                      "ldp    q15, q16, [%[ptr_out0], #32]         \n"         /* load outr00-outr80*/
                      "ldp    q17, q18, [%[ptr_out0], #64]        \n"         /* load outr00-outr80*/
                      "ldp    q19, q20, [%[ptr_out0], #96]          \n"         /* load outr00-outr80*/

                      "dup    v1.8b, v0.b[0]                       \n"         /* dup vdupq_n_s8(s[0])*/
                      "dup    v2.8b, v0.b[1]                       \n"         /* dup vdupq_n_s8(s[1])*/
                      "dup    v3.8b, v0.b[2]                       \n"         /* dup vdupq_n_s8(s[2])*/
                      "dup    v4.8b, v0.b[3]                       \n"         /* dup vdupq_n_s8(s[3])*/

                    //0
                      "smull  v5.8h , %[v0].8b, v1.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smull  v6.8h , %[v0].8b, v2.8b             \n"                /*q10 = 8x8bit*/
                      "smull  v7.8h , %[v0].8b, v3.8b             \n"                 /*q9 = 8x8bit*/
                      "smull  v8.8h , %[v0].8b, v4.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v1.8b, v0.b[4]                       \n"         /* dup vdupq_n_s8(s[3])*/
                      "add   %[r0], %[r0], #4                       \n"

                      "smlal  v5.8h , %[v1].8b, v2.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v6.8h , %[v1].8b, v3.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v7.8h , %[v1].8b, v4.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v8.8h , %[v1].8b, v1.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v2.8b, v0.b[5]                       \n"         /* dup vdupq_n_s8(s[3])*/
                      "ld1    {v0.8b}, [%[r1]]                       \n"        /* load a00-a015 to q0*/

                      "smlal  v5.8h , %[v2].8b, v3.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v6.8h , %[v2].8b, v4.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v7.8h , %[v2].8b, v1.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v8.8h , %[v2].8b, v2.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v1.8b, v0.b[0]                       \n"         /* dup vdupq_n_s8(s[0])*/
                      "dup    v2.8b, v0.b[1]                       \n"         /* dup vdupq_n_s8(s[1])*/
                      "dup    v3.8b, v0.b[2]                       \n"         /* dup vdupq_n_s8(s[2])*/
                      "dup    v4.8b, v0.b[3]                       \n"         /* dup vdupq_n_s8(s[3])*/
                    //1
                      "add   %[r1], %[r1], #4                       \n"
                      "smull  v9.8h , %[v0].8b, v1.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smull  v10.8h , %[v0].8b, v2.8b             \n"                /*q10 = 8x8bit*/
                      "smull  v11.8h , %[v0].8b, v3.8b             \n"                 /*q9 = 8x8bit*/
                      "smull  v12.8h , %[v0].8b, v4.8b              \n"                /*q10 = 8x8bit*/

                      "smlal  v5.8h , %[v3].8b, v1.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v6.8h , %[v3].8b, v2.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v7.8h , %[v3].8b, v3.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v8.8h , %[v3].8b, v4.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v1.8b, v0.b[4]                       \n"         /* dup vdupq_n_s8(s[3])*/

                      "smlal  v9.8h , %[v1].8b, v2.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v10.8h , %[v1].8b, v3.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v11.8h , %[v1].8b, v4.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v12.8h , %[v1].8b, v1.8b              \n"                /*q10 = 8x8bit*/

                      "smlal  v5.8h , %[v4].8b, v2.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v6.8h , %[v4].8b, v3.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v7.8h , %[v4].8b, v4.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v8.8h , %[v4].8b, v1.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v2.8b, v0.b[5]                       \n"         /* dup vdupq_n_s8(s[3])*/
                      "ld1    {v0.8b}, [%[r2]]                       \n"        /* load a00-a015 to q0*/

                      "smlal  v9.8h , %[v2].8b, v3.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v10.8h , %[v2].8b, v4.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v11.8h , %[v2].8b, v1.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v12.8h , %[v2].8b, v2.8b              \n"                /*q10 = 8x8bit*/

                      "smlal  v5.8h , %[v5].8b, v3.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v6.8h , %[v5].8b, v4.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v7.8h , %[v5].8b, v1.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v8.8h , %[v5].8b, v2.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v1.8b, v0.b[0]                       \n"         /* dup vdupq_n_s8(s[0])*/
                      "dup    v2.8b, v0.b[1]                       \n"         /* dup vdupq_n_s8(s[1])*/
                      "dup    v3.8b, v0.b[2]                       \n"         /* dup vdupq_n_s8(s[2])*/
                      "dup    v4.8b, v0.b[3]                       \n"         /* dup vdupq_n_s8(s[3])*/

                    //2
                      "add   %[r2], %[r2], #4                       \n"
                      "smlal  v9.8h , %[v3].8b, v1.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v10.8h , %[v3].8b, v2.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v11.8h , %[v3].8b, v3.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v12.8h , %[v3].8b, v4.8b              \n"                /*q10 = 8x8bit*/

                      "smlal  v5.8h , %[v6].8b, v1.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v6.8h , %[v6].8b, v2.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v7.8h , %[v6].8b, v3.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v8.8h , %[v6].8b, v4.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v1.8b, v0.b[4]                       \n"         /* dup vdupq_n_s8(s[3])*/

                      "smlal  v9.8h , %[v4].8b, v2.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v10.8h , %[v4].8b, v3.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v11.8h , %[v4].8b, v4.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v12.8h , %[v4].8b, v1.8b              \n"                /*q10 = 8x8bit*/

                      "smlal  v5.8h , %[v7].8b, v2.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v6.8h , %[v7].8b, v3.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v7.8h , %[v7].8b, v4.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v8.8h , %[v7].8b, v1.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v2.8b, v0.b[5]                       \n"         /* dup vdupq_n_s8(s[3])*/
                      "ld1    {v0.8b}, [%[r3]]                        \n"        /* load a00-a015 to q0*/

                      "smlal  v9.8h , %[v5].8b, v3.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v10.8h , %[v5].8b, v4.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v11.8h , %[v5].8b, v1.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v12.8h , %[v5].8b, v2.8b              \n"                /*q10 = 8x8bit*/

                      "saddw  v13.4s, v13.4s, v5.4h\n"                /*  accumulate to result low*/
                      "saddw2  v14.4s, v14.4s, v5.8h\n"                /*  accumulate to result high*/

                      "saddw  v15.4s, v15.4s, v6.4h\n"                /*  accumulate to result low*/
                      "saddw2  v16.4s, v16.4s, v6.8h\n"                /*  accumulate to result high*/

                      "saddw  v17.4s, v17.4s, v7.4h\n"                /*  accumulate to result low*/
                      "saddw2  v18.4s, v18.4s, v7.8h\n"                /*  accumulate to result high*/

                      "saddw  v19.4s, v19.4s, v8.4h\n"                /*  accumulate to result low*/
                      "saddw2  v20.4s, v20.4s, v8.8h\n"                /*  accumulate to result high*/

                      "smull  v5.8h , %[v8].8b, v3.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smull  v6.8h , %[v8].8b, v4.8b             \n"                /*q10 = 8x8bit*/
                      "smull  v7.8h , %[v8].8b, v1.8b             \n"                 /*q9 = 8x8bit*/
                      "smull  v8.8h , %[v8].8b, v2.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v1.8b, v0.b[0]                       \n"         /* dup vdupq_n_s8(s[0])*/
                      "dup    v2.8b, v0.b[1]                       \n"         /* dup vdupq_n_s8(s[1])*/
                      "dup    v3.8b, v0.b[2]                       \n"         /* dup vdupq_n_s8(s[2])*/
                      "dup    v4.8b, v0.b[3]                       \n"         /* dup vdupq_n_s8(s[3])*/

                      "saddw  v13.4s, v13.4s, v5.4h\n"                /*  accumulate to result low*/
                      "saddw2  v14.4s, v14.4s, v5.8h\n"                /*  accumulate to result high*/

                      "saddw  v15.4s, v15.4s, v6.4h\n"                /*  accumulate to result low*/
                      "saddw2  v16.4s, v16.4s, v6.8h\n"                /*  accumulate to result high*/

                      "saddw  v17.4s, v17.4s, v7.4h\n"                /*  accumulate to result low*/
                      "saddw2  v18.4s, v18.4s, v7.8h\n"                /*  accumulate to result high*/

                      "saddw  v19.4s, v19.4s, v8.4h\n"                /*  accumulate to result low*/
                      "saddw2  v20.4s, v20.4s, v8.8h\n"                /*  accumulate to result high*/

                    //r3
                      "smlal  v9.8h , %[v6].8b, v1.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v10.8h , %[v6].8b, v2.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v11.8h , %[v6].8b, v3.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v12.8h , %[v6].8b, v4.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v1.8b, v0.b[4]                       \n"         /* dup vdupq_n_s8(s[3])*/
                      "ldp    q5, q6, [%[ptr_out1]]             \n"         /* load outr00-outr80*/
                      "ldp    q7, q8, [%[ptr_out1], #32]         \n"         /* load outr00-outr80*/

                      "stp    q13, q14, [%[ptr_out0]], #32 \n"         /* save to output*/
                      "stp    q15, q16, [%[ptr_out0]], #32 \n"         /* save to output*/

                      "smlal  v9.8h , %[v7].8b, v2.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smlal  v10.8h , %[v7].8b, v3.8b             \n"                /*q10 = 8x8bit*/
                      "smlal  v11.8h , %[v7].8b, v4.8b             \n"                 /*q9 = 8x8bit*/
                      "smlal  v12.8h , %[v7].8b, v1.8b              \n"                /*q10 = 8x8bit*/

                      "dup    v2.8b, v0.b[5]                       \n"         /* dup vdupq_n_s8(s[3])*/

                      "ldp    q13, q14, [%[ptr_out1], #64]        \n"         /* load outr00-outr80*/
                      "ldp    q15, q16, [%[ptr_out1], #96]          \n"         /* load outr00-outr80*/
                      "stp    q17, q18, [%[ptr_out0]], #32 \n"         /* save to output*/
                      "stp    q19, q20, [%[ptr_out0]], #32 \n"         /* save to output*/

                      "add   %[r3], %[r3], #4                       \n"

                      "saddw  v5.4s, v5.4s, v9.4h\n"                /*  accumulate to result low*/
                      "saddw2  v6.4s, v6.4s, v9.8h\n"                /*  accumulate to result high*/

                      "saddw  v7.4s, v7.4s, v10.4h\n"                /*  accumulate to result low*/
                      "saddw2  v8.4s, v8.4s, v10.8h\n"                /*  accumulate to result high*/

                      "saddw  v13.4s, v13.4s, v11.4h\n"                /*  accumulate to result low*/
                      "saddw2  v14.4s, v14.4s, v11.8h\n"                /*  accumulate to result high*/

                      "saddw  v15.4s, v15.4s, v12.4h\n"                /*  accumulate to result low*/
                      "saddw2  v16.4s, v16.4s, v12.8h\n"                /*  accumulate to result high*/

                      "smull  v9.8h , %[v8].8b, v3.8b              \n"                 /*q9 = 8x8bit * w0*/
                      "smull  v10.8h , %[v8].8b, v4.8b             \n"                /*q10 = 8x8bit*/
                      "smull  v11.8h , %[v8].8b, v1.8b             \n"                 /*q9 = 8x8bit*/
                      "smull  v12.8h , %[v8].8b, v2.8b              \n"                /*q10 = 8x8bit*/

                      "saddw  v5.4s, v5.4s, v9.4h\n"                /*  accumulate to result low*/
                      "saddw2  v6.4s, v6.4s, v9.8h\n"                /*  accumulate to result high*/

                      "saddw  v7.4s, v7.4s, v10.4h\n"                /*  accumulate to result low*/
                      "saddw2  v8.4s, v8.4s, v10.8h\n"                /*  accumulate to result high*/

                      "saddw  v13.4s, v13.4s, v11.4h\n"                /*  accumulate to result low*/
                      "saddw2  v14.4s, v14.4s, v11.8h\n"                /*  accumulate to result high*/

                      "saddw  v15.4s, v15.4s, v12.4h\n"                /*  accumulate to result low*/
                      "saddw2  v16.4s, v16.4s, v12.8h\n"                /*  accumulate to result high*/

                      "stp    q5, q6, [%[ptr_out1]], #32 \n"         /* save to output*/
                      "stp    q7, q8, [%[ptr_out1]], #32 \n"         /* save to output*/
                      "stp    q13, q14, [%[ptr_out1]], #32 \n"         /* save to output*/
                      "stp    q15, q16, [%[ptr_out1]], #32 \n"         /* save to output*/

                      "subs   %w[cnt], %w[cnt], #1        \n"         /* loop count -1*/

                      "bne    1b                          \n"         /* jump to main loop*/

                    : [cnt]"+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), \
                        [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0), [ptr_out1] "+r"(ptr_out1)
                    : [v0]"w"(v0), [v1]"w"(v1), [v2]"w"(v2), [v3]"w"(v3),\
                        [v4]"w"(v4), [v5]"w"(v5), [v6]"w"(v6), [v7]"w"(v7), [v8] "w" (v8)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", \
                        "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20"

                    );
#else
                    const signed char* ptr_wc = wc0;
                    asm volatile(
                      "vld1.s8  {d9}, [%[r0]]           \n" /* load data din0-dinn7*/
                      "vld1.s8 {d0-d3}, [%[wc0]]!       \n"    /* wc0, wc1, wc2, wc3 */
                      "vld1.s8 {d4-d7}, [%[wc0]]!       \n"    /* wc4, wc5, wc6, wc7 */
                      "vld1.s8 {d8}, [%[wc0]]!          \n"    /* wc8 */

                    "1:                                 \n"         /* main loop*/
                      "vdup.s8    d10, d9[0]            \n"         /* dup vdupq_n_s8(s[0])*/
                      "vdup.s8    d11, d9[1]            \n"         /* dup vdupq_n_s8(s[1])*/
                      "vdup.s8    d12, d9[2]            \n"         /* dup vdupq_n_s8(s[2])*/
                      "vdup.s8    d13, d9[3]            \n"         /* dup vdupq_n_s8(s[3])*/

                      "add   %[r0], #4                   \n"
                      "vmull.s8  q7, d0, d10             \n"             /*q9 = 8x8bit * w0*/
                      "vmull.s8  q8, d0, d11             \n"                /*q10 = 8x8bit*/
                      "vmull.s8  q9, d0, d12             \n"                 /*q9 = 8x8bit*/
                      "vmull.s8  q10, d0, d13            \n"                /*q10 = 8x8bit*/

                      "vdup.s8   d10, d9[4]               \n"       /* dup vdupq_n_s8(s[3])*/

                      "vmlal.s8  q7, d1, d11              \n"             /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q8, d1, d12              \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q9, d1, d13              \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q10, d1, d10             \n"                /*q10 = 8x8bit*/

                      "vdup.s8  d11, d9[5]               \n"       /* dup vdupq_n_s8(s[3])*/

                      "vld1.s8  {d9}, [%[r1]]            \n" /* load data din0-dinn7*/

                      "vmlal.s8  q7, d2, d12             \n"        /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q8, d2, d13             \n"        /*q10 = 8x8bit*/
                      "vmlal.s8  q9, d2, d10             \n"        /*q9 = 8x8bit*/
                      "vmlal.s8  q10, d2, d11            \n"        /*q10 = 8x8bit*/

                      "add   %[r1], #4                   \n"

                      "vdup.s8    d10, d9[0]            \n"         /* dup vdupq_n_s8(s[0])*/
                      "vdup.s8    d11, d9[1]            \n"         /* dup vdupq_n_s8(s[1])*/
                      "vdup.s8    d12, d9[2]            \n"         /* dup vdupq_n_s8(s[2])*/
                      "vdup.s8    d13, d9[3]            \n"         /* dup vdupq_n_s8(s[3])*/

                      "vmlal.s8  q7, d3, d10        \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q8, d3, d11        \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q9, d3, d12        \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q10, d3, d13       \n"                /*q10 = 8x8bit*/

                      "vmull.s8  q11, d0, d10       \n"                 /*q9 = 8x8bit * w0*/
                      "vmull.s8  q12, d0, d11        \n"                /*q10 = 8x8bit*/
                      "vmull.s8  q13, d0, d12        \n"                 /*q9 = 8x8bit*/
                      "vmull.s8  q14, d0, d13        \n"                /*q10 = 8x8bit*/

                      "vdup.s8   d10, d9[4]          \n"         /* dup vdupq_n_s8(s[3])*/

                      "vmlal.s8  q7, d4, d11          \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q8, d4, d12          \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q9, d4, d13          \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q10, d4, d10         \n"                /*q10 = 8x8bit*/

                      "vmlal.s8  q11, d1, d11        \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q12, d1, d12        \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q13, d1, d13        \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q14, d1, d10        \n"                /*q10 = 8x8bit*/

                      "vdup.s8   d11, d9[5]          \n"         /* dup vdupq_n_s8(s[3])*/

                      "vmlal.s8  q11, d2, d12        \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q12, d2, d13        \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q13, d2, d10        \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q14, d2, d11        \n"                /*q10 = 8x8bit*/

                      "vld1.s8  {d9}, [%[r2]]        \n" /* load data din0-dinn7*/

                      "vmlal.s8  q7, d5, d12          \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q8, d5, d13          \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q9, d5, d10          \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q10, d5, d11         \n"                /*q10 = 8x8bit*/

                      "add      %[r2], #4                       \n"

                      "vdup.s8    d10, d9[0]          \n"         /* dup vdupq_n_s8(s[0])*/
                      "vdup.s8    d11, d9[1]          \n"         /* dup vdupq_n_s8(s[1])*/
                      "vdup.s8    d12, d9[2]          \n"         /* dup vdupq_n_s8(s[2])*/
                      "vdup.s8    d13, d9[3]          \n"         /* dup vdupq_n_s8(s[3])*/

                      "vmlal.s8  q11, d3, d10       \n"           /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q12, d3, d11       \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q13, d3, d12       \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q14, d3, d13       \n"                /*q10 = 8x8bit*/

                      "vmlal.s8  q7, d6, d10        \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q8, d6, d11        \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q9, d6, d12        \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q10, d6, d13       \n"                /*q10 = 8x8bit*/

                      "vdup.s8   d10, d9[4]         \n"         /* dup vdupq_n_s8(s[3])*/

                      "vmlal.s8  q11, d4, d11       \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q12, d4, d12       \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q13, d4, d13       \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q14, d4, d10       \n"                /*q10 = 8x8bit*/

                      "vmlal.s8  q7, d7, d11        \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q8, d7, d12        \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q9, d7, d13        \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q10, d7, d10       \n"                /*q10 = 8x8bit*/

                      "vmull.s8  q5, d8, d12       \n"                 /*q9 = 8x8bit * w0*/

                      "vld1.32    {d30-d31}, [%[ptr_out0]]   \n"   /* load  */
                      "vaddw.s16  q15, q15, d14              \n"   /*   */
                      "vaddw.s16  q15, q15, d10              \n"   /*   */
                      "vst1.32    {d30-d31}, [%[ptr_out0]]!  \n"   /* store */

                      "vld1.32    {d30-d31}, [%[ptr_out0]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q15, q15, d15                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q15, q15, d11                \n"   /* q5 = outr00[0].low  + q5  */
                      "vst1.32    {d30-d31}, [%[ptr_out0]]!  \n"   /* store q5, q6 -> ptr_out += 32 */

                      "vmull.s8  q7, d8, d13       \n"                /*q10 = 8x8bit*/

                      "vld1.32    {d30-d31}, [%[ptr_out0]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q15, q15, d14                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q15, q15, d16                \n"   /* q5 = outr00[0].low  + q5  */
                      "vst1.32    {d30-d31}, [%[ptr_out0]]!  \n"   /* store q5, q6 -> ptr_out += 32 */
                      "vld1.32    {d30-d31}, [%[ptr_out0]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q15, q15, d15                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q15, q15, d17                \n"   /* q5 = outr00[0].low  + q5  */
                      "vst1.32    {d30-d31}, [%[ptr_out0]]!  \n"   /* store q5, q6 -> ptr_out += 32 */

                      "vdup.s8   d10, d9[4]         \n"         /* dup vdupq_n_s8(s[3])*/
                      "vmull.s8  q15, d8, d10       \n"                 /*q9 = 8x8bit*/

                      "vld1.32    {d14-d17}, [%[ptr_out0]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q7, q7, d30                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q7, q7, d18                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q8, q8, d31                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q8, q8, d19                \n"   /* q5 = outr00[0].low  + q5  */
                      "vst1.32    {d14-d17}, [%[ptr_out0]]!  \n"   /* store q5, q6 -> ptr_out += 32 */

                      "vdup.s8   d11, d9[5]         \n"         /* dup vdupq_n_s8(s[3])*/
                      "vmull.s8  q15, d8, d11      \n"                /*q10 = 8x8bit*/

                      "vld1.32    {d14-d17}, [%[ptr_out0]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q7, q7, d30                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q7, q7, d20                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q8, q8, d31                \n"   /* q5 = outr00[0].low  + q5  */
                      "vaddw.s16  q8, q8, d21                \n"   /* q5 = outr00[0].low  + q5  */
                      "vst1.32    {d14-d17}, [%[ptr_out0]]!  \n"   /* store q5, q6 -> ptr_out += 32 */

                      "vmlal.s8  q11, d5, d12       \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q12, d5, d13       \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q13, d5, d10       \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q14, d5, d11       \n"                /*q10 = 8x8bit*/

                      "vld1.s8  {d9}, [%[r3]]     \n" /* load data din0-dinn7*/

                      "vdup.s8    d10, d9[0]      \n"         /* dup vdupq_n_s8(s[0])*/
                      "vdup.s8    d11, d9[1]      \n"         /* dup vdupq_n_s8(s[1])*/
                      "vdup.s8    d12, d9[2]      \n"         /* dup vdupq_n_s8(s[2])*/
                      "vdup.s8    d13, d9[3]      \n"         /* dup vdupq_n_s8(s[3])*/

                      "add   %[r3], #4                       \n"

                      "vmlal.s8  q11, d6, d10       \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q12, d6, d11       \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q13, d6, d12       \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q14, d6, d13       \n"                /*q10 = 8x8bit*/

                      "vdup.s8   d10, d9[4]         \n"         /* dup vdupq_n_s8(s[3])*/
                      "vmlal.s8  q11, d7, d11       \n"                 /*q9 = 8x8bit * w0*/
                      "vmlal.s8  q12, d7, d12       \n"                /*q10 = 8x8bit*/
                      "vmlal.s8  q13, d7, d13       \n"                 /*q9 = 8x8bit*/
                      "vmlal.s8  q14, d7, d10       \n"                /*q10 = 8x8bit*/

                      "vmull.s8  q7, d8, d12         \n"                 /*q9 = 8x8bit * w0*/

                      "vld1.32    {d18-d21}, [%[ptr_out1]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q9, q9, d22   \n"
                      "vaddw.s16  q9, q9, d14   \n"
                      "vaddw.s16  q10, q10, d23   \n"
                      "vaddw.s16  q10, q10, d15   \n"
                      "vst1.32    {d18-d21}, [%[ptr_out1]]!  \n"   /* store q5, q6 -> ptr_out += 32 */

                      "vmull.s8  q8, d8, d13         \n"                /*q10 = 8x8bit*/

                      "vld1.32    {d18-d21}, [%[ptr_out1]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q9, q9, d24   \n"
                      "vaddw.s16  q9, q9, d16   \n"
                      "vaddw.s16  q10, q10, d25   \n"
                      "vaddw.s16  q10, q10, d17   \n"
                      "vst1.32    {d18-d21}, [%[ptr_out1]]!  \n"   /* store q5, q6 -> ptr_out += 32 */

                      "vmull.s8  q9, d8, d10         \n"                 /*q9 = 8x8bit*/

                      "vld1.32    {d14-d17}, [%[ptr_out1]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q7, q7, d18   \n"
                      "vaddw.s16  q7, q7, d26   \n"
                      "vaddw.s16  q8, q8, d19   \n"
                      "vaddw.s16  q8, q8, d27   \n"
                      "vst1.32    {d14-d17}, [%[ptr_out1]]!  \n"   /* store q5, q6 -> ptr_out += 32 */

                      "vdup.s8   d11, d9[5]        \n"         /* dup vdupq_n_s8(s[3])*/
                      "vmull.s8  q10, d8, d11         \n"                /*q10 = 8x8bit*/

                      "subs  %[cnt], #1        \n"         /* loop count -1*/

                      "vld1.32    {d14-d17}, [%[ptr_out1]]   \n"   /* load ptr_out -> q5, q6    */
                      "vaddw.s16  q7, q7, d20   \n"
                      "vaddw.s16  q7, q7, d28   \n"
                      "vaddw.s16  q8, q8, d21   \n"
                      "vaddw.s16  q8, q8, d29   \n"
                      "vst1.32    {d14-d17}, [%[ptr_out1]]!  \n"   /* store q5, q6 -> ptr_out += 32 */

                      "vld1.s8  {d9}, [%[r0]]             \n" /* load data din0-dinn7*/

                      "bne    1b                          \n"         /* jump to main loop*/

                    : [cnt]"+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), \
                        [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0), [ptr_out1] "+r"(ptr_out1), \
                        [wc0] "+r" (ptr_wc)
                    :
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                        "q12", "q13", "q14", "q15"
                    );
#endif

                    wc0 += 9 * hout_c_block;
                    inr0 += win_round;
                    inr1 += win_round;
                    inr2 += win_round;
                    inr3 += win_round;
                }
                //write_to_output_c8_int32(pre_out, dout_batch, hout_c_block, hout_r_block, c, c + 8, h, h + 2, 0,\
                //wout_round, chout, hout, wout, flag_relu, ptr_write);

                if (out_type == AK_FLOAT){
                    write_to_output_c8_int32_1(pre_out, reinterpret_cast<float*>(dout_batch), hout_c_block, \
                        hout_r_block, c, c + 8, h, h + 2, 0, wout_round, chout, hout, wout, \
                        flag_relu, reinterpret_cast<float*>(ptr_write), &scale[c], out_type);
                }else if (out_type == AK_INT8){
                    write_to_output_c8_int32_1(pre_out, dout_batch, hout_c_block, \
                        hout_r_block, c, c + 8, h, h + 2, 0, wout_round, chout, hout, wout, \
                        flag_relu, reinterpret_cast<signed char*>(ptr_write), &scale[c], out_type);
                }else{//int32
                    write_to_output_c8_int32(pre_out, reinterpret_cast<int*>(dout_batch), hout_c_block, \
                         hout_r_block, c, c + 8, h, h + 2, 0, wout_round, chout, hout, wout, \
                         flag_relu, ptr_write);
                }
            }
        }
    }
  }
}
}
#endif

