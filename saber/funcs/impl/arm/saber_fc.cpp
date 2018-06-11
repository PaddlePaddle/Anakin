#include "saber_fc.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <typename Dtype>
void fill_bias_fc(Dtype* tensor, const Dtype* bias, const int num, const int channel);
template <>
void fill_bias_fc<float>(float* tensor, const float* bias, const int num, const int channel) {

    int cnt = channel >> 2;
    int remain = channel & 3;

    for (int j = 0; j < num; ++j) {

        const float* ptr_bias = bias;
        float* ptr_out = tensor + j * channel;

        if (cnt > 0) {
            asm(
            ".fill_bias_fc: \n"
            "vld1.32 {d0-d1}, [%[ptr_out]]  @ load data\n"
                    "vld1.32 {d2-d3}, [%[ptr_bias]]!  @ load data\n"
                    "vadd.f32 q2, q0, q1              @ add bias\n"
                    "vst1.32  {d4-d5}, [%[ptr_out]]!  @ store result\n"
                    "subs   %[cnt], #1                @ loop count -1\n"
                    "bne    .fill_bias_fc             @ jump to main loop\n"
            :[ptr_out] "+r"(ptr_out), [ptr_bias] "+r"(ptr_bias), \
                    [cnt] "+r"(cnt)
            :
            :"q0", "q1", "q2"
            );
        }

        for (; remain > 0; remain--) {
            *(ptr_out++) += *(ptr_bias++);
        }
    }
}

template <typename Dtype>
void fc_kernel(const Dtype* din, const Dtype* weights, const Dtype* bias, \
    Dtype* dout, const int m, const int n, const int k, bool flag_trans_weights);

template <>
void fc_kernel<float>(const float* din, const float* weights, const float* bias, \
    float* dout, const int m, const int n, const int k, \
    bool flag_bias) {

    float zero[4] = {0.f, 0.f, 0.f, 0.f};

    float* dout_ptr = dout;
    const float* din_ptr = din;
    const float* weights_ptr = weights;

    int cnt = k >> 3;
    int tail = k & 7;
    int out_cnt = n >> 2;

    for (int i = 0; i < m; ++i) {
        float* data_batch_out = dout_ptr + i * n;
        const float* data_batch_in = din_ptr + i * k;

#pragma omp parallel for
        for (int j = 0; j < out_cnt; j++) {

            float *ptr_out = data_batch_out + j * 4;
            const float *ptr_in = data_batch_in;
            const float *ptr_w0 = weights + (k * j * 4);
            const float *ptr_w1 = ptr_w0 + k;
            const float *ptr_w2 = ptr_w1 + k;
            const float *ptr_w3 = ptr_w2 + k;

            const float* ptr_bias = zero;
            if (flag_bias) {
                ptr_bias = bias + j * 4;
            }

            int loop_cnt = cnt;
            asm volatile(
            "pld [%[in], #128] @ preload cache line, input\n"

                    "vmov.u32 q0, #0 @ set q0 to 0\n"
                    "vmov.u32 q1, #0 @ set q1 to 0\n"
                    "vmov.u32 q2, #0 @ set q2 to 0\n"
                    "vmov.u32 q3, #0 @ set q3 to 0\n"

                    "pld [%[w0], #128] @ preload cache line, weights r0\n"
                    "pld [%[w1], #128] @ preload cache line, weights r1\n"
                    "pld [%[w2], #128] @ preload cache line, weights r2\n"
                    "pld [%[w3], #128] @ preload cache line, weights r3\n"

                    ".full_connect_loop: @ main loop\n"
                    // unroll 0
                    "vld1.32 {d12-d15}, [%[in]]!    @ load input, q6, q7\n"
                    "vld1.32 {d16-d19}, [%[w0]]!    @ load weights r0, q8,q9\n"
                    "vld1.32 {d20-d23}, [%[w1]]!    @ load weights r1, q10,q11\n"
                    "vld1.32 {d24-d27}, [%[w2]]!    @ load weights r2, q12,q13\n"
                    "vld1.32 {d28-d31}, [%[w3]]!    @ load weights r3, q14, q15\n"
                    "vmla.f32 q0, q6, q8            @ mul add\n"
                    "pld [%[in], #128]              @ preload cache line, input\n"
                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"
                    "vmla.f32 q1, q6, q10           @ mul add\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"
                    "vmla.f32 q2, q8, q12           @ mul add\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"
                    "vmla.f32 q3, q8, q14           @ mul add\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"


                    // unrll 1
                    "vmla.f32 q0, q7, q9            @ mul add\n"
                    "pld [%[in], #128]              @ preload cache line, input\n"
                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"
                    "vmla.f32 q1, q7, q11           @ mul add\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"
                    "vmla.f32 q2, q7, q13           @ mul add\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"
                    "vmla.f32 q3, q7, q15           @ mul add\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    // check loop end
                    "subs %[loop_cnt], #1           @ sub loop count \n"
                    "bne .full_connect_loop         @ jump to main loop\n"

                    // pair add to final result
                    "vld1.32 {d12-d13}, [%[ptr_bias]]    @ load bias, q6\n"
                    "vpadd.f32 d8, d0, d1           @ pair add, first step\n"
                    "vpadd.f32 d9, d2, d3           @ pair add, first step\n"
                    "vpadd.f32 d10, d4, d5          @ pair add, first step\n"
                    "vpadd.f32 d11, d6, d7          @ pair add, first step\n"

                    "vpadd.f32 d0, d8, d9           @ pair add, second step\n"
                    "vpadd.f32 d1, d10, d11         @ pair add, second step\n"

                    "vadd.f32  q1, q0, q6           @ add bias\n"
                    "vst1.32 {d2-d3}, [%[out]]      @ save result\n"

            :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                    [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                    [loop_cnt] "+r"(loop_cnt)
            :[ptr_bias] "r" (ptr_bias)
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q10", "q11", "q12", "q13", "q14", "q15"
            );

            for (int ii = 0; ii < tail; ++ii) {
                float data_in = *ptr_in;
                ptr_out[0] += *(ptr_w0++) * data_in;
                ptr_out[1] += *(ptr_w1++) * data_in;
                ptr_out[2] += *(ptr_w2++) * data_in;
                ptr_out[3] += *(ptr_w3++) * data_in;
                ptr_in++;
            }
        }
        //! process remains
#pragma omp parallel for
        for (int j = out_cnt * 4; j < n; ++j) {
            float *ptr_out = data_batch_out + j;
            const float *ptr_in = data_batch_in;
            const float *ptr_w0 = weights + (k * j);

            int loop_cnt = cnt;
            asm volatile(
            "pld [%[in], #128] @ preload cache line, input\n"
                    "vmov.u32 q0, #0 @ set q0 to 0\n"
                    "pld [%[w0], #128] @ preload cache line, weights r0\n"

                    ".full_connect_loop2: @ main loop\n"
                    "vld1.32 {d24-d27}, [%[in]]! @ load input, q12,q13\n"
                    "vld1.32 {d28-d29}, [%[w0]]! @ load weights r0, q14\n"
                    "vmla.f32 q0, q12, q14 @ mul add\n"
                    "vld1.32 {d30-d31}, [%[w0]]! @ load weights r0, q15\n"
                    "pld [%[in]] @ preload cache line, input\n"
                    "pld [%[w0]] @ preload cache line, weights r0\n"
                    "pld [%[in], #128] @ preload cache line, input\n"
                    "pld [%[w0], #128] @ preload cache line, weights r0\n"
                    "vmla.f32 q0, q13, q15 @ mul add\n"
                    "subs %[loop_cnt] , #1 @ sub loop count \n"
                    "bne .full_connect_loop2 @ jump to main loop\n"

                    // pair add to final result
                    "vpadd.f32 d2, d0, d1 @ pair add, first step\n"
                    "vpadd.f32 d3, d2, d2 @ pair add, final step\n"
                    "vst1.32 {d3[0]}, [%[out]] @ save result\n"
            :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
                    [loop_cnt] "+r"(loop_cnt)
            :
            :"q0", "q1", "q12", "q13", "q14", "q15"
            );

            for (int ii = 0; ii < tail; ++ii) {
                *ptr_out += *(ptr_w0++) * *(ptr_in++);
            }
            if (flag_bias) {
                ptr_out[0] += bias[j];
            }
        }
    }
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberFc<ARM, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, FcParam<OpTensor> &param) {

    const InDataType* din = inputs[0]->data();
    OutDataType* dout = outputs[0]->mutable_data();
    const OpDataType* weights = param.weights->data();
    bool flag_bias = param.bias->valid_size() > 0;
    const OpDataType* bias = nullptr;
    if (flag_bias) {
        bias = param.bias->data();
    }
#if 1
    _gemmer(din, _k, weights, (param.is_transpose_weights? _n : _k), dout, _n, 1.f, 0.f, false);
    if (flag_bias) {
        fill_bias_fc(dout, bias, _m, _n);
    }
#else
    fc_kernel(din, weights, bias, dout, _m, _n, _k, param.is_transpose_weights);
#endif
    return SaberSuccess;
}

template class SaberFc<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin

#endif

