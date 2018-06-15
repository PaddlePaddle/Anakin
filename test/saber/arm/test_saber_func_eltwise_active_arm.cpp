#include "saber/funcs/eltwise_act.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"
#include "saber_types.h"

#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(int, threads, 1);
DEFINE_GLOBAL(int, cluster_id, 0);
DEFINE_GLOBAL(int, operation, 1);
DEFINE_GLOBAL(int, num_coeff, 0);
#define USE_COMPARE

using namespace anakin::saber;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;

void eltwise_active_basic(const Context<ARM> &ctx, Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \ 
    std::vector<Tensor<ARM, AK_FLOAT, NCHW>*> &tensor_in,\
    int op_type, std::vector<float> coeffs_ptr, int num_coeff, int active) {
    CHECK_GT(tensor_out.size(), 0) << "output tensor is empty";
    CHECK_GT(tensor_in.size(), 1) << "input tensor is empty";

    int w_in = tensor_in[0]->width();
    int h_in = tensor_in[0]->height();
    int ch_in = tensor_in[0]->channel();
    int num = tensor_in[0]->num();
    int size_in = w_in * h_in;

    float* data_out = tensor_out.mutable_data();
    const float* data_in0 = tensor_in[0]->data();
    const float* data_in1 = tensor_in[1]->data();
    
    if (op_type == 1){ //Operation_PROD
        for (int n = 0; n < num; n++){
            float* data_out_batch = data_out + n * ch_in * size_in;
            const float* data_in0_batch = data_in0 + n * ch_in * size_in;
            const float* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
            for (int c = 0; c < ch_in; c++){
                float* data_out_channel = data_out_batch + c * size_in;
                const float* data_in0_channel = data_in0_batch + c * size_in;
                const float* data_in1_channel = data_in1_batch + c * size_in;
                for (int i = 0; i < size_in; i++){
                    float tmp = data_in0_channel[i] * data_in1_channel[i];
                    if(active == 2)
                        data_out_channel[i] = tmp > 0 ? tmp : 0.f;
                }
            }
        }
        for (int b = 2; b <tensor_in.size(); b++){
            const float* data_in = tensor_in[b]->data();
            for (int n = 0; n < num; n++){
                float* data_out_batch = data_out + n * ch_in * size_in;
                const float* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    float* data_out_channel = data_out_batch + c * size_in;
                    const float* data_in_channel = data_in_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        float tmp = data_out_channel[i] * data_in_channel[i];
                        if(active == 2)
                            data_out_channel[i] = tmp > 0 ? tmp : 0.f;
                    }
                }
            }
        }
    }
    if (op_type == 2){ //Operation_SUM
        if (num_coeff == 0){
            for (int n = 0; n < num; n++){
                float* data_out_batch = data_out + n * ch_in * size_in;
                const float* data_in0_batch = data_in0 + n * ch_in * size_in;
                const float* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    float* data_out_channel = data_out_batch + c * size_in;
                    const float* data_in0_channel = data_in0_batch + c * size_in;
                    const float* data_in1_channel = data_in1_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        float tmp = data_in0_channel[i] + data_in1_channel[i];
                        if(active == 2)
                            data_out_channel[i] = tmp > 0 ? tmp : 0.f;
                    }
                }
            }
            for (int b = 2; b <tensor_in.size(); b++){
                const float* data_in = tensor_in[b]->data();
                for (int n = 0; n < num; n++){
                    float* data_out_batch = data_out + n * ch_in * size_in;
                    const float* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                    for (int c = 0; c < ch_in; c++){
                        float* data_out_channel = data_out_batch + c * size_in;
                        const float* data_in_channel = data_in_batch + c * size_in;
                        for (int i = 0; i < size_in; i++){
                            float tmp = data_out_channel[i] + data_in_channel[i];
                            if(active == 2)
                                data_out_channel[i] = tmp > 0 ? tmp : 0.f;
                        }
                    }
                }
            }
        }else{
            for (int n = 0; n < num; n++){
                float* data_out_batch = data_out + n * ch_in * size_in;
                const float* data_in0_batch = data_in0 + n * ch_in * size_in;
                const float* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    float* data_out_channel = data_out_batch + c * size_in;
                    const float* data_in0_channel = data_in0_batch + c * size_in;
                    const float* data_in1_channel = data_in1_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        float tmp= data_in0_channel[i]*coeffs_ptr[0] + \ 
                        data_in1_channel[i]*coeffs_ptr[1];
                        if(active == 2)
                            data_out_channel[i] = tmp > 0 ? tmp : 0.f;
                    }
                }
            }
            for (int b = 2; b <tensor_in.size(); b++){
                const float* data_in = tensor_in[b]->data();
                for (int n = 0; n < num; n++){
                    float* data_out_batch = data_out + n * ch_in * size_in;
                    const float* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                    for (int c = 0; c < ch_in; c++){
                        float* data_out_channel = data_out_batch + c * size_in;
                        const float* data_in_channel = data_in_batch + c * size_in;
                        for (int i = 0; i < size_in; i++){
                            float tmp = data_out_channel[i] + \ 
                            data_in_channel[i] * coeffs_ptr[b];
                            if(active == 2)
                                data_out_channel[i] = tmp > 0 ? tmp : 0.f;
                        }
                    }
                }
            }
        }
    }
    if (op_type == 3){ //Operation_MAX
        for (int n = 0; n < num; n++){
            float* data_out_batch = data_out + n * ch_in * size_in;
            const float* data_in0_batch = data_in0 + n * ch_in * size_in;
            const float* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
            for (int c = 0; c < ch_in; c++){
                float* data_out_channel = data_out_batch + c * size_in;
                const float* data_in0_channel = data_in0_batch + c * size_in;
                const float* data_in1_channel = data_in1_batch + c * size_in;
                for (int i = 0; i < size_in; i++){
                    float tmp = std::max(data_in0_channel[i], data_in1_channel[i]);
                    if(active == 2)
                        data_out_channel[i] = tmp > 0 ? tmp : 0.f;
                }
            }
        }
        for (int b = 2; b <tensor_in.size(); b++){
            const float* data_in = tensor_in[b]->data();
            for (int n = 0; n < num; n++){
                float* data_out_batch = data_out + n * ch_in * size_in;
                const float* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    float* data_out_channel = data_out_batch + c * size_in;
                    const float* data_in_channel = data_in_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        float tmp = std::max(data_out_channel[i], data_in_channel[i]);
                        if(active == 2)
                            data_out_channel[i] = tmp > 0 ? tmp : 0.f;
                    }
                }
            }
        }
    }
    
}

void eltwise_act_ncnn(const Context<ARM> &ctx, Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \ 
    std::vector<Tensor<ARM, AK_FLOAT, NCHW>*> &tensor_in,\
    int op_type, std::vector<float> coeffs_ptr, int num_coeff, int active) {
    CHECK_GT(tensor_out.size(), 0) << "output tensor is empty";
    CHECK_GT(tensor_in.size(), 1) << "input tensor is empty";

    int w = tensor_in[0]->width();
    int h = tensor_in[0]->height();
    int channels = tensor_in[0]->channel();
    int num = tensor_in[0]->num();
    int size = w * h;
    int nn = size >> 2;
    int remian = size - (nn << 2);

    float* data_out = tensor_out.mutable_data();
    const float* data_in0 = tensor_in[0]->data();
    const float* data_in1 = tensor_in[1]->data();
    float32x4_t vzero = vdupq_n_f32(0.f);

    if (op_type == 0) { //Operation_PROD
        // first blob
    for (int n = 0; n < num; n++){
        const float* ptr_n = data_in0 + n * channels * size;
        const float* ptr1_n = data_in1 + n * channels * size;
        float* outptr_n = data_out + n * channels * size;
#pragma omp parallel for
        for (int q = 0; q < channels; q++){
            const float* ptr = ptr_n + q * size;
            const float* ptr1 = ptr1_n + q * size;
            float* outptr = outptr_n + q * size;

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fmul       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "pld        [%2, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vmul.f32   q0, q0, q1          \n"
                "vcgt.f32  q2, q0, %q[vzero]            @ q9 > 0 \n"
                "vbif.f32  q0, %q[vzero], q2            @ bif \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%3 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr),  // %3
                  [vzero] "+w" (vzero)
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "q0", "q1", "q2"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float tmp = *ptr * *ptr1;
                *outptr = tmp > 0 ? tmp : 0.f;
                ptr++;
                ptr1++;
                outptr++;
            }
        }
    }

        for (size_t b = 2; b < tensor_in.size(); b++)
        {
            for (int n = 0; n < num; n++){
                const float* ptr_n = data_in0 + n * channels * size;
                const float* ptr1_n = data_in1 + n * channels * size;
                float* outptr_n = data_out + n * channels * size;
#pragma omp parallel for
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = ptr_n + q * size;
                    float* outptr = outptr_n + q * size;

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fmul       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]  \n"
                    "vmul.f32   q0, q0, q1          \n"
                    "vcgt.f32   q2, q0, %q[vzero]            @ q9 > 0 \n"
                    "vbif.f32   q0, %q[vzero], q2            @ bif \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%2 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr),  // %2
                      [vzero] "+w" (vzero)
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1", "q2"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float tmp = *outptr * *ptr;
                    *outptr = tmp > 0 ? tmp : 0.f;
                    ptr++;
                    outptr++;
                }
            }
        }
    }
}
    else if (op_type == 1) //Operation_SUM
    {
        if (num_coeff == 0)
        {
            // first blob
            for (int n = 0; n < num; n++){
                const float* ptr_n = data_in0 + n * channels * size;
                const float* ptr1_n = data_in1 + n * channels * size;
                float* outptr_n = data_out + n * channels * size;
#pragma omp parallel for
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = ptr_n + q * size;
                    const float* ptr1 = ptr1_n + q * size;
                    float* outptr = outptr_n + q * size;

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2], #16    \n"
                    "fadd       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%3], #16    \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]! \n"
                    "vadd.f32   q0, q0, q1          \n"
                    "vcgt.f32   q2, q0, %q[vzero]            @ q9 > 0 \n"
                    "vbif.f32   q0, %q[vzero], q2            @ bif \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%3 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr),  // %3
                      [vzero] "+w" (vzero)
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr)
                    : "cc", "memory", "q0", "q1", "q2"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                   float tmp = *ptr + *ptr1;
                    *outptr = tmp > 0 ? tmp : 0.f;
                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }
        }

            for (size_t b = 2; b < tensor_in.size(); b++)
            {
               for (int n = 0; n < num; n++){
                    const float* ptr_n = data_in0 + n * channels * size;
                    float* outptr_n = data_out + n * channels * size;
#pragma omp parallel for
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = ptr_n + q * size;
                        float* outptr = outptr_n + q * size;
#if __ARM_NEON
                    int nn = size >> 2;
                    int remain = size - (nn << 2);
#else
                    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                               \n"
                        "prfm       pldl1keep, [%1, #128] \n"
                        "prfm       pldl1keep, [%2, #128] \n"
                        "ld1        {v0.4s}, [%1], #16    \n"
                        "ld1        {v1.4s}, [%2]         \n"
                        "fadd       v0.4s, v0.4s, v1.4s   \n"
                        "subs       %w0, %w0, #1          \n"
                        "st1        {v0.4s}, [%2], #16    \n"
                        "bne        0b                    \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr)
                        : "cc", "memory", "v0", "v1"
                    );
                    }
#else
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]! \n"
                        "vld1.f32   {d2-d3}, [%2 :128]  \n"
                        "vadd.f32   q0, q0, q1          \n"
                        "vcgt.f32   q2, q0, %q[vzero]            @ q9 > 0 \n"
                        "vbif.f32   q0, %q[vzero], q2            @ bif \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%2 :128]! \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr),  // %2
                          [vzero] "+w" (vzero)
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr)
                        : "cc", "memory", "q0", "q1", "q2"
                    );
                    }
#endif // __aarch64__
#endif // __ARM_NEON
                    for (; remain > 0; remain--)
                    {
                        float tmp = *outptr + *ptr;
                        *outptr = tmp > 0 ? tmp : 0.f;
                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }else
        {

            // first blob
            //const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs_ptr[0];
            float coeff1 = coeffs_ptr[1];
            for (int n = 0; n < num; n++){
                const float* ptr_n = data_in0 + n * channels * size;
                const float* ptr1_n = data_in1 + n * channels * size;
                float* outptr_n = data_out + n * channels * size;
#pragma omp parallel for
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = ptr_n + q * size;
                    const float* ptr1 = ptr1_n + q * size;
                    float* outptr = outptr_n + q * size;

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _coeff0 = vdupq_n_f32(coeff0);
                float32x4_t _coeff1 = vdupq_n_f32(coeff1);
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2], #16    \n"
                    "fmul       v0.4s, v0.4s, %8.4s   \n"
                    "fmla       v0.4s, v1.4s, %9.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%3], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr),
                      "w"(_coeff0), // %8
                      "w"(_coeff1)  // %9
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]! \n"
                    "vmul.f32   q0, q0, %q8         \n"
                    "vmla.f32   q0, q1, %q9         \n"
                    "vcgt.f32   q2, q0, %q8            @ q9 > 0 \n"
                    "vbif.f32   q0, %q8, q2            @ bif \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%3 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr),
                      "w"(_coeff0), // %8
                      "w"(_coeff1)  // %9
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float tmp = *ptr * coeff0 + *ptr1 * coeff1;
                    *outptr = tmp > 0 ? tmp : 0.f;
                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }
        }

            for (size_t b = 2; b < tensor_in.size(); b++)
            {
                float coeff = coeffs_ptr[b];
                for (int n = 0; n < num; n++){
                    const float* ptr_n = data_in0 + n * channels * size;
                    const float* ptr1_n = data_in1 + n * channels * size;
                    float* outptr_n = data_out + n * channels * size;
#pragma omp parallel for
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = ptr_n + q * size;
                        float* outptr = outptr_n + q * size;

#if __ARM_NEON
                    int nn = size >> 2;
                    int remain = size - (nn << 2);
#else
                    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                    float32x4_t _coeff = vdupq_n_f32(coeff);
#if __aarch64__
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                               \n"
                        "prfm       pldl1keep, [%1, #128] \n"
                        "prfm       pldl1keep, [%2, #128] \n"
                        "ld1        {v0.4s}, [%1], #16    \n"
                        "ld1        {v1.4s}, [%2]         \n"
                        "fmla       v1.4s, v0.4s, %6.4s   \n"
                        "subs       %w0, %w0, #1          \n"
                        "st1        {v1.4s}, [%2], #16    \n"
                        "bne        0b                    \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr),
                          "w"(_coeff)   // %6
                        : "cc", "memory", "v0", "v1"
                    );
                    }
#else
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]! \n"
                        "vld1.f32   {d2-d3}, [%2 :128]  \n"
                        "vmla.f32   q1, q0, %q6         \n"
                        "vcgt.f32   q2, q0, %q6            @ q9 > 0 \n"
                        "vbif.f32   q0, %q6, q2            @ bif \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d2-d3}, [%2 :128]! \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr),
                          "w"(_coeff)   // %6
                        : "cc", "memory", "q0", "q1", "q2"
                    );
                    }
#endif // __aarch64__
#endif // __ARM_NEON
                    for (; remain > 0; remain--)
                    {
                        float tmp = *outptr + *ptr * coeff;
                        *outptr = tmp > 0 ? tmp :0.f;
                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }
}
    else if (op_type == 2) //Operation_MAX
    {
        // first blob
        for (int n = 0; n < num; n++){
            const float* ptr_n = data_in0 + n * channels * size;
            const float* ptr1_n = data_in1 + n * channels * size;
            float* outptr_n = data_out + n * channels * size;
#pragma omp parallel for
            for (int q = 0; q < channels; q ++)
            {
                const float* ptr = ptr_n + q * size;
                const float* ptr1 = ptr1_n + q * size;
                float* outptr = outptr_n + q * size;
#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fmax       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "pld        [%2, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vmax.f32   q0, q0, q1          \n"
                "vcgt.f32   q2, q0, %q[vzero]            @ q9 > 0 \n"
                "vbif.f32   q0, %q[vzero], q2            @ bif \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%3 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr),  // %3
                  [vzero] "+w" (vzero)
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "q0", "q1", "q2"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
               float tmp = std::max(*ptr, *ptr1);
               *outptr = tmp;
                ptr++;
                ptr1++;
                outptr++;
            }
        }
    }

        for (size_t b = 2; b < tensor_in.size(); b++)
        {
            for (int n = 0; n < num; n++){
                const float* ptr_n = data_in0 + n * channels * size;
                const float* ptr1_n = data_in1 + n * channels * size;
                float* outptr_n = data_out + n * channels * size;
#pragma omp parallel for
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = ptr_n + q * size;
                    const float* ptr1 = ptr1_n + q * size;
                    float* outptr = outptr_n + q * size;

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fmax       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "vcgt.f32   q2, q0, %q[vzero]            @ q9 > 0 \n"
                    "vbif.f32   q0, %q[vzero], q2            @ bif \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%2 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr),  // %2
                      [vzero] "+w" (vzero)
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1", "q2"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float tmp = std::max(*ptr, *outptr);
                    *outptr = tmp > 0 ? tmp :0;

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    }
}

void test_arm_eltwise(std::vector<TensorHf4*>& tin, EltwiseType operation, \
     std::vector<float> coeffs_ptr, int num_coeff, int threads, int cluster_id) {

    int test_iter = 100;
    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;
    SaberTimer<ARM> t2;

    Context<ARM> ctx1;
    Context<ARM> ctx2;
    std::vector<int> act_ids;
    //printf("start:\n");
    //! set runtime context
    LOG(INFO) << "set runtine context";
    std::vector<int> big_cores;
    std::vector<int> small_cores;
    for (int i = 0; i < ctx1.devs[0]._info._cluster_ids.size(); ++i) {
        if (ctx1.devs[0]._info._cluster_ids[i] == 0) {
            big_cores.push_back(ctx1.devs[0]._info._core_ids[i]);
        } else {
            small_cores.push_back(ctx1.devs[0]._info._cluster_ids[i]);
        }
    }

    if (cluster_id == 0) {
        if (big_cores.size() == 0) {
            LOG(FATAL) << "big cores are not supported";
        }
        if (threads > big_cores.size()) {
            LOG(WARNING) << "not enough big cores for inference";
            act_ids = big_cores;
        } else {
            for (int i = 0; i < threads; ++i) {
                act_ids.push_back(big_cores[i]);
            }
        }
    } else {
        if (small_cores.size() == 0) {
            LOG(FATAL) << "small cores are not supported";
        }
        if (threads > small_cores.size()) {
            LOG(WARNING) << "not enough small cores for inference";
            act_ids = small_cores;
        } else {
            for (int i = 0; i < threads; ++i) {
                act_ids.push_back(small_cores[i]);
            }
        }
    }
    ctx1.set_act_cores(act_ids);
    //printf("ctx1 threads : %d\n",ctx1.get_act_ids().size());

    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int threads = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << threads;
#endif
    }
    int th_id = 0;
#pragma omp parallel private(th_id)
    {
#ifdef USE_OPENMP
        th_id = omp_get_thread_num();
#pragma omp parallel
        LOG(INFO) << "thread core ID: " << big_cores[th_id];
#endif
    }

    TensorHf4 tout_basic;
    TensorHf4 tout_saber;

    //TensorHf4* thin = tin[0];

    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;

    tvout_saber.push_back(&tout_saber);
    tvout_basic.push_back(&tout_basic);

    int numin = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();
    int pad = 0;

    LOG(INFO) << "eltwise active param: ";
    LOG(INFO) << " img_num = " << numin;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
   // enum { Eltwise_prod = 1, Eltwise_sum = 2, Eltwise_max = 3 };
    if (operation == 1)
        LOG(INFO) << " operation = " << Eltwise_prod;
    if (operation == 2)
        LOG(INFO) << " operation = " << Eltwise_sum;
    if (operation == 3)
        LOG(INFO) << " operation = " << Eltwise_max;
    LOG(INFO) << "active =" << (ActiveType) Active_relu;

    int input_dim = 1;
    Shape shape_out = tin[0]->valid_shape();
    for (int i = 0; i < 4; i++){
    	shape_out[i] = tin[0]->valid_shape()[i];
    }
   //Shape shape_out{num, ch_out, h_out, w_out}

#ifdef USE_COMPARE

/*
    LOG(INFO) << "initial input tensor data 0:";
    print_tensor_host(*tin[0]);
    LOG(INFO) << "initial input tensor data 1:";
    print_tensor_host(*tin[1]);
*/
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];

    LOG(INFO) << "run basic eltwise active for precision comparation";
    tout_basic.re_alloc(shape_out);
    size_t workspace_size = sizeof(float) * numin * chin * (hin + 2 * pad) * (win + 2 * pad);
    void* work_space_data = fast_malloc(workspace_size);
   
    to = 0;
     for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        eltwise_active_basic(ctx1, tout_basic, tin, operation, coeffs_ptr, num_coeff, Active_relu);
        
        tvout_basic[0] ->record_event(ctx1.get_compute_stream());
        tvout_basic[0] ->sync();
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    fast_free(work_space_data);
    LOG(INFO) << "basic eltwise running time, ave: " << to / test_iter << ", min time: " << min_time;
   // print_tensor_host(tout_basic);

    
  
  LOG(INFO) << "run ncnn eltwise for precision comparation";
    TensorHf4 tout_basic2;
    tout_basic2.re_alloc(shape_out);
     to = 0;
     for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);

        eltwise_act_ncnn(ctx1, tout_basic2, tin, operation, coeffs_ptr, num_coeff, Active_relu);

        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "ncnn eltwise running time, ave: " << to/test_iter << ", min time: " << min_time;
    double max_ratio1 = 0;
    double max_diff1 = 0;
    tensor_cmp_host(tout_basic.data(), tout_basic2.data(), tout_basic.valid_size(), max_ratio1, max_diff1);
   // LOG(INFO) << "tout_basic";
   // print_tensor_host(tout_basic);
  // LOG(INFO) << "tout_saber";
   // print_tensor_host(tout_saber);
    LOG(INFO) << "compare result, max diff: " << max_diff1 << ", max ratio: " << max_ratio1;
    //CHECK_EQ(fabsf(max_ratio1) < 1e-5f, true) << "compute result error";

#endif
    
    EltwiseActive<ARM, AK_FLOAT> eltwise_act_saber;
    EltwiseParam<TensorHf4> eltwise_param(operation, coeffs_ptr);
    ActivationParam<TensorHf4> activation_param(Active_relu);
    EltwiseActiveParam<TensorHf4> eltwise_act_param(eltwise_param, activation_param);

    eltwise_act_saber.compute_output_shape(tin, tvout_saber, eltwise_act_param);

    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape_1: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
        << sh_out_saber[2] << ", " << sh_out_saber[3];
    //LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber eltwise act impl init";
    SABER_CHECK(eltwise_act_saber.init(tin, tvout_saber, eltwise_act_param, SPECIFY, SABER_IMPL, ctx1));

    //! compute
    LOG(INFO) << "saber eltwise act compute";
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start(ctx1);
       // pooling_saber(tin, tvout_saber, pooling_param, ctx1);
        //eltwise_arm(ctx2, tout_saber, tin, operation, coeffs_ptr, num_coeff);
        eltwise_act_saber(tin, tvout_saber, eltwise_act_param, ctx1);
        tvout_saber[0]->record_event(ctx1.get_compute_stream());
        tvout_saber[0]->sync();
        t2.end(ctx1);
        //printf("i: %d \n",i);
        to += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
    }
    LOG(INFO) << "saber eltwise active running time, ave: " << to / test_iter << ", min time: " << min_time;
   // print_tensor_host(tout_saber);
    //print_tensor_host(*tvout_saber[0]);

#ifdef USE_COMPARE
    double max_ratio = 0;
    double max_diff = 0;
    //TensorHf4 tdiff(tout_basic.valid_shape());
    //tensor_diff(tout_basic, tout_saber, tdiff);
    //print_tensor_host(tdiff);
    tensor_cmp_host(tout_basic.data(), tout_saber.data(), tout_basic.valid_size(), max_ratio, max_diff);
   // LOG(INFO) << "tout_basic";
   // print_tensor_host(tout_basic);
  // LOG(INFO) << "tout_saber";
   // print_tensor_host(tout_saber);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
}

#if 1
TEST(TestSaberFuncTest, test_func_eltwise_arm) {

    int num = 1;
    int chin = 32;
    int hin = 112;
    int win = 112;

    int pad = 1;
    int stride = 2;
    int kernel = 3;
    //int chout = 3;

   // bool bias_term = false;
   // bool global = true;
   // PoolingType type = 1;

    Shape shape_in(num, chin, hin, win);

    
    //fill_tensor_host_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    TensorHf4 tdin;
    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    TensorHf4 tdin1;
    tdin1.re_alloc(shape_in);
    fill_tensor_host_rand(tdin1, -1.f, 1.f);
    
    tin.push_back(&tdin);
    tin.push_back(&tdin1);
    
    
    std::vector<float> coeffs_ptr;
   
	coeffs_ptr.push_back(1.0f);
	coeffs_ptr.push_back(-1.0f);
    //printf("test_arm_eltwise: GLB_operation: %d \n", GLB_operation);
    test_arm_eltwise(tin, (EltwiseType)GLB_operation, coeffs_ptr, GLB_num_coeff, GLB_threads, GLB_cluster_id);
    //LOG(WARNING) << "pooling not support yet";
}
#endif

int main(int argc, const char** argv){
    anakin::saber::Env<ARM>::env_init();

    // initial logger
    //logger::init(argv[0]);
   // printf("Test0:\n");
     if (argc < 1) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/eltwise_test\n \
            threads\n \
            cluster_id\n \
            operation\n \
            num_coeff\n  ";
        exit(0);
    } else if (argc == 3){
        GLB_threads = atoi(argv[1]);
        GLB_cluster_id = atoi(argv[2]);
    }else if (argc == 5){
        GLB_threads = atoi(argv[1]);
        GLB_cluster_id = atoi(argv[2]);
        GLB_operation = atoi(argv[3]);
        GLB_num_coeff = atoi(argv[4]);
    }
    //printf("Test:\n");
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}



