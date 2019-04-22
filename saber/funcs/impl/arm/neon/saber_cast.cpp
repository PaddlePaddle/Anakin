#include "saber/funcs/impl/arm/saber_cast.h"

namespace anakin{

namespace saber{

void cast_fp32_to_int32(const float* din, int* dout, size_t size, int threads){
    size_t cnt = size / (threads * 16);
    int th_len = cnt * 16;
    int remain = size - th_len * threads;
#pragma omp parallel for
    for (int i = 0; i < threads; ++i){
        const float* din_ptr = din + i * th_len;
        int* dout_ptr = dout + i * th_len;
#ifdef __aarch64__
        int loop = cnt;
        if (loop > 0){
            asm volatile(
                "ld1 {v0.4s}, [%[din_ptr]], #16  \n"
                "ld1 {v1.4s}, [%[din_ptr]], #16  \n"
                "ld1 {v2.4s}, [%[din_ptr]], #16  \n"
                "ld1 {v3.4s}, [%[din_ptr]], #16  \n"
                "1:    \n"
                "fcvtzs v4.4s, v0.4s  \n"
                "ld1 {v0.4s}, [%[din_ptr]], #16  \n"
                "fcvtzs v5.4s, v1.4s  \n"
                "ld1 {v1.4s}, [%[din_ptr]], #16  \n"
                "fcvtzs v6.4s, v2.4s  \n"
                "ld1 {v2.4s}, [%[din_ptr]], #16  \n"
                "fcvtzs v7.4s, v3.4s  \n"
                "ld1 {v3.4s}, [%[din_ptr]], #16  \n"

                "st1 {v4.4s}, [%[dout_ptr]], #16\n"
                "st1 {v5.4s}, [%[dout_ptr]], #16\n"
                "st1 {v6.4s}, [%[dout_ptr]], #16\n"
                "st1 {v7.4s}, [%[dout_ptr]], #16\n"
                "subs %[loop], %[loop], #1  \n"
                "bne 1b  \n"
            : [din_ptr] "+r" (din_ptr), [dout_ptr] "+r" (dout_ptr), \
              [loop] "+r" (loop)
            :
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            );
        }
#else
        int loop = cnt;
        if (loop > 0){
            asm volatile(
                "vld1.32 {d0-d1}, [%[din_ptr]]!  \n"
                "vld1.32 {d2-d3}, [%[din_ptr]]!  \n"
                "vld1.32 {d4-d5}, [%[din_ptr]]!  \n"
                "vld1.32 {d6-d7}, [%[din_ptr]]!  \n"
                "1:     \n"
                "vcvt.s32.f32 q4, q0  \n"
                "vld1.32 {d0-d1}, [%[din_ptr]]!  \n"
                "vcvt.s32.f32 q5, q1 \n"
                "vld1.32 {d2-d3}, [%[din_ptr]]!  \n"
                "vcvt.s32.f32 q6, q2 \n"
                "vld1.32 {d4-d5}, [%[din_ptr]]!  \n"
                "vcvt.s32.f32 q7, q3 \n"
                "vld1.32 {d6-d7}, [%[din_ptr]]!  \n"

                "vst1.32 {d8-d9}, [%[dout_ptr]]!  \n"
                "vst1.32 {d10-d11}, [%[dout_ptr]]!  \n"
                "vst1.32 {d12-d13}, [%[dout_ptr]]!  \n"
                "vst1.32 {d14-d15}, [%[dout_ptr]]!  \n"
                "subs %[loop], #1  \n"
                "bne 1b  \n"
            : [din_ptr] "+r" (din_ptr), [dout_ptr] "+r" (dout_ptr), \
              [loop] "+r" (loop)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
        }
#endif // __aarch64__
    }
    const float* din_ptr = din + th_len * threads;
    int* dout_ptr = dout + th_len * threads;
    for (int i = 0; i < remain; ++i){
        *dout_ptr++ = static_cast<int>(*din_ptr++);
    }

}

void cast_int32_to_fp32(const int* din, float* dout, size_t size, int threads){
    size_t cnt = size / (threads * 16);
    int th_len = cnt * 16;
    int remain = size - th_len * threads;
#pragma omp parallel for
    for (int i = 0; i < threads; ++i){
        const int* din_ptr = din + i * th_len;
        float* dout_ptr = dout + i * th_len;
#ifdef __aarch64__
        int loop = cnt;
        if (loop > 0){
            asm volatile(
                "ld1 {v0.4s}, [%[din_ptr]], #16  \n"
                "ld1 {v1.4s}, [%[din_ptr]], #16  \n"
                "ld1 {v2.4s}, [%[din_ptr]], #16  \n"
                "ld1 {v3.4s}, [%[din_ptr]], #16  \n"
                "1:    \n"
                "scvtf v4.4s, v0.4s  \n"
                "ld1 {v0.4s}, [%[din_ptr]], #16  \n"
                "scvtf v5.4s, v1.4s  \n"
                "ld1 {v1.4s}, [%[din_ptr]], #16  \n"
                "scvtf v6.4s, v2.4s  \n"
                "ld1 {v2.4s}, [%[din_ptr]], #16  \n"
                "scvtf v7.4s, v3.4s  \n"
                "ld1 {v3.4s}, [%[din_ptr]], #16  \n"

                "st1 {v4.4s}, [%[dout_ptr]], #16\n"
                "st1 {v5.4s}, [%[dout_ptr]], #16\n"
                "st1 {v6.4s}, [%[dout_ptr]], #16\n"
                "st1 {v7.4s}, [%[dout_ptr]], #16\n"
                "subs %[loop], %[loop], #1  \n"
                "bne 1b  \n"
            : [din_ptr] "+r" (din_ptr), [dout_ptr] "+r" (dout_ptr), \
              [loop] "+r" (loop)
            :
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            );
        }
#else
        int loop = cnt;
        if (loop > 0){
            asm volatile(
                "vld1.32 {d0-d1}, [%[din_ptr]]!  \n"
                "vld1.32 {d2-d3}, [%[din_ptr]]!  \n"
                "vld1.32 {d4-d5}, [%[din_ptr]]!  \n"
                "vld1.32 {d6-d7}, [%[din_ptr]]!  \n"
                "1:     \n"
                "vcvt.f32.s32 q4, q0  \n"
                "vld1.32 {d0-d1}, [%[din_ptr]]!  \n"
                "vcvt.f32.s32 q5, q1 \n"
                "vld1.32 {d2-d3}, [%[din_ptr]]!  \n"
                "vcvt.f32.s32 q6, q2 \n"
                "vld1.32 {d4-d5}, [%[din_ptr]]!  \n"
                "vcvt.f32.s32 q7, q3 \n"
                "vld1.32 {d6-d7}, [%[din_ptr]]!  \n"

                "vst1.32 {d8-d9}, [%[dout_ptr]]!  \n"
                "vst1.32 {d10-d11}, [%[dout_ptr]]!  \n"
                "vst1.32 {d12-d13}, [%[dout_ptr]]!  \n"
                "vst1.32 {d14-d15}, [%[dout_ptr]]!  \n"
                "subs %[loop], #1  \n"
                "bne 1b  \n"
            : [din_ptr] "+r" (din_ptr), [dout_ptr] "+r" (dout_ptr), \
              [loop] "+r" (loop)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
        }

#endif  //__aarch64__
    }
    const int* din_ptr = din + th_len * threads;
    float* dout_ptr = dout + th_len * threads;
    for (int i = 0; i < remain; ++i){
        *dout_ptr++ = static_cast<float>(*din_ptr++);
    }

}

template <DataType OpDtype>
SaberStatus SaberCast<ARM, OpDtype>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        CastParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    int threads = this->_ctx->get_threads();
    const void* din = inputs[0]->data();
    void* dout = outputs[0]->mutable_data();
    size_t size = inputs[0]->valid_size();
    if (param.in_type == param.out_type){
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }
    if (param.in_type == (int)AK_FLOAT && param.out_type == (int)AK_INT32){
        cast_fp32_to_int32((const float*)din, (int*)dout, size, threads);
    } else if (param.in_type == (int)AK_INT32 && param.out_type == (int)AK_FLOAT){
        cast_int32_to_fp32((const int*)din, (float*)dout, size, threads);
    } else {
        LOG(ERROR) << "unsupport cast case, in_type: %d" << param.in_type << ", out_type: %d \n" << param.out_type;
        return SaberInvalidValue;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Cast : " << this->_op_name.c_str() << " : time: " << ts;
    // GOPS ops;
    // //fixme
    // ops.ops =  2.f * inputs[0]->valid_size();
    // ops.ts = ts;
    // OpTimer::add_timer("Cast", ops);
    // OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
template class SaberCast<ARM, AK_FLOAT>;
template class SaberCast<ARM, AK_INT32>;
DEFINE_OP_TEMPLATE(SaberCast, CastParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberCast, CastParam, ARM, AK_INT8);
} //namespace anakin

} //namespace anakin
