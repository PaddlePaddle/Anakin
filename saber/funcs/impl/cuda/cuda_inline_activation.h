
#ifndef SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_CUDA_INLINE_ACTIVATION_H
#define SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_CUDA_INLINE_ACTIVATION_H

#include "saber_types.h"
#include "cuda.h"

#define SIGMOID_THRESHOLD_MIN_PADDLE -40.0
#define SIGMOID_THRESHOLD_MAX_PADDLE 13.0
#define EXP_MAX_INPUT_PADDLE 40.0

namespace anakin {

namespace saber {


template<typename Dtype>
static inline __device__ Dtype
InValidAct(Dtype
           a) {
    printf("invalid act\n");
    return static_cast<Dtype>(0);
}

template<typename Dtype>
static inline __device__ Dtype
Sigmoid(const Dtype a) {
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-a));
}


template<typename Dtype>
static inline __device__ Dtype
Tanh(const Dtype a) {
    Dtype tmp = -2.0 * a;
    return (2.0 / (1.0 + expf(tmp))) - 1.0;
}

template<typename Dtype>
static inline __device__ Dtype
Identity(const Dtype a) {
    return a;
}

template<typename Dtype>
static inline __device__ Dtype
Relu(const Dtype a) {
    return a > static_cast<Dtype>(0.0) ? a : static_cast<Dtype>(0.0);
}

template<typename Dtype>
static inline __device__ Dtype
Sigmoid_fluid(const Dtype a) {
    const Dtype min = SIGMOID_THRESHOLD_MIN_PADDLE;
    const Dtype max = SIGMOID_THRESHOLD_MAX_PADDLE;
    Dtype tmp = (a < min) ? min : ((a > max) ? max : a);

    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-tmp));
}

template<typename Dtype>
static inline __device__ Dtype
Tanh_fluid(const Dtype a) {
    Dtype tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT_PADDLE) ? EXP_MAX_INPUT_PADDLE : tmp;
    return (2.0 / (1.0 + expf(tmp))) - 1.0;
}

template<typename Dtype>
struct ACTIVATION {
    typedef  Dtype(*Act)(const Dtype);
};

template<typename Dtype>
__device__ inline typename ACTIVATION<Dtype>::Act Activate_inner(ActiveType type) {
    static  typename ACTIVATION<Dtype>::Act vec[7] = {&InValidAct<Dtype>, &Sigmoid < Dtype >, &Relu < Dtype >,
                                                     &Tanh < Dtype >,
                                                     &InValidAct<Dtype>, &InValidAct<Dtype>,
                                                     &Identity < Dtype >
                                                    };
    return vec[type];
}

}
}
#endif //SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_CUDA_INLINE_ACTIVATION_H
