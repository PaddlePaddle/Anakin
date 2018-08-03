#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_UTIL_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_UTIL_H
namespace anakin {

namespace saber {
namespace utils {

#include "core/common.h"

template<typename opTensor>
static inline void try_expand_tensor(opTensor& x, Shape shape) {
    if (x.valid_size() < shape.count()) {
        x.re_alloc(shape, x.get_dtype());
    }
}

template<typename opTensor>
static inline void try_expand_tensor(opTensor& x, int size) {
    if (x.valid_size() < size) {
        Shape shape({1, 1, 1, size}, Layout_NCHW);
        try_expand_tensor(x, shape);
    }
}


}
}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_UTIL_H
