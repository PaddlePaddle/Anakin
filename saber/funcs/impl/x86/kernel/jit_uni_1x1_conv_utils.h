/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_JIT_UNI_1X1_CONV_UTIL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_JIT_UNI_1X1_CONV_UTIL_H

#include <stdint.h>
#include <float.h>

#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {

namespace jit {
inline float loss_ratio(int amount, int divider) {
    return float(utils::rnd_up(amount, divider) - amount) / utils::rnd_up(amount, divider);
}

inline int best_divider(int value, int min_divider, int max_divider,
                        bool find_max, int step = 1) {
    max_divider = utils::max(1, utils::min(max_divider, value));
    min_divider = utils::max(1, utils::min(min_divider, max_divider));

    float min_loss = FLT_MAX;
    int x_divider = max_divider;
    for (int divider = max_divider; divider >= min_divider; divider -= step) {
        const float loss = loss_ratio(value, divider);
        if ((find_max && loss < min_loss) || (!find_max && loss <= min_loss)) {
            min_loss = loss;
            x_divider = divider;
        }
    }
    return x_divider;
}

} // namepsace jit

#define JIT_TENSOR_MAX_DIMS 12
typedef int jit_dims_t[JIT_TENSOR_MAX_DIMS];
typedef int jit_strides_t[JIT_TENSOR_MAX_DIMS];

struct conv_1x1_desc {
    int n;
    int ic;
    int ih;
    int iw;
    int oc;
    int oh;
    int ow;
    int stride_h;
    int stride_w;
    int t_pad;
    int l_pad;
};


} // namespace saber
} // namespace anakin

#endif
