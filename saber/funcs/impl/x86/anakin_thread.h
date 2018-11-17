/*******************************************************************************
* Copyright (c) 2018 Anakin Authors All Rights Reserve.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef SABER_FUNCS_IMPL_X86_ANAKIN_THREAD_H
#define SABER_FUNCS_IMPL_X86_ANAKIN_THREAD_H

#include "utils.h"
#include "z_magic.h"

#define ANAKIN_THR_SEQ 0
#define ANAKIN_THR_OMP 1
#define ANAKIN_THR_TBB 2

#if !defined(ANAKIN_THR)
#define ANAKIN_THR ANAKIN_THR_OMP
#endif

#if ANAKIN_THR == ANAKIN_THR_SEQ
#define ANAKIN_THR_SYNC 1
inline int anakin_get_max_threads() { return 1; }
inline int anakin_get_num_threads() { return 1; }
inline int anakin_get_thread_num() { return 0; }
inline int anakin_in_parallel() { return 0; }
inline void anakin_thr_barrier() {}

#elif ANAKIN_THR == ANAKIN_THR_OMP
#include <omp.h>
#define ANAKIN_THR_SYNC 1

inline int anakin_get_max_threads() { return omp_get_max_threads(); }
inline int anakin_get_num_threads() { return omp_get_num_threads(); }
inline int anakin_get_thread_num() { return omp_get_thread_num(); }
inline int anakin_in_parallel() { return omp_in_parallel(); }
inline void anakin_thr_barrier() {
#   pragma omp barrier
}

#elif ANAKIN_THR == ANAKIN_THR_TBB
#include "tbb/parallel_for.h"
#define ANAKIN_THR_SYNC 0

inline int anakin_get_max_threads()
{ return tbb::this_task_arena::max_concurrency(); }
inline int anakin_get_num_threads() { return anakin_get_max_threads(); }
inline int anakin_get_thread_num()
{ return tbb::this_task_arena::current_thread_index(); }
inline int anakin_in_parallel() { return 0; }
inline void anakin_thr_barrier() { assert(!"no barrier in TBB"); }
#endif

/* MSVC still supports omp 2.0 only */
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#   define collapse(x)
#   define PRAGMA_OMP_SIMD(...)
#else
#   define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__INTEL_COMPILER)

namespace anakin {
namespace saber {

inline bool anakin_thr_syncable() { return ANAKIN_THR_SYNC == 1; }

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = utils::div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

} // namespace saber
} // namespace anakin

#include "anakin_thread_parallel_nd.h"

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
