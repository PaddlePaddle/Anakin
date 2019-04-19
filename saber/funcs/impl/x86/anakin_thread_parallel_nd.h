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

#ifndef SABER_FUNCS_IMPL_X86_ANAKIN_THREAD_PARALLEL_ND_H
#define SABER_FUNCS_IMPL_X86_ANAKIN_THREAD_PARALLEL_ND_H

#include <utility>

/* This header must be included by anakin_thread.hpp only */

/* Functions:
 *  - parallel(nthr, f)              - executes f in parallel using at most
 *                                     nthr threads. If nthr equals 0
 *                                     anakin_get_max_threads() threads is
 *                                     used
 *  - for_nd(ithr, nthr, dims..., f) - multidimensional for loop for already
 *                                     created threads
 *  - parallel_nd(dims..., f)        - creates a parallel section and then
 *                                     calls for_nd
 *  - parallel_nd_in_omp(dims..., f) - queries current nthr and ithr and then
 *                                     calls for_nd (mostly for convenience)
 */

namespace anakin {
namespace saber {

/* general parallelization */
template <typename F>
void parallel(int nthr, F f) {
    if (nthr == 0) nthr = anakin_get_max_threads();
#if ANAKIN_THR == ANAKIN_THR_SEQ
    assert(nthr == 1);
    f(0, 1);
#elif ANAKIN_THR == ANAKIN_THR_OMP
    if (nthr == 1) { f(0, 1); return; }
#   pragma omp parallel num_threads(nthr)
    f(anakin_get_thread_num(), anakin_get_num_threads());
#elif ANAKIN_THR == ANAKIN_THR_TBB
    if (nthr == 1) { f(0, 1); return; }
    tbb::parallel_for(0, nthr, [&](int ithr) { f(ithr, nthr); });
#endif
}

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
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

template<typename T>
inline T nd_iterator_init(T start) {
    return start;
}
template<typename T, typename U, typename W, typename... Args>
inline T nd_iterator_init(T start, U& x, const W& X, Args&& ... tuple) {
    start = nd_iterator_init(start, std::forward<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool nd_iterator_step() {
    return true;
}

template<typename U, typename W, typename... Args>
inline bool nd_iterator_step(U& x, const W& X, Args&& ... tuple) {
    if (nd_iterator_step(std::forward<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }

    return false;
}

template <typename T0, typename T1, typename F>
inline void parallel_nd(const T0 D0, const T1 D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;

    if (work_amount == 0) {
        return;
    }

    #pragma omp parallel
    {
        const int ithr = anakin_get_thread_num();
        const int nthr = anakin_get_num_threads();
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        T0 d0{0};
        T1 d1{0};
        nd_iterator_init(start, d0, D0, d1, D1);

        for (size_t iwork = start; iwork < end; ++iwork) {
            f(d0, d1);
            nd_iterator_step(d0, D0, d1, D1);
        }
    }
}

template <typename T0, typename T1, typename T2, typename F>
inline void parallel_nd(const T0 D0, const T1 D1, const T2 D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;

    if (work_amount == 0) {
        return;
    }

    #pragma omp parallel
    {
        const int ithr = anakin_get_thread_num();
        const int nthr = anakin_get_num_threads();
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        T0 d0{0};
        T1 d1{0};
        T2 d2{0};
        nd_iterator_init(start, d0, D0, d1, D1, d2, D2);

        for (size_t iwork = start; iwork < end; ++iwork) {
            f(d0, d1, d2);
            nd_iterator_step(d0, D0, d1, D1, d2, D2);
        }
    }
}

template<typename U, typename W, typename Y>
inline bool nd_iterator_jump(U& cur, const U end, W& x, const Y& X) {
    U max_jump = end - cur;
    U dim_jump = X - x;

    if (dim_jump <= max_jump) {
        x = 0;
        cur += dim_jump;
        return true;
    } else {
        cur += max_jump;
        x += max_jump;
        return false;
    }
}

template<typename U, typename W, typename Y, typename... Args>
inline bool nd_iterator_jump(U& cur, const U end, W& x, const Y& X,
                             Args&& ... tuple) {
    if (nd_iterator_jump(cur, end, std::forward<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }

    return false;
}

/* for_nd section */

template <typename T0, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, F f) {
    T0 d0{0}, end{0};
    balance211(D0, nthr, ithr, d0, end);
    for (; d0 < end; ++d0) f(d0);
}

template <typename T0, typename T1, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0};
    nd_iterator_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1);
        nd_iterator_step(d0, D0, d1, D1);
    }
}

template <typename T0, typename T1, typename T2, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0}; T2 d2{0};
    nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2);
        nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0};
    nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3);
        nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
         typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0};
    nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4);
        nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
         typename T5, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, const T5 &D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0}; T5 d5{0};
    nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4, d5);
        nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

/* parallel_nd and parallel_nd_in_omp section */

#if ANAKIN_THR != ANAKIN_THR_TBB
template <typename ...Args>
void parallel_nd(Args &&...args) {
#if ANAKIN_THR == ANAKIN_THR_SEQ
    for_nd(0, 1, std::forward<Args>(args)...);
#elif ANAKIN_THR == ANAKIN_THR_OMP
#   pragma omp parallel
    for_nd(anakin_get_thread_num(), anakin_get_num_threads(),
           std::forward<Args>(args)...);
#endif
}
#else // ANAKIN_THR != ANAKIN_THR_TBB

// gcc 4.8 has a bug with passing parameter pack to lambdas.
// So have to explicitly instantiate all the cases.

template <typename T0, typename F>
void parallel_nd(const T0 &D0, F f) {
    const int nthr = anakin_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_nd(ithr, nthr, D0, f);
    });
}

template <typename T0, typename T1, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, F f) {
    const int nthr = anakin_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_nd(ithr, nthr, D0, D1, f);
    });
}

template <typename T0, typename T1, typename T2, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, F f) {
    const int nthr = anakin_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_nd(ithr, nthr, D0, D1, D2, f);
    });
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3, F f) {
    const int nthr = anakin_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_nd(ithr, nthr, D0, D1, D2, D3, f);
    });
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
         typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
        const T4 &D4, F f) {
    const int nthr = anakin_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_nd(ithr, nthr, D0, D1, D2, D3, D4, f);
    });
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
         typename T5, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
        const T4 &D4, const T5 &D5, F f) {
    const int nthr = anakin_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_nd(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
    });
}
#endif

template <typename ...Args>
void parallel_nd_in_omp(Args &&...args) {
#if ANAKIN_THR == ANAKIN_THR_SEQ
    for_nd(0, 1, std::forward<Args>(args)...);
#elif ANAKIN_THR == ANAKIN_THR_OMP
    for_nd(anakin_get_thread_num(), anakin_get_num_threads(),
           std::forward<Args>(args)...);
#elif ANAKIN_THR == ANAKIN_THR_TBB
    assert(!"unsupported parallel_nd_in_omp()");
#endif
}

} // namespace saber
} // namespace anakin

#endif
