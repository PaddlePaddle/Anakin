#include "saber/lite/funcs/neon/impl/sgemm_arm.h"
#ifdef USE_ARM_PLACE
#include <cmath>
#include "saber/lite/core/buffer_lite.h"
namespace anakin{

namespace saber{

namespace lite{

using namespace anakin::saber::lite;

#ifdef __aarch64__
const int A_INTERLEAVE = 8;
const int B_INTERLEAVE = 12;
const int OUT_WIDTH = 12;
const int OUT_HEIGHT = 8;
#else
const int A_INTERLEAVE = 6;
const int B_INTERLEAVE = 8;
const int OUT_WIDTH = 8;
const int OUT_HEIGHT = 6;
#endif //__aarch64
const bool A_TRANSPOSE = false;
const bool B_TRANSPOSE = true;

const int GEMM_ALIGN = 4096;
const int ALLOC_ROUND = 128;
#define ROUND_UP(x)	((((x) + ALLOC_ROUND-1) / ALLOC_ROUND) * ALLOC_ROUND)

inline void *mem_align(std::size_t alignment, std::size_t size, void *&ptr, std::size_t &space) {
    std::uintptr_t pn      = reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t aligned = (pn + alignment - 1) & -alignment;
    std::size_t    padding = aligned - pn;
    if (space < size + padding) {
        return nullptr;
    }
    space -= padding;
    return ptr = reinterpret_cast<void *>(aligned);
}

void sgemm_impl(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, \
    int bblocks, int K, long int row_jump=0, long int block_jump=0);

void load_apanel_no_trans(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);
void load_apanel_trans(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);
void load_bpanel_no_trans(float* out, const float* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax);
void load_bpanel_trans(float* out, const float* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax);

void merge_float_basic(float *out, const float *in, const int ldout, const int y0, \
    const int ymax, const int x0, const int xmax, const float alpha, const float beta);
void merge_float_basic_relu(float *out, const float *in, const int ldout, const int y0, \
    const int ymax, const int x0, const int xmax, const float alpha, const float beta);
void merge_float_alpha1_beta1(float *out, const float *in, const int ldout, const int y0, \
    const int ymax, const int x0, const int xmax);
void merge_float_alpha1_beta1_relu(float *out, const float *in, const int ldout, const int y0, \
    const int ymax, const int x0, const int xmax);

Sgemm::Sgemm() {}

Sgemm::~Sgemm() {
    if (_work_space_ptr != nullptr) {
        fast_free(_work_space_ptr);
        _work_space_ptr = nullptr;
    }
}

void Sgemm::init(unsigned int L1_cache, unsigned int L2_cache, unsigned int M, unsigned int N, \
    unsigned int K, bool trA, bool trB, int thread_num) {

    _M = M;
    _NN = N;
    _K = K;
    _trA = trA ^ A_TRANSPOSE;
    if (_trA) {
        //_load_a = Transform<A_INTERLEAVE, A_BLOCK, true>;
        _load_a = load_apanel_trans;
    } else {
        //_load_a = Transform<A_INTERLEAVE, A_BLOCK, false>;
        _load_a = load_apanel_no_trans;
    }
    _trB = trB ^ B_TRANSPOSE;
    if (_trB) {
        //_load_b = Transform<B_INTERLEAVE, B_BLOCK, true>;
        _load_b = load_bpanel_trans;
    } else {
        //_load_b = Transform<B_INTERLEAVE, B_BLOCK, false>;
        _load_b = load_bpanel_no_trans;
    }

    _thread_num = thread_num;

    unsigned int L1_size = L1_cache;
    if (L1_size <= 0) {
        //! 32K
        L1_size = 32000;
    }
    //! A72/A53 L1 data cache 32k//ci->L1_size;
    unsigned int L2_size = L2_cache;
    if (L2_size <= 0) {
        //! 2M
        L2_size = 2000000;
    }
    //! rockchip rk3399, with two A72, and four A53
    //! A72, 1M on big core, shared by two core,
    //! A53, 512K on little core, shared by four core //ci->L2_size;

    //! Work out blocking parameters
    //! k_block: Each iteration will consume (out_width + out_height)
    //! operands - so how many iterations will fill the L1?
    _k_block = L1_size / (sizeof(float) * (OUT_WIDTH + OUT_HEIGHT));

    int num_k_blocks = (K + (_k_block - 1)) / _k_block;
    _k_block = (K + num_k_blocks - 1) / num_k_blocks;

    //! x_block: Work out how many rows (of length k_block) will fit in the L2
    _x_block = L2_size / (sizeof(float) * _k_block);
    _x_block /= OUT_WIDTH;
    _x_block *= OUT_WIDTH;
    int num_x_blocks = (N + (_x_block - 1)) / _x_block;
    _x_block = (N + num_x_blocks - 1) / num_x_blocks;
    _x_block = (_x_block + OUT_WIDTH - 1) / OUT_WIDTH;
    _x_block *= OUT_WIDTH;

    //! Work out the rounded size of M - needed for some buffers.
    _Mround = (M + (OUT_HEIGHT - 1)) / OUT_HEIGHT;
    _Mround *= OUT_HEIGHT;

    _a_worksize = ROUND_UP(sizeof(float) * _k_block * _Mround);
    _b_worksize = ROUND_UP(sizeof(float) * _x_block * _k_block);
    //_c_worksize_per_thread = ROUND_UP(sizeof(float) * _x_block * OUT_HEIGHT);
    //_c_worksize = _thread_num * _c_worksize_per_thread;
    _cblock_size = ROUND_UP(sizeof(float) * _x_block * OUT_HEIGHT) / sizeof(float);

    _work_size = _a_worksize + _b_worksize + _cblock_size * sizeof(float) * _thread_num;

    _work_space_ptr = fast_malloc(_work_size + GEMM_ALIGN);
    _align_ptr = _work_space_ptr;
    size_t size_gemm_align = _work_size + GEMM_ALIGN - 1;
    if (mem_align(GEMM_ALIGN, _work_size, _align_ptr, \
            size_gemm_align) == nullptr) {
        LCHECK_EQ(0, 1, "Not enough space to align buffer!");
    }
    _loop_count = (_K - 1) / _k_block;
    _init_flag = true;
}

void Sgemm::operator()(const float *A, const int lda, \
    const float *B, const int ldb, \
    float *C, const int ldc, \
    const float alpha, const float beta, bool flag_relu) {

    LCHECK_EQ(_init_flag, true, "gemm is not init");

    bool flag_beta = (fabsf(beta - 1.f) < 1e-6f);
    bool flag_alpha = (fabsf(alpha -1.f) < 1e-6f);

    int8_t *working_space_bytes = reinterpret_cast<int8_t *>(_align_ptr);
    intptr_t working_space_int = reinterpret_cast<intptr_t>(working_space_bytes);
    size_t diff = 0;

    if (working_space_int & 0xF) {
        diff = 0x10 - (working_space_int & 0xF);
    }

    float* const a_panel = reinterpret_cast<float*>(working_space_bytes + diff);
    float* const b_panel = reinterpret_cast<float*>(working_space_bytes + _a_worksize + diff);
    float* const c_panel = reinterpret_cast<float*>(working_space_bytes + _a_worksize + _b_worksize + diff);

    int index = 0;

    for (unsigned int k0 = 0; k0 < _K; k0 += _k_block) {
        unsigned int kmax = k0 + _k_block;
        if (kmax > _K) {
            kmax = _K;
        }
        int kern_k = kmax - k0;
        _load_a(a_panel, A, lda, 0, _M, k0, kmax);
        for (unsigned int x0 = 0; x0 < _NN; x0 += _x_block) {

            unsigned int xmax = x0 + _x_block;
            if (xmax > _NN) {
                xmax = _NN;
            }
            int bblocks = (xmax - x0 + OUT_WIDTH - 1) / OUT_WIDTH;
            _load_b(b_panel, B, ldb, k0, kmax, x0, xmax);
#pragma omp parallel for num_threads(_thread_num)
            for (unsigned int y = 0; y < _M; y += OUT_HEIGHT) {
                unsigned int ymax = y + OUT_HEIGHT;
                if (ymax > _M) {
                    ymax = _M;
                }
#ifdef USE_OPENMP
                float* cpan1 = c_panel + omp_get_thread_num() * _cblock_size;
#else
                float* cpan1 = c_panel;
#endif
                sgemm_impl(a_panel + (y * kern_k), b_panel, cpan1, 1, bblocks, kern_k);
                //! bias must be added to output before doing gemm
                if (flag_relu && (index == _loop_count)) {
                    if ((k0 > 0) || flag_beta) {
                        merge_float_alpha1_beta1_relu(C, cpan1, ldc, y, ymax, x0, xmax);
                        //merge_float_basic_relu(C, cpan1, ldc, y, ymax, x0, xmax, alpha, (k0 == 0 ? beta : 1.f));
                    } else {
                        merge_float_basic_relu(C, cpan1, ldc, y, ymax, x0, xmax, alpha, (k0 == 0 ? beta : 1.f));
                    }
                } else {
                    if (flag_alpha && (k0 > 0) || flag_beta) {
                        merge_float_alpha1_beta1(C, cpan1, ldc, y, ymax, x0, xmax);
                    } else {
                        merge_float_basic(C, cpan1, ldc, y, ymax, x0, xmax, alpha, (k0 == 0 ? beta : 1.f));
                    }
                }
            }
        }
        index++;
    }
}

#ifdef __aarch64__
// Kernel implementation.
//
// Assume that "Apanel" points to a chunk of A blocks (each size 8xK) in read-order.
// Assume that "Bpanel" points to a chunk of B blocks (each size 12xK) in read-order.
// Assume that "Cpanel" points to a chunk of C output blocks (each size
// 8x12), the chunks being arranged in a row major fashion.
//
// Note that the intent of this is that either ablocks or bblocks will be 1
// - this construction allows the output loop to proceed in either order.
void sgemm_impl(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, \
    int bblocks, int K, long int row_jump, long int block_jump) {
    const float *a_ptr = Apanel;
    float *c_ptr = Cpanel;

    for (int yb=0; yb<ablocks; yb++) {
        const float *a_ptr0 = a_ptr;
        const float *b_ptr = Bpanel;



        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            // Fix up for odd lengths - set a flag if K is odd, but make
            // sure we round up the iteration count.
            int oddk = (K & 1);
            int k = ((K+1)/2) - 1;

            register float32x4_t a0  asm("v0");
            register float32x4_t a1  asm("v1");
            register float32x4_t b0  asm("v2");
            register float32x4_t b1  asm("v3");
            register float32x4_t b2  asm("v4");
            register float32x4_t a0a asm("v5");
            register float32x4_t a1a asm("v6");

            asm volatile (
            // Initialize result registers, load initial operands, prime prefetches.
                "movi	v8.4s, #0x0\n"
                "ldr	%q[a0], [%[a_ptr]]\n"
                "movi	v9.4s, #0x0\n"
                "ldr	%q[b0], [%[b_ptr]]\n"
                "movi	v10.4s, #0x0\n"
                "ldr	%q[a1], [%[a_ptr], #16]\n"
                "movi	v11.4s, #0x0\n"
                "ldr	%q[b1], [%[b_ptr], #16]\n"
                "movi	v12.4s, #0x0\n"
                "prfm   pldl1keep, [%[b_ptr], #64]\n"
                "movi	v13.4s, #0x0\n"
                "prfm   pldl1keep, [%[a_ptr], #64]\n"
                "movi	v14.4s, #0x0\n"
                "prfm   pldl1keep, [%[b_ptr], #128]\n"
                "movi	v15.4s, #0x0\n"
                "prfm   pldl1keep, [%[a_ptr], #128]\n"
                "movi	v16.4s, #0x0\n"
                "prfm   pldl1keep, [%[b_ptr], #192]\n"
                "movi	v17.4s, #0x0\n"
                "prfm   pldl1keep, [%[b_ptr], #256]\n"
                "movi	v18.4s, #0x0\n"
                "prfm   pldl1keep, [%[a_ptr], #192]\n"
                "movi	v19.4s, #0x0\n"
                "prfm   pldl1keep, [%[b_ptr], #320]\n"
                "movi	v20.4s, #0x0\n"
                "prfm   pldl1keep, [%[a_ptr], #256]\n"
                "movi	v21.4s, #0x0\n"
                "prfm   pldl1keep, [%[b_ptr], #384]\n"
                "movi	v22.4s, #0x0\n"
                "movi	v23.4s, #0x0\n"
                "movi	v24.4s, #0x0\n"
                "movi	v25.4s, #0x0\n"
                "movi	v26.4s, #0x0\n"
                "movi	v27.4s, #0x0\n"
                "movi	v28.4s, #0x0\n"
                "movi	v29.4s, #0x0\n"
                "movi	v30.4s, #0x0\n"
                "movi	v31.4s, #0x0\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz	%w[k], 4f\n"

                // Loop proper
                "1:\n"
                "fmla 	v8.4s , %[b0].4s, %[a0].s[0]\n"
                "fmla  	v9.4s , %[b0].4s, %[a0].s[1]\n"
                "ldr	%q[b2], [%[b_ptr], #32]\n"
                "fmla	v10.4s, %[b0].4s, %[a0].s[2]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "fmla	v11.4s, %[b0].4s, %[a0].s[3]\n"
                "ldr	%q[a0a], [%[a_ptr], #32]\n"
                "fmla 	v12.4s, %[b0].4s, %[a1].s[0]\n"
                "fmla	v13.4s, %[b0].4s, %[a1].s[1]\n"
                "ldr	%q[a1a], [%[a_ptr], #48]\n"
                "fmla	v14.4s, %[b0].4s, %[a1].s[2]\n"
                "fmla	v15.4s, %[b0].4s, %[a1].s[3]\n"
                "ldr	%q[b0], [%[b_ptr], #48]\n"

                "fmla	v16.4s, %[b1].4s, %[a0].s[0]\n"
                "fmla	v17.4s, %[b1].4s, %[a0].s[1]\n"
                "prfm   pldl1keep, [%[a_ptr], #320]\n"
                "fmla	v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla	v19.4s, %[b1].4s, %[a0].s[3]\n"
                "fmla	v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla	v21.4s, %[b1].4s, %[a1].s[1]\n"
                "fmla	v22.4s, %[b1].4s, %[a1].s[2]\n"
                "fmla	v23.4s, %[b1].4s, %[a1].s[3]\n"
                "ldr	%q[b1], [%[b_ptr], #64]\n"

                "fmla	v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla	v25.4s, %[b2].4s, %[a0].s[1]\n"
                "prfm   pldl1keep, [%[b_ptr], #448]\n"
                "fmla	v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla	v27.4s, %[b2].4s, %[a0].s[3]\n"
                "fmla	v28.4s, %[b2].4s, %[a1].s[0]\n"
                "fmla	v29.4s, %[b2].4s, %[a1].s[1]\n"
                "fmla	v30.4s, %[b2].4s, %[a1].s[2]\n"
                "fmla	v31.4s, %[b2].4s, %[a1].s[3]\n"
                "ldr	%q[b2], [%[b_ptr], #80]\n"

                "fmla 	v8.4s , %[b0].4s, %[a0a].s[0]\n"
                "fmla	v9.4s , %[b0].4s, %[a0a].s[1]\n"
                "ldr	%q[a0], [%[a_ptr], #64]\n"
                "fmla	v10.4s, %[b0].4s, %[a0a].s[2]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "fmla	v11.4s, %[b0].4s, %[a0a].s[3]\n"
                "fmla 	v12.4s, %[b0].4s, %[a1a].s[0]\n"
                "ldr	%q[a1], [%[a_ptr], #80]\n"
                "fmla   v13.4s, %[b0].4s, %[a1a].s[1]\n"
                "fmla	v14.4s, %[b0].4s, %[a1a].s[2]\n"
                "fmla	v15.4s, %[b0].4s, %[a1a].s[3]\n"
                "ldr	%q[b0], [%[b_ptr], #96]\n"

                "fmla	v16.4s, %[b1].4s, %[a0a].s[0]\n"
                "fmla	v17.4s, %[b1].4s, %[a0a].s[1]\n"
                "prfm   pldl1keep, [%[b_ptr], #512]\n"
                "fmla	v18.4s, %[b1].4s, %[a0a].s[2]\n"
                "fmla	v19.4s, %[b1].4s, %[a0a].s[3]\n"
                "fmla	v20.4s, %[b1].4s, %[a1a].s[0]\n"
                "fmla	v21.4s, %[b1].4s, %[a1a].s[1]\n"
                "fmla	v22.4s, %[b1].4s, %[a1a].s[2]\n"
                "fmla	v23.4s, %[b1].4s, %[a1a].s[3]\n"
                "ldr	%q[b1], [%[b_ptr], #112]\n"

                "fmla	v24.4s, %[b2].4s, %[a0a].s[0]\n"
                "fmla	v25.4s, %[b2].4s, %[a0a].s[1]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"
                "fmla	v26.4s, %[b2].4s, %[a0a].s[2]\n"
                "fmla	v27.4s, %[b2].4s, %[a0a].s[3]\n"
                "add	%[b_ptr], %[b_ptr], #96\n"
                "fmla	v28.4s, %[b2].4s, %[a1a].s[0]\n"
                "fmla	v29.4s, %[b2].4s, %[a1a].s[1]\n"
                "subs	%w[k], %w[k], #1\n"
                "fmla	v30.4s, %[b2].4s, %[a1a].s[2]\n"
                "fmla	v31.4s, %[b2].4s, %[a1a].s[3]\n"
                "bne	1b\n"

                // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
                "4:\n"

                // Branch to alternative tail for odd K
                "cbnz	%[oddk], 2f\n"

                // Detached final iteration (even K)
                "fmla 	v8.4s , %[b0].4s, %[a0].s[0]\n"
                "fmla   v9.4s , %[b0].4s, %[a0].s[1]\n"
                "ldr	%q[b2], [%[b_ptr], #32]\n"
                "fmla	v10.4s, %[b0].4s, %[a0].s[2]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "fmla	v11.4s, %[b0].4s, %[a0].s[3]\n"
                "ldr	%q[a0a], [%[a_ptr], #32]\n"
                "fmla 	v12.4s, %[b0].4s, %[a1].s[0]\n"
                "fmla   v13.4s, %[b0].4s, %[a1].s[1]\n"
                "ldr	%q[a1a], [%[a_ptr], #48]\n"
                "fmla	v14.4s, %[b0].4s, %[a1].s[2]\n"
                "fmla	v15.4s, %[b0].4s, %[a1].s[3]\n"
                "ldr	%q[b0], [%[b_ptr], #48]\n"

                "fmla	v16.4s, %[b1].4s, %[a0].s[0]\n"
                "fmla	v17.4s, %[b1].4s, %[a0].s[1]\n"
                "fmla	v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla	v19.4s, %[b1].4s, %[a0].s[3]\n"
                "fmla	v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla	v21.4s, %[b1].4s, %[a1].s[1]\n"
                "fmla	v22.4s, %[b1].4s, %[a1].s[2]\n"
                "fmla	v23.4s, %[b1].4s, %[a1].s[3]\n"
                "ldr	%q[b1], [%[b_ptr], #64]\n"

                "fmla	v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla	v25.4s, %[b2].4s, %[a0].s[1]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"
                "fmla	v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla	v27.4s, %[b2].4s, %[a0].s[3]\n"
                "fmla	v28.4s, %[b2].4s, %[a1].s[0]\n"
                "fmla	v29.4s, %[b2].4s, %[a1].s[1]\n"
                "fmla	v30.4s, %[b2].4s, %[a1].s[2]\n"
                "fmla	v31.4s, %[b2].4s, %[a1].s[3]\n"
                "ldr	%q[b2], [%[b_ptr], #80]\n"

                "fmla 	v8.4s , %[b0].4s, %[a0a].s[0]\n"
                "add	%[b_ptr], %[b_ptr], %[block_jump]\n"
                "fmla	v16.4s, %[b1].4s, %[a0a].s[0]\n"
                "add	%[b_ptr], %[b_ptr], #96\n"
                "fmla   v9.4s , %[b0].4s, %[a0a].s[1]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "str	q8, [%[c_ptr], #0]\n"
                "fmla	v17.4s, %[b1].4s, %[a0a].s[1]\n"
                "str	q16, [%[c_ptr], #16]\n"
                "fmla	v24.4s, %[b2].4s, %[a0a].s[0]\n"
                "str	q24, [%[c_ptr], #32]\n"

                "fmla	v25.4s, %[b2].4s, %[a0a].s[1]\n"
                "str	q9, [%[c_ptr], #48]\n"
                "fmla	v10.4s, %[b0].4s, %[a0a].s[2]\n"
                "str	q17, [%[c_ptr], #64]\n"
                "fmla	v18.4s, %[b1].4s, %[a0a].s[2]\n"
                "str	q25, [%[c_ptr], #80]\n"
                "fmla	v26.4s, %[b2].4s, %[a0a].s[2]\n"
                "str	q10, [%[c_ptr], #96]\n"

                "fmla	v11.4s, %[b0].4s, %[a0a].s[3]\n"
                "str	q18, [%[c_ptr], #112]\n"
                "fmla	v19.4s, %[b1].4s, %[a0a].s[3]\n"
                "str	q26, [%[c_ptr], #128]\n"
                "fmla	v27.4s, %[b2].4s, %[a0a].s[3]\n"
                "str	q11, [%[c_ptr], #144]\n"

                "fmla 	v12.4s, %[b0].4s, %[a1a].s[0]\n"
                "str	q19, [%[c_ptr], #160]\n"
                "fmla	v20.4s, %[b1].4s, %[a1a].s[0]\n"
                "str	q27, [%[c_ptr], #176]\n"
                "fmla	v28.4s, %[b2].4s, %[a1a].s[0]\n"
                "str	q12, [%[c_ptr], #192]\n"

                "fmla   v13.4s, %[b0].4s, %[a1a].s[1]\n"
                "str	q20, [%[c_ptr], #208]\n"
                "fmla	v21.4s, %[b1].4s, %[a1a].s[1]\n"
                "str	q28, [%[c_ptr], #224]\n"
                "fmla	v29.4s, %[b2].4s, %[a1a].s[1]\n"
                "str	q13, [%[c_ptr], #240]\n"

                "fmla	v14.4s, %[b0].4s, %[a1a].s[2]\n"
                "str	q21, [%[c_ptr], #256]\n"
                "fmla	v22.4s, %[b1].4s, %[a1a].s[2]\n"
                "str	q29, [%[c_ptr], #272]\n"
                "fmla	v30.4s, %[b2].4s, %[a1a].s[2]\n"
                "str	q14, [%[c_ptr], #288]\n"

                "fmla	v15.4s, %[b0].4s, %[a1a].s[3]\n"
                "str	q22, [%[c_ptr], #304]\n"
                "fmla	v23.4s, %[b1].4s, %[a1a].s[3]\n"
                "str	q30, [%[c_ptr], #320]\n"
                "fmla	v31.4s, %[b2].4s, %[a1a].s[3]\n"
                "str	q15, [%[c_ptr], #336]\n"

                "b	3f\n"

                // Detached final iteration (odd K)
                "2:\n"
                "fmla 	v8.4s , %[b0].4s, %[a0].s[0]\n"
                "ldr	%q[b2], [%[b_ptr], #32]\n"
                "fmla	v16.4s, %[b1].4s, %[a0].s[0]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "fmla   v9.4s , %[b0].4s, %[a0].s[1]\n"
                "str	q8, [%[c_ptr], #0]\n"
                "fmla	v17.4s, %[b1].4s, %[a0].s[1]\n"
                "str	q16, [%[c_ptr], #16]\n"
                "fmla	v24.4s, %[b2].4s, %[a0].s[0]\n"
                "add	%[b_ptr], %[b_ptr], #48\n"
                "add	%[a_ptr], %[a_ptr], #32\n"
                "str	q24, [%[c_ptr], #32]\n"
                "fmla	v25.4s, %[b2].4s, %[a0].s[1]\n"
                "str	q9, [%[c_ptr], #48]\n"

                "fmla	v10.4s, %[b0].4s, %[a0].s[2]\n"
                "str	q17, [%[c_ptr], #64]\n"
                "fmla	v18.4s, %[b1].4s, %[a0].s[2]\n"
                "str	q25, [%[c_ptr], #80]\n"
                "fmla	v26.4s, %[b2].4s, %[a0].s[2]\n"
                "str	q10, [%[c_ptr], #96]\n"

                "fmla	v11.4s, %[b0].4s, %[a0].s[3]\n"
                "str	q18, [%[c_ptr], #112]\n"
                "fmla	v19.4s, %[b1].4s, %[a0].s[3]\n"
                "str	q26, [%[c_ptr], #128]\n"
                "fmla	v27.4s, %[b2].4s, %[a0].s[3]\n"
                "str	q11, [%[c_ptr], #144]\n"

                "fmla 	v12.4s, %[b0].4s, %[a1].s[0]\n"
                "str	q19, [%[c_ptr], #160]\n"
                "fmla	v20.4s, %[b1].4s, %[a1].s[0]\n"
                "str	q27, [%[c_ptr], #176]\n"
                "fmla	v28.4s, %[b2].4s, %[a1].s[0]\n"
                "str	q12, [%[c_ptr], #192]\n"

                "fmla   v13.4s, %[b0].4s, %[a1].s[1]\n"
                "str	q20, [%[c_ptr], #208]\n"
                "fmla	v21.4s, %[b1].4s, %[a1].s[1]\n"
                "str	q28, [%[c_ptr], #224]\n"
                "fmla	v29.4s, %[b2].4s, %[a1].s[1]\n"
                "str	q13, [%[c_ptr], #240]\n"

                "fmla	v14.4s, %[b0].4s, %[a1].s[2]\n"
                "str	q21, [%[c_ptr], #256]\n"
                "fmla	v22.4s, %[b1].4s, %[a1].s[2]\n"
                "str	q29, [%[c_ptr], #272]\n"
                "fmla	v30.4s, %[b2].4s, %[a1].s[2]\n"
                "str	q14, [%[c_ptr], #288]\n"

                "fmla	v15.4s, %[b0].4s, %[a1].s[3]\n"
                "str	q22, [%[c_ptr], #304]\n"
                "fmla	v23.4s, %[b1].4s, %[a1].s[3]\n"
                "str	q30, [%[c_ptr], #320]\n"
                "fmla	v31.4s, %[b2].4s, %[a1].s[3]\n"
                "str	q15, [%[c_ptr], #336]\n"

                    // Common tail
                "3:\n"
                "str	q23, [%[c_ptr], #352]\n"
                "str	q31, [%[c_ptr], #368]\n"
                "add	%[c_ptr], %[c_ptr], #384\n"
            :
            [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
            [a0] "+w" (a0), [a1] "+w" (a1), [a0a] "+w" (a0a), [a1a] "+w" (a1a),
            [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [k] "+r" (k)
            : [oddk] "r" (oddk), [row_jump] "r" (row_jump), [block_jump] "r" (block_jump)
            : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
        }
    }
}
#else

/**
 * Sgemm Kernel implementation for arm-v7a
 * Assume that "Apanel" points to a chunk of A blocks (each size 6xK) in read-order.
 * Assume that "Bpanel" points to a chunk of B blocks (each size 8xK) in read-order.
 * Assume that "Cpanel" points to a chunk of C output blocks (each size 8x6),
 * the chunks being arranged in a row major fashion.
 * Note that the intent of this is that either ablocks or bblocks will be 1
 * - this construction allows the output loop to proceed in either order.
 */
void sgemm_impl(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, \
    int bblocks, int K, long int row_jump, long int block_jump) {

    const float *a_ptr = Apanel;
    float *c_ptr = Cpanel;

    int k_pre = ((K + 3) / 4) - 1;
    int tail_pre = (K & 3);
    if (tail_pre == 0) {
        tail_pre = 4;
    }


    for (int yb = 0; yb < ablocks; yb++) {
        const float *a_ptr0 = a_ptr;
        const float *b_ptr = Bpanel;

        for (int xb = 0; xb < bblocks; xb++) {
            a_ptr = a_ptr0;
            int tails = tail_pre;
            int k = k_pre;

            asm volatile (
            "vmov.i32	q4, #0\n"
                    "vld1.32	{d0-d1}, [%[a_ptr] :64]!\n"
                    "vmov.i32	q5, #0\n"
                    "vld1.32	{d4-d5}, [%[b_ptr] :128]!\n"
                    "vmov.i32	q6, #0\n"
                    "pld [%[a_ptr], #48]\n"
                    "vmov.i32	q7, #0\n"
                    "pld [%[b_ptr], #48]\n"
                    "vmov.i32	q8, #0\n"
                    "pld [%[a_ptr], #112]\n"
                    "vmov.i32	q9, #0\n"
                    "pld [%[b_ptr], #112]\n"
                    "vmov.i32	q10, #0\n"
                    "vmov.i32	q11, #0\n"
                    "vmov.i32	q12, #0\n"
                    "vmov.i32	q13, #0\n"
                    "pld [%[a_ptr], #176]\n"
                    "vmov.i32	q14, #0\n"
                    "pld [%[b_ptr], #176]\n"
                    "vmov.i32	q15, #0\n"
                    "cmp %[k], #0 @check weather k is bigger than 0\n"
                    "beq 0f \n"

                    "1:\n"
                    // Unroll 0
                    "vmla.f32	q4, q2, d0[0]\n"
                    "vld1.32	{d2-d3}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q5, q2, d0[1]\n"
                    "vmla.f32	q6, q2, d1[0]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"
                    "vmla.f32	q7, q2, d1[1]\n"
                    "vmla.f32	q8, q2, d2[0]\n"
                    "vmla.f32	q9, q2, d2[1]\n"
                    "vld1.32	{d4-d5}, [%[b_ptr] :128]!\n"

                    "vmla.f32	q10, q3, d0[0]\n"
                    "vmla.f32	q11, q3, d0[1]\n"
                    "vmla.f32	q12, q3, d1[0]\n"
                    "vmla.f32	q13, q3, d1[1]\n"
                    "vld1.32	{d0-d1}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q14, q3, d2[0]\n"
                    "vmla.f32	q15, q3, d2[1]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"

                    // Unroll 1
                    "vmla.f32	q4, q2, d3[0]\n"
                    "subs		%[k], %[k], #1\n"
                    "vmla.f32	q5, q2, d3[1]\n"
                    "pld [%[a_ptr], #208] \n"
                    "vmla.f32	q6, q2, d0[0]\n"
                    "vmla.f32	q7, q2, d0[1]\n"
                    "pld [%[b_ptr], #192]\n"
                    "vmla.f32	q8, q2, d1[0]\n"
                    "vmla.f32	q9, q2, d1[1]\n"
                    "vld1.32	{d4-d5}, [%[b_ptr] :128]!\n"

                    "vmla.f32	q10, q3, d3[0]\n"
                    "vmla.f32	q11, q3, d3[1]\n"
                    "vld1.32	{d2-d3}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q12, q3, d0[0]\n"
                    "vmla.f32	q13, q3, d0[1]\n"
                    "vmla.f32	q14, q3, d1[0]\n"
                    "vmla.f32	q15, q3, d1[1]\n"
                    "vld1.32	{d0-d1}, [%[a_ptr] :64]!\n"

                    // Unroll 2
                    "vmla.f32	q4, q2, d2[0]\n"
                    "vmla.f32	q5, q2, d2[1]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"
                    "vmla.f32	q6, q2, d3[0]\n"
                    "vmla.f32	q7, q2, d3[1]\n"
                    "pld [%[a_ptr], #240]\n"
                    "vmla.f32	q8, q2, d0[0]\n"
                    "vmla.f32	q9, q2, d0[1]\n"
                    "vld1.32	{d4-d5}, [%[b_ptr] :128]!\n"

                    "vmla.f32	q10, q3, d2[0]\n"
                    "vmla.f32	q11, q3, d2[1]\n"
                    "pld [%[b_ptr], #208]     \n"
                    "vmla.f32	q12, q3, d3[0]\n"
                    "vmla.f32	q13, q3, d3[1]\n"
                    "vld1.32	{d2-d3}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q14, q3, d0[0]\n"
                    "vmla.f32	q15, q3, d0[1]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"

                    // Unroll 3
                    "vmla.f32	q4, q2, d1[0]\n"
                    "vmla.f32	q5, q2, d1[1]\n"
                    "vmla.f32	q6, q2, d2[0]\n"
                    "vmla.f32	q7, q2, d2[1]\n"
                    "vmla.f32	q8, q2, d3[0]\n"
                    "vmla.f32	q9, q2, d3[1]\n"
                    "vld1.32	{d4-d5}, [%[b_ptr] :128]!\n"

                    "vmla.f32	q10, q3, d1[0]\n"
                    "vmla.f32	q11, q3, d1[1]\n"
                    "vld1.32	{d0-d1}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q12, q3, d2[0]\n"
                    "vmla.f32	q13, q3, d2[1]\n"
                    "vmla.f32	q14, q3, d3[0]\n"
                    "vmla.f32	q15, q3, d3[1]\n"
                    "bne		1b\n"

                    // "Tails" shows how many multiply blocks are needed at the
                    // end, must be 1-4 inclusive.  Bail out to alternative tail
                    // immediately if it's 1.
                    "0:                  \n"
                    "subs		%[tails], %[tails], #1\n"
                    "beq		3f\n"

                    // Detached final iteration
                    // Unroll 0
                    "vmla.f32	q4, q2, d0[0]\n"
                    "vld1.32	{d2-d3}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q5, q2, d0[1]\n"
                    "vmla.f32	q6, q2, d1[0]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"
                    "vmla.f32	q7, q2, d1[1]\n"
                    "vmla.f32	q8, q2, d2[0]\n"
                    "subs		%[tails], %[tails], #1\n"
                    "vmla.f32	q9, q2, d2[1]\n"
                    "vld1.32	{d4-d5}, [%[b_ptr] :128]!\n"

                    "vmla.f32	q10, q3, d0[0]\n"
                    "vmla.f32	q11, q3, d0[1]\n"
                    "vmla.f32	q12, q3, d1[0]\n"
                    "vmla.f32	q13, q3, d1[1]\n"
                    "vld1.32	{d0-d1}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q14, q3, d2[0]\n"
                    "vmla.f32	q15, q3, d2[1]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"
                    "beq		4f\n"

                    // Unroll 1
                    "vmla.f32	q4, q2, d3[0]\n"
                    "vmla.f32	q5, q2, d3[1]\n"
                    "subs		%[tails], %[tails], #1\n"
                    "vmla.f32	q6, q2, d0[0]\n"
                    "vmla.f32	q7, q2, d0[1]\n"
                    "vmla.f32	q8, q2, d1[0]\n"
                    "vmla.f32	q9, q2, d1[1]\n"
                    "vld1.32	{d4-d5}, [%[b_ptr] :128]!\n"

                    "vmla.f32	q10, q3, d3[0]\n"
                    "vmla.f32	q11, q3, d3[1]\n"
                    "vld1.32	{d2-d3}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q12, q3, d0[0]\n"
                    "vmla.f32	q13, q3, d0[1]\n"
                    "vmla.f32	q14, q3, d1[0]\n"
                    "vmla.f32	q15, q3, d1[1]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"
                    "beq		5f\n"

                    // Unroll 2
                    "vld1.32	{d0-d1}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q4, q2, d2[0]\n"
                    "vmla.f32	q5, q2, d2[1]\n"
                    "vmla.f32	q6, q2, d3[0]\n"
                    "vmla.f32	q7, q2, d3[1]\n"
                    "vmla.f32	q8, q2, d0[0]\n"
                    "vmla.f32	q9, q2, d0[1]\n"
                    "vld1.32	{d4-d5}, [%[b_ptr] :128]!\n"

                    "vmla.f32	q10, q3, d2[0]\n"
                    "vmla.f32	q11, q3, d2[1]\n"
                    "vmla.f32	q12, q3, d3[0]\n"
                    "vmla.f32	q13, q3, d3[1]\n"
                    "vld1.32	{d2-d3}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q14, q3, d0[0]\n"
                    "vmla.f32	q15, q3, d0[1]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"

                    // Unroll 3
                    "vmla.f32	q4, q2, d1[0]\n"
                    "vmla.f32	q10, q3, d1[0]\n"
                    "vst1.32	{d8-d9}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q5, q2, d1[1]\n"
                    "vst1.32	{d20-d21}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q11, q3, d1[1]\n"
                    "vst1.32	{d10-d11}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q6, q2, d2[0]\n"
                    "vst1.32	{d22-d23}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q12, q3, d2[0]\n"
                    "vst1.32	{d12-d13}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q7, q2, d2[1]\n"
                    "vst1.32	{d24-d25}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q13, q3, d2[1]\n"
                    "vst1.32	{d14-d15}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q8, q2, d3[0]\n"
                    "vst1.32	{d26-d27}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q14, q3, d3[0]\n"
                    "vst1.32	{d16-d17}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q9, q2, d3[1]\n"
                    "vst1.32	{d28-d29}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q15, q3, d3[1]\n"
                    "vst1.32	{d18-d19}, [%[c_ptr] :128]!\n"
                    "b		2f\n"

                    // tails==1 final tail
                    "3:\n"
                    "vmla.f32	q4, q2, d0[0]\n"
                    "vld1.32	{d2}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q5, q2, d0[1]\n"
                    "vld1.32	{d6-d7}, [%[b_ptr] :128]!\n"
                    "vmla.f32	q6, q2, d1[0]\n"
                    "vst1.32	{d8-d9}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q10, q3, d0[0]\n"
                    "vst1.32	{d20-d21}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q11, q3, d0[1]\n"
                    "vst1.32	{d10-d11}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q12, q3, d1[0]\n"
                    "vst1.32	{d22-d23}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q7, q2, d1[1]\n"
                    "vst1.32	{d12-d13}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q13, q3, d1[1]\n"
                    "vst1.32	{d24-d25}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q8, q2, d2[0]\n"
                    "vst1.32	{d14-d15}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q14, q3, d2[0]\n"
                    "vst1.32	{d26-d27}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q9, q2, d2[1]\n"
                    "vst1.32	{d16-d17}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q15, q3, d2[1]\n"
                    "vst1.32	{d28-d29}, [%[c_ptr] :128]!\n"
                    "vst1.32	{d18-d19}, [%[c_ptr] :128]!\n"
                    "b		2f\n"

                    // tails==2 final tail
                    "4:\n"
                    "vmla.f32	q4, q2, d3[0]\n"
                    "vmla.f32	q10, q3, d3[0]\n"
                    "vst1.32	{d8-d9}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q5, q2, d3[1]\n"
                    "vst1.32	{d20-d21}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q11, q3, d3[1]\n"
                    "vst1.32	{d10-d11}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q6, q2, d0[0]\n"
                    "vst1.32	{d22-d23}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q12, q3, d0[0]\n"
                    "vst1.32	{d12-d13}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q7, q2, d0[1]\n"
                    "vst1.32	{d24-d25}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q13, q3, d0[1]\n"
                    "vst1.32	{d14-d15}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q8, q2, d1[0]\n"
                    "vst1.32	{d26-d27}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q14, q3, d1[0]\n"
                    "vst1.32	{d16-d17}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q9, q2, d1[1]\n"
                    "vst1.32	{d28-d29}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q15, q3, d1[1]\n"
                    "vst1.32	{d18-d19}, [%[c_ptr] :128]!\n"
                    "b		2f\n"

                    // tails==3 final tail
                    "5:\n"
                    "vmla.f32	q4, q2, d2[0]\n"
                    "vld1.32	{d0}, [%[a_ptr] :64]!\n"
                    "vmla.f32	q5, q2, d2[1]\n"
                    "vmla.f32	q6, q2, d3[0]\n"
                    "vst1.32	{d8-d9}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q10, q3, d2[0]\n"
                    "vst1.32	{d20-d21}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q11, q3, d2[1]\n"
                    "vst1.32	{d10-d11}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q12, q3, d3[0]\n"
                    "vst1.32	{d22-d23}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q7, q2, d3[1]\n"
                    "vst1.32	{d12-d13}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q13, q3, d3[1]\n"
                    "vst1.32	{d24-d25}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q8, q2, d0[0]\n"
                    "vst1.32	{d14-d15}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q14, q3, d0[0]\n"
                    "vst1.32	{d26-d27}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q9, q2, d0[1]\n"
                    "vst1.32	{d16-d17}, [%[c_ptr] :128]!\n"
                    "vmla.f32	q15, q3, d0[1]\n"
                    "vst1.32	{d28-d29}, [%[c_ptr] :128]!\n"
                    "vst1.32	{d18-d19}, [%[c_ptr] :128]!\n"
                    "2:\n"
                    "vst1.32	{d30-d31}, [%[c_ptr] :128]!\n"
            : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr), [k] "+r" (k), [tails] "+r" (tails)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc"
            );
        }
    }
}

#endif //__aarch64__

/**
 * \brief input data is not transpose
 * for arm-v7a, transform data to block x k x 6 layout
 * for arm-v8a, transform data to block x k x 8 layout
 */
void load_apanel_no_trans(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {

    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);

#ifdef __aarch64__
    uint32_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! data A is not transposed, transpose A to k * 8
#pragma omp parallel for
    for (int y = m0; y < mmax; y += 8) {
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

        int x = kmax - k0;

        for (; x > 7; x -= 8) {
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
#else //__aarch64__

    uint32_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! data A is not transposed, transpose A to k * 6
    for (int y = m0; y < mmax; y += 6) {
        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;

        int x = kmax - k0;

        for (; x > 7; x -= 8) {
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
            //! zip load 8 elements (2 neon Q registers) from each of 6 rows
            asm volatile (
#if 0
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

                    //"pld      [%[inptr0], #128]       @ preload r0 data to cache, fill pipeline\n"
                    "vst1.32  {d1},     [%[outptr]]!  @ write d1(q0,high),r01,r11\n"
                    "vst1.32  {d9},     [%[outptr]]!  @ write d9(q4,high),r21,r31\n"
                    "vst1.32  {d17},    [%[outptr]]!  @ write d17(q8,high),r41,r51\n"

                    "vtrn.32  q1, q3                  @ trans data: q1=r02,r12,r03,r13; q3=r06,r16,r07,r17\n"
                    "vst1.32  {d2},     [%[outptr]]!  @ write d2(q1,low),r02,r12\n"
                    "vtrn.32  q5, q7                  @ trans data: q5=r22,r32,r23,r33; q7=r26,r36,r27,r37\n"
                    "vst1.32  {d10},    [%[outptr]]!  @ write d10(q5,low),r22,r32\n"
                    "vtrn.32  q9, q11                 @ trans data: q9=r42,r52,r43,r53; q11=r46,r56,r47,r57\n"
                    "vst1.32  {d18},    [%[outptr]]!  @ write d18(q9,low),r42,r52\n"

                    //"pld      [%[inptr1], #128]       @ preload r1 data to cache, fill pipeline\n"
                    "vst1.32  {d3},     [%[outptr]]!  @ write d3(q1,high),r03,r13\n"
                    "vst1.32  {d11},    [%[outptr]]!  @ write d11(q5,high),r23,r33\n"
                    "vst1.32  {d19},    [%[outptr]]!  @ write d19(q9,high),r43,r53\n"

                    //"pld      [%[inptr2], #128]       @ preload r2 data to cache, fill pipeline\n"
                    "vst1.32  {d4},     [%[outptr]]!  @ write d4(q2,low),r04,r14\n"
                    "vst1.32  {d12},    [%[outptr]]!  @ write d12(q6,low),r24,r34\n"
                    "vst1.32  {d20},    [%[outptr]]!  @ write d20(q10,low),r44,r54\n"

                    //"pld      [%[inptr3], #128]       @ preload r3 data to cache, fill pipeline\n"
                    "vst1.32  {d5},     [%[outptr]]!  @ write d5(q2,high),r05,r15\n"
                    "vst1.32  {d13},    [%[outptr]]!  @ write d13(q6,high),r25,r35\n"
                    "vst1.32  {d21},    [%[outptr]]!  @ write d21(q10,high),r45,r55\n"

                    //"pld      [%[inptr4], #128]       @ preload r4 data to cache, fill pipeline\n"
                    "vst1.32  {d6},     [%[outptr]]!  @ write d6(q3,low),r06,r16\n"
                    "vst1.32  {d14},    [%[outptr]]!  @ write d14(q7,low),r26,r36\n"
                    "vst1.32  {d22},    [%[outptr]]!  @ write d22(q11,low),r46,r56\n"

                    //"pld      [%[inptr5], #128]       @ preload r5 data to cache, fill pipeline\n"
                    "vst1.32  {d7},     [%[outptr]]!  @ write d7(q3,high),r07,r17\n"
                    "vst1.32  {d15},    [%[outptr]]!  @ write d15(q7,high),r27,r37\n"
                    "vst1.32  {d23},    [%[outptr]]!  @ write d23(q11,high),r47,r57\n"
#else
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
#endif
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
#endif //__aarch64__
}

/**
* \brief input data is transpose
* for arm-v7a, transform data to block x k x 8 layout
* for arm-v8a, transform data to block x k x 12 layout
*/
void load_bpanel_trans(float* out, const float* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {

    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + n0;

#ifdef __aarch64__
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
#if 0
        const uint32_t *ptr4 = ptr2 + ldin;
        const uint32_t *ptr5 = ptr2 + ldin;
        const uint32_t *ptr6 = ptr2 + ldin;
        const uint32_t *ptr7 = ptr2 + ldin;
#endif

        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr0], #64]   \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr1], #64]   \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr2], #64]   \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
                "prfm   pldl1keep, [%[ptr3], #64]   \n"
#if 0
                "prfm   pldl1keep, [%[ptr4]]        \n"
                "prfm   pldl1keep, [%[ptr4], #64]   \n"
                "prfm   pldl1keep, [%[ptr5]]        \n"
                "prfm   pldl1keep, [%[ptr5], #64]   \n"
                "prfm   pldl1keep, [%[ptr6]]        \n"
                "prfm   pldl1keep, [%[ptr6], #64]   \n"
                "prfm   pldl1keep, [%[ptr7]]        \n"
                "prfm   pldl1keep, [%[ptr7], #64]   \n"
#endif
        :
        :[ptr0] "r"(ptr0),[ptr1] "r"(ptr1),[ptr2] "r"(ptr2),[ptr3] "r"(ptr3)
#if 0
          , [ptr4] "r"(ptr4),[ptr5] "r"(ptr5),[ptr6] "r"(ptr6),[ptr7] "r"(ptr7)
#endif
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

#if 0
            vr00 = vld1q_u32(ptr4);
            vr01 = vld1q_u32(ptr4 + 4);
            vr02 = vld1q_u32(ptr4 + 8);

            vr10 = vld1q_u32(ptr5);
            vr11 = vld1q_u32(ptr5 + 4);
            vr12 = vld1q_u32(ptr5 + 8);

            vst1q_u32(outptr_row_col + 48, vr00);
            vst1q_u32(outptr_row_col + 52, vr01);
            vst1q_u32(outptr_row_col + 56, vr02);

            vr20 = vld1q_u32(ptr6);
            vr21 = vld1q_u32(ptr6 + 4);
            vr22 = vld1q_u32(ptr6 + 8);

            vst1q_u32(outptr_row_col + 60, vr10);
            vst1q_u32(outptr_row_col + 64, vr11);
            vst1q_u32(outptr_row_col + 68, vr12);

            vr30 = vld1q_u32(ptr7);
            vr31 = vld1q_u32(ptr7 + 4);
            vr32 = vld1q_u32(ptr7 + 8);

            vst1q_u32(outptr_row_col + 72, vr20);
            vst1q_u32(outptr_row_col + 76, vr21);
            vst1q_u32(outptr_row_col + 80, vr22);

            vst1q_u32(outptr_row_col + 84, vr30);
            vst1q_u32(outptr_row_col + 88, vr31);
            vst1q_u32(outptr_row_col + 92, vr32);
#endif
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
            vst1q_u32(outptr_row_col, vr00_1);
            vst1q_u32(outptr_row_col + 4, vr01_1);
            vst1q_u32(outptr_row_col + 8, vr02_1);

            uint32x4_t vr20 = vld1q_u32(ptr2);
            uint32x4_t vr21 = vld1q_u32(ptr2 + 4);
            uint32x4_t vr22 = vld1q_u32(ptr2 + 8);

            uint32x4_t vr10_1 = vbslq_u32(vmask1, vr10, vzero);
            uint32x4_t vr11_1 = vbslq_u32(vmask2, vr11, vzero);
            uint32x4_t vr12_1 = vbslq_u32(vmask3, vr12, vzero);
            vst1q_u32(outptr_row_col + 12, vr10_1);
            vst1q_u32(outptr_row_col + 16, vr11_1);
            vst1q_u32(outptr_row_col + 20, vr12_1);

            uint32x4_t vr30 = vld1q_u32(ptr3);
            uint32x4_t vr31 = vld1q_u32(ptr3 + 4);
            uint32x4_t vr32 = vld1q_u32(ptr3 + 8);

            uint32x4_t vr20_1 = vbslq_u32(vmask1, vr20, vzero);
            uint32x4_t vr21_1 = vbslq_u32(vmask2, vr21, vzero);
            uint32x4_t vr22_1 = vbslq_u32(vmask3, vr22, vzero);
            vst1q_u32(outptr_row_col + 24, vr20_1);
            vst1q_u32(outptr_row_col + 28, vr21_1);
            vst1q_u32(outptr_row_col + 32, vr22_1);

            uint32x4_t vr30_1 = vbslq_u32(vmask1, vr30, vzero);
            uint32x4_t vr31_1 = vbslq_u32(vmask2, vr31, vzero);
            uint32x4_t vr32_1 = vbslq_u32(vmask3, vr32, vzero);
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

#else

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
#if 0
        const uint32_t* ptr4 = ptr3 + ldin;
        const uint32_t* ptr5 = ptr4 + ldin;
        const uint32_t* ptr6 = ptr5 + ldin;
        const uint32_t* ptr7 = ptr6 + ldin;
        const uint32_t* ptr8 = ptr7 + ldin;
        const uint32_t* ptr9 = ptr8 + ldin;
#endif
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
#if 0
            "vld1.32 {d0-d3}, [%[ptr4]]!        @ load r4, 8 elements\n"
            "vld1.32 {d4-d7}, [%[ptr5]]!        @ load r5, 8 elements\n"
            "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"

            "vld1.32 {d0-d3}, [%[ptr6]]!        @ load r6, 8 elements\n"
            "vld1.32 {d4-d7}, [%[ptr7]]!        @ load r7, 8 elements\n"
            "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"

            "vld1.32 {d0-d3}, [%[ptr8]]!        @ load r8, 8 elements\n"
            "vld1.32 {d4-d7}, [%[ptr9]]!        @ load r9, 8 elements\n"
            "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"
#endif
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
#if 0
            , [ptr4] "+r" (ptr4), [ptr5] "+r" (ptr5), \
                [ptr6] "+r" (ptr6), [ptr7] "+r" (ptr7), [ptr8] "+r" (ptr8), \
                [ptr9] "+r" (ptr9)
#endif
            :
            : "q0", "q1", "q2", "q3", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_pad > 0) {
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
#if 0
            "vld1.32 {d0-d3}, [%[ptr4]]!        @ load r4, 8 elements\n"
            "vld1.32 {d4-d7}, [%[ptr5]]!        @ load r5, 8 elements\n"
            "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
            "vbif   q1, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
            "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
            "vbif   q3, %q[vzero], %q[vmask2]   @ bit select, pad zero\n"
            "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"

            "vld1.32 {d0-d3}, [%[ptr6]]!        @ load r6, 8 elements\n"
            "vld1.32 {d4-d7}, [%[ptr7]]!        @ load r7, 8 elements\n"
            "vbif   q0, %q[vzero], %q[vmask1]       @ bit select, pad zero\n"
            "vbif   q1, %q[vzero], %q[vmask2]       @ bit select, pad zero\n"
            "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            "vbif   q2, %q[vzero], %q[vmask1]       @ bit select, pad zero\n"
            "vbif   q3, %q[vzero], %q[vmask2]       @ bit select, pad zero\n"
            "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"

            "vld1.32 {d0-d3}, [%[ptr8]]!        @ load r8, 8 elements\n"
            "vld1.32 {d4-d7}, [%[ptr9]]!        @ load r9, 8 elements\n"
            "vbif   q0, %q[vzero], %q[vmask1]       @ bit select, pad zero\n"
            "vbif   q1, %q[vzero], %q[vmask2]       @ bit select, pad zero\n"
            "vst1.32 {d0-d3}, [%[outptr]]!      @ write to output ptr\n"
            "vbif   q2, %q[vzero], %q[vmask1]       @ bit select, pad zero\n"
            "vbif   q3, %q[vzero], %q[vmask2]       @ bit select, pad zero\n"
            "vst1.32 {d4-d7}, [%[outptr]]!      @ write to output ptr\n"
#endif
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
#if 0
            , [ptr4] "+r" (ptr4), [ptr5] "+r" (ptr5), \
                [ptr6] "+r" (ptr6), [ptr7] "+r" (ptr7), [ptr8] "+r" (ptr8), \
                [ptr9] "+r" (ptr9)
#endif
            : [vmask1] "w" (vmask1), [vmask2] "w" (vmask2), \
              [vzero] "w" (vzero)
            : "q0", "q1", "q2", "q3", "memory"
            );
        }
        //outptr_row += 32;
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
        if (right_pad > 0) {
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
        //outptr_row += 8;
    }
#endif //__aarch64__
}

void load_apanel_trans(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {
    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + m0;

#ifdef __aarch64__
    //todo
#else

    uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int x_len = mmax - m0;
    int y_len = kmax - k0;
    int right_remain = x_len - 6 * (x_len / 6);
    int right_pad = 6 - right_remain;

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
#endif //__aarch64__
}

void load_bpanel_no_trans(float* out, const float* in, const int ldin, const int k0, \
    const int kmax, const int n0, const int nmax) {
    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);
#ifdef __aarch64__
    // todo
#else
    uint32_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
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

        int x = kmax - k0;

        for (; x > 7; x -= 8) {
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
#endif //__aarch64__
}

void merge_float_basic(float *out, const float *in, const int ldout, const int y0, \
    const int ymax, const int x0, const int xmax, const float alpha, const float beta) {
    const float *inptr = in;

    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bv = vdupq_n_f32(beta);

#ifdef __aarch64__
    for (int y = y0; y < ymax; y += 8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;
        float *outptr6 = outptr5 + ldout;
        float *outptr7 = outptr6 + ldout;

        for (int i = x0; i < xmax; i += 12) {
            float dummyres[12];

            /* Make sure we throw away results if Y isn't a multiple of 8.
             * We do this by pointing the result pointer at a dummy buffer
             * we later discard.  */
            if ((y+7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        outptr1 = dummyres;
                    case 5:
                        outptr2 = dummyres;
                    case 4:
                        outptr3 = dummyres;
                    case 3:
                        outptr4 = dummyres;
                    case 2:
                        outptr5 = dummyres;
                    case 1:
                        outptr6 = dummyres;
                    case 0:
                        outptr7 = dummyres;
                    default:
                        break;
                }
            }

            /* For ragged X, manually copy over the valid results. */
            if ((i + 11) >= xmax) {
                for (int xi = 0; xi < 12; xi++) {
                    if ((i + xi) < xmax) {
                        *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                        outptr0++;
                        *outptr1 = (alpha * inptr[xi + 12]) + (*outptr1 * beta);
                        outptr1++;
                        *outptr2 = (alpha * inptr[xi + 24]) + (*outptr2 * beta);
                        outptr2++;
                        *outptr3 = (alpha * inptr[xi + 36]) + (*outptr3 * beta);
                        outptr3++;
                        *outptr4 = (alpha * inptr[xi + 48]) + (*outptr4 * beta);
                        outptr4++;
                        *outptr5 = (alpha * inptr[xi + 60]) + (*outptr5 * beta);
                        outptr5++;
                        *outptr6 = (alpha * inptr[xi + 72]) + (*outptr6 * beta);
                        outptr6++;
                        *outptr7 = (alpha * inptr[xi + 84]) + (*outptr7 * beta);
                        outptr7++;
                    }
                }
                inptr += 96;
            } else {
                /* Optimized routine to copy an entire block */
                asm volatile (
                // Rows 0-1
                "LDP	q16, q17, [%[outptr0]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr0], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr1]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr1], #32]\n"
                        "prfm   pldl1keep, [%[inptr], #768]\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[inptr], #832]\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr0]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q18, [%[outptr0]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[inptr], #896]\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr1]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr1]], #16\n"

                        // Rows 2-3
                        "LDP	q16, q17, [%[outptr2]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr2], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr3]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr3], #32]\n"
                        "prfm   pldl1keep, [%[inptr], #960]\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[inptr], #1024]\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr2]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q18, [%[outptr2]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[inptr], #1088]\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr3]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr3]], #16\n"

                        // Rows 4-5
                        "prfm   pldl1keep, [%[outptr0], #80]\n"
                        "LDP	q16, q17, [%[outptr4]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr4], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr5]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr5], #32]\n"
                        "prfm   pldl1keep, [%[outptr1], #80]\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[outptr2], #80]\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr4]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q18, [%[outptr4]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[outptr3], #80]\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr5]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr5]], #16\n"

                        // Rows 6-7
                        "prfm   pldl1keep, [%[outptr4], #80]\n"
                        "LDP	q16, q17, [%[outptr6]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr6], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr7]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr7], #32]\n"
                        "prfm   pldl1keep, [%[outptr5], #80]\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[outptr6], #128]\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr6]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q18, [%[outptr6]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[outptr7], #128]\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr7]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr7]], #16\n"
                        "ADD	%[inptr], %[inptr], #384\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), \
                    [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), \
                    [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), \
                    [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7), \
                    [inptr] "+r" (inptr)
                : [av] "w" (av), [bv] "w" (bv)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q16", "q17", \
                    "q18", "q19", "q20", "q21"
                );
            }
        }
    }
#else
    for (int y = y0; y < ymax; y += 8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;

        for (int i=x0; i<xmax; i+=8) {
            float dummyres[8] {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        outptr1 = dummyres;
                    case 3:
                        outptr2 = dummyres;
                    case 2:
                        outptr3 = dummyres;
                    case 1:
                        outptr4 = dummyres;
                    case 0:
                        outptr5 = dummyres;
                    default:
                        break;
                }
            }
            if ((i + 7) >= xmax) {
                for (int xi = 0; xi < 8; xi++) {
                    if ((i+xi) < xmax) {
                        *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                        outptr0++;
                        *outptr1 = (alpha * inptr[xi + 8]) + (*outptr1 * beta);
                        outptr1++;
                        *outptr2 = (alpha * inptr[xi + 16]) + (*outptr2 * beta);
                        outptr2++;
                        *outptr3 = (alpha * inptr[xi + 24]) + (*outptr3 * beta);
                        outptr3++;
                        *outptr4 = (alpha * inptr[xi + 32]) + (*outptr4 * beta);
                        outptr4++;
                        *outptr5 = (alpha * inptr[xi + 40]) + (*outptr5 * beta);
                        outptr5++;
                    }
                }
                inptr += 48;
            } else {
                asm volatile (
                //! Rows 0-1
                "VLD1.32	{d8-d11},  [%[outptr0]]\n"
                        "VMUL.f32	q4, q4, %q[bv]\n"
                        "VLD1.32	{d12-d15}, [%[outptr1]]\n"
                        "VMUL.f32	q5, q5, %q[bv]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VMUL.f32	q6, q6, %q[bv]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VMUL.f32	q7, q7, %q[bv]\n"

                        "VMLA.f32	q4, q0, %q[av]\n"
                        "pld    [%[inptr], #352]\n"
                        "VMLA.f32	q5, q1, %q[av]\n"
                        "VST1.32	{d8-d11}, [%[outptr0]]!\n"
                        "pld    [%[inptr], #416]\n"
                        "VMLA.f32	q6, q2, %q[av]\n"
                        "pld    [%[inptr], #480]\n"
                        "VMLA.f32	q7, q3, %q[av]\n"
                        "VST1.32	{d12-d15}, [%[outptr1]]!\n"

                        // Rows 2-3
                        "VLD1.32	{d8-d11},  [%[outptr2]]\n"
                        "VMUL.f32	q4, q4, %q[bv]\n"
                        "VLD1.32	{d12-d15}, [%[outptr3]]\n"
                        "VMUL.f32	q5, q5, %q[bv]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VMUL.f32	q6, q6, %q[bv]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VMUL.f32	q7, q7, %q[bv]\n"

                        "VMLA.f32	q4, q0, %q[av]\n"
                        "pld    [%[outptr0], #96]\n"
                        "VMLA.f32	q5, q1, %q[av]\n"
                        "VST1.32	{d8-d11}, [%[outptr2]]!\n"
                        "pld    [%[outptr1], #96]\n"
                        "VMLA.f32	q6, q2, %q[av]\n"
                        "pld    [%[outptr2], #96]\n"
                        "VMLA.f32	q7, q3, %q[av]\n"
                        "VST1.32	{d12-d15}, [%[outptr3]]!\n"

                        // Rows 4-5
                        "VLD1.32	{d8-d11},  [%[outptr4]]\n"
                        "VMUL.f32	q4, q4, %q[bv]\n"
                        "VLD1.32	{d12-d15}, [%[outptr5]]\n"
                        "VMUL.f32	q5, q5, %q[bv]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VMUL.f32	q6, q6, %q[bv]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VMUL.f32	q7, q7, %q[bv]\n"

                        "VMLA.f32	q4, q0, %q[av]\n"
                        "pld    [%[outptr3], #96]\n"
                        "VMLA.f32	q5, q1, %q[av]\n"
                        "VST1.32	{d8-d11}, [%[outptr4]]!\n"
                        "pld    [%[outptr4], #96]\n"
                        "VMLA.f32	q6, q2, %q[av]\n"
                        "pld    [%[outptr5], #128]\n"
                        "VMLA.f32	q7, q3, %q[av]\n"
                        "VST1.32	{d12-d15}, [%[outptr5]]!\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), \
                    [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), \
                    [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), \
                    [inptr] "+r" (inptr)
                : [av] "w" (av), [bv] "w" (bv)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                );
            }
        }
    }
#endif // end of __aarch64__
}

void merge_float_basic_relu(float *out, const float *in, const int ldout, const int y0, \
    const int ymax, const int x0, const int xmax, const float alpha, const float beta) {
    const float *inptr = in;

    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bv = vdupq_n_f32(beta);

    float32x4_t vzero = vdupq_n_f32(0.f);

#ifdef __aarch64__
    for (int y = y0; y < ymax; y += 8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;
        float *outptr6 = outptr5 + ldout;
        float *outptr7 = outptr6 + ldout;

        for (int i = x0; i < xmax; i += 12) {
            float dummyres[12];

            /* Make sure we throw away results if Y isn't a multiple of 8.
             * We do this by pointing the result pointer at a dummy buffer
             * we later discard.  */
            if ((y+7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        outptr1 = dummyres;
                    case 5:
                        outptr2 = dummyres;
                    case 4:
                        outptr3 = dummyres;
                    case 3:
                        outptr4 = dummyres;
                    case 2:
                        outptr5 = dummyres;
                    case 1:
                        outptr6 = dummyres;
                    case 0:
                        outptr7 = dummyres;
                    default:
                        break;
                }
            }

            /* For ragged X, manually copy over the valid results. */
            if ((i + 11) >= xmax) {
                for (int xi = 0; xi < 12; xi++) {
                    if ((i + xi) < xmax) {

                        outptr0[0] = alpha * inptr[xi] + beta * outptr0[0];
                        outptr0[0] = fmaxf(outptr0[0], 0.f);
                        outptr0++;

                        outptr1[0] = alpha * inptr[xi + 12] + beta * outptr1[0];
                        outptr1[0] = fmaxf(outptr1[0], 0.f);
                        outptr1++;

                        outptr2[0] = alpha * inptr[xi + 24] + beta * outptr2[0];
                        outptr2[0] = fmaxf(outptr2[0], 0.f);
                        outptr2++;

                        outptr3[0] = alpha * inptr[xi + 36] + beta * outptr3[0];
                        outptr3[0] = fmaxf(outptr3[0], 0.f);
                        outptr3++;

                        outptr4[0] = alpha * inptr[xi + 48] + beta * outptr4[0];
                        outptr4[0] = fmaxf(outptr4[0], 0.f);
                        outptr4++;

                        outptr5[0] = alpha * inptr[xi + 60] + beta * outptr5[0];
                        outptr5[0] = fmaxf(outptr5[0], 0.f);
                        outptr5++;

                        outptr6[0] = alpha * inptr[xi + 72] + beta * outptr6[0];
                        outptr6[0] = fmaxf(outptr6[0], 0.f);
                        outptr6++;

                        outptr7[0] = alpha * inptr[xi + 84] + beta * outptr7[0];
                        outptr7[0] = fmaxf(outptr7[0], 0.f);
                        outptr7++;
                    }
                }
                inptr += 96;
            } else {
                /* Optimized routine to copy an entire block */
                asm volatile (
                // Rows 0-1
                "LDP	q16, q17, [%[outptr0]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr0], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr1]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr1], #32]\n"
                        "prfm   pldl1keep, [%[inptr], #768]\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[inptr], #832]\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "FMAX   v16.4s, v16.4s, %[vzero].4s\n"
                        "FMAX   v17.4s, v17.4s, %[vzero].4s\n"
                        "STP	q16, q17, [%[outptr0]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "FMAX   v18.4s, v18.4s, %[vzero].4s\n"
                        "STR	q18, [%[outptr0]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[inptr], #896]\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "FMAX   v19.4s, v19.4s, %[vzero].4s\n"
                        "FMAX   v20.4s, v20.4s, %[vzero].4s\n"
                        "STP	q19, q20, [%[outptr1]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "FMAX   v21.4s, v21.4s, %[vzero].4s\n"
                        "STR	q21, [%[outptr1]], #16\n"

                        // Rows 2-3
                        "LDP	q16, q17, [%[outptr2]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr2], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr3]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr3], #32]\n"
                        "prfm   pldl1keep, [%[inptr], #960]\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[inptr], #1024]\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "FMAX   v16.4s, v16.4s, %[vzero].4s\n"
                        "FMAX   v17.4s, v17.4s, %[vzero].4s\n"
                        "STP	q16, q17, [%[outptr2]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "FMAX   v18.4s, v18.4s, %[vzero].4s\n"
                        "STR	q18, [%[outptr2]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[inptr], #1088]\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "FMAX   v19.4s, v19.4s, %[vzero].4s\n"
                        "FMAX   v20.4s, v20.4s, %[vzero].4s\n"
                        "STP	q19, q20, [%[outptr3]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "FMAX   v21.4s, v21.4s, %[vzero].4s\n"
                        "STR	q21, [%[outptr3]], #16\n"

                        // Rows 4-5
                        "prfm   pldl1keep, [%[outptr0], #80]\n"
                        "LDP	q16, q17, [%[outptr4]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr4], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr5]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr5], #32]\n"
                        "prfm   pldl1keep, [%[outptr1], #80]\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[outptr2], #80]\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "FMAX   v16.4s, v16.4s, %[vzero].4s\n"
                        "FMAX   v17.4s, v17.4s, %[vzero].4s\n"
                        "STP	q16, q17, [%[outptr4]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "FMAX   v18.4s, v18.4s, %[vzero].4s\n"
                        "STR	q18, [%[outptr4]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[outptr3], #80]\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "FMAX   v19.4s, v19.4s, %[vzero].4s\n"
                        "FMAX   v20.4s, v20.4s, %[vzero].4s\n"
                        "STP	q19, q20, [%[outptr5]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "FMAX   v21.4s, v21.4s, %[vzero].4s\n"
                        "STR	q21, [%[outptr5]], #16\n"

                        // Rows 6-7
                        "prfm   pldl1keep, [%[outptr4], #80]\n"
                        "LDP	q16, q17, [%[outptr6]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr6], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr7]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr7], #32]\n"
                        "prfm   pldl1keep, [%[outptr5], #80]\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[outptr6], #128]\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "FMAX   v16.4s, v16.4s, %[vzero].4s\n"
                        "FMAX   v17.4s, v17.4s, %[vzero].4s\n"
                        "STP	q16, q17, [%[outptr6]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "FMAX   v18.4s, v18.4s, %[vzero].4s\n"
                        "STR	q18, [%[outptr6]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "prfm   pldl1keep, [%[outptr7], #128]\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "FMAX   v19.4s, v19.4s, %[vzero].4s\n"
                        "FMAX   v20.4s, v20.4s, %[vzero].4s\n"
                        "STP	q19, q20, [%[outptr7]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "FMAX   v21.4s, v21.4s, %[vzero].4s\n"
                        "STR	q21, [%[outptr7]], #16\n"
                        "ADD	%[inptr], %[inptr], #384\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), \
                    [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), \
                    [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), \
                    [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7), \
                    [inptr] "+r" (inptr)
                : [av] "w" (av), [bv] "w" (bv), [vzero] "w" (vzero)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q16", "q17", \
                    "q18", "q19", "q20", "q21"
                );
            }
        }
    }
#else
    for (int y = y0; y < ymax; y += 8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;

        for (int i=x0; i<xmax; i+=8) {
            float dummyres[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        outptr1 = dummyres;
                    case 3:
                        outptr2 = dummyres;
                    case 2:
                        outptr3 = dummyres;
                    case 1:
                        outptr4 = dummyres;
                    case 0:
                        outptr5 = dummyres;
                    default:
                        break;
                }
            }

            if ((i + 7) >= xmax) {
                for (int xi = 0; xi < 8; xi++) {
                    if ((i + xi) < xmax) {
                        outptr0[0] = alpha * inptr[xi] + beta * outptr0[0];
                        outptr0[0] = fmaxf(outptr0[0], 0.f);
                        outptr0++;

                        outptr1[0] = alpha * inptr[xi + 8] + beta * outptr1[0];
                        outptr1[0] = fmaxf(outptr1[0], 0.f);
                        outptr1++;

                        outptr2[0] = alpha * inptr[xi + 16] + beta * outptr2[0];
                        outptr2[0] = fmaxf(outptr2[0], 0.f);
                        outptr2++;

                        outptr3[0] = alpha * inptr[xi + 24] + beta * outptr3[0];
                        outptr3[0] = fmaxf(outptr3[0], 0.f);
                        outptr3++;

                        outptr4[0] = alpha * inptr[xi + 32] + beta * outptr4[0];
                        outptr4[0] = fmaxf(outptr4[0], 0.f);
                        outptr4++;

                        outptr5[0] = alpha * inptr[xi + 40] + beta * outptr5[0];
                        outptr5[0] = fmaxf(outptr5[0], 0.f);
                        outptr5++;
                    }
                }
                inptr += 48;
            } else {
                asm volatile (
                //! Rows 0-1
                "VLD1.32	{d8-d11},  [%[outptr0]]\n"
                        "VMUL.f32	q4, q4, %q[bv]\n"
                        "VLD1.32	{d12-d15}, [%[outptr1]]\n"
                        "VMUL.f32	q5, q5, %q[bv]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VMUL.f32	q6, q6, %q[bv]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VMUL.f32	q7, q7, %q[bv]\n"

                        "VMLA.f32	q4, q0, %q[av]\n"
                        "pld    [%[inptr], #352]\n"
                        "VMLA.f32	q5, q1, %q[av]\n"
                        "VMAX.f32   q4, q4, %q[vzero]\n"
                        "VMAX.f32   q5, q5, %q[vzero]\n"
                        "VST1.32	{d8-d11}, [%[outptr0]]!\n"
                        "pld    [%[inptr], #416]\n"
                        "VMLA.f32	q6, q2, %q[av]\n"
                        "pld    [%[inptr], #480]\n"
                        "VMLA.f32	q7, q3, %q[av]\n"
                        "VMAX.f32   q6, q6, %q[vzero]\n"
                        "VMAX.f32   q7, q7, %q[vzero]\n"
                        "VST1.32	{d12-d15}, [%[outptr1]]!\n"

                        // Rows 2-3
                        "VLD1.32	{d8-d11},  [%[outptr2]]\n"
                        "VMUL.f32	q4, q4, %q[bv]\n"
                        "VLD1.32	{d12-d15}, [%[outptr3]]\n"
                        "VMUL.f32	q5, q5, %q[bv]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VMUL.f32	q6, q6, %q[bv]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VMUL.f32	q7, q7, %q[bv]\n"

                        "VMLA.f32	q4, q0, %q[av]\n"
                        "pld    [%[outptr0], #96]\n"
                        "VMLA.f32	q5, q1, %q[av]\n"
                        "VMAX.f32   q4, q4, %q[vzero]\n"
                        "VMAX.f32   q5, q5, %q[vzero]\n"
                        "VST1.32	{d8-d11}, [%[outptr2]]!\n"
                        "pld    [%[outptr1], #96]\n"
                        "VMLA.f32	q6, q2, %q[av]\n"
                        "pld    [%[outptr2], #96]\n"
                        "VMLA.f32	q7, q3, %q[av]\n"
                        "VMAX.f32   q6, q6, %q[vzero]\n"
                        "VMAX.f32   q7, q7, %q[vzero]\n"
                        "VST1.32	{d12-d15}, [%[outptr3]]!\n"

                        // Rows 4-5
                        "VLD1.32	{d8-d11},  [%[outptr4]]\n"
                        "VMUL.f32	q4, q4, %q[bv]\n"
                        "VLD1.32	{d12-d15}, [%[outptr5]]\n"
                        "VMUL.f32	q5, q5, %q[bv]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VMUL.f32	q6, q6, %q[bv]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VMUL.f32	q7, q7, %q[bv]\n"

                        "VMLA.f32	q4, q0, %q[av]\n"
                        "pld    [%[outptr3], #96]\n"
                        "VMLA.f32	q5, q1, %q[av]\n"
                        "VMAX.f32   q4, q4, %q[vzero]\n"
                        "VMAX.f32   q5, q5, %q[vzero]\n"
                        "VST1.32	{d8-d11}, [%[outptr4]]!\n"
                        "pld    [%[outptr4], #96]\n"
                        "VMLA.f32	q6, q2, %q[av]\n"
                        "pld    [%[outptr5], #128]\n"
                        "VMLA.f32	q7, q3, %q[av]\n"
                        "VMAX.f32   q6, q6, %q[vzero]\n"
                        "VMAX.f32   q7, q7, %q[vzero]\n"
                        "VST1.32	{d12-d15}, [%[outptr5]]!\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), \
                    [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), \
                    [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), \
                    [inptr] "+r" (inptr)
                : [av] "w" (av), [bv] "w" (bv), [vzero] "w" (vzero)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                );
            }
        }
    }
#endif // end of __aarch64__
}

void merge_float_alpha1_beta1(float *out, const float *in, const int ldout, const int y0, \
    const int ymax, const int x0, const int xmax) {
    const float *inptr = in;

#ifdef __aarch64__

    for (int y = y0; y < ymax; y += 8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;
        float *outptr6 = outptr5 + ldout;
        float *outptr7 = outptr6 + ldout;

        for (int i = x0; i < xmax; i += 12) {
            float dummyres[12];

            /* Make sure we throw away results if Y isn't a multiple of 8.
             * We do this by pointing the result pointer at a dummy buffer
             * we later discard.  */
            if ((y+7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        outptr1 = dummyres;
                    case 5:
                        outptr2 = dummyres;
                    case 4:
                        outptr3 = dummyres;
                    case 3:
                        outptr4 = dummyres;
                    case 2:
                        outptr5 = dummyres;
                    case 1:
                        outptr6 = dummyres;
                    case 0:
                        outptr7 = dummyres;
                    default:
                        break;
                }
            }

            /* For ragged X, manually copy over the valid results. */
            if ((i + 11) >= xmax) {
                for (int xi = 0; xi < 12; xi++) {
                    if ((i + xi) < xmax) {
                        *outptr0 = inptr[xi] + *outptr0;
                        outptr0++;
                        *outptr1 = inptr[xi + 12] + *outptr1;
                        outptr1++;
                        *outptr2 = inptr[xi + 24] + *outptr2;
                        outptr2++;
                        *outptr3 = inptr[xi + 36] + *outptr3;
                        outptr3++;
                        *outptr4 = inptr[xi + 48] + *outptr4;
                        outptr4++;
                        *outptr5 = inptr[xi + 60] + *outptr5;
                        outptr5++;
                        *outptr6 = inptr[xi + 72] + *outptr6;
                        outptr6++;
                        *outptr7 = inptr[xi + 84] + *outptr7;
                        outptr7++;
                    }
                }
                inptr += 96;
            } else {
                /* Optimized routine to copy an entire block */
                asm volatile (
                        // Rows 0-1
                "LDP	q16, q17, [%[outptr0]]\n"
                        "LDR	q18, [%[outptr0], #32]\n"
                        "LDP	q19, q20, [%[outptr1]]\n"
                        "LDR	q21, [%[outptr1], #32]\n"
                        "prfm   pldl1keep, [%[inptr], #768]\n"
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FADD	v16.4s, v0.4s, v16.4s\n"
                        "prfm   pldl1keep, [%[inptr], #832]\n"
                        "FADD	v17.4s, v1.4s, v17.4s\n"
                        "STP	q16, q17, [%[outptr0]], #32\n"
                        "FADD	v18.4s, v2.4s, v18.4s\n"
                        "STR	q18, [%[outptr0]], #16\n"
                        "FADD	v19.4s, v3.4s, v19.4s\n"
                        "prfm   pldl1keep, [%[inptr], #896]\n"
                        "FADD	v20.4s, v4.4s, v20.4s\n"
                        "STP	q19, q20, [%[outptr1]], #32\n"
                        "FADD	v21.4s, v5.4s, v21.4s\n"
                        "STR	q21, [%[outptr1]], #16\n"

                        // Rows 2-3
                        "LDP	q16, q17, [%[outptr2]]\n"
                        "LDR	q18, [%[outptr2], #32]\n"
                        "LDP	q19, q20, [%[outptr3]]\n"
                        "LDR	q21, [%[outptr3], #32]\n"
                        "prfm   pldl1keep, [%[inptr], #960]\n"
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FADD	v16.4s, v0.4s, v16.4s\n"
                        "prfm   pldl1keep, [%[inptr], #1024]\n"
                        "FADD	v17.4s, v1.4s, v17.4s\n"
                        "STP	q16, q17, [%[outptr2]], #32\n"
                        "FADD	v18.4s, v2.4s, v18.4s\n"
                        "STR	q18, [%[outptr2]], #16\n"
                        "FADD	v19.4s, v3.4s, v19.4s\n"
                        "prfm   pldl1keep, [%[inptr], #1088]\n"
                        "FADD	v20.4s, v4.4s, v20.4s\n"
                        "STP	q19, q20, [%[outptr3]], #32\n"
                        "FADD	v21.4s, v5.4s, v21.4s\n"
                        "STR	q21, [%[outptr3]], #16\n"

                        // Rows 4-5
                        "prfm   pldl1keep, [%[outptr0], #80]\n"
                        "LDP	q16, q17, [%[outptr4]]\n"
                        "LDR	q18, [%[outptr4], #32]\n"
                        "LDP	q19, q20, [%[outptr5]]\n"
                        "LDR	q21, [%[outptr5], #32]\n"
                        "prfm   pldl1keep, [%[outptr1], #80]\n"
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FADD	v16.4s, v0.4s, v16.4s\n"
                        "prfm   pldl1keep, [%[outptr2], #80]\n"
                        "FADD	v17.4s, v1.4s, v17.4s\n"
                        "STP	q16, q17, [%[outptr4]], #32\n"
                        "FADD	v18.4s, v2.4s, v18.4s\n"
                        "STR	q18, [%[outptr4]], #16\n"
                        "FADD	v19.4s, v3.4s, v19.4s\n"
                        "prfm   pldl1keep, [%[outptr3], #80]\n"
                        "FADD	v20.4s, v4.4s, v20.4s\n"
                        "STP	q19, q20, [%[outptr5]], #32\n"
                        "FADD	v21.4s, v5.4s, v21.4s\n"
                        "STR	q21, [%[outptr5]], #16\n"

                        // Rows 6-7
                        "prfm   pldl1keep, [%[outptr4], #80]\n"
                        "LDP	q16, q17, [%[outptr6]]\n"
                        "LDR	q18, [%[outptr6], #32]\n"
                        "LDP	q19, q20, [%[outptr7]]\n"
                        "LDR	q21, [%[outptr7], #32]\n"
                        "prfm   pldl1keep, [%[outptr5], #80]\n"
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FADD	v16.4s, v0.4s, v16.4s\n"
                        "prfm   pldl1keep, [%[outptr6], #128]\n"
                        "FADD	v17.4s, v1.4s, v17.4s\n"
                        "STP	q16, q17, [%[outptr6]], #32\n"
                        "FADD	v18.4s, v2.4s, v18.4s\n"
                        "STR	q18, [%[outptr6]], #16\n"
                        "FADD	v19.4s, v3.4s, v19.4s\n"
                        "prfm   pldl1keep, [%[outptr7], #128]\n"
                        "FADD	v20.4s, v4.4s, v20.4s\n"
                        "STP	q19, q20, [%[outptr7]], #32\n"
                        "FADD	v21.4s, v5.4s, v21.4s\n"
                        "STR	q21, [%[outptr7]], #16\n"
                        "ADD	%[inptr], %[inptr], #384\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), \
                    [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), \
                    [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), \
                    [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7), \
                    [inptr] "+r" (inptr)
                :
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q16", "q17", \
                    "q18", "q19", "q20", "q21"
                );
            }
        }
    }

#else
    for (int y = y0; y < ymax; y += 8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;

        for (int i = x0; i < xmax; i += 8) {
            float dummyres[8];

            /* Make sure we throw away results if Y isn't a multiple of 8.
             * We do this by pointing the result pointer at a dummy buffer
             * we later discard.  */
            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        outptr1 = dummyres;
                    case 3:
                        outptr2 = dummyres;
                    case 2:
                        outptr3 = dummyres;
                    case 1:
                        outptr4 = dummyres;
                    case 0:
                        outptr5 = dummyres;
                    default:
                        break;
                }
            }

            /* For ragged X, manually copy over the valid results. */
            if ((i+7) >= xmax) {
                for (int xi=0; xi<8; xi++) {
                    if ((i+xi) < xmax) {
                        *outptr0 = inptr[xi] + *outptr0;
                        outptr0++;
                        *outptr1 = inptr[xi + 8] + *outptr1;
                        outptr1++;
                        *outptr2 = inptr[xi + 16] + *outptr2;
                        outptr2++;
                        *outptr3 = inptr[xi + 24] + *outptr3;
                        outptr3++;
                        *outptr4 = inptr[xi + 32] + *outptr4;
                        outptr4++;
                        *outptr5 = inptr[xi + 40] + *outptr5;
                        outptr5++;
                    }
                }
                inptr += 48;
            } else {
                /* Optimized routine to copy an entire block */
                __asm __volatile (
                //! Rows 0-1
                "VLD1.32	{d8-d11},  [%[outptr0]]\n"
                        "VLD1.32	{d12-d15}, [%[outptr1]]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VADD.f32	q4, q0, q4\n"
                        "pld    [%[inptr], #352]\n"
                        "VADD.f32	q5, q1, q5\n"
                        "VST1.32	{d8-d11}, [%[outptr0]]!\n"
                        "pld    [%[inptr], #416]\n"
                        "VADD.f32	q6, q2, q6\n"
                        "pld    [%[inptr], #480]\n"
                        "VADD.f32	q7, q3, q7\n"
                        "VST1.32	{d12-d15}, [%[outptr1]]!\n"

                        //! Rows 2-3
                        "VLD1.32	{d8-d11},  [%[outptr2]]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VADD.f32	q4, q0, q4\n"
                        "pld    [%[outptr0], #96]\n"
                        "VLD1.32	{d12-d15}, [%[outptr3]]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VADD.f32	q5, q1, q5\n"
                        "VST1.32	{d8-d11}, [%[outptr2]]!\n"
                        "pld    [%[outptr1], #96]\n"
                        "VADD.f32	q6, q2, q6\n"
                        "pld    [%[outptr2], #96]\n"
                        "VADD.f32	q7, q3, q7\n"
                        "VST1.32	{d12-d15}, [%[outptr3]]!\n"

                        // Rows 4-5
                        "VLD1.32	{d8-d11},  [%[outptr4]]\n"
                        "VLD1.32	{d12-d15}, [%[outptr5]]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VADD.f32	q4, q0, q4\n"
                        "pld    [%[outptr3], #96]\n"
                        "VADD.f32	q5, q1, q5\n"
                        "VST1.32	{d8-d11}, [%[outptr4]]!\n"
                        "pld    [%[outptr4], #96]\n"
                        "VADD.f32	q6, q2, q6\n"
                        "pld    [%[outptr5], #128]\n"
                        "VADD.f32	q7, q3, q7\n"
                        "VST1.32	{d12-d15}, [%[outptr5]]!\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), \
                    [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), \
                    [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), \
                    [inptr] "+r" (inptr)
                :
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                );
            }
        }
    }
#endif // end of __aarch64__
}

void merge_float_alpha1_beta1_relu(float *out, const float *in, const int ldout, const int y0, \
    const int ymax, const int x0, const int xmax) {
    const float *inptr = in;

    float32x4_t vzero = vdupq_n_f32(0.f);

#ifdef __aarch64__
    for (int y = y0; y < ymax; y += 8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;
        float *outptr6 = outptr5 + ldout;
        float *outptr7 = outptr6 + ldout;

        for (int i = x0; i < xmax; i += 12) {
            float dummyres[12];

            /* Make sure we throw away results if Y isn't a multiple of 8.
             * We do this by pointing the result pointer at a dummy buffer
             * we later discard.  */
            if ((y+7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        outptr1 = dummyres;
                    case 5:
                        outptr2 = dummyres;
                    case 4:
                        outptr3 = dummyres;
                    case 3:
                        outptr4 = dummyres;
                    case 2:
                        outptr5 = dummyres;
                    case 1:
                        outptr6 = dummyres;
                    case 0:
                        outptr7 = dummyres;
                    default:
                        break;
                }
            }

            /* For ragged X, manually copy over the valid results. */
            if ((i + 11) >= xmax) {
                for (int xi = 0; xi < 12; xi++) {
                    if ((i + xi) < xmax) {

                        outptr0[0] = inptr[xi] + outptr0[0];
                        outptr0[0] = fmaxf(outptr0[0], 0.f);
                        outptr0++;

                        outptr1[0] = inptr[xi + 12] + outptr1[0];
                        outptr1[0] = fmaxf(outptr1[0], 0.f);
                        outptr1++;

                        outptr2[0] = inptr[xi + 24] + outptr2[0];
                        outptr2[0] = fmaxf(outptr2[0], 0.f);
                        outptr2++;

                        outptr3[0] = inptr[xi + 36] + outptr3[0];
                        outptr3[0] = fmaxf(outptr3[0], 0.f);
                        outptr3++;

                        outptr4[0] = inptr[xi + 48] + outptr4[0];
                        outptr4[0] = fmaxf(outptr4[0], 0.f);
                        outptr4++;

                        outptr5[0] = inptr[xi + 60] + outptr5[0];
                        outptr5[0] = fmaxf(outptr5[0], 0.f);
                        outptr5++;

                        outptr6[0] = inptr[xi + 72] + outptr6[0];
                        outptr6[0] = fmaxf(outptr6[0], 0.f);
                        outptr6++;

                        outptr7[0] = inptr[xi + 84] + outptr7[0];
                        outptr7[0] = fmaxf(outptr7[0], 0.f);
                        outptr7++;
                    }
                }
                inptr += 96;
            } else {

                /* Optimized routine to copy an entire block */
                asm volatile (
                        // Rows 0-1
                "LDP	q16, q17, [%[outptr0]]\n"
                        "LDR	q18, [%[outptr0], #32]\n"
                        "LDP	q19, q20, [%[outptr1]]\n"
                        "LDR	q21, [%[outptr1], #32]\n"
                        "prfm   pldl1keep, [%[inptr], #768]\n"
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FADD	v16.4s, v0.4s, v16.4s\n"
                        "prfm   pldl1keep, [%[inptr], #832]\n"
                        "FADD	v17.4s, v1.4s, v17.4s\n"
                        "FMAX   v16.4s, v16.4s, %[vzero].4s\n"
                        "FMAX   v17.4s, v17.4s, %[vzero].4s\n"
                        "STP	q16, q17, [%[outptr0]], #32\n"
                        "FADD	v18.4s, v2.4s, v18.4s\n"
                        "FMAX   v18.4s, v18.4s, %[vzero].4s\n"
                        "STR	q18, [%[outptr0]], #16\n"
                        "FADD	v19.4s, v3.4s, v19.4s\n"
                        "prfm   pldl1keep, [%[inptr], #896]\n"
                        "FADD	v20.4s, v4.4s, v20.4s\n"
                        "FMAX   v19.4s, v19.4s, %[vzero].4s\n"
                        "FMAX   v20.4s, v20.4s, %[vzero].4s\n"
                        "STP	q19, q20, [%[outptr1]], #32\n"
                        "FADD	v21.4s, v5.4s, v21.4s\n"
                        "FMAX   v21.4s, v21.4s, %[vzero].4s\n"
                        "STR	q21, [%[outptr1]], #16\n"

                        // Rows 2-3
                        "LDP	q16, q17, [%[outptr2]]\n"
                        "LDR	q18, [%[outptr2], #32]\n"
                        "LDP	q19, q20, [%[outptr3]]\n"
                        "LDR	q21, [%[outptr3], #32]\n"
                        "prfm   pldl1keep, [%[inptr], #960]\n"
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FADD	v16.4s, v0.4s, v16.4s\n"
                        "prfm   pldl1keep, [%[inptr], #1024]\n"
                        "FADD	v17.4s, v1.4s, v17.4s\n"
                        "FMAX   v16.4s, v16.4s, %[vzero].4s\n"
                        "FMAX   v17.4s, v17.4s, %[vzero].4s\n"
                        "STP	q16, q17, [%[outptr2]], #32\n"
                        "FADD	v18.4s, v2.4s, v18.4s\n"
                        "FMAX   v18.4s, v18.4s, %[vzero].4s\n"
                        "STR	q18, [%[outptr2]], #16\n"
                        "FADD	v19.4s, v3.4s, v19.4s\n"
                        "prfm   pldl1keep, [%[inptr], #1088]\n"
                        "FADD	v20.4s, v4.4s, v20.4s\n"
                        "FMAX   v19.4s, v19.4s, %[vzero].4s\n"
                        "FMAX   v20.4s, v20.4s, %[vzero].4s\n"
                        "STP	q19, q20, [%[outptr3]], #32\n"
                        "FADD	v21.4s, v5.4s, v21.4s\n"
                        "FMAX   v21.4s, v21.4s, %[vzero].4s\n"
                        "STR	q21, [%[outptr3]], #16\n"

                        // Rows 4-5
                        "prfm   pldl1keep, [%[outptr0], #80]\n"
                        "LDP	q16, q17, [%[outptr4]]\n"
                        "LDR	q18, [%[outptr4], #32]\n"
                        "LDP	q19, q20, [%[outptr5]]\n"
                        "LDR	q21, [%[outptr5], #32]\n"
                        "prfm   pldl1keep, [%[outptr1], #80]\n"
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FADD	v16.4s, v0.4s, v16.4s\n"
                        "prfm   pldl1keep, [%[outptr2], #80]\n"
                        "FADD	v17.4s, v1.4s, v17.4s\n"
                        "FMAX   v16.4s, v16.4s, %[vzero].4s\n"
                        "FMAX   v17.4s, v17.4s, %[vzero].4s\n"
                        "STP	q16, q17, [%[outptr4]], #32\n"
                        "FADD	v18.4s, v2.4s, v18.4s\n"
                        "FMAX   v18.4s, v18.4s, %[vzero].4s\n"
                        "STR	q18, [%[outptr4]], #16\n"
                        "FADD	v19.4s, v3.4s, v19.4s\n"
                        "prfm   pldl1keep, [%[outptr3], #80]\n"
                        "FADD	v20.4s, v4.4s, v20.4s\n"
                        "FMAX   v19.4s, v19.4s, %[vzero].4s\n"
                        "FMAX   v20.4s, v20.4s, %[vzero].4s\n"
                        "STP	q19, q20, [%[outptr5]], #32\n"
                        "FADD	v21.4s, v5.4s, v21.4s\n"
                        "FMAX   v21.4s, v21.4s, %[vzero].4s\n"
                        "STR	q21, [%[outptr5]], #16\n"

                        // Rows 6-7
                        "prfm   pldl1keep, [%[outptr4], #80]\n"
                        "LDP	q16, q17, [%[outptr6]]\n"
                        "LDR	q18, [%[outptr6], #32]\n"
                        "LDP	q19, q20, [%[outptr7]]\n"
                        "LDR	q21, [%[outptr7], #32]\n"
                        "prfm   pldl1keep, [%[outptr5], #80]\n"
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FADD	v16.4s, v0.4s, v16.4s\n"
                        "prfm   pldl1keep, [%[outptr6], #128]\n"
                        "FADD	v17.4s, v1.4s, v17.4s\n"
                        "FMAX   v16.4s, v16.4s, %[vzero].4s\n"
                        "FMAX   v17.4s, v17.4s, %[vzero].4s\n"
                        "STP	q16, q17, [%[outptr6]], #32\n"
                        "FADD	v18.4s, v2.4s, v18.4s\n"
                        "FMAX   v18.4s, v18.4s, %[vzero].4s\n"
                        "STR	q18, [%[outptr6]], #16\n"
                        "FADD	v19.4s, v3.4s, v19.4s\n"
                        "prfm   pldl1keep, [%[outptr7], #128]\n"
                        "FADD	v20.4s, v4.4s, v20.4s\n"
                        "FMAX   v19.4s, v19.4s, %[vzero].4s\n"
                        "FMAX   v20.4s, v20.4s, %[vzero].4s\n"
                        "STP	q19, q20, [%[outptr7]], #32\n"
                        "FADD	v21.4s, v5.4s, v21.4s\n"
                        "FMAX   v21.4s, v21.4s, %[vzero].4s\n"
                        "STR	q21, [%[outptr7]], #16\n"
                        "ADD	%[inptr], %[inptr], #384\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), \
                    [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), \
                    [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), \
                    [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7), \
                    [inptr] "+r" (inptr)
                : [vzero] "w" (vzero)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q16", "q17", \
                    "q18", "q19", "q20", "q21"
                );
            }
        }
    }
#else
    for (int y = y0; y < ymax; y += 8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;

        for (int i = x0; i < xmax; i += 8) {
            float dummyres[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        outptr1 = dummyres;
                    case 3:
                        outptr2 = dummyres;
                    case 2:
                        outptr3 = dummyres;
                    case 1:
                        outptr4 = dummyres;
                    case 0:
                        outptr5 = dummyres;
                    default:
                        break;
                }
            }
            if ((i + 7) >= xmax) {
                for (int xi = 0; xi < 8; xi++) {
                    if ((i + xi) < xmax) {
                        outptr0[0] = inptr[xi] + outptr0[0];
                        outptr0[0] = fmaxf(outptr0[0], 0.f);
                        outptr0++;
                        *outptr1 = inptr[xi + 8] + *outptr1;
                        outptr1[0] = fmaxf(outptr1[0], 0.f);
                        outptr1++;
                        *outptr2 = inptr[xi + 16] + *outptr2;
                        outptr2[0] = fmaxf(outptr2[0], 0.f);
                        outptr2++;
                        *outptr3 = inptr[xi + 24] + *outptr3;
                        outptr3[0] = fmaxf(outptr3[0], 0.f);
                        outptr3++;
                        *outptr4 = inptr[xi + 32] + *outptr4;
                        outptr4[0] = fmaxf(outptr4[0], 0.f);
                        outptr4++;
                        *outptr5 = inptr[xi + 40] + *outptr5;
                        outptr5[0] = fmaxf(outptr5[0], 0.f);
                        outptr5++;
                    }
                }
                inptr += 48;
            } else {
                asm volatile (
                //! Rows 0-1
                "VLD1.32	{d8-d11},  [%[outptr0]]\n"
                        "VLD1.32	{d12-d15}, [%[outptr1]]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VADD.f32	q4, q0, q4\n"
                        "pld    [%[inptr], #352]\n"
                        "VADD.f32	q5, q1, q5\n"
                        "VMAX.f32   q4, q4, %q[vzero]\n"
                        "VMAX.f32   q5, q5, %q[vzero]\n"
                        "VST1.32	{d8-d11}, [%[outptr0]]!\n"
                        "pld    [%[inptr], #416]\n"
                        "VADD.f32	q6, q2, q6\n"
                        "pld    [%[inptr], #480]\n"
                        "VADD.f32	q7, q3, q7\n"
                        "VMAX.f32   q6, q6, %q[vzero] \n"
                        "VMAX.f32   q7, q7, %q[vzero]\n"
                        "VST1.32	{d12-d15}, [%[outptr1]]!\n"

                        //! Rows 2-3
                        "VLD1.32	{d8-d11},  [%[outptr2]]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VADD.f32	q4, q0, q4\n"
                        "pld    [%[outptr0], #96]\n"
                        "VLD1.32	{d12-d15}, [%[outptr3]]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VADD.f32	q5, q1, q5\n"
                        "VMAX.f32   q4, q4, %q[vzero]\n"
                        "VMAX.f32   q5, q5, %q[vzero]\n"
                        "VST1.32	{d8-d11}, [%[outptr2]]!\n"
                        "pld    [%[outptr1], #96]\n"
                        "VADD.f32	q6, q2, q6\n"
                        "pld    [%[outptr2], #96]\n"
                        "VADD.f32	q7, q3, q7\n"
                        "VMAX.f32   q6, q6, %q[vzero]\n"
                        "VMAX.f32   q7, q7, %q[vzero]\n"
                        "VST1.32	{d12-d15}, [%[outptr3]]!\n"

                        // Rows 4-5
                        "VLD1.32	{d8-d11},  [%[outptr4]]\n"
                        "VLD1.32	{d12-d15}, [%[outptr5]]\n"
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VADD.f32	q4, q0, q4\n"
                        "pld    [%[outptr3], #96]\n"
                        "VADD.f32	q5, q1, q5\n"
                        "VMAX.f32   q4, q4, %q[vzero]\n"
                        "VMAX.f32   q5, q5, %q[vzero]\n"
                        "VST1.32	{d8-d11}, [%[outptr4]]!\n"
                        "pld    [%[outptr4], #96]\n"
                        "VADD.f32	q6, q2, q6\n"
                        "pld    [%[outptr5], #128]\n"
                        "VADD.f32	q7, q3, q7\n"
                        "VMAX.f32   q6, q6, %q[vzero]\n"
                        "VMAX.f32   q7, q7, %q[vzero]\n"
                        "VST1.32	{d12-d15}, [%[outptr5]]!\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), \
                    [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), \
                    [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), \
                    [inptr] "+r" (inptr)
                : [vzero] "w" (vzero)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                );
            }
        }
    }
#endif // end of __aarch64__
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE