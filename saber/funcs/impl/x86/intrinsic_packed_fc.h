#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_INTRINSIC_PACKED_FC_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_INTRINSIC_PACKED_FC_H
#include "saber/core/tensor.h"
#include "saber/funcs/gemm.h"
#include "jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"

namespace anakin {
namespace saber {
namespace jit{
static int print_buffer[32] {0};
struct jit_s8s8s32_packed_gemm: public jit_generator {

    jit_s8s8s32_packed_gemm(jit_int8_packed_fc_config_t ajcp) : jcp(ajcp) {

//        real_printf(123);
//        real_printf_fp32();
        print_func_ptr = (void*)&real_printf;
        print_vec_func_ptr = (void*)&real_printf_fp32;
        this->generate();
        jit_ker = (void (*)(jit_int8_packed_fc_call_t*))this->getCode();
//                LOG(INFO) << "gen done";

    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_s8s8s32_packed_gemm);

    void (*jit_ker)(jit_int8_packed_fc_call_t*);



private:
    void cal_one_block();
    void load_and_init();
    void reduction_and_store2mem();
    static void real_printf(size_t x) {
        printf("real_printf %d , %p \n", x, x);
    }
    static void real_printf_fp32() {
        for (int i = 0; i < 8; i++) {
            printf("avx printf[%d] = %d\n",i, print_buffer[i]);
        }
        for (int i = 0; i < 8; i++) {
            print_buffer[i]=-i;
        }
    }



    void* print_func_ptr{nullptr};
    void* print_vec_func_ptr{nullptr};
    void print_jit(Xbyak::Reg64 reg) {
        save_common_regs();
        mov(rax, (size_t)print_func_ptr);
        mov(abi_param1, reg);
        call(rax);
        restore_common_regs();
    }

    void print_jit_vec(Xbyak::Ymm reg) {
        save_common_regs();
        mov(rax, (size_t)print_vec_func_ptr);
        mov(r15, (size_t)&print_buffer[0]);
        vmovdqu(ptr[r15], reg);
        call(rax);
        restore_common_regs();
    }

    void print_jit_vec(Xbyak::Xmm reg) {
        save_common_regs();
        mov(rax, (size_t)print_vec_func_ptr);
        mov(r15, (size_t)&print_buffer[0]);
        movdqu(ptr[r15], reg);
        call(rax);
        restore_common_regs();
    }

    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t reg_output = rbx;
    reg64_t reg_weights = rcx;
    reg64_t reg_k_block_size = rdx;
    reg64_t reg_k_block_num = r8;
    //    reg64_t reg_debug=r9;

    reg64_t reg_lda = rsi;
    reg64_t reg_ldb = r9;
    reg64_t temp_0 = rsi;
    reg64_t temp_1 = r9;
    reg64_t reg_ldc = rsi;



    reg64_t address_a_0 = r10;
    reg64_t address_a_1 = r11;
    reg64_t address_b_0 = r12;
    reg64_t address_b_1 = r13;
    reg64_t address_b_2 = r14;
    reg64_t address_b_3 = r15;



    Xbyak::Ymm sum_row0_col0 = Xbyak::Ymm(0);
    Xbyak::Ymm sum_row0_col1 = Xbyak::Ymm(1);
    Xbyak::Ymm sum_row0_col2 = Xbyak::Ymm(2);
    Xbyak::Ymm sum_row0_col3 = Xbyak::Ymm(3);
    Xbyak::Ymm c_row0_col0_1 = Xbyak::Ymm(0);
    Xbyak::Ymm c_row0_col2_3 = Xbyak::Ymm(1);
    Xbyak::Ymm c_row0_col0_1_2_3 = Xbyak::Ymm(0);
    Xbyak::Xmm c_row0_col0_1_2_3_m128 = Xbyak::Xmm(0);

    Xbyak::Ymm sum_row1_col0 = Xbyak::Ymm(4);
    Xbyak::Ymm sum_row1_col1 = Xbyak::Ymm(5);
    Xbyak::Ymm sum_row1_col2 = Xbyak::Ymm(6);
    Xbyak::Ymm sum_row1_col3 = Xbyak::Ymm(7);
    Xbyak::Ymm c_row1_col0_1 = Xbyak::Ymm(4);
    Xbyak::Ymm c_row1_col2_3 = Xbyak::Ymm(5);
    Xbyak::Ymm c_row1_col0_1_2_3 = Xbyak::Ymm(4);
    Xbyak::Xmm c_row1_col0_1_2_3_m128 = Xbyak::Xmm(4);


    Xbyak::Ymm a0 = Xbyak::Ymm(8);
    Xbyak::Ymm a1 = Xbyak::Ymm(9);
    Xbyak::Ymm b0 = Xbyak::Ymm(10);
    Xbyak::Ymm b1 = Xbyak::Ymm(11);
    Xbyak::Ymm b2 = Xbyak::Ymm(12);
    Xbyak::Ymm b3 = Xbyak::Ymm(13);
    Xbyak::Xmm a0_xmm = Xbyak::Xmm(8);
    Xbyak::Xmm a1_xmm = Xbyak::Xmm(9);
    Xbyak::Xmm b0_xmm = Xbyak::Xmm(10);
    Xbyak::Xmm b1_xmm = Xbyak::Xmm(11);
    Xbyak::Xmm b2_xmm = Xbyak::Xmm(12);
    Xbyak::Xmm b3_xmm = Xbyak::Xmm(13);
    Xbyak::Ymm zero_in_reduction = Xbyak::Ymm(8);
    Xbyak::Ymm temp0_in_reduction = Xbyak::Ymm(9);
    Xbyak::Ymm temp1_in_reduction = Xbyak::Ymm(10);
    Xbyak::Ymm temp2_in_reduction = Xbyak::Ymm(11);
    Xbyak::Ymm temp3_in_reduction = Xbyak::Ymm(12);

    Xbyak::Ymm vtemp_0 = Xbyak::Ymm(14);
    Xbyak::Ymm vtemp_1 = Xbyak::Ymm(15);
    Xbyak::Ymm vtemp_3 = Xbyak::Ymm(16);
    Xbyak::Ymm vtemp_4 = Xbyak::Ymm(17);
    jit_int8_packed_fc_config_t jcp;
    const size_t aligned_length = 16;

    void generate();
};
}

enum PackedFCAlg : int{
    DotReduction=0,
    DotAdd,
    DotReductionPacked,
    DotSplitK,
};

template <DataType A_Dtype,DataType B_Dtype,DataType C_Dtype>
class PackedFC {
public:
    PackedFC(){
        _scale_inputs.re_alloc(Shape({1,1,1,64}),AK_INT8);
    }
    ~PackedFC(){
        delete _packed_gemm;
    }
//    SaberStatus init(int n,int k,int8_t* weights);
    SaberStatus init(int n, int k, Tensor<X86>& weights_tensor,float input_scale=1.f,float output_scale=1.f,PackedFCAlg alg=DotReduction);

    SaberStatus dispatch(const int m, const int n, const int k, const Tensor<X86>&tensor_a,
                         Tensor<X86> &tensor_c);

    Tensor<X86> _inner_weights;
private:

    Tensor<X86> _scale_inputs;
    jit::jit_s8s8s32_packed_gemm* _packed_gemm{nullptr};
    std::vector<float> _scale;
    PackedFCAlg _alg;
};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_X86_INTRINSIC_PACKED_FC_H
