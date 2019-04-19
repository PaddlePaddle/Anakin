#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX512_EXPAND_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX512_EXPAND_H
namespace anakin {
namespace saber {
#if defined(__AVX512F__)
inline  __mmask16 __mm512_get_mask(int k) {
    __mmask16 mask = 0xffff;
    return mask >> (16 - k);
}
#endif
}
}

#endif //ANAKIN_SABER_AVX512_EXPAND_H
