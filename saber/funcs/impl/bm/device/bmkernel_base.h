#ifndef ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BMKERNEL_BASE_H
#define ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BMKERNEL_BASE_H
#ifdef __cplusplus
extern "C" {
#endif

namespace anakin {
namespace saber {

typedef struct {
    char op[64];
} __attribute__((packed)) bmkernel_api_base;

}
}
#ifdef __cplusplus
}
#endif
#endif /* ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BMKERNEL_BASE_H */
