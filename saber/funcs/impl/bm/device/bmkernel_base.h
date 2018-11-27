#ifndef ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BMKERNEL_BASE_H
#define ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BMKERNEL_BASE_H
#ifdef __cplusplus
extern "C" {
#endif

enum BmOpType {
    ACTIVATION, 
    CONV
};

typedef struct {
    unsigned long long             ifmap_offset_global;
    unsigned long long             ofmap_offset_global;
    unsigned long long             weight_offset_global;
    unsigned long long             bias_offset_global;
    int                            input_n;   // note this is total input_n
    int                            input_c;
    int                            input_h;
    int                            input_w;
    int                            groups;
    int                            output_c;
    int                            kh;
    int                            kw;
    int                            dh;
    int                            dw;
    int                            pad_h;
    int                            pad_w;
    int                            stride_h;
    int                            stride_w;
    int                            using_bias;
    int                            result_add;
    int                            icsecs;
    int                            ocsecs;
    int                            nsecs;
    int                            hsecs;
} __attribute__((packed)) bm_api_conv_forward;

typedef struct {
    enum BmOpType op; // Flag to determine the operation type.
    union U1{
      bm_api_conv_forward convParam;
    } opParam;
} __attribute__((packed)) bmkernel_api_base;

#ifdef __cplusplus
}
#endif
#endif /* ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BMKERNEL_BASE_H */
