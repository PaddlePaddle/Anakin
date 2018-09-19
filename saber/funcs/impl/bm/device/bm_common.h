#ifndef ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_COMMON_H
#define ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include "bm_config.h"
#include "bm_atomic.h"
#include "bm_memmap.h"
#ifdef __cplusplus
extern "C" {
#endif

//#define DEBUG_MESSAGE
#ifdef DEBUG_MESSAGE
#define MSG_DBG(fmt, ...)       printf("MSG: "fmt, ##__VA_ARGS__)
#else
#define MSG_DBG(fmt, ...)
#endif


#define INLINE                  inline

#define UNUSED(x)               (void)(x)

#define __ALIGN_MASK(x,mask)    (((x)+(mask))&~(mask))
#define ALIGN(x,a)              __ALIGN_MASK(x,(__typeof__(x))(a)-1)

#define ROUND_UP(A, B)  ((A)/(B) + ((A) % (B) == 0 ? 0 : 1))

#define bm_min(x, y)               ((x) < (y) ? (x) : (y))
#define bm_max(x, y)               ((x) > (y) ? (x) : (y))


typedef unsigned char           u8;
typedef unsigned short          u16;
typedef unsigned int            u32;
typedef unsigned long long      u64;

typedef union {
  int ival;
  float fval;
} IF_VAL;

typedef u32 tuple4_u32[4];

typedef struct tensor_info{
    u32 n,c,h,w;
    u32 w_stride, n_stride, c_stride, h_stride;
    u32 address;
    u32 data_format;
    u32 neuron_matrix;		//0: neuron, 1: matrix
    u32 matrix_col_magin;	//the magin is not 0, when column_num%w_param!=0
}TENSOR_INFO;


typedef struct shape{
    u16 n, c, h, w;
}local_shape_t;

#define FLOAT_SIZE              4
#define INT8_SIZE               1
#define FLOAT_BITWIDTH          32
#define GET_U64(U32_H, U32_L)   (((u64)(U32_H) << 32) | (u64)(U32_L))

typedef enum {
    CAFFE_SUPPORT             = 0,
    TENSORFLOW_SUPPORT        = 1
} PLATFORM_SUPPORT;

typedef enum {
    NODECHIP_REG    = 0,
    HOST_REG        = 1
} REG_TYPE;

typedef enum {
  ENGINE_BD                     = 0,
  ENGINE_GDMA                   = 1,
  ENGINE_CDMA                   = 2,
  ENGINE_HDMA                   = 3,
  ENGINE_END
} ENGINE_ID;

typedef struct tensor_4d_t {
    int n;
    int c;
    int h;
    int w;
} bm_tensor_4d_t;

typedef struct kernel_param{
    int g;
    int oc;
    int ic;
    int h;
    int w;
} bm_kernel_param_t;

typedef struct bm_conv_param{
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dilation_h;
    int dilation_w;
    bool result_add;
} bm_conv_param_t;

typedef struct conv_secs_info{
    int ocsecs;
    int icsecs;
    int nsecs;
    int hsecs;
} conv_secs_info_t;

static INLINE int ceiling_func(int numerator, int denominator)
{
  return (numerator + denominator - 1) / denominator;
}

static INLINE int ceiling_func_shift(int numerator, int shift)
{
  return (numerator + (1 << shift) - 1) >> shift;
}

static int INLINE calc_offset(int *shape, int *offset)
{
  return ((offset[0] * shape[1] + offset[1]) * shape[2] + offset[2])
      * shape[3] + offset[3];
}

//All the size are in the units of bytes
static int INLINE get_index_csize_global(int h, int w, int index_bitwidth)
{
  int size = h * w * index_bitwidth;
  //32 bit align
  return (((size >> 5)) + ((size & 0x1f) != 0)) * FLOAT_SIZE;
}

static int INLINE get_index_cstride_global(int h, int w, int index_bitwidth)
{
  int size = h * w * index_bitwidth;
  //32 bit align
  return (((size >> 5)) +
          ((size & 0x1f) != 0)) * FLOAT_BITWIDTH / index_bitwidth;
}

static int INLINE get_neuron_csize_local(int h, int w)
{
  int size = h * w;
  //EU_NUM neurons align
  return ALIGN(size,EU_NUM) * FLOAT_SIZE;
}

static int INLINE addr_EU_align(int addr){
  addr = addr / FLOAT_SIZE;
  return ALIGN( addr, EU_NUM ) * FLOAT_SIZE;
}

static INLINE int get_align_tensor_size(bm_tensor_4d_t shape){
  int c_per_npu = ceiling_func_shift(shape.c, NPU_SHIFT);
  return shape.n * c_per_npu * get_neuron_csize_local(shape.h, shape.w);
}

static int INLINE get_cstride_local(int h, int w)
{
  int size = h * w;
  //EU_NUM neurons align
  return ALIGN(size,EU_NUM);
}

#ifdef __cplusplus
}
#endif
#endif /* ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_COMMON_H */
