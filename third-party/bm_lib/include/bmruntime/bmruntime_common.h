#ifndef BMRUNTIME_COMMON_H
#define BMRUNTIME_COMMON_H

#define BMRT_ASSERT(_cond)                       \
  do {                                           \
    if (!(_cond)) {                              \
      printf("ASSERT %s: %s: %d: %s\n",          \
          __FILE__, __func__, __LINE__, #_cond); \
      exit(-1);                                  \
    }                                            \
  } while(0)

typedef enum neuron_device_mem_type {
    INPUT_NEURON_TENSOR = 0,
    INTERMEDIATE_NEURON_TENSOR = 1,
    OUTPUT_NEURON_TENSOR = 2,
    CMD_BUF_TENSOR = 3,
    CMD_NUM_TENSOR = 4
} NEURON_DEVICE_MEM_TYPE;

typedef enum device_mem_type {
    NEURON = 0,
    COEFF = 1,
#ifdef INT8_COEFF_FUNC
    COEFF_INT8 = 2,
    COEFF_INT8SCALE = 3,
    LOCAL = 4
#else
    LOCAL = 2
#endif
} DEVICE_MEM_TYPE;

typedef struct device_mem_info {
    DEVICE_MEM_TYPE device_mem_type;
    NEURON_DEVICE_MEM_TYPE neuron_device_mem_type;
    int n;
    int c;
    int h;
    int w;
    int coeff_count;
    int groups;
    unsigned long long address;
    unsigned long size;
} DEVICE_MEM_INFO;

//info for compute output tensor
typedef struct tensor_max_shape {
  int max_n;
  int channel;
  int max_h;
  int max_w;
} tensor_max_shape_t;

typedef struct global_output_tensor_param {
  int input_idx;
  int global_kh;
  int global_kw;
  int global_stride_h;
  int global_stride_w;
  int global_pad_h;
  int global_pad_w;
  int global_pool_kh;
  int global_pool_kw;
} global_output_tensor_param_t; 

#endif
