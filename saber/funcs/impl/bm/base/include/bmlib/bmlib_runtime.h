#ifndef BMLIB_RUNTIME_H_
#define BMLIB_RUNTIME_H_
#include <stdbool.h>
#include <stddef.h>

#if !defined(__x86_64__) && !defined(__aarch64__)
#error "BM needs 64-bit to compile"
#endif

#if defined (__cplusplus)
extern "C" {
#endif

typedef enum {
  BM_SUCCESS                 = 0,
  BM_ERR_DEVNOTREADY          = 1,   /* Device not ready yet */
  BM_ERR_FAILURE             = 2,   /* General failure */
  BM_ERR_TIMEOUT             = 3,   /* Timeout */
  BM_ERR_PARAM               = 4,   /* Parameters invalid */
  BM_ERR_NOMEM               = 5,   /* Not enough memory */
  BM_ERR_DATA                = 6,   /* Data error */
  BM_ERR_BUSY                = 7,   /* Busy */
  BM_ERR_NOFEATURE           = 8,    /* Not supported yet */
  BM_NOT_SUPPORTED           = 9
} bm_status_t;

typedef enum {
  BM_MEM_TYPE_DEVICE  = 0,
  BM_MEM_TYPE_HOST    = 1,
  BM_MEM_TYPE_SYSTEM  = 2,
  BM_MEM_TYPE_INT8_DEVICE  = 3,
  BM_MEM_TYPE_INVALID = 4
} bm_mem_type_t;

#define BM_MEM_ADDR_NULL     (0xfffffffff)

typedef struct bm_mem_desc {
  unsigned char                 desc[16];
} bm_mem_desc_t;

struct bm_context;
typedef struct bm_context *  bm_handle_t;
typedef struct bm_mem_desc   bm_device_mem_t;
typedef struct bm_mem_desc   bm_host_mem_t;
typedef struct bm_mem_desc   bm_system_mem_t;

#define BM_CHECK_RET(call)                         \
    do {                                        \
      bm_status_t ret = call;                \
	  if ( ret != BM_SUCCESS ) {             \
        printf("BM_CHECK_RET failed %d\n", ret);   \
        ASSERT(0);                              \
        exit(-ret);                             \
      }                                         \
    } while(0)

/*
 * control 
 */
void bm_flush(
    bm_handle_t      handle);
/*
 * brief malloc host memory according to a tensor shape(each neuron is 32 bits)
*/

bm_status_t bm_malloc_neuron_device(
    bm_handle_t      handle,
    bm_device_mem_t *pmem,
    int              n,
    int              c,
    int              h,
    int              w);

/*
 * brief malloc host memory in size of dword(32 bits)
*/

bm_status_t bm_malloc_device_dword(
    bm_handle_t      handle,
    bm_device_mem_t *pmem,
    int              count);
bm_status_t bm_malloc_ctx_dword(
    bm_handle_t      handle,
    bm_device_mem_t *pmem,
    int              count,
    unsigned long long ctx_addr);
/*
 * brief malloc host memory in size of byte
*/

bm_status_t bm_malloc_device_byte(
    bm_handle_t      handle,
    bm_device_mem_t *pmem,
    unsigned int     size);

void bm_free_device(
    bm_handle_t      handle,
    bm_device_mem_t  mem);

/*
 * brief malloc host memory in size of byte
 */
bm_status_t bm_malloc_host(
    bm_handle_t      handle,
    bm_host_mem_t   *pmem,
    unsigned int     size);

bm_status_t bm_free_host(
    bm_handle_t      handle,
    bm_host_mem_t    mem);

void *bm_host_mem_get_pointer(
    bm_host_mem_t    mem);

/*
 * Memory copy and set
 */
bm_status_t bm_memcpy_h2d(
    bm_handle_t      handle,
    bm_device_mem_t  dst,
    bm_host_mem_t    src);

bm_status_t bm_memcpy_d2h(
    bm_handle_t      handle,
    bm_host_mem_t    dst,
    bm_device_mem_t  src);


bm_status_t bm_memcpy_s2d(
    bm_handle_t      handle,
    bm_device_mem_t  dst,
    bm_system_mem_t  src);

bm_status_t bm_memcpy_d2s(
    bm_handle_t      handle,
    bm_system_mem_t  dst,
    bm_device_mem_t  src);

bm_status_t bm_memcpy_d2d(
    bm_handle_t     handle,
    bm_device_mem_t dst,
    int             dst_offset,
    bm_device_mem_t src,
    int             src_offset,
    int             len);

bm_status_t bm_memset_device(
    bm_handle_t      handle,
    const int        value,
    bm_device_mem_t  mem);

bm_device_mem_t bm_mem_from_system(
    void *              system_addr);

/*
*brief malloc one device memory with the shape of (N,C,H,W), copy the sys_mem to
device mem if need_copy is true
*/

bm_status_t bm_mem_convert_system_to_device_neuron(
    bm_handle_t          handle,
    struct bm_mem_desc  *dev_mem,
    struct bm_mem_desc   sys_mem,
    bool                 need_copy,
    int                  n,
    int                  c,
    int                  h,
    int                  w);

/*
*brief malloc one device memory with the size of coeff_count, copy the sys_mem to
device mem if need_copy is true
*/
bm_status_t bm_mem_convert_system_to_device_coeff(
    bm_handle_t          handle,
    struct bm_mem_desc  *dev_mem,
    struct bm_mem_desc   sys_mem,
    bool                 need_copy,
    int                  coeff_count);

/*
 * memory info get and set
 */
unsigned long long bm_mem_get_device_addr(struct bm_mem_desc mem);
void               bm_mem_set_device_addr(struct bm_mem_desc & mem, unsigned long long addr);
unsigned int       bm_mem_get_device_size(struct bm_mem_desc mem);
void               bm_mem_set_device_size(struct bm_mem_desc & mem, unsigned int size);
bm_mem_type_t      bm_mem_get_type(struct bm_mem_desc mem);

unsigned long long bm_gmem_arm_reserved_request(bm_handle_t handle);
void bm_gmem_arm_reserved_release(bm_handle_t handle);

/* 
* brief Get the handle of bmlib_runtime
* return : If the handle has been inited, return the handle it self , else init one and return it
*/

bm_status_t bm_init(bm_handle_t *handle, bool bmkernel_used);
void bm_deinit(bm_handle_t handle);

/*
 * Helper functions
 */

/**
* \brief Get the number of nodechip (Constant 1 in bm1682)
* \return
* \ref NO
*/
int bm_get_nodechip_num(
    bm_handle_t      handle);

/**
* \brief Get the number of nodechip (Constant 64 in bm1682)
* \return
* \ref NO
*/
int bm_get_npu_num(
    bm_handle_t      handle);
int bm_get_eu_num( bm_handle_t handle);
/**
* \brief Get the number of nodechip (Constant 64 in bm1682)
* \return
* \ref NO
*/
bm_device_mem_t bm_mem_null(void);
#define BM_MEM_NULL  (bm_mem_null())

bm_status_t bm_dev_getcount(int* count);
bm_status_t bm_dev_query(int devid);
bm_status_t bm_dev_request(bm_handle_t *handle, bool bmkernel_used, int devid);
void bm_dev_free(bm_handle_t handle);

typedef struct bm_fw_desc {
	unsigned int *itcm_fw;
	int itcmfw_size;
	unsigned int *ddr_fw;
	int ddrfw_size;
} bm_fw_desc, *pbm_fw_desc;
bm_status_t bm_update_firmware(bm_handle_t handle, pbm_fw_desc pfw);

#if defined (__cplusplus)
}
#endif

#endif /* BM_RUNTIME_H_ */
