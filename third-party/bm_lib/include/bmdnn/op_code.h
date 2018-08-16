#ifndef OP_CODE_H_
#define OP_CODE_H_


typedef enum align_tensor_op {
    ALIGN_TENSOR_ADD,
    ALIGN_TENSOR_SUB,
    ALIGN_TENSOR_MUL,
    ALIGN_TENSOR_DIV,
    TENSOR_INVALID
} ALIGN_TENSOR_OP;

typedef enum linear_op {
    LINEAR_MAC,
    LINEAR_ADD_SQR,
    LINEAR_SUB_SQR
} LINEAR_OP;

typedef enum sfu_op {
    SFU_XN,
    SFU_EX,
    SFU_LNX,
    SFU_RSQ,
    SFU_INVALID
} SFU_OP;
typedef struct tensor_4d_t {
    int n;
    int c;
    int h;
    int w;
}bm_tensor_4d_t;


#define TENSOR_ADD 0
#define TENSOR_SUB 1
#define TENSOR_MUL 2
//Note the div should be implmented by KAMAKE algorithm
#define TENSOR_DIV 3
#define TENSOR_MAX 4
#define TENSOR_CPY 5
#define TENSOR_MAC 6

#define TENSOR_N_DIM 0
#define TENSOR_C_DIM 1
#define TENSOR_H_DIM 2
#define TENSOR_W_DIM 3

#define SHARE_REG_MESSAGE_WP            0
#define SHARE_REG_MESSAGE_RP            1
#define SHARE_REG_MESSAGE_IRQSTATUS     2
#define SHARE_REG_CDMA_IRQSTATUS    3 
#define SHARE_REG_MSGIRQ_NUM_LO     4
#define SHARE_REG_MSGIRQ_NUM_HI     5

#define SHAREMEM_MSG_FIXED_OFFSET  (8192)
#define SHAREMEM_SIZE_BIT  8
#define SHAREMEM_MASK      ((1<<SHAREMEM_SIZE_BIT) - 1)
#define SHARE_REG_CNT      16

#define IRQ_STATUS_CDMA_INT             0x1111
#define IRQ_STATUS_MSG_DONE_INT         0x2222

 
#endif /* OP_CODE_H_ */
