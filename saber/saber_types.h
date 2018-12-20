/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef ANAKIN_SABER_CORE_TYPES_H
#define ANAKIN_SABER_CORE_TYPES_H
#include "anakin_config.h"
namespace anakin{
namespace saber{
struct __true_type{};
struct __false_type{};
struct __invalid_type{};
//! target type
enum TargetTypeEnum {
    eINVALID = -1,
    eNV = 1,
    eAMD = 2,
    eARM = 3,
    eX86 = 4,
    eNVHX86 = 5,
    eNVHARM = 6,
    eARMGPU = 7,
    eARMDSP = 8,
    eBM = 9,
    eAMDHX86 = 10,
};
template <TargetTypeEnum T>
struct TargetType {};
// NV device without pinned memory
typedef TargetType<eNV> NV;
typedef TargetType<eBM> BM;
typedef TargetType<eARM> ARM;
typedef TargetType<eARMGPU> ARMGPU;
typedef TargetType<eAMD> AMD;
typedef TargetType<eX86> X86;
// NV device with pinned memory
typedef TargetType<eNVHX86> NVHX86;
//typedef TargetType<eNVHARM> NVHARM;
// invalid target type, for target has only one memory block
typedef TargetType<eAMDHX86> AMDHX86;
typedef TargetType<eINVALID> INVLD;
enum LayoutType {
    Layout_invalid = 0,
    Layout_W = 1,
    Layout_HW = 2,
    Layout_WH = 3,
    Layout_NC = 4,
    Layout_NH = 5,
    Layout_NW = 6,
    Layout_NHW = 7,
    Layout_NCHW = 8,
    Layout_NHWC = 9,
    Layout_NCHW_C4 = 10,
    Layout_NCHW_C8 = 11,
    Layout_NCHW_C16 = 12,
    Layout_OIHW16I16O = 13,
    Layout_GOIHW16I16O = 14,
    Layout_NCHW_C8R=15,
    Layout_NCHW_C16R=16,
};
//! target_type struct
struct Layout {
    virtual int num_index() {return -1;}
    virtual int channel_index() {return -1;}
    virtual int height_index() {return -1;}
    virtual int width_index() {return -1;}
    virtual int depth_index() {return -1;}
    virtual int inner_c() {return -1;}
    virtual int dims() {return -1;}
    virtual LayoutType type() {return Layout_invalid;}
};
struct W : public Layout {
    int width_index() {return 0;}
    int dims() {return 1;}
    LayoutType type() {return Layout_W;}
};
struct HW : public Layout {
    int height_index() {return 0;}
    int width_index() {return 1;}
    int dims() {return 2;}
    LayoutType type() {return Layout_HW;}
};
struct WH : public Layout {
    int height_index() {return 1;}
    int width_index() {return 0;}
    int dims() {return 2;}
    LayoutType type() {return Layout_WH;}
};
struct NC : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int dims() {return 2;}
    LayoutType type() {return Layout_NC;}
};
struct NH : public Layout {
    int num_index() {return 0;}
    int height_index() {return 1;}
    int dims() {return 2;}
    LayoutType type() {return Layout_NH;}
};
struct NW : public Layout {
    int num_index() {return 0;}
    int width_index() {return 1;}
    int dims() {return 2;}
    LayoutType type() {return Layout_NW;}
};
struct NHW : public Layout {
    int num_index() {return 0;}
    int height_index() {return 1;}
    int width_index() {return 2;}
    int dims() {return 3;}
    LayoutType type() {return Layout_NHW;}
};
struct NCHW : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int dims() {return 4;}
    LayoutType type() {return Layout_NCHW;}
};
struct NHWC : public Layout {
    int num_index() {return 0;}
    int height_index() {return 1;}
    int width_index() {return 2;}
    int channel_index() {return 3;}
    int dims() {return 4;}
    LayoutType type() {return Layout_NHWC;}
};
struct NCHW_C4 : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int inner_c() {return 4;}
    int dims() {return 5;}
    LayoutType type() {return Layout_NCHW_C4;}
};
struct NCHW_C8 : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int inner_c() {return 8;}
    int dims() {return 5;}
    LayoutType type() {return Layout_NCHW_C8;}
};
struct NCHW_C8R : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int dims() {return 4;}
    LayoutType type() {return Layout_NCHW_C8R;}
};
struct NCHW_C16 : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int inner_c() {return 16;}
    int dims() {return 5;}
    LayoutType type() {return Layout_NCHW_C16;}
};

struct NCHW_C16R : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int dims() {return 4;}
    LayoutType type() {return Layout_NCHW_C16R;}
};

enum DataType {
    AK_INVALID      =       -1,
    AK_HALF         =       0,
    AK_FLOAT        =       1,
    AK_DOUBLE       =       2,
    AK_INT8         =       3,
    AK_INT16        =       4,
    AK_INT32        =       5,
    AK_INT64        =       6,
    AK_UINT8        =       7,
    AK_UINT16       =       8,
    AK_UINT32       =       9,
    AK_UINT64       =       10,
    AK_STRING       =       11,
    AK_BOOL         =       12,
    AK_SHAPE        =       13,
    AK_TENSOR       =       14
};
typedef enum {
    SaberSuccess         = -1,                             /*!< No errors */
    SaberNotInitialized  = 1,                              /*!< Data not initialized. */
    SaberInvalidValue    = (1 << 1) + SaberNotInitialized, /*!< Incorrect variable value. */
    SaberMemAllocFailed  = (1 << 2) + SaberInvalidValue,   /*!< Memory allocation error. */
    SaberUnKownError     = (1 << 3) + SaberMemAllocFailed, /*!< Unknown error. */
    SaberOutOfAuthority  = (1 << 4) + SaberUnKownError,    /*!< Try to modified data not your own*/
    SaberOutOfMem        = (1 << 5) + SaberOutOfAuthority, /*!< OOM error*/
    SaberUnImplError     = (1 << 6) + SaberOutOfMem,       /*!< Unimplement error. */
    SaberWrongDevice     = (1 << 7) + SaberUnImplError     /*!< un-correct device. */
} SaberStatus;
typedef enum{
    STATIC = 1, /*!< choose impl by static policy */
    RUNTIME = 2, /*!< choose impl by compare performance at runtime */
    SPECIFY = 3,
    UNKNOWN = 4
}SaberImplStrategy;

typedef enum {
    nearest = 0,
    down
} round_mode;

//should design this one for pick_best_specify()
enum ImplEnum{
    VENDER_IMPL = 0,
    SABER_IMPL
};
enum SequencePoolType{
    Sequence_pool_unknow = 0,
    Sequence_pool_average,
    Sequence_pool_sum,
    Sequence_pool_sqrt,
    Sequence_pool_last,
    Sequence_pool_first,
    Sequence_pool_max
};
/**
 * GRU_Formula,origin for paddle,Cudnn for cudnn,difference is w_h_r and weighted mean
 * weight for origin is [W_h_o][W_h_r,W_h_z]
 * weight for cudnn is [W_h_o,W_h_r,W_h_z]
 */
enum GruFormula {
    GRU_ORIGIN = 0,
    GRU_CUDNN
};
typedef enum{
    Active_unknow = 0,
    Active_sigmoid = 1,
    Active_relu = 2,
    Active_tanh = 3,
    Active_clipped_relu = 4,
    Active_elu = 5,
    Active_identity = 6,
    Active_stanh = 9,
    Active_prelu = 10
} ActiveType;
typedef enum{
    Pooling_unknow = 0,
    Pooling_max = 1,
    Pooling_average_include_padding = 2,
    Pooling_average_exclude_padding = 3,
    Pooling_max_deterministic
} PoolingType;
typedef enum{
    Eltwise_unknow = 0,
    Eltwise_prod = 1,
    Eltwise_sum = 2,
    Eltwise_max = 3
} EltwiseType;
typedef enum{
    ACROSS_CHANNELS = 0,
    WITHIN_CHANNEL = 1
} NormRegion;
enum BoxCoderType {
    ENCODE_CENTER = 0,
    DECODE_CENTER = 1
};
enum CodeType {
    CORNER      = 1,
    CENTER_SIZE = 2,
    CORNER_SIZE = 3
};
typedef enum {
    SABER_POWER_HIGH = 0,
    SABER_POWER_LOW  = 1,
    SABER_POWER_FULL = 2,
    SABER_POWER_NO_BIND = 3,
    SABER_POWER_RAND_HIGH = 4,
    SABER_POWER_RAND_LOW = 5
} PowerMode;
typedef enum {
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE
} BorderType;
typedef enum {
    PRIOR_MIN = 0,
    PRIOR_MAX = 1,
    PRIOR_COM = 2
} PriorType;

typedef enum{
    RANDOM=0,
    SPECIAL,
    CUSTOM
} TestDataType;
typedef enum{
    ENTROPY= 0,
    MAXABS = 1
} CalibrationAlgoType;
typedef enum{
    BILINEAR_ALIGN = 0,
    BILINEAR_NO_ALIGN = 1,
    RESIZE_CUSTOM = 2
} ResizeType;
} //namespace saber
} //namespace anakin
#endif //ANAKIN_SABER_CORE_TYPES_H
