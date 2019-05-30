/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#pragma once

#include <type_traits>
#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <unordered_map>

#include <CL/cl.h>

#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include "saber/core/tensor_op.h"

#define RADIX_SORT_FILL_DIGIT_COUNTS_LONG_ITER      1
#define RADIX_SORT_SCAN_BATCH_LONG_ITER             2
#define RADIX_SORT_SCAN_DIGITS_LONG_ITER            3
#define RADIX_SORT_SORT_SCATTER_LONG_ITER           4

#define RADIX_SORT_FILL_DIGIT_COUNTS_SHORT_ITER      5
#define RADIX_SORT_SCAN_BATCH_SHORT_ITER             6
#define RADIX_SORT_SCAN_DIGITS_SHORT_ITER            7
#define RADIX_SORT_SORT_SCATTER_SHORT_ITER           8

#define RADIX_SORT_OUTPUT_COMBINE                    9

namespace anakin {
namespace saber {

template<class T>
inline
constexpr T max(const T& a, const T& b) {
    return a < b ? b : a;
}

template<class T>
inline
constexpr T min(const T& a, const T& b) {
    return a < b ? a : b;
}


template<class T>
inline
unsigned int ceiling_div(T a, T b) {
    return (a + b - 1) / b;
}

inline
size_t align_size(size_t size, size_t alignment = 256) {
    return ceiling_div(size, alignment) * alignment;
}

template<unsigned int BlockSize, unsigned int ItemsPerThread>
struct kernel_config {
    static const unsigned int block_size = BlockSize;
    static const unsigned int items_per_thread = ItemsPerThread;
};

template <
unsigned int LongRadixBits,
         unsigned int ShortRadixBits,
         class ScanConfig,
         class SortConfig
         >
struct radix_sort_config {
    static const unsigned int long_radix_bits = LongRadixBits;
    static const unsigned int short_radix_bits = ShortRadixBits;
    using scan = ScanConfig;
    using sort = SortConfig;
};

struct radix_sort_paramter {
    unsigned int batches;

    unsigned int blocks_per_full_batch;
    unsigned int full_batches;

    unsigned int short_iterations;
    unsigned int long_iterations;
};

template<DataType valueType, typename TargetType_D>
void radix_sort_genIndexBuffer(Tensor<TargetType_D>* keys_input,
                               Tensor<TargetType_D>& values_input) {
    Shape keys_input_shape = keys_input->valid_shape();
    values_input.re_alloc(keys_input_shape, valueType);

    Tensor<AMDHX86> temp_tensor(values_input.valid_shape(), values_input.get_dtype());
    fill_tensor_seq(temp_tensor);

    values_input.copy_from(temp_tensor);
}

template<DataType keyType, DataType valueType, typename TargetType>
void radix_sort_genRadixSortBuffer(unsigned int size,
                                   Tensor<TargetType>& keys_output,
                                   Tensor<TargetType>& values_output,
                                   Tensor<TargetType>& batch_digit_counts,
                                   Tensor<TargetType>& digit_counts,
                                   Tensor<TargetType>& keys_tmp,
                                   Tensor<TargetType>& values_tmp,
                                   radix_sort_paramter& param,
                                   unsigned int begin_bit,
                                   unsigned int end_bit) {

    size_t keyType_len = type_length(keyType);
    size_t valueType_len = type_length(valueType);

    using config = radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 15>>;

    size_t storage_size;

    unsigned int max_radix_size = 1 << config::long_radix_bits;

    int tmp_block_size = config::scan::block_size;
    int tmp_scan_item_per_thread = config::scan::items_per_thread;
    int tmp_sort_items_per_thread = config::sort::items_per_thread;

    unsigned int scan_size = config::scan::block_size * config::scan::items_per_thread;
    unsigned int sort_size = config::sort::block_size * config::sort::items_per_thread;

    const unsigned int blocks = max(1u, ceiling_div(size, sort_size));
    const unsigned int blocks_per_full_batch = ceiling_div(blocks, scan_size);
    const unsigned int full_batches = blocks % scan_size != 0
                                      ? blocks % scan_size
                                      : scan_size;
    const unsigned int batches = (blocks_per_full_batch == 1 ? full_batches : scan_size);

    const unsigned int bits = end_bit - begin_bit;
    const unsigned int iterations = ceiling_div(bits, config::long_radix_bits);
    const unsigned int radix_bits_diff = config::long_radix_bits - config::short_radix_bits;
    unsigned int short_iterations = radix_bits_diff != 0
                                    ? min(iterations, (config::long_radix_bits * iterations - bits) / radix_bits_diff)
                                    : 0;
    unsigned int long_iterations = iterations - short_iterations;

    const size_t batch_digit_counts_bytes =
        align_size(batches * max_radix_size * sizeof(int));
    const size_t digit_counts_bytes = align_size(max_radix_size * sizeof(int));
    const size_t keys_bytes = align_size(size * keyType_len);
    const size_t values_bytes = align_size(size * valueType_len);

    keys_output.re_alloc(Shape({1, 1, keys_bytes / keyType_len, 1}), keyType);
    values_output.re_alloc(Shape({1, 1, values_bytes / valueType_len, 1}), valueType);

    batch_digit_counts.re_alloc(Shape({1, 1, batch_digit_counts_bytes / sizeof(int), 1}), AK_INT32);
    digit_counts.re_alloc(Shape({1, 1, digit_counts_bytes / sizeof(int), 1}), AK_INT32);

    keys_tmp.re_alloc(Shape({1, 1, keys_bytes / keyType_len, 1}), keyType);
    values_tmp.re_alloc(Shape({1, 1, values_bytes / valueType_len, 1}), valueType);

    param.batches = batches;
    param.blocks_per_full_batch = blocks_per_full_batch;
    param.full_batches = full_batches;
    param.short_iterations = short_iterations;
    param.long_iterations = long_iterations;
}

template<unsigned int RadixBits>
SaberStatus radix_sort_gen_kernel(std::unordered_map<int, AMDKernelPtr>&  kernel_map, int type,
                                  int device_id, radix_sort_paramter param) {
    int kernel_type;
    unsigned int radix_size = 1 << RadixBits;

    using config = radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 15>>;

    switch (type) {
    case 0: { //fill_digit_counts
        KernelInfo kernelInfo;
        kernelInfo.kernel_file = AMD_RADIX_SORT_KERNEL_PATH;

        if (RadixBits == 7) {
            kernel_type = RADIX_SORT_FILL_DIGIT_COUNTS_LONG_ITER;
            kernelInfo.kernel_name =
                "_ZN7rocprim6detail24fill_digit_counts_kernelILj256ELj15ELj7ELb0EPfEEvT3_jPjjjjj";
        } else {
            kernel_type = RADIX_SORT_FILL_DIGIT_COUNTS_SHORT_ITER;
            kernelInfo.kernel_name =
                "_ZN7rocprim6detail24fill_digit_counts_kernelILj256ELj15ELj6ELb0EPfEEvT3_jPjjjjj";
        }

        kernelInfo.wk_dim      = 3;
        kernelInfo.kernel_type = SOURCE;

        kernelInfo.g_wk        = {param.batches* config::sort::block_size, 1, 1};
        kernelInfo.l_wk        = {config::sort::block_size, 1, 1};

        AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        kernel_map[kernel_type] = kptr;
    }
    break;

    case 1: { //scan_batches
        KernelInfo kernelInfo;
        kernelInfo.kernel_file = AMD_RADIX_SORT_KERNEL_PATH;

        if (RadixBits == 7) {
            kernel_type = RADIX_SORT_SCAN_BATCH_LONG_ITER;
            kernelInfo.kernel_name = "_ZN7rocprim6detail19scan_batches_kernelILj256ELj2ELj7EEEvPjS2_j";
        } else {
            kernel_type = RADIX_SORT_SCAN_BATCH_SHORT_ITER;
            kernelInfo.kernel_name = "_ZN7rocprim6detail19scan_batches_kernelILj256ELj2ELj6EEEvPjS2_j";
        }

        kernelInfo.wk_dim      = 3;
        kernelInfo.kernel_type = SOURCE;

        kernelInfo.g_wk        = {radix_size* config::scan::block_size, 1, 1};
        kernelInfo.l_wk        = {config::scan::block_size, 1, 1};

        AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        kernel_map[kernel_type] = kptr;
    }
    break;

    case 2: { //scan_digits
        KernelInfo kernelInfo;
        kernelInfo.kernel_file = AMD_RADIX_SORT_KERNEL_PATH;

        if (RadixBits == 7) {
            kernel_type = RADIX_SORT_SCAN_DIGITS_LONG_ITER;
            kernelInfo.kernel_name = "_ZN7rocprim6detail18scan_digits_kernelILj7EEEvPj";
        } else {
            kernel_type = RADIX_SORT_SCAN_DIGITS_SHORT_ITER;
            kernelInfo.kernel_name = "_ZN7rocprim6detail18scan_digits_kernelILj6EEEvPj";
        }

        kernelInfo.wk_dim      = 3;
        kernelInfo.kernel_type = SOURCE;

        kernelInfo.g_wk        = {radix_size, 1, 1};
        kernelInfo.l_wk        = {radix_size, 1, 1};

        AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        kernel_map[kernel_type] = kptr;
    }
    break;

    case 3: { //scan_scatter
        KernelInfo kernelInfo;
        kernelInfo.kernel_file = AMD_RADIX_SORT_KERNEL_PATH;

        if (RadixBits == 7) {
            kernel_type = RADIX_SORT_SORT_SCATTER_LONG_ITER;
            kernelInfo.kernel_name =
                "_ZN7rocprim6detail23sort_and_scatter_kernelILj256ELj15ELj7ELb0EPfS2_PiS3_EEvT3_T4_T5_T6_jPKjS9_jjjj";
        } else {
            kernel_type = RADIX_SORT_SORT_SCATTER_SHORT_ITER;
            kernelInfo.kernel_name =
                "_ZN7rocprim6detail23sort_and_scatter_kernelILj256ELj15ELj6ELb0EPfS2_PiS3_EEvT3_T4_T5_T6_jPKjS9_jjjj";
        }

        kernelInfo.wk_dim      = 3;
        kernelInfo.kernel_type = SOURCE;

        kernelInfo.g_wk        = {param.batches* config::sort::block_size, 1, 1};
        kernelInfo.l_wk        = {config::sort::block_size, 1, 1};

        AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        kernel_map[kernel_type] = kptr;
    }
    break;
    }
}

template<unsigned int RadixBits>
SaberStatus radix_sort_launch_kernel_t(std::unordered_map<int, AMDKernelPtr> kernel_map,
                                       cl_mem keys_input) {

    return SaberSuccess;
}

template<unsigned int RadixBits>
SaberStatus radix_sort_launch_kernel(std::unordered_map<int, AMDKernelPtr> kernel_map,
                                     cl_mem keys_input,
                                     cl_mem keys_tmp,
                                     cl_mem keys_output,
                                     cl_mem values_input,
                                     cl_mem values_tmp,
                                     cl_mem values_output,
                                     cl_mem batch_digit_counts,
                                     cl_mem digit_counts,
                                     unsigned int size,
                                     bool to_output,
                                     unsigned int bit,
                                     unsigned int begin_bit,
                                     unsigned int end_bit,
                                     radix_sort_paramter param,
                                     AMDStream_t cm) {

    amd_kernel_list kernels;

    unsigned int radix_size = 1 << RadixBits;
    unsigned int current_radix_bits = min(RadixBits, end_bit - bit);

    const bool is_first_iteration = (bit == begin_bit);

    bool err = false;

    AMDKernelPtr kptr;

    if (RadixBits == 7) {
        kptr = kernel_map[RADIX_SORT_FILL_DIGIT_COUNTS_LONG_ITER];
    } else {
        kptr = kernel_map[RADIX_SORT_FILL_DIGIT_COUNTS_SHORT_ITER];
    }

    if (is_first_iteration) {
        kptr->SetKernelArgs(
            (cl_mem)keys_input,
            (int)size,
            (cl_mem)batch_digit_counts,
            (int)bit,
            (int)current_radix_bits,
            (int)param.blocks_per_full_batch,
            (int)param.full_batches);
    } else {
        if (to_output) {
            kptr->SetKernelArgs(
                (cl_mem)keys_tmp,
                (int)size,
                (cl_mem)batch_digit_counts,
                (int)bit,
                (int)current_radix_bits,
                (int)param.blocks_per_full_batch,
                (int)param.full_batches);
        } else {
            kptr->SetKernelArgs(
                (cl_mem)keys_output,
                (int)size,
                (cl_mem)batch_digit_counts,
                (int)bit,
                (int)current_radix_bits,
                (int)param.blocks_per_full_batch,
                (int)param.full_batches);
        }
    }

    kernels.push_back(kptr);
    //err = LaunchKernel(cm, kernels);
    //kernels.clear();

    //if (!err) {
    //    LOG(ERROR) << "Fail to set execution fill digits";
    //    return SaberInvalidValue;
    //}

    if (RadixBits == 7) {
        kptr = kernel_map[RADIX_SORT_SCAN_BATCH_LONG_ITER];
    } else {
        kptr = kernel_map[RADIX_SORT_SCAN_BATCH_SHORT_ITER];
    }

    kptr->SetKernelArgs(
        (cl_mem)batch_digit_counts,
        (cl_mem)digit_counts,
        (int)param.batches);

    kernels.push_back(kptr);
    //err = LaunchKernel(cm, kernels);
    //kernels.clear();

    //if (!err) {
    //    LOG(ERROR) << "Fail to set execution scan batch";
    //    return SaberInvalidValue;
    //}

    if (RadixBits == 7) {
        kptr = kernel_map[RADIX_SORT_SCAN_DIGITS_LONG_ITER];
    } else {
        kptr = kernel_map[RADIX_SORT_SCAN_DIGITS_SHORT_ITER];
    }

    kptr->SetKernelArgs(
        (cl_mem)digit_counts);

    kernels.push_back(kptr);
    //err = LaunchKernel(cm, kernels);
    //kernels.clear();

    //if (!err) {
    //    LOG(ERROR) << "Fail to set execution scan digits";
    //    return SaberInvalidValue;
    //}

    if (RadixBits == 7) {
        kptr = kernel_map[RADIX_SORT_SORT_SCATTER_LONG_ITER];
    } else {
        kptr = kernel_map[RADIX_SORT_SORT_SCATTER_SHORT_ITER];
    }

    if (is_first_iteration) {
        if (to_output) {
            kptr->SetKernelArgs(
                (cl_mem)keys_input,
                (cl_mem)keys_output,
                (cl_mem)values_input,
                (cl_mem)values_output,
                (int)size,
                (cl_mem)batch_digit_counts,
                (cl_mem)digit_counts,
                (int)bit,
                (int)current_radix_bits,
                (int)param.blocks_per_full_batch,
                (int)param.full_batches);
        } else {
            kptr->SetKernelArgs(
                (cl_mem)keys_input,
                (cl_mem)keys_tmp,
                (cl_mem)values_input,
                (cl_mem)values_tmp,
                (int)size,
                (cl_mem)batch_digit_counts,
                (cl_mem)digit_counts,
                (int)bit,
                (int)current_radix_bits,
                (int)param.blocks_per_full_batch,
                (int)param.full_batches);
        }
    } else {
        if (to_output) {
            kptr->SetKernelArgs(
                (cl_mem)keys_tmp,
                (cl_mem)keys_output,
                (cl_mem)values_tmp,
                (cl_mem)values_output,
                (int)size,
                (cl_mem)batch_digit_counts,
                (cl_mem)digit_counts,
                (int)bit,
                (int)current_radix_bits,
                (int)param.blocks_per_full_batch,
                (int)param.full_batches);
        } else {
            kptr->SetKernelArgs(
                (cl_mem)keys_output,
                (cl_mem)keys_tmp,
                (cl_mem)values_output,
                (cl_mem)values_tmp,
                (int)size,
                (cl_mem)batch_digit_counts,
                (cl_mem)digit_counts,
                (int)bit,
                (int)current_radix_bits,
                (int)param.blocks_per_full_batch,
                (int)param.full_batches);
        }
    }

    kernels.push_back(kptr);
    err = LaunchKernel(cm, kernels);
    kernels.clear();

    if (!err) {
        LOG(ERROR) << "Fail to set execution sort and scatter";
        return SaberInvalidValue;
    }

    return SaberSuccess;
}

}// namespace saber
}// namesapce anakin
