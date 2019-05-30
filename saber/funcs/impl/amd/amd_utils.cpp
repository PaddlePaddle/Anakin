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
/*
   MIT License

   Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#include "saber/funcs/impl/amd/include/amd_utils.h"

namespace anakin {
namespace saber {
// so that MIOpen works whether or not recent MIOpenGEMM changes pulled:
// convert size_t and ulong kernel function parameters to unsigned.
namespace tempfix {
void add_bias_relu(std::string& clstr) {
    clstr = clstr.insert(clstr.find("miog_betac_alphaab") + 20,
                         "__constant TFLOAT * restrict bias,\nTFLOAT slope,");

    std::string search = "c[index] += alpha*rC";
    std::string sub_search1 = "c[index] += alpha*rC[dima][dimb]";
    std::string sub_search2 =
        "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb]";
    std::string sub_search3 =
        "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
        "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string sub_search4 =
        "c[index] += alpha*rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string add1 =
        "rC[dima][dimb] += bias[write_start_b + dimb];\nrC[dima][dimb] *= "
        "(rC[dima][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add2 =
        "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] += "
        "bias[write_start_b + "
        "dimb];\nrC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] *= "
        "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] > 0.0f ? 1.0f : "
        "slope);\n";
    std::string add3 = "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                       "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
                       "dimbi_v] += bias[write_start_b + "
                       "dimb];\nrC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                       "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
                       "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                       "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
                       "dimbi_v] > 0.0f ? 1.0f : slope);\n";
    std::string add4 =
        "rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] += "
        "bias[write_start_b + "
        "dimb];\nrC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
        "(rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > 0.0f ? 1.0f : "
        "slope);\n";

    for (size_t pos = clstr.find(search); pos != std::string::npos;
            pos = clstr.find(search, pos)) {
        size_t temp = clstr.find(sub_search2);

        if (clstr.find(sub_search1) != std::string::npos) {
            clstr = clstr.insert(pos, add1);
            pos += add1.length() + sub_search1.length();
        } else if (clstr.find(sub_search2) != std::string::npos) {
            clstr = clstr.insert(pos, add2);
            pos += add2.length() + sub_search2.length();
        } else if (clstr.find(sub_search3) != std::string::npos) {
            clstr = clstr.insert(pos, add3);
            pos += add3.length() + sub_search3.length();
        } else if (clstr.find(sub_search4) != std::string::npos) {
            clstr = clstr.insert(pos, add4);
            pos += add4.length() + sub_search4.length();
        } else {
            break;
        }
    }
}

void add_relu(std::string& clstr) {
    clstr = clstr.insert(clstr.find("miog_betac_alphaab") + 20, "TFLOAT slope,");

    std::string search = "c[index] += alpha*rC";
    std::string sub_search1 = "c[index] += alpha*rC[dima][dimb]";
    std::string sub_search2 =
        "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb]";
    std::string sub_search3 =
        "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
        "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string sub_search4 =
        "c[index] += alpha*rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string add1 =
        "rC[dima][dimb] *= (rC[dima][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add2 = "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] *= "
                       "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] > "
                       "0.0f ? 1.0f : slope);\n";
    std::string add3 = "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                       "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
                       "dimbi_v] *= (rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                       "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > "
                       "0.0f ? 1.0f : slope);\n";
    std::string add4 = "rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
                       "(rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > "
                       "0.0f ? 1.0f : slope);\n";

    for (size_t pos = clstr.find(search); pos != std::string::npos;
            pos = clstr.find(search, pos)) {
        size_t temp = clstr.find(sub_search2);

        if (clstr.find(sub_search1) != std::string::npos) {
            clstr = clstr.insert(pos, add1);
            pos += add1.length() + sub_search1.length();
        } else if (clstr.find(sub_search2) != std::string::npos) {
            clstr = clstr.insert(pos, add2);
            pos += add2.length() + sub_search2.length();
        } else if (clstr.find(sub_search3) != std::string::npos) {
            clstr = clstr.insert(pos, add3);
            pos += add3.length() + sub_search3.length();
        } else if (clstr.find(sub_search4) != std::string::npos) {
            clstr = clstr.insert(pos, add4);
            pos += add4.length() + sub_search4.length();
        } else {
            break;
        }
    }
}

void set_offsets_to_uint(std::string& clstr, int times) {
    for (int i = 0; i < times; i++) {
        clstr = clstr.replace(clstr.find("const ulong"), 11, "const uint");
    }
}
void set_offsets_to_uint(std::string& clstr) {
    auto get_target = [](std::string inttype, char x) {
        std::stringstream ss;
        ss << "const " << inttype << ' ' << std::string(1, x) << "_offset";
        return std::regex(ss.str());
    };

    for (char x : {
                'a', 'b', 'c'
            }) {
        std::string replacement = "const unsigned " + std::string(1, x) + "_offset";

        for (auto inttype : {
                    "size_t", "ulong"
                }) {
            clstr = std::regex_replace(clstr, get_target(inttype, x), replacement);
        }
    }
}
} // namespace tempfix

#define WG_SIZE 256
#define MAX_ACTIVE_THREADS (64 * 4 * 64)
//The below section of code are as MIT license, the permission notice is from above (line 16 to 36)
void Im2ColGPU(AMDKernelPtr& kptr, int device_id, int c_in,
               int h_in, int w_in, int h_wei, int w_wei, int h_out, int w_out,
               int h_pad, int w_pad, int h_stride, int w_stride, int h_dilation,
               int w_dilation) {
    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "MIOpenUtilKernels.cl";
    kernelInfo.kernel_name = "Im2Col";
    kernelInfo.kernel_type = MIOPEN;

    int tile_sz_x = 32;
    int tile_sz_y = 8;
    int num_im_blks_x = std::ceil(static_cast<float>(w_out) / tile_sz_x);
    int num_im_blks = num_im_blks_x * std::ceil(static_cast<float>(h_out) / tile_sz_y);
    int local_mem_size = 0;
    int local_mem_size_x = (tile_sz_x - 1) * w_stride + (w_wei - 1) * w_dilation + 1;
    int local_mem_size_y = (tile_sz_y - 1) * h_stride + (h_wei - 1) * h_dilation + 1;
    int num_ch_per_wg = 0;

    if ((c_in % 4 == 0) && (h_out <= 8 && w_out <= 8) && (h_stride == 1 && w_stride == 1)) {
        num_ch_per_wg = 4;
    } else {
        num_ch_per_wg = 1;
    }

    if (num_ch_per_wg != 1) {
        local_mem_size = std::max(
                             local_mem_size_x *
                             ((std::floor(static_cast<float>(tile_sz_y) / num_ch_per_wg)) *
                              h_stride + (h_wei - 1) * h_dilation + 1) *
                             num_ch_per_wg,
                             local_mem_size_y *
                             ((std::floor(static_cast<float>(tile_sz_x) / num_ch_per_wg)) *
                              w_stride + (w_wei - 1) * w_dilation + 1) *
                             num_ch_per_wg);
    } else {
        local_mem_size = local_mem_size_x * local_mem_size_y;
    }

    kernelInfo.comp_options = "";
    kernelInfo.comp_options += " -DLOCAL_MEM_SIZE=" + std::to_string(local_mem_size);
    kernelInfo.comp_options += " -DNUM_IM_BLKS=" + std::to_string(num_im_blks);
    kernelInfo.comp_options += " -DSTRIDE_GT_1=" +
                               std::to_string(static_cast<int>(h_stride * w_stride > 1));
    kernelInfo.comp_options += " -DTILE_SZ_X=" + std::to_string(tile_sz_x);
    kernelInfo.comp_options += " -DTILE_SZ_Y=" + std::to_string(tile_sz_y);
    kernelInfo.comp_options += " -DNUM_CH_PER_WG=" + std::to_string(num_ch_per_wg);
    kernelInfo.comp_options += " -DNUM_IM_BLKS_X=" + std::to_string(num_im_blks_x);
    kernelInfo.comp_options += " -DUSE_IM_OFF_GUARD=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1";

    kernelInfo.l_wk = {256, 1, 1};
    kernelInfo.g_wk = {256 * std::max(1, (c_in / num_ch_per_wg))* num_im_blks, 1, 1};

    kptr = CreateKernel(device_id, &kernelInfo);
}

void transpose_NCHW2CNHW(AMDKernelPtr& kptr,
                         int device_id, int n, int c, int h_in, int w_in,
                         int h_out, int w_out, int in_offset, int out_offset,
                         int h_stride, int w_stride) {
    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "MIOpenUtilKernels4.cl";

    if (h_stride == 1 && w_stride == 1) {
        kernelInfo.kernel_name = "transpose_NCHW2CNHW_opt";
        kernelInfo.kernel_type = MIOPEN;

        int RD_BLCK =
            ((h_in * w_in) % 4 == 0) ? 4 : ((h_in * w_in) % 2 == 0) ? 2 : 1;
        int HW_RD = (h_in * w_in) / RD_BLCK;
        size_t MAP_RD = HW_RD * c;
        size_t lcl_size0 =
            WG_SIZE; //((MAP_RD + 63)/64 < 4) ? ((MAP_RD + 63)/64)*64 : 256;

        std::string READ_TYPE =
            (RD_BLCK == 1) ? "float" : "float" + std::to_string(RD_BLCK);

        kernelInfo.comp_options = "";
        kernelInfo.comp_options += " -DNC_TRANS_NCHW_OPT";
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);
        kernelInfo.comp_options += " -DH=" + std::to_string(h_in);
        kernelInfo.comp_options += " -DW=" + std::to_string(w_in);
        kernelInfo.comp_options += " -DN=" + std::to_string(n);
        kernelInfo.comp_options += " -DC=" + std::to_string(c);
        kernelInfo.comp_options += " -DRD_BLCK=" + std::to_string(RD_BLCK);
        kernelInfo.comp_options += " -DHW_RD=" + std::to_string(HW_RD);
        kernelInfo.comp_options += " -DMAP_RD=" + std::to_string(MAP_RD);
        kernelInfo.comp_options += " -DREAD_TYPE=" + READ_TYPE;

        kernelInfo.l_wk = {lcl_size0, 1, 1};
        kernelInfo.g_wk = {MAP_RD, 1, 1};

        if (MAP_RD < MAX_ACTIVE_THREADS) {
            kernelInfo.g_wk = {MAP_RD, static_cast<size_t>(n), 1};
            kernelInfo.comp_options += " -D_2D_WG";
            kernelInfo.wk_dim = 3;
        }

        kptr = CreateKernel(device_id, &kernelInfo);

    } else {
        kernelInfo.kernel_name = "transpose_NCHW2CNHW";
        kernelInfo.kernel_type = MIOPEN;

        kernelInfo.comp_options = "";
        kernelInfo.comp_options += " -DNC_TRANS_NCHW";
        kernelInfo.comp_options += " -DN=" + std::to_string(n);
        kernelInfo.comp_options += " -DC=" + std::to_string(c);
        kernelInfo.comp_options += " -DHW_IN=" + std::to_string(h_in * w_in);
        kernelInfo.comp_options += " -DHW_OUT=" + std::to_string(h_out * w_out);
        kernelInfo.comp_options += " -DW_IN=" + std::to_string(w_in);
        kernelInfo.comp_options += " -DW_OUT=" + std::to_string(w_out);
        kernelInfo.comp_options += " -DH_STRIDE=" + std::to_string(h_stride);
        kernelInfo.comp_options += " -DW_STRIDE=" + std::to_string(w_stride);
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);

        size_t ld0 = WG_SIZE;
        size_t gd0 = c * h_out * w_out;
        kernelInfo.l_wk = {ld0, 1, 1};
        kernelInfo.g_wk = {gd0, 1, 1};

        if (gd0 < MAX_ACTIVE_THREADS) {
            kernelInfo.g_wk = {gd0, static_cast<size_t>(n), 1};
            kernelInfo.comp_options += " -D_2D_WG";
            kernelInfo.wk_dim = 3;
        }

        kptr = CreateKernel(device_id, &kernelInfo);
    }
}

void transpose_CNHW2NCHW(AMDKernelPtr& kptr,
                         int device_id, int n, int c, int h_out, int w_out,
                         int h_in, int w_in, int in_offset, int out_offset,
                         int h_stride, int w_stride, bool isBias) {
    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "MIOpenUtilKernels4.cl";

    if (h_stride == 1 && w_stride == 1) {
        kernelInfo.kernel_name = "transpose_CNHW2NCHW_opt_bias_prelu";
        kernelInfo.kernel_type = MIOPEN;

        int RD_BLCK =
            ((h_out * w_out) % 4 == 0) ? 4 : ((h_out * w_out) % 2 == 0) ? 2 : 1;
        int HW_RD = (h_out * w_out) / RD_BLCK;
        size_t MAP_RD = HW_RD * c;
        size_t lcl_size0 =
            WG_SIZE; //((MAP_RD + 63)/64 < 4) ? ((MAP_RD + 63)/64)*64 : 256;

        std::string READ_TYPE =
            (RD_BLCK == 1) ? "float" : "float" + std::to_string(RD_BLCK);

        kernelInfo.comp_options = "";
        kernelInfo.comp_options += " -DNC_TRANS_CNHW_OPT";
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);
        kernelInfo.comp_options += " -DH=" + std::to_string(h_out);
        kernelInfo.comp_options += " -DW=" + std::to_string(w_out);
        kernelInfo.comp_options += " -DN=" + std::to_string(n);
        kernelInfo.comp_options += " -DC=" + std::to_string(c);
        kernelInfo.comp_options += " -DRD_BLCK=" + std::to_string(RD_BLCK);
        kernelInfo.comp_options += " -DHW_RD=" + std::to_string(HW_RD);
        kernelInfo.comp_options += " -DMAP_RD=" + std::to_string(MAP_RD);
        kernelInfo.comp_options += " -DREAD_TYPE=" + READ_TYPE;

        if (isBias) {
            kernelInfo.comp_options += " -DBIAS";
        }

        kernelInfo.l_wk = {lcl_size0, 1, 1};
        kernelInfo.g_wk = {MAP_RD, 1, 1};

        if (MAP_RD < MAX_ACTIVE_THREADS) {
            kernelInfo.g_wk = {MAP_RD, static_cast<size_t>(n), 1};
            kernelInfo.comp_options += " -D_2D_WG";
            kernelInfo.wk_dim = 3;
        }

        kptr = CreateKernel(device_id, &kernelInfo);
    } else {
        kernelInfo.kernel_name = "transpose_CNHW2NCHW";
        kernelInfo.kernel_type = MIOPEN;

        kernelInfo.comp_options = "";
        kernelInfo.comp_options += " -DNC_TRANS_CNHW";
        kernelInfo.comp_options += " -DN=" + std::to_string(n);
        kernelInfo.comp_options += " -DC=" + std::to_string(c);
        kernelInfo.comp_options += " -DHW_IN=" + std::to_string(h_in * w_in);
        kernelInfo.comp_options += " -DHW_OUT=" + std::to_string(h_out * w_out);
        kernelInfo.comp_options += " -DW_IN=" + std::to_string(w_in);
        kernelInfo.comp_options += " -DW_OUT=" + std::to_string(w_out);
        kernelInfo.comp_options += " -DH_STRIDE=" + std::to_string(h_stride);
        kernelInfo.comp_options += " -DW_STRIDE=" + std::to_string(w_stride);
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);

        size_t ld0 = WG_SIZE;
        size_t gd0 = c * h_out * w_out;

        kernelInfo.l_wk = {ld0, 1, 1};
        kernelInfo.g_wk = {gd0, 1, 1};

        if (gd0 < MAX_ACTIVE_THREADS) {
            kernelInfo.g_wk = {gd0, static_cast<size_t>(n), 1};
            kernelInfo.comp_options += " -D_2D_WG";
            kernelInfo.wk_dim = 3;
        }

        kptr = CreateKernel(device_id, &kernelInfo);
    }
}

static bool tryPoolingGeneral(KernelInfo& kernelInfo,
                              int bt_size, int in_h, int in_w, int in_c, int out_h, int out_w,
                              int out_c, int pooling_w_h, int pooling_w_w, int pooling_s_h,
                              int pooling_s_w, int pooling_p_h, int pooling_p_w,
                              int pooling_type, int pooling_global, bool isBias, bool isActive) {
    if (pooling_w_h * pooling_w_w < 32) {
        return false;
    }

    if ((pooling_w_h > pooling_s_h || pooling_w_w > pooling_s_w)
            && out_h * out_w != 1) {
        return false;
    }

    int output_size = bt_size * out_c * out_h * out_w;

    int group_size   = 256;
    int group_size_0 = 256;  // adder

    while (group_size_0 * 8 > pooling_w_h * pooling_w_w && group_size_0 > 1) {
        group_size_0 = group_size_0 >> 1;
    }

    int group_size_1 = group_size / group_size_0;

    int global_size_0 = group_size_0;
    int global_size_1 = (output_size + group_size_1 - 1) / group_size_1 * group_size_1;

    kernelInfo.wk_dim      = 3;
    kernelInfo.l_wk        = {group_size_0, group_size_1, 1};
    kernelInfo.g_wk        = {global_size_0, global_size_1, 1};
    kernelInfo.kernel_file = "PoolingGeneral.cl";
    kernelInfo.kernel_name = "PoolingGeneral";
    kernelInfo.kernel_type = SABER;

    kernelInfo.comp_options = std::string(" -DGROUP_SIZE=") + std::to_string(group_size)
                              + std::string(" -DGROUP_SIZE_0=") + std::to_string(group_size_0)
                              + std::string(" -DGROUP_SIZE_1=") + std::to_string(group_size_1)
                              + std::string(" -DPOOLING_TYPE=") + std::to_string(pooling_type)
                              + std::string(" -DADDER=") + std::to_string(group_size_0)
                              + std::string(" -DMLO_CONV_BIAS=") + std::to_string(isBias)
                              + std::string(" -DMLO_CONV_PRELU=") + std::to_string(isActive);

    return true;
}

static bool tryPoolingWithShare(KernelInfo& kernelInfo,
                                int bt_size, int in_h, int in_w, int in_c, int out_h, int out_w,
                                int out_c, int pooling_w_h, int pooling_w_w, int pooling_s_h,
                                int pooling_s_w, int pooling_p_h, int pooling_p_w,
                                int pooling_type, int pooling_global, bool isBias, bool isActive) {
    if (bt_size * out_c * out_h * out_w < 1024 * 32 * 32) {
        return false;
    }

    if (pooling_w_h <= pooling_s_h || pooling_w_w <= pooling_s_w) {
        return false;
    }

    int group_size_1 = 256;
    int group_size_0 = 256;

    while (group_size_1 * 2 > out_h && group_size_1 > 1) {
        group_size_1 = group_size_1 >> 1;
    }

    while (group_size_0 * 2 > out_w && group_size_0 > 1) {
        group_size_0 = group_size_0 >> 1;
    }

    while (group_size_0 * group_size_1 > 256) {
        if (group_size_0 > group_size_1) {
            group_size_0 = group_size_0 >> 1;
        } else {
            group_size_1 = group_size_1 >> 1;
        }
    }

    int y_tile = (out_h + group_size_1 - 1) / group_size_1;
    int x_tile = (out_w + group_size_0 - 1) / group_size_0;

    int y_cache_size = (group_size_1 - 1) * pooling_s_h + pooling_w_h;
    int x_cache_size = (group_size_0 - 1) * pooling_s_w + pooling_w_w;

    if (group_size_0 * group_size_1 < 64) {
        return false;
    }

    if (y_cache_size * x_cache_size > 2000) {
        return false;
    }

    kernelInfo.wk_dim      = 3;
    kernelInfo.l_wk        = {group_size_0, group_size_1, 1};
    kernelInfo.g_wk        = {x_tile * group_size_0, y_tile * group_size_1, bt_size * out_c};
    kernelInfo.kernel_file = "PoolingWithShare.cl";
    kernelInfo.kernel_name = "PoolingWithShare";
    kernelInfo.kernel_type = SABER;

    kernelInfo.comp_options = std::string(" -DGROUP_SIZE_0=") + std::to_string(group_size_0)
                              + std::string(" -DGROUP_SIZE_1=") + std::to_string(group_size_1)
                              + std::string(" -DPOOLING_TYPE=") + std::to_string(pooling_type)
                              + std::string(" -DCACHE_SIZE_0=") + std::to_string(x_cache_size)
                              + std::string(" -DCACHE_SIZE_1=") + std::to_string(y_cache_size)
                              + std::string(" -DMLO_CONV_BIAS=") + std::to_string(isBias)
                              + std::string(" -DMLO_CONV_PRELU=") + std::to_string(isActive);

    return true;
}

static bool tryPoolingGen(KernelInfo& kernelInfo,
                          int bt_size, int in_h, int in_w, int in_c, int out_h, int out_w,
                          int out_c, int pooling_w_h, int pooling_w_w, int pooling_s_h,
                          int pooling_s_w, int pooling_p_h, int pooling_p_w,
                          int pooling_type, int pooling_global, bool isBias, bool isActive) {
    int _grp_tile0 = 8;
    int _grp_tile1 = 8;

    int _out_pix_tile0 = std::max(1, 8 / pooling_s_w);
    int _out_pix_tile1 = std::max(1, 8 / pooling_s_h);

    kernelInfo.wk_dim      = 3;
    kernelInfo.l_wk        = {256, 1, 1};
    kernelInfo.g_wk        = {64 * 64 * 40, 1, 1};
    kernelInfo.kernel_file = "PoolingGen.cl";
    kernelInfo.kernel_name = "mloPooling";
    kernelInfo.kernel_type = SABER;

    kernelInfo.comp_options =
        std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(pooling_type) +
        std::string(" -DMLO_POOLING_KERNEL_SZ0=") + std::to_string(pooling_w_w) +
        std::string(" -DMLO_POOLING_KERNEL_SZ1=") + std::to_string(pooling_w_h) +
        std::string(" -DMLO_POOLING_PAD0=") + std::to_string(pooling_p_w) +
        std::string(" -DMLO_POOLING_PAD1=") + std::to_string(pooling_p_h) +
        std::string(" -DMLO_POOLING_STRIDE0=") + std::to_string(pooling_s_w) +
        std::string(" -DMLO_POOLING_STRIDE1=") + std::to_string(pooling_s_h) +
        std::string(" -DMLO_POOLING_N_OUTPUTS=") + std::to_string(out_c) +
        std::string(" -DMLO_POOLING_N_CHANNELS=") + std::to_string(in_c) +
        std::string(" -DMLO_POOLING_N_HORIZ_OUT_PIX=") + std::to_string(_out_pix_tile0) +
        std::string(" -DMLO_POOLING_N_VERT_OUT_PIX=") + std::to_string(_out_pix_tile1) +
        std::string(" -DMLO_POOLING_GROUP_SZ0=") + std::to_string(_grp_tile0) +
        std::string(" -DMLO_POOLING_GROUP_SZ1=") + std::to_string(_grp_tile1) +
        std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(in_w) +
        std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(in_h) +
        std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(in_w) +
        std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(in_w * in_h) +
        std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") + std::to_string(in_w * in_h * in_c) +
        std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string(out_w) +
        std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string(out_h) +
        std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string(out_w) +
        std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string(out_w * out_h) +
        std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") + std::to_string(out_w * out_h * out_c) +
        std::string(" -DBATCH_NUM=") + std::to_string(bt_size) +
        std::string(" -DCU_NUM=64") +
        std::string(" -DMLO_CONV_BIAS=") + std::to_string(isBias) +
        std::string(" -DMLO_CONV_PRELU=") + std::to_string(isActive) +
        std::string(" -DMIOPEN_USE_FP32=1");

    return true;
}

void BiasReluPool(std::vector<AMDKernelPtr>& vkptr, int device_id, int bt_size,
                  int n_wei, int in_h, int in_w, int in_c, int out_h, int out_w,
                  int out_c, int pooling_w_h, int pooling_w_w, int pooling_s_h,
                  int pooling_s_w, int pooling_p_h, int pooling_p_w,
                  int pooling_type, int pooling_global, bool isBias, bool isActive) {
    AMDKernelPtr kptr;
    KernelInfo kernelInfo;

    if (pooling_w_h != 0 || pooling_w_w != 0) {
        if (tryPoolingGeneral(kernelInfo, bt_size, in_h, in_w, in_c, out_h, out_w, out_c,
                              pooling_w_h, pooling_w_w, pooling_s_h, pooling_s_w, pooling_p_h, pooling_p_w,
                              pooling_type, pooling_global, isBias, isActive)) {
            ; // tryPoolingGeneral will get kernel info
        } else if (tryPoolingWithShare(kernelInfo, bt_size, in_h, in_w, in_c, out_h, out_w, out_c,
                                       pooling_w_h, pooling_w_w, pooling_s_h, pooling_s_w, pooling_p_h, pooling_p_w,
                                       pooling_type, pooling_global, isBias, isActive)) {
            ; // tryPoolingWithShare will get kernel info
        } else if (tryPoolingGen(kernelInfo, bt_size, in_h, in_w, in_c, out_h, out_w, out_c,
                                 pooling_w_h, pooling_w_w, pooling_s_h, pooling_s_w, pooling_p_h, pooling_p_w,
                                 pooling_type, pooling_global, isBias, isActive)) {
            ; // tryPoolingGen will get kernel info
        } else {
            LOG(ERROR) << " no Pooling kernel is selected.";
        }

        // To create the program
        kptr = CreateKernel(device_id, &kernelInfo);
        vkptr.push_back(kptr);
    } else {
        if (isBias) {
            kernelInfo.kernel_file = "BiasReLuUni.cl";

            if (isActive) {
                kernelInfo.kernel_name = "BiasReluBoth";
            } else {
                kernelInfo.kernel_name = "BiasOnly";
            }

            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {bt_size* n_wei* out_h * out_w, 1, 1};

            kernelInfo.kernel_type = SABER;

            // To create the program
            kptr = CreateKernel(device_id, &kernelInfo);
            vkptr.push_back(kptr);
        } else {
            if (isActive) {
                kernelInfo.kernel_file = "ReluUni.cl";
                kernelInfo.kernel_name = "ReluUni";

                kernelInfo.l_wk = {256, 1, 1};
                kernelInfo.g_wk = {bt_size* n_wei* out_h * out_w, 1, 1};
                kernelInfo.kernel_type = SABER;

                // To create the program
                kptr = CreateKernel(device_id, &kernelInfo);
                vkptr.push_back(kptr);
            }
        }
    }
}

std::vector<KernelInfo> FindSolution(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param) {
    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; // anakin device id to AMD device
    cl_device_id device = dev.get_device();
    cl_context context  = dev.get_context();
    miopen::ConvolutionContext convContext;
    convContext.direction.Set(1);
    convContext.general_compile_options += "";
    convContext.n_inputs           = inputs[0]->channel();
    convContext.in_height          = inputs[0]->height();
    convContext.in_width           = inputs[0]->width();
    convContext.kernel_size0       = param.weight()->width();
    convContext.kernel_size1       = param.weight()->height();
    convContext.n_outputs          = param.weight()->num();
    convContext.out_height         = outputs[0]->height();
    convContext.out_width          = outputs[0]->width();
    convContext.batch_sz           = inputs[0]->num();
    convContext.pad0               = param.pad_w;
    convContext.pad1               = param.pad_h;
    convContext.group_counts       = param.group;
    convContext.kernel_stride0     = param.stride_w;
    convContext.kernel_stride1     = param.stride_h;
    convContext.kernel_dilation0   = param.dilation_w;
    convContext.kernel_dilation1   = param.dilation_h;
    convContext.bias               = (param.bias()->size() > 0) ? 1 : 0;
    convContext.float_size         = 32;
    convContext.in_stride          = inputs[0]->get_stride()[2];
    convContext.out_stride         = outputs[0]->get_stride()[2];
    convContext.in_channel_stride  = convContext.in_stride * convContext.in_height;
    convContext.in_batch_stride    = convContext.in_channel_stride * convContext.n_inputs;
    convContext.out_channel_stride = convContext.out_stride * convContext.out_height;
    convContext.out_batch_stride   = convContext.out_channel_stride * convContext.n_outputs;
    convContext.has_active         = param.activation_param.has_active ? 1 : 0;
    convContext.negative_slope =
        param.activation_param.has_active ? param.activation_param.negative_slope : 0;
    convContext.rmv             = rocm_meta_version::AMDHSA_1_0;
    convContext.use_binaries    = true;
    convContext.use_asm_kernels = true;
#ifdef ENABLE_AMD_DO_SEARCH
    convContext.do_search       = true;
#else
    convContext.do_search       = false;
#endif

#ifdef ENABLE_AMD_EXPAND_ALL_SEARCH
    convContext.do_all_search    = true;
#else
    convContext.do_all_search    = false;
#endif
    convContext.save_srch_req   = true;
    convContext.in_layout       = "NCHW";
    convContext.out_layout      = "NCHW";
    convContext.in_data_type    = "FP32";
    convContext.out_data_type   = "FP32";
    int data_len                = convContext.in_data_type == "FP32" ? 4 : 2;
    convContext.bot_sz = convContext.batch_sz * convContext.n_inputs * convContext.in_height
                         * convContext.in_width * data_len;
    convContext.top_sz = convContext.batch_sz * convContext.n_outputs * convContext.out_height
                         * convContext.out_width * data_len;
    convContext.weights_sz = convContext.n_outputs * convContext.n_inputs * convContext.kernel_size0
                             * convContext.kernel_size1 * data_len;
    convContext.bias_sz       = (param.bias()->size() > 0) ? convContext.n_outputs * data_len : 0;
    convContext.deconvolution = 0;
    convContext.general_compile_options = " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";

    miopen::Db db = anakin::saber::GetDb(dev._info._device_name, dev._info._compute_core_num);
    miopen::Handle::setClEnv(context, device);
    miopen::Handle handle;
    convContext.SetStream(&handle);
    miopen::solver::ConvSolution solution;

    if (convContext.group_counts > 1) {
        solution = miopen::solver::SearchForSolution <
                   // miopen::solver::ConvBinWinograd3x3U,
                   miopen::solver::ConvOclDirectFwd
                   > (convContext, db);
    } else {

#ifdef ENABLE_AMD_EXPAND_ALL_SEARCH
        auto candidate_solutions = miopen::solver::SearchForAllSolutions <
                                   miopen::solver::ConvBinWinograd3x3U,
                                   miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                                   miopen::solver::ConvOclDirectFwd1x1Tensile,
                                   miopen::solver::ConvOclDirectFwd1x1AMD,
                                   miopen::solver::ConvOclDirectFwd1x1Gemm,
                                   //miopen::solver::ConvAsm3x3U,
                                   //miopen::solver::ConvAsm1x1U,
                                   miopen::solver::ConvOclDirectFwdGen,
                                   miopen::solver::ConvOclDirectFwd3x3,
                                   miopen::solver::ConvOclDirectFwd1x1,
                                   miopen::solver::ConvOclDirectFwd > (convContext, db);
        //solution = candidate_solutions[0];
        double min_time = std::numeric_limits<float>::max();

        for (int i = 0; i < candidate_solutions.size(); i++) {
            auto tmp_solution = candidate_solutions[i];

            if (i == 0) {
                solution = tmp_solution;
            }

            if (min_time > tmp_solution.min_proc_time) {
                min_time = tmp_solution.min_proc_time;
                solution = tmp_solution;
            }
        }

#else

        if (param.weight()->width() == 1
                && param.weight()->height() == 1) {
            auto candidate_solutions = miopen::solver::SearchForAllSolutions <
                                       miopen::solver::ConvOclDirectFwd1x1Tensile,
                                       miopen::solver::ConvOclDirectFwd1x1AMD,
                                       miopen::solver::ConvOclDirectFwd1x1Gemm,
                                       miopen::solver::ConvOclDirectFwdGen,
                                       miopen::solver::ConvOclDirectFwd1x1,
                                       miopen::solver::ConvOclDirectFwd > (convContext, db);
            //solution = candidate_solutions[0];
            double min_time = std::numeric_limits<float>::max();

            for (int i = 0; i < candidate_solutions.size(); i++) {
                auto tmp_solution = candidate_solutions[i];

                if (i == 0) {
                    solution = tmp_solution;
                }

                if (min_time > tmp_solution.min_proc_time) {
                    min_time = tmp_solution.min_proc_time;
                    solution = tmp_solution;
                }
            }
        } else {
            solution = miopen::solver::SearchForSolution <
                       miopen::solver::ConvBinWinograd3x3U,
                       miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                       miopen::solver::ConvBinWinogradRxS,
                       // miopen::solver::ConvAsm3x3U,
                       // miopen::solver::ConvAsm1x1U,
                       miopen::solver::ConvOclDirectFwdGen,
                       miopen::solver::ConvOclDirectFwd3x3,
                       miopen::solver::ConvOclDirectFwd > (convContext, db);
        }

#endif

    }

    miopen::Handle::clearClEnv();
    std::vector<KernelInfo> solution_vector;
    KernelInfo kernelInfo;

    for (auto s : solution.construction_params) {
        kernelInfo = s;
        solution_vector.push_back(kernelInfo);
    }

    return solution_vector;
}

std::vector<KernelInfo> FindSolutionWithPooling(
    const std::vector<Tensor<AMD>*>& inputs,
    Tensor<AMD>*& workspace,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param) {
    cl_context context  = 0;
    cl_device_id device = 0;
    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; // anakin device id to AMD device
    device          = dev.get_device();
    context         = dev.get_context();

    std::vector<KernelInfo> solution_vector;
    KernelInfo kernelInfo;
    solution_vector.clear();
    int data_len;
    miopen::ConvolutionContext convContext;
    convContext.direction.Set(1);
#ifdef ENABLE_AMD_DO_SEARCH
    convContext.do_search        = true;
#else
    convContext.do_search        = false;
#endif

#ifdef ENABLE_AMD_EXPAND_ALL_SEARCH
    convContext.do_all_search    = true;
#else
    convContext.do_all_search    = false;
#endif

    convContext.general_compile_options += "";
    // context.SetStream(&profile_h);
    convContext.n_inputs         = inputs[0]->channel();
    convContext.in_height        = inputs[0]->height();
    convContext.in_width         = inputs[0]->width();
    convContext.kernel_size1     = param.conv_param.weight()->width();
    convContext.kernel_size0     = param.conv_param.weight()->height();
    convContext.n_outputs        = param.conv_param.weight()->num();
    convContext.out_height       = workspace->height();
    convContext.out_width        = workspace->width();
    convContext.batch_sz         = inputs[0]->num();
    convContext.pad0             = param.conv_param.pad_w;
    convContext.pad1             = param.conv_param.pad_h;
    convContext.group_counts     = param.conv_param.group;
    convContext.kernel_stride0   = param.conv_param.stride_h;
    convContext.kernel_stride1   = param.conv_param.stride_w;
    convContext.kernel_dilation0 = param.conv_param.dilation_w;
    convContext.kernel_dilation1 = param.conv_param.dilation_h;
    convContext.bias             = (param.conv_param.bias()->size() > 0) ? 1 : 0;;
    convContext.float_size       = 32;
    convContext.in_layout        = "NCHW";
    convContext.in_data_type     = "FP32";
    convContext.save_srch_req    = true;
    convContext.use_asm_kernels  = true;
    convContext.use_binaries     = true;
    convContext.weights_layout   = "";
    convContext.out_data_type    = "FP32";
    convContext.out_layout       = "NCHW";
    data_len                     = convContext.in_data_type == "FP32" ? 4 : 2;
    convContext.bot_sz = convContext.batch_sz * convContext.n_inputs * convContext.in_height
                         * convContext.in_width * data_len;
    convContext.top_sz = convContext.batch_sz * convContext.n_outputs * convContext.out_height
                         * convContext.out_width * data_len;
    convContext.weights_sz = convContext.n_outputs * convContext.n_inputs * convContext.kernel_size0
                             * convContext.kernel_size1 * data_len;
    convContext.bias_sz                 = outputs[0]->channel();
    convContext.deconvolution           = 0;
    convContext.in_stride               = inputs[0]->get_stride()[2];
    convContext.out_stride              = workspace->get_stride()[2];
    convContext.in_channel_stride       = convContext.in_stride * convContext.in_height;
    convContext.in_batch_stride         = convContext.in_channel_stride * convContext.n_inputs;
    convContext.out_channel_stride      = convContext.out_stride * convContext.out_height;
    convContext.out_batch_stride        = convContext.out_channel_stride * convContext.n_outputs;
    convContext.rmv                     = rocm_meta_version::AMDHSA_1_0;
    convContext.general_compile_options = " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";

    convContext.has_active = param.conv_param.activation_param.has_active;

    convContext.has_pooling               = true;
    convContext.poolingContext.batch_sz   = workspace->num();
    convContext.poolingContext.n_inputs   = workspace->channel();
    convContext.poolingContext.in_height  = workspace->height();
    convContext.poolingContext.in_width   = workspace->width();
    convContext.poolingContext.n_outputs  = outputs[0]->channel();
    convContext.poolingContext.out_height = outputs[0]->height();
    convContext.poolingContext.out_width  = outputs[0]->width();
    convContext.poolingContext.pooling_type   = param.pooling_param.pooling_type;
    convContext.poolingContext.pooling_global = param.pooling_param.global_pooling;
    convContext.poolingContext.pad1           = param.pooling_param.pad_h;
    convContext.poolingContext.pad0           = param.pooling_param.pad_w;
    convContext.poolingContext.kernel_size1   = param.pooling_param.window_h;
    convContext.poolingContext.kernel_size0   = param.pooling_param.window_w;
    convContext.poolingContext.kernel_stride1 = param.pooling_param.stride_h;
    convContext.poolingContext.kernel_stride0 = param.pooling_param.stride_w;

    miopen::Db db = anakin::saber::GetDb(dev._info._device_name, dev._info._compute_core_num);
    miopen::Handle::setClEnv(context, device);
    miopen::Handle handle /*(context, device)*/;
    convContext.SetStream(&handle);
    miopen::solver::ConvSolution solution;

    if (convContext.group_counts > 1) {
        solution = miopen::solver::SearchForSolution <miopen::solver::ConvOclDirectFwd > (convContext, db);
    } else {
#ifdef ENABLE_AMD_EXPAND_ALL_SEARCH
        auto candidate_solutions = miopen::solver::SearchForAllSolutions <
                                   miopen::solver::ConvBinWinograd3x3U,
                                   miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                                   miopen::solver::ConvOclDirectFwd1x1Tensile,
                                   miopen::solver::ConvOclDirectFwd1x1AMD,
                                   miopen::solver::ConvOclDirectFwd1x1Gemm,
                                   //miopen::solver::ConvAsm3x3U,
                                   //miopen::solver::ConvAsm1x1U,
                                   miopen::solver::ConvOclDirectFwdGen,
                                   miopen::solver::ConvOclDirectFwd3x3,
                                   miopen::solver::ConvOclDirectFwd1x1,
                                   miopen::solver::ConvOclDirectFwd > (convContext, db);
        //solution = candidate_solutions[0];
        double min_time = std::numeric_limits<float>::max();

        for (int i = 0; i < candidate_solutions.size(); i++) {
            auto tmp_solution = candidate_solutions[i];

            if (i == 0) {
                solution = tmp_solution;
            }

            if (min_time > tmp_solution.min_proc_time) {
                min_time = tmp_solution.min_proc_time;
                solution = tmp_solution;
            }
        }

#else

        if (param.conv_param.weight()->width() == 1
                && param.conv_param.weight()->height() == 1) {
            auto candidate_solutions = miopen::solver::SearchForAllSolutions <
                                       miopen::solver::ConvOclDirectFwd1x1Tensile,
                                       miopen::solver::ConvOclDirectFwd1x1AMD,
                                       miopen::solver::ConvOclDirectFwd1x1Gemm,
                                       miopen::solver::ConvOclDirectFwdGen,
                                       miopen::solver::ConvOclDirectFwd1x1,
                                       miopen::solver::ConvOclDirectFwd > (convContext, db);
            //solution = candidate_solutions[0];
            double min_time = std::numeric_limits<float>::max();

            for (int i = 0; i < candidate_solutions.size(); i++) {
                auto tmp_solution = candidate_solutions[i];

                if (i == 0) {
                    solution = tmp_solution;
                }

                if (min_time > tmp_solution.min_proc_time) {
                    min_time = tmp_solution.min_proc_time;
                    solution = tmp_solution;
                }
            }
        } else {
            solution = miopen::solver::SearchForSolution <
                       miopen::solver::ConvBinWinograd3x3U,
                       miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                       miopen::solver::ConvBinWinogradRxS,
                       // miopen::solver::ConvAsm3x3U,
                       // miopen::solver::ConvAsm1x1U,
                       miopen::solver::ConvOclDirectFwdGen,
                       miopen::solver::ConvOclDirectFwd3x3,
                       miopen::solver::ConvOclDirectFwd > (convContext, db);
        }

#endif
    }

    miopen::Handle::clearClEnv();

    for (auto s : solution.construction_params) {
        kernelInfo = s;
        solution_vector.push_back(kernelInfo);
    }

    return solution_vector;
}

std::vector<KernelInfo> FindDeconvSolution(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param) {
    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; // anakin device id to AMD device
    cl_device_id device = dev.get_device();
    cl_context context  = dev.get_context();
    miopen::ConvolutionContext convContext;
    convContext.direction.Set(0); //backward solutions in MIOpen
    convContext.general_compile_options += "";
    convContext.n_inputs           = inputs[0]->channel();
    convContext.in_height          = inputs[0]->height();
    convContext.in_width           = inputs[0]->width();
    convContext.kernel_size0       = param.weight()->width();
    convContext.kernel_size1       = param.weight()->height();
    convContext.n_outputs          = param.weight()->num() * param.group;
    convContext.out_height         = outputs[0]->height();
    convContext.out_width          = outputs[0]->width();
    convContext.batch_sz           = inputs[0]->num();
    convContext.pad0               = param.pad_w;
    convContext.pad1               = param.pad_h;
    convContext.group_counts       = param.group;
    convContext.kernel_stride0     = param.stride_w;
    convContext.kernel_stride1     = param.stride_h;
    convContext.kernel_dilation0   = param.dilation_w;
    convContext.kernel_dilation1   = param.dilation_h;
    convContext.bias               = (param.bias()->size() > 0) ? 1 : 0;
    convContext.float_size         = 32;
    convContext.in_stride          = inputs[0]->get_stride()[2];
    convContext.out_stride         = outputs[0]->get_stride()[2];
    convContext.in_channel_stride  = convContext.in_stride * convContext.in_height;
    convContext.in_batch_stride    = convContext.in_channel_stride * convContext.n_inputs;
    convContext.out_channel_stride = convContext.out_stride * convContext.out_height;
    convContext.out_batch_stride   = convContext.out_channel_stride * convContext.n_outputs;
    convContext.has_active         = param.activation_param.has_active ? 1 : 0;
    convContext.negative_slope =
        param.activation_param.has_active ? param.activation_param.negative_slope : 0;
    convContext.rmv             = rocm_meta_version::AMDHSA_1_0;
    convContext.use_binaries    = true;
    convContext.use_asm_kernels = true;
#ifdef ENABLE_AMD_DO_SEARCH
    convContext.do_search       = true;
#else
    convContext.do_search       = false;
#endif
    convContext.save_srch_req   = true;
    convContext.in_layout       = "NCHW";
    convContext.out_layout      = "NCHW";
    convContext.in_data_type    = "FP32";
    convContext.out_data_type   = "FP32";
    int data_len                = convContext.in_data_type == "FP32" ? 4 : 2;
    convContext.bot_sz = convContext.batch_sz * convContext.n_inputs * convContext.in_height
                         * convContext.in_width * data_len;
    convContext.top_sz = convContext.batch_sz * convContext.n_outputs * convContext.out_height
                         * convContext.out_width * data_len;
    convContext.weights_sz = convContext.n_outputs * convContext.n_inputs * convContext.kernel_size0
                             * convContext.kernel_size1 * data_len;
    convContext.bias_sz       = (param.bias()->size() > 0) ? convContext.n_outputs * data_len : 0;
    convContext.deconvolution = 0;
    convContext.general_compile_options = " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";

    miopen::Db db = anakin::saber::GetDb(dev._info._device_name, dev._info._compute_core_num);
    miopen::Handle::setClEnv(context, device);
    miopen::Handle handle;
    convContext.SetStream(&handle);
    miopen::solver::ConvSolution solution;

    std::vector<KernelInfo> solution_vector;
#ifdef ENABLE_AMD_EXPAND_ALL_SEARCH
    auto candidate_solutions = miopen::solver::SearchForAllSolutions <
                               miopen::solver::ConvBinWinogradRxS,
                               miopen::solver::ConvOclDirectFwd > (convContext, db);
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "candidate_solutions.size()=" << candidate_solutions.size();

    if (candidate_solutions.size() > 0) {
        double min_proc_time = std::numeric_limits<double>::max();
        int solution_index = 0;

        for (int i = 0; i < candidate_solutions.size(); i++) {
            if (candidate_solutions[i].min_proc_time < min_proc_time) {
                min_proc_time = candidate_solutions[i].min_proc_time;
                solution_index = i;
            }

            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "candidate_solutions[" << i << "]=" <<
                                                 candidate_solutions[i].construction_params[0].kernel_name;
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "min_proc_time=" << candidate_solutions[i].min_proc_time;
        }

        solution = candidate_solutions[solution_index];

    }

#else
    solution = miopen::solver::SearchForSolution <
               miopen::solver::ConvBinWinogradRxS,
               miopen::solver::ConvOclDirectFwd > (convContext, db);

#endif
    miopen::Handle::clearClEnv();
    KernelInfo kernelInfo;

    for (auto s : solution.construction_params) {
        kernelInfo = s;
        solution_vector.push_back(kernelInfo);
    }

    return solution_vector;
}

} // namespace saber
} // namespace anakin
