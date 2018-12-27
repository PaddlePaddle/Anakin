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

void Im2ColGPU(AMDKernelPtr& kptr, int device_id, int c,
               int h, int w, int wei_h, int wei_w, int out_h, int out_w,
               int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
               int dilation_w) {
    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "MIOpenUtilKernels.cl";
    kernelInfo.kernel_name = "Im2Col";
    kernelInfo.kernel_type = MIOPEN;

    std::string params;
    int num_ch_per_wg;

    if ((out_h <= 8 && out_w <= 8) && (stride_h == 1 && stride_w == 1) &&
            (c % 4 == 0)) {
        num_ch_per_wg = 4;
    } else {
        num_ch_per_wg = 1;
    }

    int tile_sz_x = 32;
    int tile_sz_y = 8;
    int num_blks_x = std::ceil(static_cast<float>(out_w) / tile_sz_x);
    int num_blks = num_blks_x * std::ceil(static_cast<float>(out_h) / tile_sz_y);
    int local_mem_sz;

    if (num_ch_per_wg == 1)
        local_mem_sz = ((tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1) *
                       ((tile_sz_y - 1) * stride_h + (wei_h - 1) * dilation_h + 1);
    else
        local_mem_sz = std::max(
                           num_ch_per_wg *
                           ((std::ceil(static_cast<float>(tile_sz_x) / num_ch_per_wg) - 1) *
                            stride_w +
                            (wei_w - 1) * dilation_w + 1) *
                           ((tile_sz_y - 1) * stride_h + (wei_h - 1) * dilation_h + 1),
                           num_ch_per_wg *
                           ((tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1) *
                           ((std::ceil(static_cast<float>(tile_sz_y) / num_ch_per_wg) - 1) *
                            stride_h +
                            (wei_h - 1) * dilation_h + 1));

    // int data_size_off = data_size - im_offset;

    params += " -DNUM_CH_PER_WG=" + std::to_string(num_ch_per_wg);
    params += " -DNUM_IM_BLKS_X=" + std::to_string(num_blks_x);
    params += " -DNUM_IM_BLKS=" + std::to_string(num_blks);
    params += " -DLOCAL_MEM_SIZE=" + std::to_string(local_mem_sz);
    params += " -DSTRIDE_GT_1=" +
              std::to_string(static_cast<int>(stride_h * stride_w > 1));
    params += " -DTILE_SZ_X=" + std::to_string(tile_sz_x);
    params += " -DTILE_SZ_Y=" + std::to_string(tile_sz_y);
    params += " -DUSE_IM_OFF_GUARD=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1";

    kernelInfo.l_wk = {256, 1, 1};
    kernelInfo.g_wk = {256 * std::max(1, (c / num_ch_per_wg))* num_blks, 1, 1};

    kernelInfo.comp_options = params;

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
        //kernelInfo.comp_options += " -DHW_OUT=" + std::to_string(h_out * w_out);
        kernelInfo.comp_options += " -DHW_OUT=" + std::to_string((h_in / h_stride) * (w_in / w_stride));
        kernelInfo.comp_options += " -DW_IN=" + std::to_string(w_in);
        kernelInfo.comp_options += " -DW_OUT=" + std::to_string((w_in / w_stride));
        kernelInfo.comp_options += " -DH_STRIDE=" + std::to_string(h_stride);
        kernelInfo.comp_options += " -DW_STRIDE=" + std::to_string(w_stride);
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);

        size_t ld0 = WG_SIZE;
        size_t gd0 = c * (h_in / h_stride) * (w_in / w_stride);
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

void BiasReluPool(std::vector<AMDKernelPtr>& vkptr, int device_id, int bt_size,
                  int n_wei, int in_h, int in_w, int in_c, int out_h, int out_w,
                  int out_c, int pooling_w_h, int pooling_w_w, int pooling_s_h,
                  int pooling_s_w, int pooling_p_h, int pooling_p_w,
                  int pooling_type, bool isBias, bool isActive) {
    AMDKernelPtr kptr;
    KernelInfo kernelInfo;

    if (pooling_w_h != 0 || pooling_w_w != 0) {
        int _grp_tile0 = 8;
        int _grp_tile1 = 8;

        int _out_pix_tile0 = std::max(1, 8 / pooling_s_w);
        int _out_pix_tile1 = std::max(1, 8 / pooling_s_h);

        if (pooling_w_h == 2 && pooling_w_w == 2) {
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {64 * 64 * 40, 1, 1};
            kernelInfo.kernel_file = "BiasReLuPooling.cl";
            kernelInfo.kernel_name = "mloPooling";
            kernelInfo.kernel_type = SABER;
        } else {
            // Bias relu kernel
            while (_out_pix_tile0 * _grp_tile0 > out_w * 2 && _out_pix_tile0 > 1) {
                _out_pix_tile0 >>= 1;
            }

            while (_out_pix_tile1 * _grp_tile1 > out_h * 2 && _out_pix_tile1 > 1) {
                _out_pix_tile1 >>= 1;
            }

            int g_wk_width = ((out_w + _grp_tile0 * _out_pix_tile0 - 1) /
                              (_grp_tile0 * _out_pix_tile0));
            int g_wk_height = ((out_h + _grp_tile1 * _out_pix_tile1 - 1) /
                               (_grp_tile1 * _out_pix_tile1));

            kernelInfo.l_wk = {_grp_tile0, _grp_tile1, 1};
            kernelInfo.g_wk = {g_wk_width * _grp_tile0,
                               g_wk_height * _grp_tile1,
                               out_c* bt_size
                              };
            kernelInfo.kernel_file = "MIOpenPooling.cl";
            kernelInfo.kernel_name = "mloPoolingG";
            kernelInfo.kernel_type = MIOPEN;
            kernelInfo.wk_dim = 3;
        }

        int ptype = MLO_POOLING_OP_MAX;

        if (pooling_type == Pooling_max) {
            ptype = MLO_POOLING_OP_MAX;
        } else if (pooling_type == Pooling_average_exclude_padding ||
                   pooling_type == Pooling_average_include_padding) {
            ptype = MLO_POOLING_OP_AVE;
        }

        kernelInfo.comp_options =
            std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(ptype) +
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
        solution = miopen::solver::SearchForSolution <miopen::solver::ConvOclDirectFwd > (convContext, db);
    } else {
        solution = miopen::solver::SearchForSolution <
                   miopen::solver::ConvBinWinograd3x3U,
                   miopen::solver::ConvOclDirectFwd1x1AMD,
                   // miopen::solver::ConvAsm3x3U,
                   // miopen::solver::ConvAsm1x1U,
                   miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                   miopen::solver::ConvOclDirectFwdGen,
                   miopen::solver::ConvOclDirectFwd3x3,
                   miopen::solver::ConvOclDirectFwd1x1,
                   miopen::solver::ConvOclDirectFwd > (convContext, db);
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

    switch (param.pooling_param.pooling_type) {
    case Pooling_max:
        convContext.poolingContext.pooling_type = (PoolingType)MLO_POOLING_OP_MAX;
        break;

    case Pooling_average_exclude_padding:
    case Pooling_average_include_padding:
        convContext.poolingContext.pooling_type = (PoolingType)MLO_POOLING_OP_AVE;
        break;

    case Pooling_unknow:
    case Pooling_max_deterministic:
    default:
        LOG(ERROR) << "Unknown polling type";
        return solution_vector;
    }

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

    miopen::solver::ConvSolution solution = miopen::solver::SearchForSolution <
                                            miopen::solver::ConvBinWinograd3x3U,
                                            miopen::solver::ConvOclDirectFwd1x1AMD,
                                            // miopen::solver::ConvAsm3x3U,
                                            // miopen::solver::ConvAsm1x1U,
                                            miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                                            miopen::solver::ConvOclDirectFwdGen,
                                            miopen::solver::ConvOclDirectFwd3x3,
                                            miopen::solver::ConvOclDirectFwd1x1,
                                            miopen::solver::ConvOclDirectFwd > (convContext, db);
    miopen::Handle::clearClEnv();

    for (auto s : solution.construction_params) {
        kernelInfo = s;
        solution_vector.push_back(kernelInfo);
    }

    return solution_vector;
}
} // namespace saber
} // namespace anakin
