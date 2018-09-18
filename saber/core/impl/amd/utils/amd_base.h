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
#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_UTILS_AMDBASE_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_UTILS_AMDBASE_H
#include <vector>
#include <string>
#include "amd_logger.h"
#include <miopen/kernel_info.hpp>

namespace anakin {
namespace saber {

typedef struct ExtSolutionConfigTpye {
    int in_tile0, in_tile1;
    int grp_tile0, grp_tile1;
    int out_pix_tile0, out_pix_tile1;
    int n_stacks;
    int n_out_pix_tiles;
    int n_out_tiles_perstack;
    int n_in_data_tiles;
    int n_read_procs;
    int alu_tile0, alu_tile1;
    int horiz_out_pix;
    int vert_out_pix;
} T_ExtSolutionConfig;

enum KernelType {
    DEFAULT = 0,
    SABER   = DEFAULT,
    SOURCE,
    MIOPEN,
};

struct KernelInfo {
    std::string comp_options;
    int wk_dim = 1;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    std::vector<size_t> g_wk_offset;
    std::string kernel_file;
    std::string kernel_name;
    KernelType kernel_type = DEFAULT;
    friend std::ostream& operator<<(std::ostream& os, const KernelInfo& k);

    KernelInfo& operator=(KernelInfo right) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "assign KerneInfo";
        clone(&right);
    };
    KernelInfo& operator=(KernelInfo* right) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "assign KerneInfo *";
        clone(right);
    };
    KernelInfo& operator=(miopen::solver::KernelInfo& right) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "assign KerneInfo *";
        clone(&right);
    };

    void printE() {
        std::string s = "";

        LOG(INFO) << "======================================================================";

        if (kernel_type == KernelType::SOURCE) {
            LOG(INFO) << "kernel is cl string";
        } else {
            LOG(INFO) << "kernel file name: " << kernel_file;
        }

        LOG(INFO) << "kernel name:      " << kernel_name;

        for (auto v : l_wk) {
            s += std::to_string(v);
            s += " ";
        }

        LOG(INFO) << "local work size:  " << s;
        s = "";

        for (auto v : g_wk) {
            s += std::to_string(v);
            s += " ";
        }

        LOG(INFO) << "global work size: " << s;
        LOG(INFO) << "compile option:   " << comp_options;
        LOG(INFO) << "kernel type:      " << kernel_type;
        LOG(INFO) << "======================================================================";
    };

    void print() {
#ifdef ENABLE_DEBUG
        printE();
#endif
    };

private:
    template <typename T>
    void copy_vector(std::vector<T>* l, std::vector<T>* r) {
        l->clear();
        typename std::vector<T>::iterator i;

        for (i = r->begin(); i < r->end(); i++) {
            l->push_back(*i);
        }
    }
    void clone(KernelInfo* ki) {
        comp_options.assign(ki->comp_options);
        wk_dim = ki->wk_dim;
        copy_vector(&l_wk, &ki->l_wk);
        copy_vector(&g_wk, &ki->g_wk);
        copy_vector(&g_wk_offset, &ki->g_wk_offset);
        kernel_file.assign(ki->kernel_file);
        kernel_name.assign(ki->kernel_name);
        kernel_type = ki->kernel_type;
    }

    void clone(miopen::solver::KernelInfo* ki) {
        comp_options.assign(ki->comp_options);
        wk_dim = ki->g_wk.size();
        copy_vector(&l_wk, &ki->l_wk);
        copy_vector(&g_wk, &ki->g_wk);
        // copy_vector(&g_wk_offset, &ki->g_wk_offset);
        kernel_file.assign(ki->kernel_file);
        kernel_name.assign(ki->kernel_name);
        kernel_type = ki->isMIOpenKernel ? MIOPEN : SABER;
    }
};

} // namespace saber
} // namespace anakin
#endif
