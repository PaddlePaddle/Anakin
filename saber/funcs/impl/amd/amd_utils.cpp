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
    clstr = clstr.insert(
                clstr.find("miog_betac_alphaab") + 20,
                "__constant TFLOAT * restrict bias,\nTFLOAT slope,");

    std::string search      = "c[index] += alpha*rC";
    std::string sub_search1 = "c[index] += alpha*rC[dima][dimb]";
    std::string sub_search2 =
        "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb]";
    std::string sub_search3 = "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                              "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string sub_search4 =
        "c[index] += alpha*rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string add1 = "rC[dima][dimb] += bias[write_start_b + dimb];\nrC[dima][dimb] *= "
                       "(rC[dima][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add2 =
        "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] += bias[write_start_b + "
        "dimb];\nrC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] *= "
        "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add3 =
        "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
        "dimbi_v] += bias[write_start_b + dimb];\nrC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
        "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
        "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
        "dimbi_v] > 0.0f ? 1.0f : slope);\n";
    std::string add4 =
        "rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] += bias[write_start_b + "
        "dimb];\nrC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
        "(rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > 0.0f ? 1.0f : slope);\n";

    for (size_t pos = clstr.find(search); pos != std::string::npos; pos = clstr.find(search, pos)) {
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

    std::string search      = "c[index] += alpha*rC";
    std::string sub_search1 = "c[index] += alpha*rC[dima][dimb]";
    std::string sub_search2 =
        "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb]";
    std::string sub_search3 = "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                              "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string sub_search4 =
        "c[index] += alpha*rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string add1 = "rC[dima][dimb] *= (rC[dima][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add2 =
        "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] *= "
        "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add3 =
        "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
        "dimbi_v] *= (rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
        "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > 0.0f ? 1.0f : slope);\n";
    std::string add4 =
        "rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
        "(rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > 0.0f ? 1.0f : slope);\n";

    for (size_t pos = clstr.find(search); pos != std::string::npos; pos = clstr.find(search, pos)) {
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
} // namespace saber
} // namespace anakin
