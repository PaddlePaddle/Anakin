/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef ANAKIN_TEST_SABER_X86_TEST_POOL_COMMON_UTIL_H
#define ANAKIN_TEST_SABER_X86_TEST_POOL_COMMON_UTIL_H

#include "saber/saber_funcs_param.h"
#include "saber/saber_types.h"

struct test_pool_desc_t {
    int mb, c;
    int id, ih, iw;
    int od, oh, ow;
    int kd, kh, kw;
    int padf, padt, padl;
    int strd, strh, strw;
};

struct pool_test_params {
    PoolingType aalgorithm;
    int ndims;
    test_pool_desc_t test_pd;
    bool expect_to_fail;
    SaberStatus expected_status;
};

typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;
typedef Tensor<X86, AK_FLOAT, NCHW_C16> Tensor5f_C16; 

template <typename data_t>
bool check_pool_fwd(const pool_test_params &p, const std::vector<Tensor5f_C16 *> inputs,
        std::vector<Tensor5f_C16 *> outputs) {

    data_t *src_data = (data_t *)inputs[0]->get_buf()->get_data();
    data_t *dst_data = (data_t *)outputs[0]->get_buf()->get_data();

    auto pd = p.test_pd;

    for (int n = 0; n < pd.mb; n++) {
        for (int c = 0; c < pd.c / 16; c++) {
            for (int od = 0; od < pd.od; od++) {
                for (int oh = 0; oh < pd.oh; oh++) {
                    for (int c_b = 0; c_b < 16; c_b++) {
                        for (int ow = 0; ow < pd.ow; ow++) {
                            int oidx = n * pd.c * pd.od * pd.oh * pd.ow +
                                   c * pd.od * pd.oh * pd.ow * 16 + od * pd.oh * pd.ow * 16 +
                                   oh * pd.ow * 16 + ow * 16 + c_b;
                            data_t out = dst_data[oidx];
                            data_t out_ref = data_t(0);
                            bool is_initialized = false;
                            int num_summands = 0;

                            for (int kd = 0; kd < pd.kd; ++kd) {
                                for (int kh = 0; kh < pd.kh; ++kh) {
                                    for (int kw = 0; kw < pd.kw; ++kw) {
                                        const int id = od * pd.strd - pd.padf + kd;
                                        const int ih = oh * pd.strh - pd.padt + kh;
                                        const int iw = ow * pd.strw - pd.padl + kw;

                                        if (id < 0 || id >= pd.id) continue;
                                        if (ih < 0 || ih >= pd.ih) continue;
                                        if (iw < 0 || iw >= pd.iw) continue;

                                        int iidx = n * pd.c * pd.id * pd.ih * pd.iw +
                                               c * pd.id * pd.ih * pd.iw * 16 +
                                               id * pd.ih * pd.iw * 16 + ih * pd.iw * 16 + iw * 16 + c_b;
                                        data_t d = src_data[iidx];
                                        if (p.aalgorithm == Pooling_max) {
                                            if (!is_initialized) {
                                                out_ref = d;
                                                is_initialized = true;
                                            } else {
                                                if (out_ref < d) {
                                                    out_ref = d;
                                                }
                                            }
                                        } else if (p.aalgorithm == Pooling_average_include_padding ||
                                               p.aalgorithm == Pooling_average_exclude_padding) {
                                            out_ref += d;
                                            num_summands++;
                                        }
                                    }
                                }
                            }

                            if (p.aalgorithm == Pooling_average_include_padding) {
                                num_summands = pd.kw * pd.kh * pd.kd;
                            }

                            if (p.aalgorithm == Pooling_average_include_padding ||
                                p.aalgorithm == Pooling_average_exclude_padding) {
                                out_ref = ((data_t)out_ref / num_summands);
                            }

                            float absdiff = std::fabs(out - out_ref);
                            float absref = std::fabs(out_ref);
                            float eps = 1e-4;
                            float e = (absdiff > eps) ? absdiff / absref : absdiff;
                            if (e > eps) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}
#endif //ANAKIN_TEST_SABER_X86_TEST_POOL_COMMON_UTIL_H
