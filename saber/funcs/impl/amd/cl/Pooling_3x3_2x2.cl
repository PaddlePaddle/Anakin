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


#ifndef N
#define N   1
#endif

#ifndef C
#define C   64
#endif

#ifndef H
#define H   112
#endif

#ifndef W
#define W   112
#endif

#ifndef K
#define K   64
#endif

#define OH  (H >> 1)
#define OW  (W >> 1)

#define IN_CSTRIDE  (H * W)
#define OUT_CSTRIDE (OH * OW)
#define IN_BSTRIDE  (C * H * W)
#define OUT_BSTRIDE (K * OH * OW)

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
pooling_f3x3_s2x2(const __global float* in, __global float* out) {
    uint lidx = get_local_id(0);
    uint gridx = get_group_id(0);

    __local float buffer[112 * 66 * 2];

    __global float* pIn;
    __global float* pOut;

    float max_temp;
    uint in_index;
    uint out_index;
    uint in_lds_index;
    uint out_lds_index;

    for (uint n = 0; n < N; n++) {
        pIn = (__global float*)(in + gridx * IN_CSTRIDE + n * IN_BSTRIDE);
        pOut = (__global float*)(out + gridx * OUT_CSTRIDE + n * OUT_BSTRIDE);

        if ((lidx & 63) < OW) {
            for (uint i = 0; i < 7; i++) {
                in_index = ((lidx >> 6) * 7 + i) * W;
                in_lds_index = ((lidx >> 6) * 7 + i);
                max_temp = fmax(pIn[in_index + (lidx & 63) * 2], pIn[in_index + (lidx & 63) * 2 + 1]);
                buffer[in_lds_index * 66 + (lidx & 63) + ((lidx & 63) >> 5)] = (lidx & 63) * 2 + 2 < W ? fmax(
                            max_temp, pIn[in_index + (lidx & 63) * 2 + 2]) : max_temp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if ((lidx >> 6) < 14 && (lidx & 63) < OW) {
            for (uint i = 0; i < 4; i++) {
                out_index = ((lidx >> 6) * 4 + i) * OW;
                out_lds_index = ((lidx >> 6) * 4 + i) * 2;
                max_temp = fmax(buffer[out_lds_index * 66 + (lidx & 63) + ((lidx & 63) >> 5)],
                                buffer[(out_lds_index + 1) * 66 + (lidx & 63) + ((lidx & 63) >> 5)]);
                pOut[out_index + (lidx & 63)] = out_lds_index + 2 < H ? fmax(max_temp,
                                                buffer[(out_lds_index + 2) * 66 + (lidx & 63) + ((lidx & 63) >> 5)]) : max_temp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
