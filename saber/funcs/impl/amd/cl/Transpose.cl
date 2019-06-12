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
#define SABER_TRANSPOSE_TILE_DIM 16
__kernel void transpose_2d(
    global float* odata,
    const global float* idata,
    int num,
    int channel,
    int height,
    int width) {
    // Handle to thread block group
    __local float tile[SABER_TRANSPOSE_TILE_DIM][SABER_TRANSPOSE_TILE_DIM + 1];
    unsigned int group_idx = get_group_id(0);
    unsigned int group_idy = get_group_id(1);
    unsigned int local_idx = get_local_id(0);
    unsigned int local_idy = get_local_id(1);

    for (int i = 0; i < num * channel; ++i) {
        unsigned int offset = i * height * width;
        unsigned int yIndex;
        unsigned int xIndex;

        xIndex = group_idx * SABER_TRANSPOSE_TILE_DIM + local_idx;
        yIndex = group_idy * SABER_TRANSPOSE_TILE_DIM + local_idy;

        if (xIndex < width && yIndex < height) {

            unsigned int index_in      = xIndex + (yIndex) * width;
            tile[local_idy][local_idx] = idata[offset + index_in];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        xIndex = group_idy * SABER_TRANSPOSE_TILE_DIM + local_idx;
        yIndex = group_idx * SABER_TRANSPOSE_TILE_DIM + local_idy;

        if (xIndex < height && yIndex < width) {
            unsigned int index_out    = xIndex + (yIndex) * height;
            odata[offset + index_out] = tile[local_idx][local_idy];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// local = 16,16,1
// global = (w_in + 16 - 1), (h_in + 16 - 1), 1
