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
__kernel void Embedding(
    __global float* out_data,
    __global const float* in_data,
    __global const float* tabel,
    const int emb_dim,
    const int word_num,
    const int padding_idx,
    const int out_count) {

    int tid = get_global_id(0);

    if (tid < out_count) {

        int emb_id            = tid % emb_dim;
        int word_id           = tid / emb_dim;
        int word_idx_in_tabel = (int)(in_data[word_id]);

        if (word_idx_in_tabel != padding_idx) {
            out_data[tid] = (float)tabel[word_idx_in_tabel * emb_dim + emb_id];
        } else {
            out_data[tid] = (0.f);
        }
    }
}
