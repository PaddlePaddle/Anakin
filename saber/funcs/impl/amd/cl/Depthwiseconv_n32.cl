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
__kernel void DepthwiseconvDw21n32(
        __global float* out,
        __global const float* in,
        __global const float* weights
#if MLO_CONV_BIAS
        ,
        __global float* bias
#endif
) {
    in            = (in + -113);
    int local_id0 = get_local_id(0);
    float agg[16] = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
    };
    __local float lds_in[4492];
    __local float lds_weights[9];
    int x0_gid = (get_group_id(0) * 8);
    int c_gid  = get_group_id(2);
    int n_gid  = (get_group_id(1) * 4);
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3) {
        for (int k1_gid = 0; k1_gid < 3; k1_gid += 3) {
            {
                int gbase =
                        ((((k1_gid + (k0_gid * 112)) + (x0_gid * 112)) + (c_gid * 12544))
                         + (n_gid * 401408));
                int k1_x1_k0_x0_tid = (local_id0 % 256);
                for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 5; k1_x1_k0_x0_lid += 1) {
                    int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 4) || (k1_x1_k0_x0_tid < 98));
                    if (k1_x1_k0_x0_cond) {
                        int k1_x1_k0_x0 = ((256 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
                        for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                            int lidx     = (k1_x1_k0_x0 + (1123 * n_lid));
                            int gidx     = ((gbase + k1_x1_k0_x0) + (401408 * (int)n_lid));
                            lds_in[lidx] = in[clamp((int)gidx, (int)113, (int)12845168)];
                        }
                    }
                }
            }
            {
                int gbase        = ((k1_gid + (k0_gid * 3)) + (c_gid * 9));
                int k1_k0_c_tid  = (local_id0 % 16);
                int k1_k0_c_cond = (k1_k0_c_tid < 9);
                if (k1_k0_c_cond) {
                    if ((local_id0 < 16)) {
                        int gidx                 = (gbase + k1_k0_c_tid);
                        lds_weights[k1_k0_c_tid] = weights[clamp((int)gidx, (int)0, (int)287)];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1)
                   && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 112))
                  && ((-1 * k1_gid) <= -1))
                 && ((((k1_gid + 3) - 1) + 111) <= 112))) {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x1_lid = 0; x1_lid < 4; x1_lid += 1) {
                            int x1_cond = ((x1_lid < 3) || (x1_tid < 16));
                            int x1 = select((int)0, (int)((32 * x1_lid) + x1_tid), (int)x1_cond);
                            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                                int x0 = ((4 * x0_lid) + x0_tid);
                                for (int n_lid = 0; n_lid < 2; n_lid += 1) {
                                    int n = ((2 * n_lid) + n_tid);
                                    float val1 =
                                            lds_in[((((k1_lid + x1) + (112 * k0_lid)) + (112 * x0))
                                                    + (1123 * n))];
                                    float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                    int agg_idx   = ((x1_lid + (x0_lid * 4)) + (n_lid * 8));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx], (float)agg_rhs, (int)x1_cond);
                                }
                            }
                        }
                    }
                }
            } else {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x1_lid = 0; x1_lid < 4; x1_lid += 1) {
                            int x1_cond = ((x1_lid < 3) || (x1_tid < 16));
                            int x1 = select((int)0, (int)((32 * x1_lid) + x1_tid), (int)x1_cond);
                            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                                int x0 = ((4 * x0_lid) + x0_tid);
                                for (int n_lid = 0; n_lid < 2; n_lid += 1) {
                                    int n = ((2 * n_lid) + n_tid);
                                    float val1 =
                                            lds_in[((((k1_lid + x1) + (112 * k0_lid)) + (112 * x0))
                                                    + (1123 * n))];
                                    float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                    int agg_idx   = ((x1_lid + (x0_lid * 4)) + (n_lid * 8));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx],
                                            (float)agg_rhs,
                                            (int)(x1_cond
                                                  && ((((((-1 * (k0_gid + k0_lid))
                                                          + (-1 * (x0_gid + x0)))
                                                         <= -1)
                                                        && (((k0_gid + k0_lid) + (x0_gid + x0))
                                                            <= 112))
                                                       && (((-1 * (k1_gid + k1_lid)) + (-1 * x1))
                                                           <= -1))
                                                      && (((k1_gid + k1_lid) + x1) <= 112))));
                                }
                            }
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    int x1_tid = (local_id0 % 32);
    int x0_tid = ((local_id0 / 32) % 4);
    int n_tid  = ((local_id0 / 128) % 2);
    for (int x1_lid = 0; x1_lid < 4; x1_lid += 1) {
        int x1_cond = ((x1_lid < 3) || (x1_tid < 16));
        if (x1_cond) {
            int x1 = ((32 * x1_lid) + x1_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                int x0 = ((4 * x0_lid) + x0_tid);
                for (int n_lid = 0; n_lid < 2; n_lid += 1) {
                    int n         = ((2 * n_lid) + n_tid);
                    float agg_out = agg[((x1_lid + (x0_lid * 4)) + (n_lid * 8))];
                    int gout_idx =
                            ((((12544 * c_gid) + (401408 * (int)(n_gid + n)))
                              + (112 * (x0_gid + x0)))
                             + x1);
                    if (((gout_idx >= 0) && (gout_idx < 12845056))) {
#if MLO_CONV_BIAS
                        agg_out += bias[((get_group_id(2)) % 32)];
#endif
#if MLO_CONV_ACTIVE_RELU
                        agg_out = agg_out > 0.0f ? agg_out : 0.0f;
#endif
                        out[gout_idx] = agg_out;
                    }
                }
            }
        }
    }
}

__kernel void DepthwiseconvDw22n32(
        __global float* out,
        __global const float* in,
        __global const float* weights
#if MLO_CONV_BIAS
        ,
        __global float* bias
#endif
) {
    in            = (in + -113);
    int local_id0 = get_local_id(0);
    float agg[8]  = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
    };
    __local float lds_in[7620];
    __local float lds_weights[9];
    int v2_gid = (get_group_id(0) * 8);
    int v1_gid = get_group_id(2);
    int v0_gid = (get_group_id(1) * 4);
    for (int v5_gid = 0; v5_gid < 3; v5_gid += 3) {
        for (int v6_gid = 0; v6_gid < 3; v6_gid += 3) {
            {
                int gbase =
                        ((((v6_gid + (v5_gid * 112)) + (v2_gid * 224)) + (v1_gid * 12544))
                         + (v0_gid * 802816));
                int v6_v3_v5_v2_tid = (local_id0 % 256);
                for (int v6_v3_v5_v2_lid = 0; v6_v3_v5_v2_lid < 8; v6_v3_v5_v2_lid += 1) {
                    int v6_v3_v5_v2_cond = ((v6_v3_v5_v2_lid < 7) || (v6_v3_v5_v2_tid < 113));
                    if (v6_v3_v5_v2_cond) {
                        int v6_v3_v5_v2 = ((256 * v6_v3_v5_v2_lid) + v6_v3_v5_v2_tid);
                        for (int v0_lid = 0; v0_lid < 4; v0_lid += 1) {
                            int lidx     = (v6_v3_v5_v2 + (1905 * v0_lid));
                            int gidx     = ((gbase + v6_v3_v5_v2) + (802816 * (int)v0_lid));
                            lds_in[lidx] = in[clamp((int)gidx, (int)113, (int)25690224)];
                        }
                    }
                }
            }
            {
                int gbase         = ((v6_gid + (v5_gid * 3)) + (v1_gid * 9));
                int v6_v5_v1_tid  = (local_id0 % 16);
                int v6_v5_v1_cond = (v6_v5_v1_tid < 9);
                if (v6_v5_v1_cond) {
                    if ((local_id0 < 16)) {
                        int gidx                  = (gbase + v6_v5_v1_tid);
                        lds_weights[v6_v5_v1_tid] = weights[clamp((int)gidx, (int)0, (int)575)];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (((((-2 * v2_gid) + (-1 * v5_gid)) <= -1) && ((-1 * v6_gid) <= -1))) {
                for (int v5_lid = 0; v5_lid < 3; v5_lid += 1) {
                    for (int v6_lid = 0; v6_lid < 3; v6_lid += 1) {
                        int v3_tid = (local_id0 % 32);
                        int v2_tid = ((local_id0 / 32) % 4);
                        int v0_tid = ((local_id0 / 128) % 2);
                        for (int v3_lid = 0; v3_lid < 2; v3_lid += 1) {
                            int v3_cond = ((v3_lid < 1) || (v3_tid < 24));
                            int v3 = select((int)0, (int)((32 * v3_lid) + v3_tid), (int)v3_cond);
                            for (int v2_lid = 0; v2_lid < 2; v2_lid += 1) {
                                int v2 = ((4 * v2_lid) + v2_tid);
                                for (int v0_lid = 0; v0_lid < 2; v0_lid += 1) {
                                    int v0        = ((2 * v0_lid) + v0_tid);
                                    float val1    = lds_in[(
                                            (((v6_lid + (2 * v3)) + (112 * v5_lid)) + (224 * v2))
                                            + (1905 * v0))];
                                    float val2    = lds_weights[(v6_lid + (3 * v5_lid))];
                                    int agg_idx   = ((v3_lid + (v2_lid * 2)) + (v0_lid * 4));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx], (float)agg_rhs, (int)v3_cond);
                                }
                            }
                        }
                    }
                }
            } else {
                for (int v5_lid = 0; v5_lid < 3; v5_lid += 1) {
                    for (int v6_lid = 0; v6_lid < 3; v6_lid += 1) {
                        int v3_tid = (local_id0 % 32);
                        int v2_tid = ((local_id0 / 32) % 4);
                        int v0_tid = ((local_id0 / 128) % 2);
                        for (int v3_lid = 0; v3_lid < 2; v3_lid += 1) {
                            int v3_cond = ((v3_lid < 1) || (v3_tid < 24));
                            int v3 = select((int)0, (int)((32 * v3_lid) + v3_tid), (int)v3_cond);
                            for (int v2_lid = 0; v2_lid < 2; v2_lid += 1) {
                                int v2 = ((4 * v2_lid) + v2_tid);
                                for (int v0_lid = 0; v0_lid < 2; v0_lid += 1) {
                                    int v0        = ((2 * v0_lid) + v0_tid);
                                    float val1    = lds_in[(
                                            (((v6_lid + (2 * v3)) + (112 * v5_lid)) + (224 * v2))
                                            + (1905 * v0))];
                                    float val2    = lds_weights[(v6_lid + (3 * v5_lid))];
                                    int agg_idx   = ((v3_lid + (v2_lid * 2)) + (v0_lid * 4));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx],
                                            (float)agg_rhs,
                                            (int)(v3_cond
                                                  && ((((-2 * (v2_gid + v2))
                                                        + (-1 * (v5_gid + v5_lid)))
                                                       <= -1)
                                                      && (((-2 * v3) + (-1 * (v6_gid + v6_lid)))
                                                          <= -1))));
                                }
                            }
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    int v3_tid = (local_id0 % 32);
    int v2_tid = ((local_id0 / 32) % 4);
    int v0_tid = ((local_id0 / 128) % 2);
    for (int v3_lid = 0; v3_lid < 2; v3_lid += 1) {
        int v3_cond = ((v3_lid < 1) || (v3_tid < 24));
        if (v3_cond) {
            int v3 = ((32 * v3_lid) + v3_tid);
            for (int v2_lid = 0; v2_lid < 2; v2_lid += 1) {
                int v2 = ((4 * v2_lid) + v2_tid);
                for (int v0_lid = 0; v0_lid < 2; v0_lid += 1) {
                    int v0        = ((2 * v0_lid) + v0_tid);
                    float agg_out = agg[((v3_lid + (v2_lid * 2)) + (v0_lid * 4))];
                    int gout_idx =
                            ((((200704 * (int)(v0_gid + v0)) + (int)(3136 * v1_gid))
                              + (int)(56 * (v2_gid + v2)))
                             + (int)v3);
                    if (((gout_idx >= 0) && (gout_idx < 6422528))) {
#if MLO_CONV_BIAS
                        agg_out += bias[((get_group_id(2)) % 64)];
#endif
#if MLO_CONV_ACTIVE_RELU
                        agg_out = agg_out > 0.0f ? agg_out : 0.0f;
#endif
                        out[gout_idx] = agg_out;
                    }
                }
            }
        }
    }
}

__kernel void DepthwiseconvDw31n32(
        __global float* out,
        __global const float* in,
        __global const float* weights
#if MLO_CONV_BIAS
        ,
        __global float* bias
#endif
) {
    in            = (in + -57);
    int local_id0 = get_local_id(0);
    float agg[16] = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
    };
    __local float lds_in[4504];
    __local float lds_weights[9];
    int x0_gid = (get_group_id(0) * 8);
    int c_gid  = get_group_id(2);
    int n_gid  = (get_group_id(1) * 8);
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3) {
        for (int k1_gid = 0; k1_gid < 3; k1_gid += 3) {
            {
                int gbase =
                        ((((k1_gid + (k0_gid * 56)) + (x0_gid * 56)) + (c_gid * 3136))
                         + (n_gid * 401408));
                int k1_x1_k0_x0_tid = (local_id0 % 256);
                for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 3; k1_x1_k0_x0_lid += 1) {
                    int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 2) || (k1_x1_k0_x0_tid < 50));
                    if (k1_x1_k0_x0_cond) {
                        int k1_x1_k0_x0 = ((256 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
                        for (int n_lid = 0; n_lid < 8; n_lid += 1) {
                            int lidx     = (k1_x1_k0_x0 + (563 * n_lid));
                            int gidx     = ((gbase + k1_x1_k0_x0) + (401408 * (int)n_lid));
                            lds_in[lidx] = in[clamp((int)gidx, (int)57, (int)12845112)];
                        }
                    }
                }
            }
            {
                int gbase        = ((k1_gid + (k0_gid * 3)) + (c_gid * 9));
                int k1_k0_c_tid  = (local_id0 % 16);
                int k1_k0_c_cond = (k1_k0_c_tid < 9);
                if (k1_k0_c_cond) {
                    if ((local_id0 < 16)) {
                        int gidx                 = (gbase + k1_k0_c_tid);
                        lds_weights[k1_k0_c_tid] = weights[clamp((int)gidx, (int)0, (int)1151)];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1)
                   && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 56))
                  && ((-1 * k1_gid) <= -1))
                 && ((((k1_gid + 3) - 1) + 55) <= 56))) {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x1_lid = 0; x1_lid < 2; x1_lid += 1) {
                            int x1_cond = ((x1_lid < 1) || (x1_tid < 24));
                            int x1 = select((int)0, (int)((32 * x1_lid) + x1_tid), (int)x1_cond);
                            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                                int x0 = ((4 * x0_lid) + x0_tid);
                                for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                                    int n = ((2 * n_lid) + n_tid);
                                    float val1 =
                                            lds_in[((((k1_lid + x1) + (56 * k0_lid)) + (56 * x0))
                                                    + (563 * n))];
                                    float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                    int agg_idx   = ((x1_lid + (x0_lid * 2)) + (n_lid * 4));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx], (float)agg_rhs, (int)x1_cond);
                                }
                            }
                        }
                    }
                }
            } else {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x1_lid = 0; x1_lid < 2; x1_lid += 1) {
                            int x1_cond = ((x1_lid < 1) || (x1_tid < 24));
                            int x1 = select((int)0, (int)((32 * x1_lid) + x1_tid), (int)x1_cond);
                            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                                int x0 = ((4 * x0_lid) + x0_tid);
                                for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                                    int n = ((2 * n_lid) + n_tid);
                                    float val1 =
                                            lds_in[((((k1_lid + x1) + (56 * k0_lid)) + (56 * x0))
                                                    + (563 * n))];
                                    float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                    int agg_idx   = ((x1_lid + (x0_lid * 2)) + (n_lid * 4));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx],
                                            (float)agg_rhs,
                                            (int)(x1_cond
                                                  && ((((((-1 * (k0_gid + k0_lid))
                                                          + (-1 * (x0_gid + x0)))
                                                         <= -1)
                                                        && (((k0_gid + k0_lid) + (x0_gid + x0))
                                                            <= 56))
                                                       && (((-1 * (k1_gid + k1_lid)) + (-1 * x1))
                                                           <= -1))
                                                      && (((k1_gid + k1_lid) + x1) <= 56))));
                                }
                            }
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    int x1_tid = (local_id0 % 32);
    int x0_tid = ((local_id0 / 32) % 4);
    int n_tid  = ((local_id0 / 128) % 2);
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1) {
        int x1_cond = ((x1_lid < 1) || (x1_tid < 24));
        if (x1_cond) {
            int x1 = ((32 * x1_lid) + x1_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                int x0 = ((4 * x0_lid) + x0_tid);
                for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                    int n         = ((2 * n_lid) + n_tid);
                    float agg_out = agg[((x1_lid + (x0_lid * 2)) + (n_lid * 4))];
                    int gout_idx =
                            ((((3136 * c_gid) + (401408 * (int)(n_gid + n))) + (56 * (x0_gid + x0)))
                             + x1);
                    if (((gout_idx >= 0) && (gout_idx < 12845056))) {
#if MLO_CONV_BIAS
                        agg_out += bias[((get_group_id(2)) % 128)];
#endif
#if MLO_CONV_ACTIVE_RELU
                        agg_out = agg_out > 0.0f ? agg_out : 0.0f;
#endif
                        out[gout_idx] = agg_out;
                    }
                }
            }
        }
    }
}

__kernel void DepthwiseconvDw41n32(
        __global float* out,
        __global const float* in,
        __global const float* weights
#if MLO_CONV_BIAS
        ,
        __global float* bias
#endif
) {
    in            = (in + -29);
    int local_id0 = get_local_id(0);
    float agg[16] = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
    };
    __local float lds_in[4056];
    __local float lds_weights[9];
    int x0_gid = (get_group_id(0) * 16);
    int c_gid  = get_group_id(2);
    int n_gid  = (get_group_id(1) * 8);
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3) {
        for (int k1_gid = 0; k1_gid < 3; k1_gid += 3) {
            {
                int gbase =
                        ((((k1_gid + (k0_gid * 28)) + (x0_gid * 28)) + (c_gid * 784))
                         + (n_gid * 200704));
                int k1_x1_k0_x0_tid = (local_id0 % 256);
                for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 2; k1_x1_k0_x0_lid += 1) {
                    int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 1) || (k1_x1_k0_x0_tid < 250));
                    if (k1_x1_k0_x0_cond) {
                        int k1_x1_k0_x0 = ((256 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
                        for (int n_lid = 0; n_lid < 8; n_lid += 1) {
                            int lidx     = (k1_x1_k0_x0 + (507 * n_lid));
                            int gidx     = ((gbase + k1_x1_k0_x0) + (200704 * (int)n_lid));
                            lds_in[lidx] = in[clamp((int)gidx, (int)29, (int)6422556)];
                        }
                    }
                }
            }
            {
                int gbase        = ((k1_gid + (k0_gid * 3)) + (c_gid * 9));
                int k1_k0_c_tid  = (local_id0 % 16);
                int k1_k0_c_cond = (k1_k0_c_tid < 9);
                if (k1_k0_c_cond) {
                    if ((local_id0 < 16)) {
                        int gidx                 = (gbase + k1_k0_c_tid);
                        lds_weights[k1_k0_c_tid] = weights[clamp((int)gidx, (int)0, (int)2303)];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1)
                   && ((((k0_gid + 3) - 1) + ((x0_gid + 16) - 1)) <= 28))
                  && ((-1 * k1_gid) <= -1))
                 && ((((k1_gid + 3) - 1) + 27) <= 28))) {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1) {
                            int x0      = ((4 * x0_lid) + x0_tid);
                            int x1_cond = (x1_tid < 28);
                            int x1      = select((int)0, (int)x1_tid, (int)x1_cond);
                            for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                                int n         = ((2 * n_lid) + n_tid);
                                float val1    = lds_in[(
                                        (((k1_lid + x1) + (28 * k0_lid)) + (28 * x0)) + (507 * n))];
                                float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                int agg_idx   = (x0_lid + (n_lid * 4));
                                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                agg[agg_idx] =
                                        select((float)agg[agg_idx], (float)agg_rhs, (int)x1_cond);
                            }
                        }
                    }
                }
            } else {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1) {
                            int x0      = ((4 * x0_lid) + x0_tid);
                            int x1_cond = (x1_tid < 28);
                            int x1      = select((int)0, (int)x1_tid, (int)x1_cond);
                            for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                                int n         = ((2 * n_lid) + n_tid);
                                float val1    = lds_in[(
                                        (((k1_lid + x1) + (28 * k0_lid)) + (28 * x0)) + (507 * n))];
                                float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                int agg_idx   = (x0_lid + (n_lid * 4));
                                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                agg[agg_idx]  = select(
                                        (float)agg[agg_idx],
                                        (float)agg_rhs,
                                        (int)(x1_cond
                                              && ((((((-1 * (k0_gid + k0_lid))
                                                      + (-1 * (x0_gid + x0)))
                                                     <= -1)
                                                    && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 28))
                                                   && (((-1 * (k1_gid + k1_lid)) + (-1 * x1))
                                                       <= -1))
                                                  && (((k1_gid + k1_lid) + x1) <= 28))));
                            }
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    int x1_tid = (local_id0 % 32);
    int x0_tid = ((local_id0 / 32) % 4);
    int n_tid  = ((local_id0 / 128) % 2);
    for (int x0_lid = 0; x0_lid < 4; x0_lid += 1) {
        int x0_cond = ((x0_lid < 3) || (x0_gid != 16));
        if (x0_cond) {
            int x0      = ((4 * x0_lid) + x0_tid);
            int x1_cond = (x1_tid < 28);
            if (x1_cond) {
                for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                    int n         = ((2 * n_lid) + n_tid);
                    float agg_out = agg[(x0_lid + (n_lid * 4))];
                    int gout_idx =
                            ((((784 * c_gid) + (200704 * (int)(n_gid + n))) + (28 * (x0_gid + x0)))
                             + x1_tid);
                    if (((gout_idx >= 0) && (gout_idx < 6422528))) {
#if MLO_CONV_BIAS
                        agg_out += bias[((get_group_id(2)) % 256)];
#endif
#if MLO_CONV_ACTIVE_RELU
                        agg_out = agg_out > 0.0f ? agg_out : 0.0f;
#endif
                        out[gout_idx] = agg_out;
                    }
                }
            }
        }
    }
}

__kernel void DepthwiseconvDw222n32(
        __global float* out,
        __global const float* in,
        __global const float* weights
#if MLO_CONV_BIAS
        ,
        __global float* bias
#endif
) {
    in            = (in + -113);
    int local_id0 = get_local_id(0);
    float agg[8]  = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
    };
    __local float lds_in[7620];
    __local float lds_weights[9];
    int v2_gid = (get_group_id(0) * 8);
    int v1_gid = get_group_id(1);
    int v0_gid = (get_group_id(2) * 4);
    for (int v5_gid = 0; v5_gid < 3; v5_gid += 3) {
        for (int v6_gid = 0; v6_gid < 3; v6_gid += 3) {
            {
                int gbase =
                        ((((v6_gid + (v5_gid * 112)) + (v2_gid * 224)) + (v1_gid * 12544))
                         + (v0_gid * 1204224));
                int v6_v3_v5_v2_tid = (local_id0 % 256);
                for (int v6_v3_v5_v2_lid = 0; v6_v3_v5_v2_lid < 8; v6_v3_v5_v2_lid += 1) {
                    int v6_v3_v5_v2_cond = ((v6_v3_v5_v2_lid < 7) || (v6_v3_v5_v2_tid < 113));
                    if (v6_v3_v5_v2_cond) {
                        int v6_v3_v5_v2 = ((256 * v6_v3_v5_v2_lid) + v6_v3_v5_v2_tid);
                        for (int v0_lid = 0; v0_lid < 4; v0_lid += 1) {
                            int lidx     = (v6_v3_v5_v2 + (1905 * v0_lid));
                            int gidx     = ((gbase + v6_v3_v5_v2) + (1204224 * (int)v0_lid));
                            lds_in[lidx] = in[clamp((int)gidx, (int)113, (int)38535280)];
                        }
                    }
                }
            }
            {
                int gbase         = ((v6_gid + (v5_gid * 3)) + (v1_gid * 9));
                int v6_v5_v1_tid  = (local_id0 % 16);
                int v6_v5_v1_cond = (v6_v5_v1_tid < 9);
                if (v6_v5_v1_cond) {
                    if ((local_id0 < 16)) {
                        int gidx                  = (gbase + v6_v5_v1_tid);
                        lds_weights[v6_v5_v1_tid] = weights[clamp((int)gidx, (int)0, (int)863)];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (((((-2 * v2_gid) + (-1 * v5_gid)) <= -1) && ((-1 * v6_gid) <= -1))) {
                for (int v5_lid = 0; v5_lid < 3; v5_lid += 1) {
                    for (int v6_lid = 0; v6_lid < 3; v6_lid += 1) {
                        int v3_tid = (local_id0 % 32);
                        int v2_tid = ((local_id0 / 32) % 4);
                        int v0_tid = ((local_id0 / 128) % 2);
                        for (int v3_lid = 0; v3_lid < 2; v3_lid += 1) {
                            int v3_cond = ((v3_lid < 1) || (v3_tid < 24));
                            int v3 = select((int)0, (int)((32 * v3_lid) + v3_tid), (int)v3_cond);
                            for (int v2_lid = 0; v2_lid < 2; v2_lid += 1) {
                                int v2 = ((4 * v2_lid) + v2_tid);
                                for (int v0_lid = 0; v0_lid < 2; v0_lid += 1) {
                                    int v0        = ((2 * v0_lid) + v0_tid);
                                    float val1    = lds_in[(
                                            (((v6_lid + (2 * v3)) + (112 * v5_lid)) + (224 * v2))
                                            + (1905 * v0))];
                                    float val2    = lds_weights[(v6_lid + (3 * v5_lid))];
                                    int agg_idx   = ((v3_lid + (v2_lid * 2)) + (v0_lid * 4));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx], (float)agg_rhs, (int)v3_cond);
                                }
                            }
                        }
                    }
                }
            } else {
                for (int v5_lid = 0; v5_lid < 3; v5_lid += 1) {
                    for (int v6_lid = 0; v6_lid < 3; v6_lid += 1) {
                        int v3_tid = (local_id0 % 32);
                        int v2_tid = ((local_id0 / 32) % 4);
                        int v0_tid = ((local_id0 / 128) % 2);
                        for (int v3_lid = 0; v3_lid < 2; v3_lid += 1) {
                            int v3_cond = ((v3_lid < 1) || (v3_tid < 24));
                            int v3 = select((int)0, (int)((32 * v3_lid) + v3_tid), (int)v3_cond);
                            for (int v2_lid = 0; v2_lid < 2; v2_lid += 1) {
                                int v2 = ((4 * v2_lid) + v2_tid);
                                for (int v0_lid = 0; v0_lid < 2; v0_lid += 1) {
                                    int v0        = ((2 * v0_lid) + v0_tid);
                                    float val1    = lds_in[(
                                            (((v6_lid + (2 * v3)) + (112 * v5_lid)) + (224 * v2))
                                            + (1905 * v0))];
                                    float val2    = lds_weights[(v6_lid + (3 * v5_lid))];
                                    int agg_idx   = ((v3_lid + (v2_lid * 2)) + (v0_lid * 4));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx],
                                            (float)agg_rhs,
                                            (int)(v3_cond
                                                  && ((((-2 * (v2_gid + v2))
                                                        + (-1 * (v5_gid + v5_lid)))
                                                       <= -1)
                                                      && (((-2 * v3) + (-1 * (v6_gid + v6_lid)))
                                                          <= -1))));
                                }
                            }
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    int v3_tid = (local_id0 % 32);
    int v2_tid = ((local_id0 / 32) % 4);
    int v0_tid = ((local_id0 / 128) % 2);
    for (int v3_lid = 0; v3_lid < 2; v3_lid += 1) {
        int v3_cond = ((v3_lid < 1) || (v3_tid < 24));
        if (v3_cond) {
            int v3 = ((32 * v3_lid) + v3_tid);
            for (int v2_lid = 0; v2_lid < 2; v2_lid += 1) {
                int v2 = ((4 * v2_lid) + v2_tid);
                for (int v0_lid = 0; v0_lid < 2; v0_lid += 1) {
                    int v0        = ((2 * v0_lid) + v0_tid);
                    float agg_out = agg[((v3_lid + (v2_lid * 2)) + (v0_lid * 4))];
                    int gout_idx =
                            ((((301056 * (int)(v0_gid + v0)) + (int)(3136 * v1_gid))
                              + (int)(56 * (v2_gid + v2)))
                             + (int)v3);
                    if (((gout_idx >= 0) && (gout_idx < 9633792))) {
#if MLO_CONV_BIAS
                        agg_out += bias[((get_group_id(1)) % 96)];
#endif
#if MLO_CONV_ACTIVE_RELU
                        agg_out = agg_out > 0.0f ? agg_out : 0.0f;
#endif
                        out[gout_idx] = agg_out;
                    }
                }
            }
        }
    }
}

__kernel void DepthwiseconvDw231n32(
        __global float* out,
        __global const float* in,
        __global const float* weights
#if MLO_CONV_BIAS
        ,
        __global float* bias
#endif
) {
    in            = (in + -57);
    int local_id0 = get_local_id(0);
    float agg[16] = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
    };
    __local float lds_in[4504];
    __local float lds_weights[9];
    int x0_gid = (get_group_id(0) * 8);
    int c_gid  = get_group_id(1);
    int n_gid  = (get_group_id(2) * 8);
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3) {
        for (int k1_gid = 0; k1_gid < 3; k1_gid += 3) {
            {
                int gbase =
                        ((((k1_gid + (k0_gid * 56)) + (x0_gid * 56)) + (c_gid * 3136))
                         + (n_gid * 451584));
                int k1_x1_k0_x0_tid = (local_id0 % 256);
                for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 3; k1_x1_k0_x0_lid += 1) {
                    int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 2) || (k1_x1_k0_x0_tid < 50));
                    if (k1_x1_k0_x0_cond) {
                        int k1_x1_k0_x0 = ((256 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
                        for (int n_lid = 0; n_lid < 8; n_lid += 1) {
                            int lidx     = (k1_x1_k0_x0 + (563 * n_lid));
                            int gidx     = ((gbase + k1_x1_k0_x0) + (451584 * (int)n_lid));
                            lds_in[lidx] = in[clamp((int)gidx, (int)57, (int)14450744)];
                        }
                    }
                }
            }
            {
                int gbase        = ((k1_gid + (k0_gid * 3)) + (c_gid * 9));
                int k1_k0_c_tid  = (local_id0 % 16);
                int k1_k0_c_cond = (k1_k0_c_tid < 9);
                if (k1_k0_c_cond) {
                    if ((local_id0 < 16)) {
                        int gidx                 = (gbase + k1_k0_c_tid);
                        lds_weights[k1_k0_c_tid] = weights[clamp((int)gidx, (int)0, (int)1295)];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1)
                   && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 56))
                  && ((-1 * k1_gid) <= -1))
                 && ((((k1_gid + 3) - 1) + 55) <= 56))) {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x1_lid = 0; x1_lid < 2; x1_lid += 1) {
                            int x1_cond = ((x1_lid < 1) || (x1_tid < 24));
                            int x1 = select((int)0, (int)((32 * x1_lid) + x1_tid), (int)x1_cond);
                            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                                int x0 = ((4 * x0_lid) + x0_tid);
                                for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                                    int n = ((2 * n_lid) + n_tid);
                                    float val1 =
                                            lds_in[((((k1_lid + x1) + (56 * k0_lid)) + (56 * x0))
                                                    + (563 * n))];
                                    float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                    int agg_idx   = ((x1_lid + (x0_lid * 2)) + (n_lid * 4));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx], (float)agg_rhs, (int)x1_cond);
                                }
                            }
                        }
                    }
                }
            } else {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x1_lid = 0; x1_lid < 2; x1_lid += 1) {
                            int x1_cond = ((x1_lid < 1) || (x1_tid < 24));
                            int x1 = select((int)0, (int)((32 * x1_lid) + x1_tid), (int)x1_cond);
                            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                                int x0 = ((4 * x0_lid) + x0_tid);
                                for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                                    int n = ((2 * n_lid) + n_tid);
                                    float val1 =
                                            lds_in[((((k1_lid + x1) + (56 * k0_lid)) + (56 * x0))
                                                    + (563 * n))];
                                    float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                    int agg_idx   = ((x1_lid + (x0_lid * 2)) + (n_lid * 4));
                                    float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                    agg[agg_idx]  = select(
                                            (float)agg[agg_idx],
                                            (float)agg_rhs,
                                            (int)(x1_cond
                                                  && ((((((-1 * (k0_gid + k0_lid))
                                                          + (-1 * (x0_gid + x0)))
                                                         <= -1)
                                                        && (((k0_gid + k0_lid) + (x0_gid + x0))
                                                            <= 56))
                                                       && (((-1 * (k1_gid + k1_lid)) + (-1 * x1))
                                                           <= -1))
                                                      && (((k1_gid + k1_lid) + x1) <= 56))));
                                }
                            }
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    int x1_tid = (local_id0 % 32);
    int x0_tid = ((local_id0 / 32) % 4);
    int n_tid  = ((local_id0 / 128) % 2);
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1) {
        int x1_cond = ((x1_lid < 1) || (x1_tid < 24));
        if (x1_cond) {
            int x1 = ((32 * x1_lid) + x1_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1) {
                int x0 = ((4 * x0_lid) + x0_tid);
                for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                    int n         = ((2 * n_lid) + n_tid);
                    float agg_out = agg[((x1_lid + (x0_lid * 2)) + (n_lid * 4))];
                    int gout_idx =
                            ((((3136 * c_gid) + (451584 * (int)(n_gid + n))) + (56 * (x0_gid + x0)))
                             + x1);
                    if (((gout_idx >= 0) && (gout_idx < 14450688))) {
#if MLO_CONV_BIAS
                        agg_out += bias[((get_group_id(1)) % 144)];
#endif
#if MLO_CONV_ACTIVE_RELU
                        agg_out = agg_out > 0.0f ? agg_out : 0.0f;
#endif
                        out[gout_idx] = agg_out;
                    }
                }
            }
        }
    }
}

__kernel void DepthwiseconvDw242n32(
        __global float* out,
        __global const float* in,
        __global const float* weights
#if MLO_CONV_BIAS
        ,
        __global float* bias
#endif
) {
    in            = (in + -29);
    int local_id0 = get_local_id(0);
    float agg[16] = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
    };
    __local float lds_in[4056];
    __local float lds_weights[9];
    int x0_gid = (get_group_id(1) * 16);
    int c_gid  = get_group_id(0);
    int n_gid  = (get_group_id(2) * 8);
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3) {
        for (int k1_gid = 0; k1_gid < 3; k1_gid += 3) {
            {
                int gbase =
                        ((((k1_gid + (k0_gid * 28)) + (x0_gid * 28)) + (c_gid * 784))
                         + (n_gid * 301056));
                int k1_x1_k0_x0_tid = (local_id0 % 256);
                for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 2; k1_x1_k0_x0_lid += 1) {
                    int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 1) || (k1_x1_k0_x0_tid < 250));
                    if (k1_x1_k0_x0_cond) {
                        int k1_x1_k0_x0 = ((256 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
                        for (int n_lid = 0; n_lid < 8; n_lid += 1) {
                            int lidx     = (k1_x1_k0_x0 + (507 * n_lid));
                            int gidx     = ((gbase + k1_x1_k0_x0) + (301056 * (int)n_lid));
                            lds_in[lidx] = in[clamp((int)gidx, (int)29, (int)9633820)];
                        }
                    }
                }
            }
            {
                int gbase        = ((k1_gid + (k0_gid * 3)) + (c_gid * 9));
                int k1_k0_c_tid  = (local_id0 % 16);
                int k1_k0_c_cond = (k1_k0_c_tid < 9);
                if (k1_k0_c_cond) {
                    if ((local_id0 < 16)) {
                        int gidx                 = (gbase + k1_k0_c_tid);
                        lds_weights[k1_k0_c_tid] = weights[clamp((int)gidx, (int)0, (int)3455)];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1)
                   && ((((k0_gid + 3) - 1) + ((x0_gid + 16) - 1)) <= 28))
                  && ((-1 * k1_gid) <= -1))
                 && ((((k1_gid + 3) - 1) + 27) <= 28))) {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1) {
                            int x0      = ((4 * x0_lid) + x0_tid);
                            int x1_cond = (x1_tid < 28);
                            int x1      = select((int)0, (int)x1_tid, (int)x1_cond);
                            for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                                int n         = ((2 * n_lid) + n_tid);
                                float val1    = lds_in[(
                                        (((k1_lid + x1) + (28 * k0_lid)) + (28 * x0)) + (507 * n))];
                                float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                int agg_idx   = (x0_lid + (n_lid * 4));
                                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                agg[agg_idx] =
                                        select((float)agg[agg_idx], (float)agg_rhs, (int)x1_cond);
                            }
                        }
                    }
                }
            } else {
                for (int k0_lid = 0; k0_lid < 3; k0_lid += 1) {
                    for (int k1_lid = 0; k1_lid < 3; k1_lid += 1) {
                        int x1_tid = (local_id0 % 32);
                        int x0_tid = ((local_id0 / 32) % 4);
                        int n_tid  = ((local_id0 / 128) % 2);
                        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1) {
                            int x0      = ((4 * x0_lid) + x0_tid);
                            int x1_cond = (x1_tid < 28);
                            int x1      = select((int)0, (int)x1_tid, (int)x1_cond);
                            for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                                int n         = ((2 * n_lid) + n_tid);
                                float val1    = lds_in[(
                                        (((k1_lid + x1) + (28 * k0_lid)) + (28 * x0)) + (507 * n))];
                                float val2    = lds_weights[(k1_lid + (3 * k0_lid))];
                                int agg_idx   = (x0_lid + (n_lid * 4));
                                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                                agg[agg_idx]  = select(
                                        (float)agg[agg_idx],
                                        (float)agg_rhs,
                                        (int)(x1_cond
                                              && ((((((-1 * (k0_gid + k0_lid))
                                                      + (-1 * (x0_gid + x0)))
                                                     <= -1)
                                                    && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 28))
                                                   && (((-1 * (k1_gid + k1_lid)) + (-1 * x1))
                                                       <= -1))
                                                  && (((k1_gid + k1_lid) + x1) <= 28))));
                            }
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    int x1_tid = (local_id0 % 32);
    int x0_tid = ((local_id0 / 32) % 4);
    int n_tid  = ((local_id0 / 128) % 2);
    for (int x0_lid = 0; x0_lid < 4; x0_lid += 1) {
        int x0_cond = ((x0_lid < 3) || (x0_gid != 16));
        if (x0_cond) {
            int x0      = ((4 * x0_lid) + x0_tid);
            int x1_cond = (x1_tid < 28);
            if (x1_cond) {
                for (int n_lid = 0; n_lid < 4; n_lid += 1) {
                    int n         = ((2 * n_lid) + n_tid);
                    float agg_out = agg[(x0_lid + (n_lid * 4))];
                    int gout_idx =
                            ((((784 * c_gid) + (301056 * (int)(n_gid + n))) + (28 * (x0_gid + x0)))
                             + x1_tid);
                    if (((gout_idx >= 0) && (gout_idx < 9633792))) {
#if MLO_CONV_BIAS
                        agg_out += bias[((get_group_id(0)) % 384)];
#endif
#if MLO_CONV_ACTIVE_RELU
                        agg_out = agg_out > 0.0f ? agg_out : 0.0f;
#endif
                        out[gout_idx] = agg_out;
                    }
                }
            }
        }
    }
}
