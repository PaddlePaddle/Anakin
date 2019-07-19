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
#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8
#define dtype _FLOAT

#define BLOCK_SIZE 16
#define MAX_FILTER_SIZE 32
#define LOCAL_INPUT_SIZE (BLOCK_SIZE + MAX_FILTER_SIZE - 1)


__kernel void direct_deconv(__global const dtype* din,
                            __global const dtype* bias_data, __global const dtype* const weight_data,
                            const int num, const int in_channels, const int out_channels,
                            const int hout, const int wout, const int channel_out_stride,
                            const int hin, const int win, const int channel_in_stride,
                            const int kernel_h, const int kernel_w, const int kernel_size,
                            const int stride_h, const int stride_w,
                            const int pad_h, const int pad_w,
                            const int dilation_h, const int dilation_w,
                            __global dtype* dout, const int bias_flag, const int relu_flag) {

    __local dtype local_inputs[LOCAL_INPUT_SIZE * LOCAL_INPUT_SIZE];

    const int group_i = get_group_id(0);
    const int group_j = get_group_id(1);

    const int local_id_x = get_local_id(0); // 0 ~ BLOCK_SIZE-1
    const int local_id_y = get_local_id(1); // 0 ~ BLOCK_SIZE-1

    const int global_id_x = get_global_id(0);
    const int global_id_y = get_global_id(1);

    const int input_in_tile = (BLOCK_SIZE + kernel_h - 1) * (BLOCK_SIZE + kernel_w - 1);

    if (global_id_x < wout && global_id_y < hout) {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = local_id_x; i < input_in_tile; i += BLOCK_SIZE) {
                //local_inputs[] = din[input_start_x]
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    //__local dtype local_inputs[]
    //__local dtype local_weights[1024] // Must smaller than 32*32

    /*
        //int wo = blockIdx.x * blockDim.x + threadIdx.x;
        int wo = get_global_id(0);
        int w =  wo + pad_w;

        //int ho = blockIdx.y * blockDim.y + threadIdx.y;
        int ho = get_global_id(1);
        int h =  ho + pad_h;

        //int iout = blockIdx.z;
        int iout = get_group_id(2);

        int cout = iout % out_channels;
        int n = iout / out_channels;
        int iin = n * in_channels;
        int idx_out = iout * channel_out_stride + ho * wout + wo;
        __local dtype local_weights[1024]; // must smaller than 32*32 per channel
        __local dtype local_inputs[1024]; // must smaller than 32*32 per channel

        dtype val = 0;

        if (wo < wout && ho < hout) {
        //printf("idx_out(%d) = iout(%d) * channel_out_stride(%d) + ho(%d) * wout(%d) + wo(%d)", idx_out, iout, channel_out_stride, ho, wout, wo);
            for(int ic = 0; ic < in_channels; ic++) {
                //! read weights
                //int idx_weight = threadIdx.y * blockDim.x + threadIdx.x;
                //int idx_weight = get_local_id(1) * get_local_size(0) + get_local_id(0);

                // Support without pad cases if channel_out_stride < kernel_size
                int idx_weight = idx_out % kernel_size;

                //How to enumerate all index from 0 to kernel_size - 1
                //Only run here wout * hout times
                //int idx_weight = get_local_id(0)
                //printf("idx_out = %d, idx_weight = %d", idx_out, get_local_id(0));
                if (idx_weight < kernel_size) {
                   //local_weights[idx_weight] = weight_data[(cout * in_channels + ic) * kernel_size + idx_weight];
                   //local_weights[idx_weight] = weight_data[cout * kernel_size + ic * out_channels * kernel_size + idx_weight];
                   local_weights[idx_weight] = weight_data[(cout + ic * out_channels) * kernel_size + idx_weight];
                }
                //__syncthreads();
                barrier(CLK_LOCAL_MEM_FENCE);
                //if (idx_weight==2 || idx_weight==3) {
                    //printf("local_weights[%d] = weight_data[%d] (iout=%d,cout=%d,in_channels=%d,ic=%d) = %2.5f", idx_weight, (cout * in_channels + ic) * kernel_size + idx_weight, iout, cout, in_channels, ic, local_weights[idx_weight]);
                //}
                //! get start and end index
                const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
                const int phend = min(h / stride_h + 1, hin);
                const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
                const int pwend = min(w / stride_w + 1, win);

                const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
                const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

                __global const dtype* din_c = din + (iin + ic) * channel_in_stride;

                if (idx_out == 20) {
                    printf("phstart=%d ~ phend=%d", phstart, phend);
                    printf("khstart=%d", khstart);
                    printf("kwstart=%d", kwstart);
                }

                //printf("local_weights[%d] = %2.5f", idx_weight, local_weights[idx_weight]);

                //! start computation
                for (int ph = phstart; ph < phend; ++ph) {
                    for (int pw = pwstart; pw < pwend; ++pw) {
                        int kh = khstart - (ph - phstart) * stride_h;
                        int kw = kwstart - (pw - pwstart) * stride_w;
                        val += din_c[ph * win + pw] * local_weights[kh * kernel_w + kw];
                        if (idx_out == 20) {
                            printf("ph * win + pw = %d * %d + %d", ph, win, pw);
                            printf("kh * kernel_w + kw = %d * %d + %d", kh, kernel_w, kw);
                            printf("[idx_out=%d] += val = din_c[%d] * local_weights[%d] = %2.5f * %2.5f", idx_out, ph * win + pw, kh * kernel_w + kw, din_c[ph * win + pw], local_weights[kh * kernel_w + kw]);
                        }

                    }
                }
                //barrier(CLK_LOCAL_MEM_FENCE);

            }
            //! final computation
            if (bias_flag) {
                val += bias_data[cout];
            }
            if (relu_flag) {
                val = val > (dtype)0? val : (dtype)0;
            }
            dout[idx_out] = val;
            //printf("dout[%d] = %2.5f\n", idx_out, val);
        }
    */
}

//template <typename dtype, bool flag_bias, bool flag_act>
__kernel void direct_deconv_v0_1(__global const dtype* din,
                                 __global const dtype* bias_data, __global const dtype* const weight_data,
                                 const int num, const int in_channels, const int out_channels,
                                 const int hout, const int wout, const int channel_out_stride,
                                 const int hin, const int win, const int channel_in_stride,
                                 const int kernel_h, const int kernel_w, const int kernel_size,
                                 const int stride_h, const int stride_w,
                                 const int pad_h, const int pad_w,
                                 const int dilation_h, const int dilation_w,
                                 __global dtype* dout, const int bias_flag, const int relu_flag) {
    //int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int wo = get_global_id(0);
    int w =  wo + pad_w;

    //int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int ho = get_global_id(1);
    int h =  ho + pad_h;

    //int iout = blockIdx.z;
    int iout = get_group_id(2);

    int cout = iout % out_channels;
    int n = iout / out_channels;
    int iin = n * in_channels;
    int idx_out = iout * channel_out_stride + ho * wout + wo;
    __local dtype sharedw[1024]; // must smaller than 32*32

    if (wo < wout && ho < hout) {
        //printf("idx_out(%d) = iout(%d) * channel_out_stride(%d) + ho(%d) * wout(%d) + wo(%d)", idx_out, iout, channel_out_stride, ho, wout, wo);
        for (int ic = 0; ic < in_channels; ic++) {
            //! read weights
            //int idx_weight = threadIdx.y * blockDim.x + threadIdx.x;
            //int idx_weight = get_local_id(1) * get_local_size(0) + get_local_id(0);

            // Support without pad cases if channel_out_stride < kernel_size
            int idx_weight = idx_out % kernel_size;

            //How to enumerate all index from 0 to kernel_size - 1
            //Only run here wout * hout times
            //int idx_weight = get_local_id(0)
            //printf("idx_out = %d, idx_weight = %d", idx_out, get_local_id(0));
            if (idx_weight < kernel_size) {
                //sharedw[idx_weight] = weight_data[(cout * in_channels + ic) * kernel_size + idx_weight];
                //sharedw[idx_weight] = weight_data[cout * kernel_size + ic * out_channels * kernel_size + idx_weight];
                sharedw[idx_weight] = weight_data[(cout + ic * out_channels) * kernel_size + idx_weight];
            }

            //__syncthreads();
            barrier(CLK_LOCAL_MEM_FENCE);
            //if (idx_weight==2 || idx_weight==3) {
            //printf("sharedw[%d] = weight_data[%d] (iout=%d,cout=%d,in_channels=%d,ic=%d) = %2.5f", idx_weight, (cout * in_channels + ic) * kernel_size + idx_weight, iout, cout, in_channels, ic, sharedw[idx_weight]);
            //}
            //! get start and end index
            const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
            const int phend = min(h / stride_h + 1, hin);
            const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
            const int pwend = min(w / stride_w + 1, win);

            const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
            const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

            __global const dtype* din_c = din + (iin + ic) * channel_in_stride;
            /*
                        if (idx_out == 0) {
                            printf("phstart=%d ~ phend=%d", phstart, phend);
                            printf("khstart=%d", khstart);
                            printf("kwstart=%d", kwstart);
                        }
            */
            //printf("sharedw[%d] = %2.5f", idx_weight, sharedw[idx_weight]);

            dtype val = 0;
            //! start computation
            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    int kh = khstart - (ph - phstart) * stride_h;
                    int kw = kwstart - (pw - pwstart) * stride_w;
                    val += din_c[ph * win + pw] * sharedw[kh * kernel_w + kw];
                    /*                    if (idx_out == 0) {
                                            printf("ph * win + pw = %d * %d + %d", ph, win, pw);
                                            printf("kh * kernel_w + kw = %d * %d + %d", kh, kernel_w, kw);
                                            printf("[idx_out=%d] += val = din_c[%d] * sharedw[%d] = %2.5f * %2.5f", idx_out, ph * win + pw, kh * kernel_w + kw, din_c[ph * win + pw], sharedw[kh * kernel_w + kw]);
                                        }
                    */
                }
            }

            //barrier(CLK_LOCAL_MEM_FENCE);

        }

        //! final computation
        if (bias_flag) {
            val += bias_data[cout];
        }

        if (relu_flag) {
            val = val > (dtype)0 ? val : (dtype)0;
        }

        dout[idx_out] = val;
        //printf("dout[%d] = %2.5f\n", idx_out, val);
    }
}

__kernel void direct_deconv_ck_equal(__global const dtype* din,
                                     __global const dtype* bias_data, __global const dtype* const weight_data,
                                     const int num, const int in_channels, const int out_channels,
                                     const int hout, const int wout, const int channel_out_stride,
                                     const int hin, const int win, const int channel_in_stride,
                                     const int kernel_h, const int kernel_w, const int kernel_size,
                                     const int stride_h, const int stride_w,
                                     const int pad_h, const int pad_w,
                                     const int dilation_h, const int dilation_w,
                                     __global dtype* dout, const int bias_flag, const int relu_flag) {
    //int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int wo = get_global_id(0);
    int w =  wo + pad_w;

    //int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int ho = get_global_id(1);
    int h =  ho + pad_h;

    //int iout = blockIdx.z;
    int iout = get_group_id(2);

    int cout = iout % out_channels;
    int n = iout / out_channels;
    int iin = n * in_channels;
    int idx_out = iout * channel_out_stride + ho * wout + wo;
    __local dtype sharedw[1024]; //k <= 16, 16*16*4

    if (wo < wout && ho < hout) {
        //if (wo < wout && ho < hout) {
        printf("idx_out(%d) = iout(%d) * channel_out_stride(%d) + ho(%d) * wout(%d) + wo(%d)", idx_out,
               iout, channel_out_stride, ho, wout, wo);

        for (int ic = 0; ic < in_channels; ic++) {
            //! read weights
            //int idx_weight = threadIdx.y * blockDim.x + threadIdx.x;
            //int idx_weight = get_local_id(1) * get_local_size(0) + get_local_id(0);
            //idx_weight = idx_weight % kernel_size;
            int idx_weight = idx_out % kernel_size;
            printf("idx_out = %d, idx_weight = %d * %d + %d", idx_out, get_local_id(1), get_local_size(0),
                   get_local_id(0));

            if (idx_weight < kernel_size) {
                //sharedw[idx_weight] = weight_data[(cout * in_channels + ic) * kernel_size + idx_weight];
                sharedw[idx_weight] = weight_data[(ic * in_channels + cout) * kernel_size + idx_weight];
            }

            //__syncthreads();
            barrier(CLK_LOCAL_MEM_FENCE);
            //if (idx_weight==2 || idx_weight==3) {
            printf("sharedw[%d] = weight_data[%d] (iout=%d,cout=%d,in_channels=%d,ic=%d) = %2.5f", idx_weight,
                   (cout * in_channels + ic) * kernel_size + idx_weight, iout, cout, in_channels, ic,
                   sharedw[idx_weight]);
            //}
            //! get start and end index
            const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
            const int phend = min(h / stride_h + 1, hin);
            const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
            const int pwend = min(w / stride_w + 1, win);

            const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
            const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

            __global const dtype* din_c = din + (iin + ic) * channel_in_stride;

            if (idx_out == 7) {
                printf("phstart=%d ~ phend=%d", phstart, phend);
                printf("khstart=%d", khstart);
                printf("kwstart=%d", kwstart);
            }

            //printf("sharedw[%d] = %2.5f", idx_weight, sharedw[idx_weight]);

            dtype val = 0;
            //! start computation
            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    int kh = khstart - (ph - phstart) * stride_h;
                    int kw = kwstart - (pw - pwstart) * stride_w;
                    val += din_c[ph * win + pw] * sharedw[kh * kernel_w + kw];

                    if (idx_out == 7) {
                        printf("ph * win + pw = %d * %d + %d", ph, win, pw);
                        printf("kh * kernel_w + kw = %d * %d + %d", kh, kernel_w, kw);
                        printf("[idx_out=%d] += val = din_c[%d] * sharedw[%d] = %2.5f * %2.5f", idx_out, ph * win + pw,
                               kh * kernel_w + kw, din_c[ph * win + pw], sharedw[kh * kernel_w + kw]);
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

        }

        //! final computation
        if (bias_flag) {
            val += bias_data[cout];
        }

        if (relu_flag) {
            val = val > (dtype)0 ? val : (dtype)0;
        }

        dout[idx_out] = val;
        //printf("dout[%d] = %2.5f\n", idx_out, val);
    }
}

__kernel void depthwise_deconv_2d(const int channel_in_stride, const int channel_out_stride,
                                  const int kernel_size,
                                  __global const dtype* din, const int num, const int channels,
                                  const int hin, const int win, const int hout,
                                  const int wout, const int kernel_h, const int kernel_w,
                                  const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                  __global dtype* dout, __global const dtype* weight, __global const dtype* const bias,
                                  const int bias_flag, const int relu_flag) {

    //int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int wo = get_global_id(0);
    int w =  wo + pad_w;
    //int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int ho = get_global_id(1);
    int h =  ho + pad_h;
    //int c = blockIdx.z % channels;
    //int i = blockIdx.z;
    int c = get_group_id(2) % channels;
    int i = get_group_id(2);
    int index = i * channel_out_stride + ho * wout + wo;

    __local dtype sharedw[256];
    //int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int idx = get_local_id(1) * get_local_size(0) + get_local_id(0);

    if (idx < kernel_size) {
        sharedw[idx] = weight[c * kernel_size + idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (wo < wout && ho < hout) {
        const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        const int phend = min(h / stride_h + 1, hin);
        const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        const int pwend = min(w / stride_w + 1, win);

        const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
        const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

        dtype gradient = 0;
        const global dtype* top_diff_slice = din + i * channel_in_stride;

        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int kh = khstart - (ph - phstart) * stride_h;
                int kw = kwstart - (pw - pwstart) * stride_w;
                gradient += top_diff_slice[ph * win + pw] * sharedw[kh * kernel_w + kw];
            }
        }

        if (bias_flag) {
            gradient += bias[c];
        }

        if (relu_flag) {
            gradient = gradient > (dtype)0 ? gradient : (dtype)0;
        }

        dout[index] = gradient;
    }
}

