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
__kernel void Depthwiseconv(
    global float* din,
    const int num,
    const int channels,
    const int hin,
    const int win,
    const int hout,
    const int wout,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    global float* dout,
    global float* weight
#if MLO_CONV_BIAS
    ,
    global float* bias
#endif
) {
    int local_idx        = get_global_id(0);
    int size_channel_in  = hin * win;
    int size_channel_out = hout * wout;
    int size_kernel      = kernel_h * kernel_w;

    const int pw = local_idx % wout;
    const int ph = (local_idx / wout) % hout;
    const int c  = (local_idx / size_channel_out) % channels;
    const int n  = local_idx / size_channel_out / channels;
    int hstart   = ph * stride_h - pad_h;
    int wstart   = pw * stride_w - pad_w;
    int hend     = min(hstart + kernel_h, hin + pad_h);
    int wend     = min(wstart + kernel_w, win + pad_w);

    hstart                     = max(hstart, 0);
    wstart                     = max(wstart, 0);
    hend                       = min(hend, hin);
    wend                       = min(wend, win);
    float aveval               = 0;
    global float* bottom_slice = din + (n * channels + c) * size_channel_in;
    global float* weight_slice = weight + c * size_kernel;

    int khstart = hend < kernel_h ? kernel_h - hend : 0;
    int kwstart = wend < kernel_w ? kernel_w - wend : 0;

    for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
            aveval += bottom_slice[h * win + w]
                      * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
        }
    }

#if MLO_CONV_BIAS
    aveval += bias[c];
#endif
#if MLO_CONV_ACTIVE_RELU
    aveval = max(aveval, (float)0);
#endif
    dout[local_idx] = aveval;
}
