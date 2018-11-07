

__kernel void
Crop(__global float* __restrict out_data,
     __global const float* __restrict in_data,
     int in_n_stride,
     int in_c_stride,
     int in_h_stride,
     int in_w_stride,
     int out_n_stride,
     int out_c_stride,
     int out_h_stride,
     int out_w_stride,
     int out_n,
     int out_c,
     int out_h,
     int out_w,
     int img_offset) {

    // img_offset
    in_data += img_offset;
    int tid         = get_global_id(0);
    int global_size = get_global_size(0);
    int count       = out_n * out_c * out_h * out_w;

    for (; tid < count; tid += global_size) {
        int n = (tid / out_n_stride) % out_n;
        int c = (tid / out_c_stride) % out_c;
        int h = (tid / out_h_stride) % out_h;
        int w = (tid / out_w_stride) % out_w;

        int in_offset = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;
        out_data[tid] = in_data[in_offset];
    }
}