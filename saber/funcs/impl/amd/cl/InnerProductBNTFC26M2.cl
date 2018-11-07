
#define STRIDE (4096)
#define HSTRIDE (2048)
#define ITER (32)
#define HWG (384 >> 1)

#define OUTPUT 1470

void reduce(__local float* buffer, int tid)
{
    if(tid < 64)
    {
        buffer[tid] += buffer[tid + 64];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid < 32)
    {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid < 16)
    {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid < 8)
    {
        buffer[tid << 3] += buffer[(tid << 3) + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void InnerProduct(
    __constant float* a, __global const float* b, __global const float* bias, __global float* c)
{
    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __local float shared_b[16][65];
    __local float result[129];

    __constant float* pA     = (__constant float*)(a + ((grid_x / HWG) * STRIDE));
    __global const float* pB = (__global const float*)(b);

    int offset = (((grid_x % HWG) << 6)) % HSTRIDE;

    float sum = 0.0f;

    for(int i = 0; i < ITER; i++, offset = (offset + 64) % HSTRIDE)
    {
        for(int j = 0; j < 8; j++)
        {
            shared_b[(lid_x >> 6 << 3) + j][(lid_x & 63)] =
                (offset + ((lid_x >> 6) * (HSTRIDE)) + (j + ((grid_x % HWG) << 3)) * STRIDE +
                             (lid_x & 63) <
                         OUTPUT * STRIDE
                     ? pB[offset + ((lid_x >> 6) * (HSTRIDE)) +
                          (j + ((grid_x % HWG) << 3)) * STRIDE + (lid_x & 63)]
                     : 0.0f); // correct
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < 8; k++)
        {
            sum += pA[(offset + ((lid_x & 7) << 3) + k) % HSTRIDE + ((lid_x >> 6) * (HSTRIDE))] *
                   shared_b[(lid_x >> 6 << 3) + ((lid_x & 63) >> 3)]
                           [((lid_x & 7) << 3) + k]; // correct
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[lid_x] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    reduce(result, lid_x);

    if(lid_x < 8 && ((grid_x % HWG) << 3) + lid_x < OUTPUT)
    {
        int out_offset = ((grid_x / HWG) * OUTPUT + ((grid_x % HWG) << 3) + lid_x);
        c[out_offset]  = bias[((grid_x % HWG) << 3) + lid_x] + result[(lid_x << 3)];
    }
}
