/////////////////////////////////////////////////////////
// FC6 batch 32 Version 4 2018.7.21

#define STRIDE (25088 * 2)
#define QSTRIDE (6272 * 2)
#define CSTRIDE (4096)
#define ITER (196 * 2)

#define ALTREAD

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c)
{
    __local float shared_a[1536];
    __local float shared_b[2560];
    __local float2* pShared_a = (__local float2*)shared_a;
    __local float4* pShared_b = (__local float4*)shared_b;

    float2 sha;
    float* pSha = (float*)&sha;
    float4 shb;
    float* pShb = (float*)&shb;
    float4 sum[2];
    float* pSum = (float*)sum;

    uint lid_x  = get_local_id(0);
    uint grid_x = get_group_id(0);

    __global const float* pA =
        (__global const float*)(a + (lid_x >> 6 << 3) * STRIDE + (grid_x >> 6) * QSTRIDE);
    __global const float* pB =
        (__global const float*)(b + (((grid_x & 63) << 6) + (lid_x >> 6 << 4)) * STRIDE +
                                (grid_x >> 6) * QSTRIDE);
    __global float4* pC    = (__global float4*)(c + ((lid_x >> 4 << 1) * CSTRIDE +
                                                  ((grid_x & 63) << 6) + ((lid_x & 15) << 2)));
    __global float4* pBias = (__global float4*)(bias + ((grid_x & 63) << 6) + ((lid_x & 15) << 2));

    int offset = (((grid_x & 63) << 5)) % QSTRIDE;

    for(uint i = 0; i < 2; i++)
    {
        sum[i] = 0.0f;
    }

    for(ushort i = 0; i < ITER; i++, offset = (offset + 32) % QSTRIDE)
    {
        for(uint j = 0; j < 4; j++)
        {
            shared_a[((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 3)) + ((lid_x & 31)) * 32 +
                     ((((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 3)) +
                       ((lid_x & 31)) * 32) >>
                      5)] = pA[((j << 1) + ((lid_x & 63) >> 5)) * STRIDE + (lid_x & 31) + offset];
        }
        for(uint j = 0; j < 8; j++)
        {
            shared_b[((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 4)) + ((lid_x & 31)) * 64 +
                     ((((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 4)) +
                       ((lid_x & 31)) * 64) >>
                      5)] = pB[((j << 1) + ((lid_x & 63) >> 5)) * STRIDE + (lid_x & 31) + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint k = 0; k < 32; k++)
        {
            for(uint m = 0; m < 2; m++)
            {
                pSha[m] = shared_a[((lid_x >> 4) << 1) + m + k * 32 +
                                   ((((lid_x >> 4) << 1) + m + k * 32) >> 5)];
            }

            for(uint l = 0; l < 4; l++)
            {
                pShb[l] = shared_b[((lid_x & 15) << 2) + l + k * 64 +
                                   ((((lid_x & 15) << 2) + l + k * 64) >> 5)];
            }

            for(uint m = 0; m < 2; m++)
            {
                for(uint l = 0; l < 4; l++)
                {
                    pSum[m * 4 + l] += pSha[m] * pShb[l];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if((grid_x >> 6) == 0)
    {
        for(uint i = 0; i < 2; i++)
        {
            pC[i * CSTRIDE >> 2] = sum[i] + pBias[0];
        }
    }
    else
    {
        for(uint i = 0; i < 2; i++)
        {
            pC[i * CSTRIDE >> 2] += sum[i];
        }
    }
}
