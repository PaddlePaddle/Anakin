# Benchmark

## Benchmark Model  

> Please note that you should transform caffe model or others into anakin model with the help of [`external converter ->`](#)

The following convolutional neural networks are tested with both `Anakin` and `TenorRT5` on GPU.
 You can use pretrained caffe model or the model trained by youself.

- [Vgg16]()  *caffe model can be found [here->](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)*
- [Resnet50]()  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Resnet101]()  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Mobilenet v1]()  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [Mobilenet v2]()  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [mobilenet-ssd]()  *caffe model can be found [here->](https://github.com/chuanqi305/MobileNet-SSD)*


## NV GPU Benchmark
### Machine And Enviornment
>  CPU: `Intel(R) Xeon(R) CPU 5117 @ 2.0GHz`
>  GPU: `Tesla P4`
>  cuda: `CUDA8`
>  cuDNN: `v7`

* Time：warmup 10，running 1000 times to get average time
* Latency (`ms`) and Memory(MB) of different batch

> The counterpart of **`Anakin`** is the acknowledged high performance inference engine **`NVIDIA TensorRT 5`** ,   The models which TensorRT 5 doesn't support we use the custom plugins  to support.

### <span id = '1'> VGG16 </span>

| Batch_Size | RT latency FP32(ms) | Anakin2 Latency FP32 (ms) |RT Memory (MB) | Anakin2 Memory (MB) |
|------------|---------------------|---------------------------|---------------|---------------------|
| 1          | 8.52532             | 8.2387                    |1090.89        | 702                 |
| 2          | 14.1209             | 13.8772                   |1056.02        | 768.76              |
| 4          | 24.4529             | 24.3391                   |1002.17        | 840.54              |
| 8          | 46.7956             | 46.3309                   |1098.98        | 935.61              |


### <span id = '2'> Resnet50 </span>

| Batch_Size | RT latency FP32(ms) | Anakin2 Latency FP32 (ms) | RT Latency INT8 (ms) | Anakin2 Latency INT8 (ms) | RT Memory FP32(MB) | Anakin2 Memory FP32(MB) |
|------------|---------------------|---------------------------|----------------------|---------------------------|--------------------|-------------------------|
| 1          | 4.6447              | 3.0863                    | 1.78892              | 1.61537                   | 1134.88            | 311.25                  |
| 2          | 6.69187             | 5.13995                   | 2.71136              | 2.70022                   | 1108.86            | 382                     |
| 4          | 11.1943             | 9.20513                   | 4.16771              | 4.77145                   | 885.96             | 406.86                  |
| 8          | 19.8769             | 17.1976                   | 6.2798               | 8.68197                   | 813.84             | 532.61                  |


### <span id = '3'> Resnet101 </span>

| Batch_Size | RT latency (ms) | Anakin2 Latency (ms) | RT Latency INT8 (ms) | Anakin2 Latency INT8 (ms) | RT Memory (MB) | Anakin2 Memory (MB) |
|------------|-----------------|----------------------|----------------------|---------------------------|----------------|---------------------|
| 1          | 9.98695         | 5.44947              | 2.81031              | 2.74399                   | 1159.16        | 500.5               |
| 2          | 17.3489         | 8.85699              | 4.8641               | 4.69473                   | 1158.73        | 492                 |
| 4          | 20.6198         | 16.8214              | 7.11608              | 8.45324                   | 1021.68        | 541.08              |
| 8          | 31.9653         | 33.5015              | 11.2403              | 15.4336                   | 914.49         | 611.54              |


## X86 CPU Benchmark
### Machine And Enviornment
>  CPU: `Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz` with HT, for FP32 test
>  CPU: `Intel(R) Xeon(R) Gold 6271 CPU @ 2.60GHz` with HT, for INT8 test
>  System: `CentOS 6.3` with `GCC 4.8.2`, for benchmark between Anakin and Intel Caffe

* All test enable `8 thread parallel`
* Time：warmup 10，running 200 times to get average time

> The counterpart of **`Anakin`** is **`Intel Cafe`(1.1.6)** with mklml.

| Net_Name    | Batch_Size | Anakin2 Latency(2650v4) fp32 (ms) | caffe Latency(2650v4) fp32 (ms) | Anakin2 Latency int8(6271) (ms) |
|-------------|----|-------------------------------------|-----------------------------------|---------------------------------|
| resnet50    | 1  | 20.6201                             | 24.1369                           | 3.20866                         |
| resnet50    | 2  | 39.2286                             | 43.1096                           | 5.44311                         |
| resnet50    | 4  | 77.1392                             | 81.8814                           | 9.93424                         |
| resnet50    | 8  | 152.941                             | 158.321                           | 19.5618                         |
| vgg16       | 1  | 55.6132                             | 70.532                            | 15.3181                         |
| vgg16       | 2  | 96.5034                             | 131.451                           | 22.5082                         |
| vgg16       | 4  | 180.479                             | 247.926                           | 37.2974                         |
| vgg16       | 8  | 346.619                             | 485.44                            | 67.6682                         |
| mobilenetv1 | 1  | 3.98104                             | 5.42775                           | 0.926546                        |
| mobilenetv1 | 2  | 7.27079                             | 9.16058                           | 1.35007                         |
| mobilenetv1 | 4  | 14.4029                             | 16.2505                           | 2.37271                         |
| mobilenetv1 | 8  | 29.1651                             | 29.8381                           | 3.75992                         |
| vgg16_ssd   | 1  | 125.948                             | 143.412                           |                                 |
| vgg16_ssd   | 2  | 247.242                             | 266.22                            |                                 |
| vgg16_ssd   | 4  | 488.377                             | 510.978                           |                                 |
| vgg16_ssd   | 8  | 972.762                             | 995.407                           |                                 |
| mobilenetv2 | 1  | 3.78504                             | 23.0066                           |                                 |
| mobilenetv2 | 2  | 7.24622                             | 65.9301                           |                                 |
| mobilenetv2 | 4  | 13.7638                             | 85.3893                           |                                 |
| mobilenetv2 | 8  | 28.4093                             | 131.669                           |


## ARM CPU Benchmark
### Machine And Enviornment
>  CPU: `Kirin 980`
>  CPU: `Snapdragon 652`
>  CPU: `Snapdragon 855`
>  CPU: `RK3399`

* Compile circumstance: Android ndk cross compile，gcc 4.9，enable neon
* Time：warmup 10，running 10 times to get average time
* Note: 1、shufflenetv2 int8 model add swish operator
> The counterpart of **`Anakin`** is **`ncnn`(20190320)**. This benchmark we test ARMv7 ARMv8 splitly

### ARMv8 TEST
* ABI： arm64-v8a
- Latency (`ms`) of `one batch`

| Kirin 980       | Anakin fp32 |          |          | Anakin int8 |          |          | NCNN fp32 |          |          | NCNN int8 |          |          |
|---------------|-------------|----------|----------|-------------|----------|----------|-----------|----------|----------|-----------|----------|----------|
|               | 1 thread        | 2 thread     | 4 thread     | 1 thread        | 2 thread     | 4 thread     | 1 thread      | 2 thread     | 4 thread     | 1 thread      | 2 thread     | 4 thread     |
| mobilenet_v1  | 34.172      | 19.369   | 12.723   | 37.588      | 20.692   | 13.280   | 45.420    | 24.220   | 16.730   | 50.560    | 27.820   | 20.010   |
| mobilenet_v2  | 30.489      | 17.784   | 12.327   | 29.581      | 17.208   | 15.307   | 30.390    | 17.310   | 12.900   |           |          |          |
| mobilenet_ssd | 71.609      | 37.477   | 28.952   |             |          |          | 88.220    | 70.070   | 66.430   | 103.700   | 85.160   | 85.320   |
| resnet50      | 255.748     | 137.842  | 104.628  |             |          |          | 1299.480  | 695.830  | 498.010  | 243.360   | 131.100  | 89.800   |
| shufflenetv1  | 11.544      | 8.931    | 7.027    |             |          |          | 12.810    | 9.390    | 8.030    |           |          |          |
| shufflenetv2  | 11.687      | 7.899    | 5.321    | 20.402      | 11.529   | 9.061    |           |          |          |           |          |          |
| squeezenet    | 28.580      | 16.638   | 14.435   |             |          |          |           |          |          |           |          |          |
| googlenet     | 93.917      | 52.742   | 40.301   | 130.875     | 72.522   | 54.204   |

---
---

| Snapdragon 855        | Anakin fp32 |          |         | Anakin int8 |          |          | NCNN fp32 |           |          | NCNN int8 |          |          |
|---------------|-------------|----------|---------|-------------|----------|----------|-----------|-----------|----------|-----------|----------|----------|
|               | 1 thread        | 2 thread     | 4 thread    | 1 thread        | 2 thread     | 4 thread     | 1 thread      | 2 thread      | 4 thread     | 1 thread      | 2 thread     | 4 thread     |
| mobilenet_v1  | 32.019      | 19.024   | 10.491  | 34.363      | 20.292   | 10.382   | 37.110    | 22.310    | 13.520   | 47.430    | 28.350   | 15.830   |
| mobilenet_v2  | 28.533      | 17.455   | 10.433  | 24.487      | 15.182   | 9.133    | 25.060    | 15.970    | 11.250   |           |          |          |
| mobilenet_ssd | 66.454      | 41.397   | 23.639  |             |          |          | 101.560   | 69.380    | 43.930   | 136.420   | 91.010   | 47.490   |
| resnet50      | 201.362     | 132.133  | 78.300  |             |          |          | 1141.290  | 724.090   | 385.990  | 229.020   | 138.450  | 82.060   |
| shufflenetv1  | 10.153      | 7.101    | 5.327   |             |          |          | 11.610    | 8.020     | 5.870    |           |          |          |
| shufflenetv2  | 10.868      | 6.713    | 4.526   | 17.306      | 10.987   | 6.788    |           |           |          |           |          |          |
| squeezenet    | 25.880      | 16.134   | 9.697   |             |          |          |           |           |          |           |          |          |
| googlenet     | 85.774      | 54.518   | 34.025  | 118.120     | 73.686   | 41.865   |

---
---

| Snapdragon 652        | Anakin fp32 |          |          | Anakin int8 |          |          | NCNN fp32 |           |          | NCNN int8 |          |          |
|---------------|-------------|----------|----------|-------------|----------|----------|-----------|-----------|----------|-----------|----------|----------|
|               | 1 thread        | 2 thread     | 4 thread     | 1 thread        | 2 thread     | 4 thread     | 1 thread      | 2 thread      | 4 thread     | 1 thread      | 2 thread     | 4 thread     |
| mobilenet_v1  | 109.994     | 54.937   | 33.174   | 83.887      | 43.639   | 24.665   | 123.320   | 122.670   | 65.100   | 128.800   | 154.370  | 125.570  |
| mobilenet_v2  | 80.712      | 46.314   | 30.874   | 69.340      | 43.590   | 31.864   | 89.920    | 90.900    | 55.320   |           |          |          |
| mobilenet_ssd | 246.459     | 121.684  | 134.019  |             |          |          | 248.190   | 138.170   | 142.350  | 247.020   | 145.080  | 211.000  |
| resnet50      | 673.285     | 346.287  | 378.065  |             |          |          | 880.940   | 514.190   |          | 533.760   | 313.630  |          |
| shufflenetv1  | 34.948      | 26.635   | 21.571   |             |          |          | 39.950    | 25.520    | 20.180   |           |          |          |
| shufflenetv2  | 35.530      | 21.440   | 16.434   | 49.498      | 29.116   | 19.346   |           |           |          |           |          |          |
| squeezenet    | 87.037      | 47.192   | 28.663   |             |          |          |           |           |          |           |          |          |
| googlenet     | 268.023     | 148.533  | 95.624   | 236.492     | 131.510  | 81.561   |

---
---

| RK3399        | Anakin fp32 |          |      | Anakin int8 |          |      | NCNN fp32 |          |      | NCNN int8 |          |      |
|---------------|-------------|----------|------|-------------|----------|------|-----------|----------|------|-----------|----------|------|
|               | 1 thread        | 2 thread     | 4 thread | 1 thread        | 2 thread     | 4 thread | 1 thread      | 2 thread     | 4 thread | 1 thread      | 2 thread     | 4 thread |
| mobilenet_v1  | 111.317     | 60.008   |      | 87.201      | 45.693   |      | 149.270   | 91.200   |      | 142.790   | 86.140   |      |
| mobilenet_v2  | 105.767     | 60.899   |      | 79.065      | 53.914   |      | 118.530   | 86.900   |      |           |          |      |
| mobilenet_ssd | 232.923     | 128.337  |      |             |          |      | 268.900   | 157.860  |      | 256.560   | 149.730  |      |
| resnet50      | 671.800     | 369.386  |      |             |          |      | 1029.300  | 571.230  |      | 569.250   | 344.830  |      |
| shufflenetv1  | 38.761      | 25.971   |      |             |          |      |           |          |      |           |          |      |
| shufflenetv2  | 36.220      | 22.095   |      | 51.879      | 30.351   |      |           |          |      |           |          |      |
| squeezenet    | 98.489      | 54.863   |      |             |          |      |           |          |      |           |          |      |
| googlenet     | 274.166     | 159.429  |      | 235.085     | 133.044  |



### ARMv7 TEST
* ABI： armveabi-v7a with neon
- Latency (`ms`) of `one batch`

| Kirin 980       | Anakin fp32 |          |          | Anakin int8 |          |          | NCNN fp32 |          |          | NCNN int8 |          |          |
|---------------|-------------|----------|----------|-------------|----------|----------|-----------|----------|----------|-----------|----------|----------|
|               | 1 thread        | 2 thread     | 4 thread     | 1 thread        | 2 thread     | 4 thread     | 1 thread      | 2 thread     | 4 thread     | 1 thread      | 2 thread     | 4 thread     |
| mobilenet_v1  | 39.051      | 19.813   | 14.184   | 39.026      | 22.048   | 14.250   | 50.240    | 26.850   | 20.010   | 92.900    | 49.420   | 37.160   |
| mobilenet_v2  | 36.052      | 19.550   | 14.507   | 32.656      | 19.641   | 15.735   | 35.890    | 20.730   | 18.550   |           |          |          |
| mobilenet_ssd | 83.474      | 44.530   | 33.116   |             |          |          | 99.960    | 53.160   | 84.360   | 180.000   | 91.380   | 68.140   |
| resnet50      | 291.478     | 158.954  | 129.484  |             |          |          | 1412.37   | 766.62   | 560.760  | 355.010   | 189.18   | 133.410  |
| shufflenetv1  | 11.909      | 9.761    | 7.441    |             |          |          | 16.030    | 10.660   | 8.120    |           |          |          |
| shufflenetv2  | 11.755      | 7.983    | 6.289    | 21.968      | 14.111   | 9.888    |           |          |          |           |          |          |
| squeezenet    | 30.148      | 20.908   | 17.084   |             |          |          |           |          |          |           |          |          |
| googlenet     | 108.210     | 65.798   | 58.630   | 140.886     | 79.910   | 60.693   |

---
---

| Snapdragon 855        | Anakin fp32 |          |         | Anakin int8 |          |          | NCNN fp32 |           |          | NCNN int8 |          |          |
|---------------|-------------|----------|---------|-------------|----------|----------|-----------|-----------|----------|-----------|----------|----------|
|               | 1 thread        | 2 thread     | 4 thread    | 1 thread        | 2 thread     | 4 thread     | 1 thread      | 2 thread      | 4 thread     | 1 thread      | 2 thread     | 4 thread     |
| mobilenet_v1  | 34.015      | 20.064   | 11.410  | 42.222      | 21.532   | 11.746   | 41.150    | 24.870    | 18.420   | 79.180    | 48.470   | 24.530   |
| mobilenet_v2  | 30.742      | 18.507   | 11.354  | 24.628      | 15.133   | 9.079    | 30.060    | 19.220    | 15.520   |           |          |          |
| mobilenet_ssd | 69.749      | 44.010   | 26.000  |             |          |          | 85.030    | 62.770    | 48.940   | 154.600   | 138.700  | 82.140   |
| resnet50      | 218.581     | 146.509  | 92.899  |             |          |          | 1380.340  | 996.410   | 540.660  | 324.720   | 261.920  | 126.270  |
| shufflenetv1  | 11.032      | 7.430    | 5.369   |             |          |          | 13.390    | 9.270     | 6.360    |           |          |          |
| shufflenetv2  | 11.372      | 7.120    | 4.728   | 19.393      | 12.278   | 7.719    |           |           |          |           |          |          |
| squeezenet    | 27.860      | 17.538   | 10.729  |             |          |          |           |           |          |           |          |          |
| googlenet     | 100.719     | 69.509   | 49.021  | 127.982     | 83.369   | 50.275   |

---
---

| Snapdragon 652        | Anakin fp32 |          |          | Anakin int8 |          |          | NCNN fp32 |           |           | NCNN int8 |          |          |
|---------------|-------------|----------|----------|-------------|----------|----------|-----------|-----------|-----------|-----------|----------|----------|
|               | 1 thread        | 2 thread     | 4 thread     | 1 thread        | 2 thread     | 4 thread     | 1 thread      | 2 thread      | 4 thread      | 1 thread      | 2 thread     | 4 thread     |
| mobilenet_v1  | 121.982     | 63.004   | 37.325   | 86.672      | 45.728   | 26.354   | 130.740   | 140.850   | 81.810    | 184.630   | 192.730  | 144.740  |
| mobilenet_v2  | 89.113      | 50.609   | 35.291   | 72.679      | 45.888   | 33.887   | 94.520    | 101.380   | 65.570    |           |          |          |
| mobilenet_ssd | 236.466     | 132.293  | 86.335   |             |          |          | 270.630   | 295.520   | 174.280   | 350.640   | 286.420  | 243.850  |
| resnet50      | 751.528     | 405.433  | 255.699  |             |          |          | 2762.890  | 1447.070  | 883.730   | 664.180   | 369.020  |          |
| shufflenetv1  | 36.883      | 23.718   | 15.144   |             |          |          | 53.660    | 33.450    | 23.330    |           |          |          |
| shufflenetv2  | 36.933      | 26.353   | 20.507   | 53.243      | 31.083   | 21.550   |           |           |           |           |          |          |
| squeezenet    | 92.748      | 51.936   | 33.027   |             |          |          |           |           |           |           |          |          |
| googlenet     | 296.092     | 179.542  | 125.509  | 242.505     | 140.083  | 89.646   |

---
---

| RK3399        | Anakin fp32 |          | Anakin int8 |          | NCNN fp32 |           | NCNN int8 |          |
|---------------|-------------|----------|-------------|----------|-----------|-----------|-----------|----------|
|               | 1 thread        | 2 thread     | 1 thread        | 2 thread     | 1 thread      | 2 thread      | 1 thread      | 2 thread     |
| mobilenet_v1  | 116.981     | 65.033   | 87.768      | 47.617   | 155.830   | 98.520    | 201.800   | 116.440  |
| mobilenet_v2  | 118.229     | 70.567   | 83.790      | 55.413   | 126.530   | 90.930    |           |          |
| mobilenet_ssd | 237.196     | 134.508  |             |          | 292.130   | 183.650   | 361.570   | 200.370  |
| resnet50      | 725.582     | 413.995  |             |          | 2883.120  | 1632.800  | 702.660   | 404.970  |
| shufflenetv1  | 41.094      | 27.353   |             |          |           |           |           |          |
| shufflenetv2  | 37.660      | 23.489   | 53.558      | 32.122   |           |           |           |          |
| squeezenet    | 104.519     | 59.402   |             |          |           |           |           |          |
| googlenet     | 305.304     | 190.897  | 244.855     | 142.493  |
