# BenchMark

## Machine:

+ Compile circumstance: Android ndk cross compile，gcc 4.9，enable neon
+ ABI： armveabi-v7a with neon -mfloat-abi=softfp
+ Testing platform
   - honor v9(root): Kirin960, 4 big cores in 2.36GHz, 4 little cores in 1.8GHz
   - nubia z17:Qualcomm835, 4 big cores in 2.36GHz, 4 little cores in 1.9GHz
   - 360 N5:Qualcomm653, 4 big cores in 1.8GHz, 4 little cores in 1.4GHz
+ Time：warmup 10，running 10 times to get average time
+ ncnn ：git clone on github master branch and commits ID is 307a77f04be29875f40d337cfff6df747df09de6（msg:convert            LogisticRegressionOutput)
+ TFlite：git clone on github master branch and commits ID is 65c05bc2ac19f51f7027e66350bc71652662125c（msg:Removed unneeded file copy that was causing failure in Pi builds)

## Counterpart of Anakin

The counterpart of **`Anakin`** are **`ncnn`** and **`TFlite`**.

## BenchMark model

> Please note that you should transform caffe model or others into anakin model with the help of [`external converter ->`](../docs/Manual/Converter_en.md)

- [Mobilenet v1](#11)  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [Mobilenet v2](#22)  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [mobilenet-ssd](#33)  *caffe model can be found [here->](https://github.com/chuanqi305/MobileNet-SSD)*

We tested them on ARM with multi-thread and single-batchsize.

### <span id = '11'> mobilenetv1 </span>

- Latency (`ms`) of different thread  

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |Kirin960|107.7|61.1ms|38.2 |152.8 |85.2 |51.9 |152.6 |nan|nan|
   |Qualcomm835|105.7 |63.1 |~~46.8 ~~|152.7 |87.0 |~~92.7 ~~|146.9 |nan|nan|
   |Qualcomm653|120.3 |64.2 |46.6 |202.5 |117.6 |84.8 |158.6 |nan|nan| 

### <span id = '22'> mobilenetv2 </span>

- Latency (`ms`) of different thread  

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |Kirin960|93.1 |53.9 |34.8 |144.4 |84.3 |55.3 |100.6 |nan|nan|
   |Qualcomm835|93.0 |55.6 |41.1 |139.1 |88.4 |58.1 |95.2 |nan|nan|
   |Qualcomm653|106.6 |64.2 |48.0 |199.9 |125.1 |98.9 |108.5 |nan|nan|

### <span id = '33'> mobilenet-ssd </span>

- Latency (`ms`) of different thread  

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |Kirin960|213.9 |120.5 |74.5 |307.9 |166.5 |104.2 |nan|nan|nan|
   |Qualcomm835|213.0 |125.7 |~~98.4 ~~|292.9 |177.9 |~~167.8 ~~|nan|nan|nan|
   |Qualcomm653|236.0 |129.6 |96.0 |377.7 |228.9 |165.0 |nan|nan|nan

## How to run those Benchmark models?

1. At first, you should parse the caffe model with [External Converter](../docs/Manual/Converter_en.md)
2. Second, adb push Anakin model and benchmark_arm bin to testing phone
3. Then, switch to /data/local/tmp/ directory on testing phone, run `./benchmark_arm ./ anakin_model.anakin.bin 1 10 10 1` command
4. Finally，model latency summary will be displayed on the screen.
5. You can see the detailed parameters meaning by running `/benchmark_arm`

