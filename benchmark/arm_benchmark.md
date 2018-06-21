# 测试环境和参数:
1. 测试模型Mobilenetv1, mobilenetv2, mobilenet-ssd
2. 采用android ndk交叉编译，gcc 4.9，enable neon， ABI： armveabi-v7a with neon -mfloat-abi=softfp
3. 测试平台
	a)荣耀v9(root): 处理器:麒麟960,    4 big cores in 2.36GHz, 4 little cores in 1.8GHz
	b)nubia z17:处理器:高通835,    4 big cores in 2.36GHz, 4 little cores in 1.9GHz
	c)小米5(root):  处理器:高通820,    2 big cores in 1.8GHz,  2 little cores in 1.36GHz
	d)360 N5:处理器:高通653,    4 big cores in 1.8GHz, 4 little cores in 1.4GHz
4. 多线程：openmp
5. 时间：warmup10次，运行10次取均值

## 测试结果
### mobilenetv1
   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|107.7ms|61.1ms|38.2ms|152.8ms|85.2ms|51.9ms|151ms|81ms|78ms|
   |高通835|105.7ms|63.1ms|~~46.8ms~~|152.7ms|87.0ms|~~92.7ms~~|145ms|101ms|92ms|
   |高通820|129.9ms|70.8ms|53.1ms|201.1ms|111.9ms|88.5ms|134ms|87ms|81ms|
   |高通653|120.3ms|64.2ms|46.6ms|202.5ms|117.6ms|84.8ms|152ms|97ms|80ms| 

### mobilenetv2

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|93.1ms|53.9ms|34.8ms|144.4ms|84.3ms|55.3ms|99ms|65ms|60ms|
   |高通835|93.0ms|55.6ms|41.1ms|139.1ms|88.4ms|58.1ms|98ms|82ms|72ms|
   |高通820|111.2ms|63.9ms|47.6ms|207.3ms|123.5ms|97.1ms|90ms|76ms|71ms|
   |高通653|106.6ms|64.2ms|48.0ms|199.9ms|125.1ms|98.9ms|104ms|85ms|74ms|

### mobilenet-ssd

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|213.9ms|120.5ms|74.5ms|307.9ms|166.5ms|104.2ms|nan|nan|nan|
   |高通835|213.0ms|125.7ms|~~98.4ms~~|292.9ms|177.9ms|~~167.8ms~~|nan|nan|nan|
   |高通820|260.7ms|142.3ms|105.8ms|382.4ms|214.8ms|169.9ms|nan|nan|nan|
   |高通653|236.0ms|129.6ms|96.0ms|377.7ms|228.9ms|165.0ms|nan|nan|nan|