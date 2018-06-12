# 测试环境和参数:
1. 测试模型Mobilenetv1, mobilenetv2, mobilenet-ssd
2. 采用android ndk交叉编译，gcc 4.9，enable neon， ABI： armveabi-v7a with neon -mfloat-abi=softfp
3. 测试平台
	a)荣耀v9(root): 处理器:麒麟960,    4 big cores in 2.36GHz, 4 little cores in 1.8GHz
	b)nubia z17:处理器:高通835,    4 big cores in 2.36GHz, 4 little cores in 1.9GHz
	c)小米5(root):  处理器:高通820,    2 big cores in 1.8GHz,  2 little cores in 1.36GHz
	d)360 N5:处理器:高通653,    4 big cores in 1.8GHz, 4 little cores in 1.4GHz
4. 多线程：openmp
5. 时间：20次取最小值

## 测试结果
### mobilenetv1
   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|104.8ms|58.6ms|36.3ms|156.6ms|86.0ms|51.9ms|151ms|81ms|78ms|
   |高通835|102.8ms|60.3ms|~~60.5ms~~|149.3ms|82.1ms|~~85.3ms~~|145ms|101ms|92ms|
   |高通820|127.2ms|69.3ms|50.8ms|195.1ms|108.7ms|83.7ms|134ms|87ms|81ms|
   |高通653|115.2ms|62.1ms|42.6ms|201.78ms|116.5ms|83.8ms|152ms|97ms|80ms| 

### mobilenetv2

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|89.8ms|51.4ms|32.5ms|145.4ms|83.7ms|54.9ms|99ms|65ms|60ms|
   |高通835|89.0ms|53.3ms|37.0ms|133.1ms|76.4ms|55.1ms|98ms|82ms|72ms|
   |高通820|109.4ms|61.9ms|45.2ms|203.1ms|122.1ms|93.9ms|90ms|76ms|71ms|    |高通653|106.0ms|58.4ms|42.7ms|194.3ms|120.3ms|97.8ms|104ms|85ms|74ms|

### mobilenet-ssd

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|211.5ms|117.8ms|72.8ms|300.6ms|165.2ms|100.6ms|
   |高通835|207.1ms|121.3ms|~~119.4ms~~|273.4ms|154.4ms|~~162.8ms~~|
   |高通820|258.3ms|141.7ms|104.2ms|376.1ms|209.6ms|166.1ms|
   |高通653|253.3ms|124.1ms|87.3ms|384.9ms|225.2ms|159.7ms|