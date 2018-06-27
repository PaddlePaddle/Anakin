# **Anakin developer guide**

本文主要包含以下四个方面内容：
  
+ [C++ APIs](#0001)
+ [如何贡献代码](#0002)
+ [如何添加新的Operator](#0003)
+ [如何添加新的设备](#0004)

## <span id = '0001'> C++ APIs </span>
---

### <span id ='api'>Anakin APIs </span> ###
#### Tensor ####

1. <span id =' '> Tensor结构 </span>

  `Tensor`提供基础的数据操作和管理，为ops提供统一的数据接口。`Tensor`包含以下几个属性：   

  - Buffer  
    数据存储区
  - Shape  
    数据的维度信息
  - Event  
    用于异步计算的同步

  > `Tensor` 类包含三个`Shape`对象， 分别是`_shape`, `_valid_shape`和 `offset`  
  > `_shape`为`tensor`真正空间信息  
  > `_valid_shape`表示当前`tensor`使用的空间信息  
  > `_offset`表示当前`tensor`数据指针相对于真正数据空间的信息  

  `Tensor`不同维度与分别与数学中的向量、矩阵等相对应如下表所示 

  Dimentions | Math entity |
  :----: | :----:
  1 | vector
  2 | matrix
  3 | 3-tensor
  n | n-tensor


2. <span id =' '> 声明tensor对象 </span>

  `Tensor`接受三个模板参数:


```c++
 template<typename TargetType, DataType datatype, typename LayOutType = NCHW>
 class Tensor .../* Inherit other class */{
  //some implements
  ...
 };
```

  > TargetType是平台类型，如X86，GPU等等，在Anakin内部有相应的标识与之对应  
  > datatype是普通的数据类型，在Anakin内部也有相应的标志与之对应   
  > [LayOutType](#layout)是数据分布类型，如batch x channel x height x width [NxCxHxW], 在Anakin内部用一个struct来标识  

  Anakin中数据类型与基本数据类型的对应如下:

  2.1. <span id='target'> TargetType </sapn>

  Anakin TargetType | platform
  :----: | :----:|
  NV | NVIDIA GPU
  ARM | ARM
  AMD | AMD GPU
  X86 | X86
  NVHX86 | NVIDIA GPU with Pinned Memory


  2.2. <sapn id='datatype'> DataType </span>

  Anakin DataType | C++ | Description 
  :---: | :---: | :---: |
  AK_HALF | short | fp16
  AK_FLOAT | float | fp32
  AK_DOUBLE | double | fp64
  AK_INT8 | char | int8
  AK_INT16 | short | int16
  AK_INT32 | int | int32
  AK_INT64 | long | int64
  AK_UINT8 | unsigned char | uint8
  AK_UINT16 | unsigned short | uint8
  AK_UINT32 | unsigned int | uint32
  AK_STRING | std::string | /
  AK_BOOL | bool | /
  AK_SHAPE | / | Anakin Shape 
  AK_TENSOR | / | Anakin Tensor 

  2.3. <span id = 'layout'> LayOutType </span>

  Anakin LayOutType ( Tensor LayOut ) | Tensor Dimention | Tensor Support | Op Support
  :---: | :---: | :---: | :---: |
  W | 1-D | YES | NO
  HW | 2-D | YES | NO
  WH | 2-D | YES | NO
  NW | 2-D | YES | YES
  NHW | 3-D | YES |YES
  NCHW ( default ) | 4-D | YES | YES
  NHWC | 4-D | YES | NO
  NCHW_C4 | 5-D | YES | YES

    理论上，Anakin支持申明1维以上的tensor。但是对于Anakin中的OP来说，只支持NW、NHW、NCHW、NCHW_C4这四种LayOut，
    其中NCHW是默认的LayOutType，NCHW_C4是专门针对于int8这种数据类型的。

3. 例子

  > 下面的代码将展示如何使用tensor， 建议先看看这些示例。
  > 要想获得更多关于tensor的信息， 请参考 *soure_path/core/tensor.h*

  + 使用shape对象初始化tensor

```c++  
  //create a null tensor. A null tensor holds for nothing.
  //tensor's buffer  is resident at CPU and its datatype is AK_FLOAT.
  //tensor's Layout is NCHW(default)
   Tensor<X86, AK_FLOAT> mytensor;

   //1. using shape object to create a tensor.
   Shape shape1(NUM); //1-D shape. NUM is the number of dimention.
   Tensor<X86, AK_FLOAT, W> mytensor1(shape1); //1-D tensor.

  // A 4-D shape
   Shape shape2(N, C, H, W); // batch x channel x height x width
```

    >`注意：Shape的维度必须和tensor的`[LayoutType](#layout)`相同，比如Shape(N,C,H,W), 那么Tensor的 LayoutType必须是NCHW，否则会出错。如下列代码所示`  

```c++
   // A 4-D tensor.
   Tensor<X86, AK_FLOAT> mytensor2(shape2);  //right

   //A 4-D tensor which is resident at GPU and its datatype is AK_INT8
   Tensor<NV, AK_INT8> mytensor3(shape2);   //right
   
   Tensor<X86, AK_FLOAT, NHW> mytensor4(shape2); //wrong!! shape's dimetion must be equal to tensor's Layout.
   Tensor<NV, AK_FLOAT, NCHW_C4> mytensor5(shape2); //wrong!!!!

```

  + 使用现有的数据和shape初始化tensor

```c++

   /**
   *  A construtor of Tensor.
   *  data_ptr is a pointer to any data type of data
   *  TargetType is type of a platform [Anakin TargetType]
   *  id : device id
   *  shape: a Anakin shape
   */
   Tensor(Dtype* data_ptr, TargetType_t target, int id, Shape shape);

   //using existing data feed to a tensor
   Tensor<X86, AK_FLOAT> mytensor(data_ptr, TargetType, device_id, shape); //shape must has dimention (N, C, H, W).

```

  + 使用tensor初始化tensor

```c++
   Tensor<NV, AK_FLOAT> tensor(exist_tensor);
```

    > 提示： 你可以用` typedef Tensor<X86, AK_FLOAT> Tensor4d_X86 `方便定义tensor


4. <span id =' '> 填充tensor数据区 </span>


  填充数据区得看你申明tensor的方式， 下面展示了如何填充tensor的数据区。

```c++
  首先来看看tensor的四种声明方式：

  1. Tensor<X86, AK_FLOAT> mytensor;
  2. Tensor<X86, AK_FLOAT, W> mytensor1(shape1);
  3. Tensor<X86, AK_FLOAT> mytensor(data_ptr, TargetType, device_id, shape);
  4. Tensor<NV, AK_FLOAT> tensor(exist_tensor);


  相关的声明方式的数据填充方法如下：

  1：声明一个空的tensor，此时没有为其分配内存，所以，我们需要手动的为其分配内存。
            
            //parama shape
            mytensor.re_alloc(Shape shape); 

            //Get writable pointer to mytensor.
            //parama index (int): where you start to write.
            //Dtype is your data type such int, float or double.
            Dtype *p = mytensor.mutable_data(index/*=0*/);
            //write data to mytensor
            for(int i = 0; i < mytensor.size(); i++){
              p[i] = 1.0f;
            }
            //do something ...

  2: 这种声明方式会自动分配内存 

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor1.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...

 
  3：在该种声明方式中，我们仍不需要手动为其分配内存。但在构造函数内部是否为其分配内存，得依情况而定。如果data_ptr和申明的
  tensor都在都一个目标平台上，那么该tensor就会与data_ptr共享内存空间，相反，如果他们不在同一个平台上（如data_ptr在X86上，而
  tensor在GPU上），那么此时tensor就会开辟一个新的内存空间，并将data_ptr所指向的数据拷贝到tensor的buffer中。

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...

  4：该种方式仍不需要手动分配内存

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...


  另外，你还可以获取一个tensor的可读指针，示例如下：
        //Get read-only pointer to mytensor.
        //parama index (int): where you start to read.
        //Dtype is your data type such int, float or double.
         Dtype *p = mytensor.data(index/*=0*/);
        //do something ...
```

  如果想更详细的了解tensor，请查阅*soure_path/saber/core/tensor.h*

5. 获取tensor的shape

```c++
//some declarations
// ...
Shape shape = mytensor.shape();

//Get a first dimetion size of tesor, if it has.
int d1 = shape[0];

//Get a second dimention size of tensor, if it has.
int d2 = shape[1];

...

//Get a n-th dimention size of tensor, if it has.
int dn = shape[n-1];


//Get a tensor's dimention
int dims = mytensor.dims();

//Get the size of tensor.
//size = d1 x d2 x ... x dn.
int size = mytensor.size();

//Get the size of tensor at interval [Di, Dj)
// form i-th dimention to j-th dimention, but not including the j-th dimention.
// which means di x (di+1) x ... x (dj -1)
int size = mytensor.count(start, end);
```

6. 设置tensor的shape

  我们可以用tensor的成员函数set_shape来设置tensor的shape。 下面是set_shape的定义


```c++
/**
 * \brief set a tensor's shape
 * \param valid_shape [a Shape object]
 * \param shape [a Shape object]
 * \param offset [a Shape object]
 * \return the status of this operation, that means whether it success * or not.
 */
SaberStatus set_shape(Shape valid_shape, Shape shape = Shape::zero(TensorAPI::layout_dims::value), Shape offset = Shape::minusone(TensorAPI::layout_dims::value)); 
```

  这个成员函数只设置tensor的shape。这些shape对象(valid_shape, shape, offset)的[LayOutType](#layout)必须和当前的tensor的相应三个shape对象的LayOutType相同，如果不同就会出错，返回SaberInvalidValue。 如果相同，那么将成功设置tensor的shape。

```c++

// some declarations
// ...
//valid_shape, shape , offset are Shape object;
//All these Shape object's LayOutType must be equal to mytensor's.
mytensor.set_shape(valid_shape, shape, offset);

```

7. 重置 tensor的shape

```c++
//some declarations
Shape shape, valid_shape, offset;

//do some initializations
... 
mytensor.reshape(valid_shape, shape, offset);
```

  注意： Reshape操作仍然需要shape的[LayOutType](#layout) 与tensor的相同


#### Graph ####

`Graph`类负责加载Anakin模型生成计算图、对图进行优化、存储模型等操作。

1. <span id =' '> 图的声明 </span>

  与`Tensor`一样，graph也接受三个模板参数。

```c++

template<typename TargetType, DataType Dtype, Precision Ptype>
class Graph ... /* inherit other class*/{
  
  //some implements
  ...

};
```

  前面已经介绍过[TargetType](#target)和[DataType](#datatype)是Anakin内部自定义数据类型。[TargetType](#target)表示平台类型 (如NV、X86), [DataType](#datatype)是Anakin基本数据类型与C++/C中的基本数据类型相对应。 [Precision](#precision)为op所支持的精度类型, 稍后我们在介绍它。


```c++

//Create a empty graph object.
Graph graph = Graph<NV, AK_FLOAT, Precision::FP32> tmp();

//Create a pointer to a empty graph.
Graph *graph = new Graph<NV, AK_FLOAT, Precision::FP32>();

//Create a pointer to a empty graph.
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();

```

2. 加载 Anakin 模型

```c++
//some declarations
...
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
std::string model_path = "the/path/to/where/your/models/are";
const char *model_path1 = "the/path/to/where/your/models/are";

//Loading Anakin model to generate a compute graph.
auto status = graph->load(model_path);

//Or this way.
auto status = graph->load(model_path1);
//Check whether load operation success.
if(!status){
  std::cout << "error" << endl;
  //do something...
}

```

3. 优化计算图

```c++
//some declarations
...
//Load graph.
...
//According to the ops of loaded graph, optimize compute graph.
graph->Optimize();

```

  > 注意： 第一次加载原始图，必须要优化。

4. 保存模型
  
  你可以在任何时候保存模型， 特别的， 你可以保存一个优化的模型，这样，下次再加载模型时，就不必进行优化操作。


```c++
//some declarations
...
//Load graph.
...
// save a model
//save_model_path: the path to where your model is.
auto status = graph->save(save_model_path);

//Checking
if(!status){
  cout << "error" << endl;
  //do somethin...
}
```

5. 重新设置计算图里的tensor的shape

```c++
//some declarations
...
//Load graph.
...
vector<int> shape{10, 256, 256, 10};
//input_name : std::string.
//Reshape a tensor named input_name.
graph->Reshape(input_name, shape);//Note: shape is a vector, not a Shape object.
```

6. 设置 batch size

`Graph` 支持重新设置batch size的大小。

```c++
//some declarations
...
//Load graph.
...
//input_name : std::string.
//Reset a tensor named input_name.
int new_batch_size = 4;
graph->ResetBatchSize(input_name, new_batch_size);
```

####  Net ####

`Net` 是计算图的执行器。你可以通过Net对象获得输入和输出

1. Creating a graph executor

  `Net`接受四个模板参数。  


```c++
template<typename TargetType, DataType Dtype, Precision PType OpRunType RunType = OpRunType::ASYNC>
class Net{
  //some implements
  ...

};
```
  由于有些Op可能支持多种精度，我们可以通过Precision来指定 

  > OpRunType表示同步或异步类型，异步是默认类型  
  > OpRunType::SYNC表示同步，在GPU上只有单个流  
  > OpRunType::ASYNC表示异步，在GPU上有多个流并以异步方式执行  

  实际上，Precision和OpRunType都是enum class, 详细设计请参考*source_root/framework/core/types.h*.


  + <span id = 'precision'> Precision </span>

  Precision | Op support
  :---: | :---:
  Precision::INT4 | NO
  Precision::INT8 | NO
  Precision::FP16 | NO
  Precision::FP32 | YES
  Precision::FP64 | NO

  > 现在Op的精度只支持FP32， 但在将来我们会支持剩下的Precision.

  + OpRunType

  OpRunType | Sync/Aync |Description
  :---: | :---: | :---:
  OpRunType::SYNC | Synchronization | single-stream on GPU
  OpRunType::ASYNC | Asynchronization | multi-stream on GPU

  用graph对象创建一个执行器。

```c++
//some declarations
...
//Create a pointer to a graph.
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
//do something...
...

//create a executor
Net<NV, AK_FLOAT, Precision::FP32> executor(*graph);

```

2. 获取输入输出tensor

  > 获取输入输出tensor，并填充输入tensor的buffer  
  > 如果想要获取输入和输出tensor，那么必须指定输入的名字，如"input_0", "input_1", "input_2", ..., 必须传入如上字符串才能够获得输入tensor  
  > 另外，如果想知道input_i对应哪个输入，你需要去dash board查看，如何使用dash board请看[Anakin Parser](./Converter_ch.md)  

  请看如下示例代码:

```c++
//some declaratinos
...

//create a executor
//TargetType is NV [NVIDIA GPU]
Net<NV, AK_FLOAT, Precision::FP32> executor(*graph);

//Get the first input tensor.
//The following tensors(tensor_in0, tensor_in2 ...) are resident at GPU.
//Note: Member function get_in returns an pointer to tensor.
Tensor<NV, AK_FLOAT>* tensor_in0 = executor.get_in("input_0");

//If you have multiple input tensors
//You just type this code below.
Tensor<NV, AK_FLOAT>* tensor_in1 = executor.get_in("input_1");
...
auto tensor_inn = executor.get_in("input_n");
```

  当得到输入tensor之后，就可以填充它的数据区了。

```c++
//This tensor is resident at GPU.
auto tensor_d_in = executor.get_in("input_0");

//If we want to feed above tensor, we must feed the tensor which is resident at host. And then copy the host tensor to the device's one.

//using Tensor4d = Tensor<Ttype, Dtype>;
Tensor4d<X86, AK_FLOAT> tensor_h_in; //host tensor;
//Tensor<X86, AK_FLOAT> tensor_h_in; 

//Allocate memory for host tensor.
tensor_h_in.re_alloc(tensor_d_in->valid_shape());
//Get a writable pointer to tensor.
float *h_data = tensor_h_in.mutable_data();

//Feed your tensor.
/** example
for(int i = 0; i < tensor_h_in.size(); i++){
  h_data[i] = 1.0f;
}
*/
//Copy host tensor's data to device tensor.
tensor_d_in->copy_from(tensor_h_in);

// And then
```

  > 类似的，我们可以利用成员函数get_out来获得输出tensor
  > 但与获得输入tensor不同的是， 我们需要指定输入tensor结点的名字，这个可以从dash board中看到，请从[Anakin Parser](./Converter_ch.md)中查看dash board的使用方法

  假如有个输出结点叫pred_out, 那么我们可以通过如下代码获得相应的输出tensor：

```c++
//Note: this tensor are resident at GPU.
Tensor<NV, AK_FLOAT>* tensor_out_d = executor.get_out("pred_out");

```

3. Executing graph

  当一切准备就绪后，我们就可以执行真正的计算了！
```c++
executor.prediction();
```

## <span id = '0002'> 如何贡献代码 </span>
---
我们真诚地感谢您的贡献，欢迎通过 GitHub 的 fork 和 pull request 流程来提交代码。

***代码要求:***

- 代码注释请遵守[Doxygen](http://www.stack.nl/~dimitri/doxygen/)的样式
- 所有代码必须具有单元测试
- 通过所有单元测试
- 请遵守提交代码的一些约定

以下教程将指导您提交代码

<span id = ''> 1. Fork </span>

  首先跳转到[Anakin](https://github.com/PaddlePaddle/Anakin)的github首页，然后点击`Fork`, 生成自己目录下的仓库

<span id = ''> 2. 克隆（clone）</span>

  将远程仓库clone到本地：

```bash
git clone YOUR_REPOSITORY_URL
cd Anakin
```

<span id = ''> 3. 创建本地分支 </span>

  > Anakin目前使用[Git流分支模型](https://nvie.com/posts/a-successful-git-branching-model/)进行开发, 测试和维护

  > 所有的feature和bug fix的开发工作都应该在一个新的分支上完成，根据需要从现有分支上创建新分支

  > 使用`git checkout -b`创建并切换到新分支

```bash
git checkout -b YOUR_NEW_BRANCH
```

<span id = ''> 4. 开发 </span>

  4.1. 编写代码

  4.2. 构建和测试

    详细请参考 [Instal and Compile](../docker/README.md)

  4.3. 提交(commit)

    提交代码时，请认真写好提交说明，这样其他人就可以清楚的知道这次提交做了哪些改变：

  ```bash
  git commit -m 'description'
  ```

<span id = ''> 5. 保持本地仓库最新 </span>

  在发起Pull Request之前，需要与原始仓库同步。

  如果还没添加原仓库，请先添加源，可通过`git remote -v`查看是否添加源：

```bash
git remote -v
origin .... (fetch)
origin .... (push)
```
  如果只出现origin，说明还未添加源，可通过如下命令添加源：

```bash
git remote add upstream ORIGIN_REPOSITORY_URL
```
  获取 upstream 的最新代码并更新当前分支

```bash
git fetch upstream
git pull upstream BRANCH_NAME
```

6. Push到远程仓库

  将本地的修改push到远程仓库上

```bash
git push origin BRANCH_NAME
```

7. 提交Pull Request

  切换到所建分支，然后点击`New pull request`

  ![contri1](./contri1.JPG)

  选择目标分支：

  ![contri2](./contri2.JPG)

  接下来等待review

8. 删除远程分支

  当PR被合进主仓库后，可以在PR的界面删除远程仓库的分支

  也可以通过以下命令删除远程分支：

```bash
git push origin :YOUR_NEW_BRANCH
```

9. 删除本地分支

  可以通过以下命令删除本地分支:

```bash
#切换到其他分支
git checkout OTHER_BRANCH

#删除YOUR_NEW_BRANCH分支
git branch -D YOUR_NEW_BRANCH
```

  至此，我们就完成了一次代码贡献的过程

  ***提交代码的一些约定***

  为了使评审人在评审代码时更好地专注于代码本身，请您每次提交代码时，遵守以下约定：

+ 提交Pull Request前：  
- 注意commit的数量

  - 原因：如果仅仅修改一个文件但提交了十几个commit，每个commit只做了少量的修改，这会给评审人带来很大困扰。评审人需要逐一查看每个commit才能知道做了哪些修改，且不排除commit之间的修改存在相互覆盖的情况。

  - 建议：每次提交时，保持尽量少的commit，可以通过`git commit --amend`补充上次的commit。对已经Push到远程仓库的多个commit，可以参考[squash commits after push](https://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed)
  
- 注意每个commit的名称：应能反映当前commit的内容，不能太随意。

+ 如果解决了某个Issue的问题，请在该Pull Request的第一个评论框中加上：`fix #issue_number`，这样当该Pull Request被合并后，会自动关闭对应的Issue。关键词包括：close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved，请选择合适的词汇。详细可参考[Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages)。

  在回复评审人意见时，请您遵守以下约定：  
+ 评审人的每个意见都必须回复
   - 对评审意见同意且按其修改完的，给个简单的Done即可
   - 对评审意见不同意的，请给出您自己的反驳理由 
+ 如果评审意见比较多
   - 请给出总体的修改情况 
   - 请采用[start a review](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/)进行回复，而非直接回复的方式 


## <span id = '0003'> 如何添加新的Operator </span>
---

<span id = ''> 1. 基本概念 </span>

  1.1. 与Operator相关的基本概念

    简单介绍下几个与Operator相关的基本概念，详情请参考设计文档。

    + ```framework```: 上层的逻辑代码，负责从parser中获取参数及weights，添加op时主要修改framework/operator目录下的内容。

    + ```saber```: 底层的实现代码，Anakin通过saber封装了不同的backends，不同的实现(impl)分别特化出自己的实现，外层framework通过不同的template进入各自的impl完成调用。各个op的parameter放在saber/saber_funcs_param.h文件中，增加op主要修改saber/funcs下的内容。

    + saber的文件结构：
      - saber/funcs下的是各个funcs的外部接口，这一层的op与具体的设备实现无关，只与各op完成的功能有关。由于跟实现(impl)无关，本层文件明均不带impl。
      - saber/funcs/impl下是各个op的impl声明，特定设备需要完成该层声明的特化版本，如saber/funcs/impl/x86实现了上一层impl声明的x86特化版本，saber/funcs/impl/cuda实现了上一层impl声明的NV特化版本。当增加新的backends时需要特化出新的实现。本层代码同实现相关，均带有```impl_```前缀。
      - saber/funcs/impl/cuda/base/cuda_c内有cuda```.cu```扩展名的文件，添加cuda的kernel需要在该文件目录下添加。
      - saber/funcs/impl/cuda/base/sass 内有不同架构的汇编代码编译的静态库。

  2.2. 涉及到的基类及各个类之前的关系

    简单介绍相关的基类

    + ```anakin::Operator```: framework的operator基类，位于framework/core/operator/operator.h

    + ```anakin::saber::BaseFunc```: saber对外的op接口基类，提供统一的对外接口，位于saber/funcs/base.h。BaseFunc的```compute_output_shape```接口只根据input的shape和param的参数计算输出的shape，并通过```tensor```的```set_shape```接口(只设置shape，不分配空间)设置到output中。```operator()```接口为各个op的计算接口。

    + ```ankain::saber::ImplBase```: saber设备实现的op的接口，所有设备相关实现的基类。位于saber/funcs/impl/impl_base.h。实现版本中这里分为两类，一类以```vender_```为前缀，带有```vender_```代码意为使用第三方库来实现该op，如cudnn的conv，或mkl的conv等等，这类op的性能我们难以调优，因此单独列为一类。另一类是带有源码的saber实现，这些实现都带有```saber_```为前缀，此类实现带有源码，能够通过后续优化不断提升性能，实现起名时需要注意这一点。

<span id = ''> 2. 添加operator </span>

  添加一个新的op需要以下几步：

  - 添加saber的param
  - 定义saber的Operator类
  - 定义新的impl声明
  - 完成新的impl实现
  - 增加framework的实现或特化

  接下来就针对这几步，以一个简单例子为例介绍实现。

  例如我们要添加新的Mul op，给出计算公式如下：$$Out = alpha \dot X * Y$$

  2.1. 为operator增加param

    涉及到的文件：```saber/saber_funcs_param.h```。如果之前已经存在需要添加的op的param，这一步可以跳过

    这里```XXXParam```是一个```struct```。包含一个无参数的构造函数，含参数的构造函数，复制构造函数，```operator=()```及```operator==()```

```bash
template <typename opTensor> // 能够获得target, datatype, layout
struct MulParam{
  MulParam()
    : alpha(0)
  {}
  MulParam(float alpha_in)
    : alpha(alpha_in)
  {}
  MulParam(const MulParam& right)
    : alpha(right.alpha)
  {}
  MulParam &operator=(const MulParam &right) {
    alpha = right.alpha;
  }
  bool operator==(const MulParam &right) {
    return alpha == right.alpha;
  }
  float alpha;
};
```

  2.2. 定义Operator类

    涉及到的文件:```saber/funcs/mul.h```。如果之前定义过该op的类，这里需要修改输入的impl定义头文件

    下面给出一个相对完整的定义结构供参考:

```bash
//不同的设备需要包含对应的operator实现
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_mul.h"
#include "saber/funcs/impl/cuda/vender_mul.h"
#endif
//如果一个设备现在还没有对应的operator实现，需要包含声明
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/impl_mul.h"
#endif
namespace anakin {
namespace saber {
template<typename TargetType,
        DataType OpDtype,
        DataType inDtype = AK_FLOAT,
        DataType outDtype = AK_FLOAT,
        typename LayOutType_op = NCHW,
        typename LayOutType_in = NCHW,
        typename LayOutType_out = NCHW>
class Mul : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase, MulParam> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase, MulParam>::BaseFunc;
    Mul() = default;
    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef MulParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        //计算输出的shape，
        Shape output_shape = (input[0]->valid_shape());
        /* code */
        return output[0]->set_shape(output_shape);
    }
    virtual SaberStatus init_impl(ImplEnum implenum) override {
      // 不同设备均使用此init_impl, 此接口创建对应impl的实现。
      switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderMul <TargetType,
                OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            case SABER_IMPL:
                this->_impl.push_back(new SaberMul <TargetType,
                OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            default:
                return SaberUnImplError;
        }
    }
private:
    virtual void pick_best_static() override {
        if (true) // some condition?
            this->_best_impl = this->_impl[0];
    }
    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
};
} // namespace saber
} // namespace anakin
```

  2.3. 为operator增加新的impl声明

    涉及的文件:```saber/funcs/impl/impl_mul.h```。不同的设备都特化同一个声明，特化版本放在对应的文件夹下，这里的声明就是给出所有设备的统一声明。

    下面给出一个参考:

```bash
#include "saber/funcs/impl/impl_macro.h"
namespace anakin{
namespace saber{
DEFINE_OP_CLASS(Mul, MulParam); // 第一个参数是op的名字，第二个是对应param的名字
}
}
```

  2.4. 完成新的operator特定后端实现

    涉及的文件:```saber/funcs/impl/xxx/vender_mul.h```或```saber/funcs/impl/xxx/saber_mul.h```

- ```xxx```指代特定的一种设备
- ```vender```是指的使用第三方库实现的op
- ```saber```指的源码实现的op

    这里以cuda的vender实现为例，简单介绍一下特化出的函数的几个基本接口:

```bash
// include 对应的声明
#include "saber/funcs/impl/impl_mul.h"

namespace anakin{
namespace saber{
template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderMul<NV, //偏特化出需要的后端。
    OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out> :
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>,
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        MulParam<Tensor<NV, OpDtype, LayOutType_op> > >
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;
    VenderMul(){}
    ~VenderMul() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            MulParam<OpTensor>& param, Context<NV>& ctx) {
        this->_ctx = ctx;
        create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            MulParam<OpTensor>& param, Context<NV>& ctx) {
        // set内部参数
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                        MulParam<OpTensor>& param) {
        // dispatch kernel.
    }

private:
};
}
}
```

    > 注意：
    ```init```和```create```的区别：```init```接口是第一次初始化op的时候进入的接口，此函数只在第一次初始化op时调用，这个接口一般放一些只需要执行一次的代码，如malloc或者create之类的函数。```create```函数除了第一次init执行外，在输入发生变化或者param发生变化时会再次触发，create一般放置set函数，设置内部变量，当input发生变化时这里执行一些同input或weights直接相关的代码。但create因为触发位置在网络内，如果```create```函数执行了一些严重耗时的操作，这里会拖慢整个op的执行时间，需要慎重选择操作放置的位置。

  2.5. 添加framework的特化

    涉及的文件:```framework/operators/mul.h```和```framework/operators/mul.cpp```

    这里简单介绍下如果添加或修改framework内的operator:

```bash
#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/mul.h" // 需要包对应的saber头文件
namespace anakin {
namespace ops {
template<typename Ttype, DataType Dtype, Precision Ptype>
class MulHelper;

template<typename Ttype, DataType Dtype, Precision Ptype>
class Mul : public Operator<Ttype, Dtype, Ptype> {
public:
    Mul() {}
    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx,
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }
    friend class MulHelper<Ttype, Dtype, Ptype>;
};
template<typename Ttype, DataType Dtype, Precision Ptype>
class MulHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    MulHelper() = default;
    ~MulHelper();
    Status InitParam() override;

    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

public:
    saber::MulParam<Tensor4d<Ttype, Dtype>> _param_mul;
    saber::Mul<Ttype, Dtype> _funcs_mul;
};
}
} /* namespace anakin */
```

    对应的```.cpp```文件如下：

```bash
#include "framework/operators/mul.h"

namespace anakin {
namespace ops {

#ifdef USE_CUDA
template<>
void Mul<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<MulHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<MulHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_mul;
    impl->_funcs_mul(ins, outs, param, ctx);
}
#endif

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::InitParam() {
    auto alpha = GET_PARAMETER(float, alpha);
    MulParam<Tensor4d<Ttype, Dtype>> param_mul(alpha);
    _param_mul = param_mul;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {

    SABER_CHECK(_funcs_mul.init(ins, outs, _param_mul, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status MulHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_mul.compute_output_shape(ins, outs, _param_mul));
    return Status::OK();
}

#ifdef USE_CUDA
template class MulHelper<NV, AK_FLOAT, Precision::FP32>;
#endif
#ifdef USE_ARM_PLACE
template class MulHelper<ARM, AK_FLOAT, Precision::FP32>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Mul, MulHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Mul, MulHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Mul)
.Doc("Mul operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("mul")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("mul")
#endif
.num_in(1)
.num_out(1)
.Args<float>("alpha", " alpha of Mul "); //注册

} /* namespace ops */

} /* namespace anakin */
```

  2.6. 实现单元测试

    涉及的文件:```test/saber/xxx/test_saber_funcs_mul_xxx.cpp```

    在对应的test下需要添加新的单元测试如下所示:

```bash
TEST(TestSaberFuncNV, test_depthwise_conv) {

    // init tensors and some param.

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // create param
    MulParam<Tensor<NV, AK_FLOAT, NCHW> > param(alpha);

    std::vector<Tensor<NV, AK_FLOAT, NCHW>*> input;
    std::vector<Tensor<NV, AK_FLOAT, NCHW>*> output;

    // create saber op
    Mul<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> mul;

    // compute output shape
    mul.compute_output_shape(input, output, param);

    // re_alloc output tensors memory based on output shape
    output[0]->re_alloc(output[0]->shape());

    // init saber op(calling init and create)
    mul.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    // call operator()
    mul(input, output, param, ctx1);

    // cuda specified, record events
    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    
    // param changed 
    param.alpha = 2.0;
    // auto calling saber op(create and dispatch)
    mul(input, output, param, ctx1);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv){
    anakin::saber::Env<NV>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

```

<span id = ''> 3. 调试及注意事项 </span>

  一个op需要有对外的op接口和内部实现，由于存在saber/funcs/impl的非特化版本声明，当有op在某种设备下没有对应实现时，也能够编译，但此时是没有任何实现的空实现


## <span id = '0004'> 如何添加新的设备 </span>
---

添加新设备流程：

  * [在`CMakeList`中添加设备的支持](#21001)
  * [在`saber`中添加设备的实现](#21002)
  * [在`framework`中添加设备的具体化或实例化](#21003)

假设新设备的名称为`TNEW`, 以下将以这个设备名称进行演示。


1. <span id = '21001'> 在`CMakeList`中添加设备的支持 </span>

  * 修改根目录`CMakeList.txt`

    ```cmake
    #select the plantform to build
    anakin_option(USE_GPU_PLACE "Select the build mode for GPU place." NO)
    anakin_option(USE_X86_PLACE "Select the build mode for X86 place." NO)
    anakin_option(USE_ARM_PLACE "Select the build mode for ARM place." NO)
    anakin_option(USE_TNEW_PLACE "Select the build mode for ARM place." YES)
    ```

  * 修改`saber/CMakeList.txt`

    根据新增设备的目录完善`saber`目录下的`CMakeList.txt`

    ```cmake
    if(USE_TNEW_PLACE)
      anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/tnew "cpp" ANAKIN_SABER_BASE_SRC)
       anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/tnew "cpp" ANAKIN_SABER_BASE_SRC)
    endif()
    ```

  * 修改`test/CMakeList.txt`

    新增设备的单测文件放在`test/saber/tnew`目录下，修改`test`目录下的`CMakeList.txt`

    ```cmake
    if(USE_TNEW_PLACE)
      anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/saber/tnew "cpp" ANAKIN_TEST_CASE_SRC)
    endif()
    ```

  * 修改`cmake/anakin_config.h.in`

```c++
    // plantform to use
    #cmakedefine USE_GPU_PLACE

    #cmakedefine USE_X86_PLACE

    #cmakedefine USE_ARM_PLACE

    #cmakedefine USE_TNEW_PLACE
```

  * 其他依赖和编译选项    
    修改`cmake`目录下的`compiler_options.cmake`和`find_modules.cmake`


2. <span id = '21002'> 在`saber`中添加设备的实现 </span> 
  
  `saber`是`Anakin`的基础计算库，对外提供设备无关的统一的API，设备相关的实现都会封装到`TargetWrapper`中 

  2.1. 在`saber/saber_types.h`中添加设备

```c++
enum TargetTypeEnum {
    eINVALID = -1,
    eNV = 1,
    eAMD = 2,
    eARM = 3,
    eX86 = 4,
    eNVHX86 = 5,
    eTNEW = 6
};

typedef TargetType<eNV> NV;
typedef TargetType<eARM> ARM;
typedef TargetType<eAMD> AMD;
typedef TargetType<eX86> X86;
typedef TargetType<eTNEW> TNEW;

```

  2.2. 在`saber/core`中添加设备的实现

  (1) 在`target_traits.h`中添加新设备

    * 增加设备类型

```c++
  struct __cuda_device{};
  struct __arm_device{};
  struct __amd_device{};
  struct __x86_device{};
  struct __tnew_device{};
```

    * `TargetTypeTraits`模板具体化

```c++
template <>
struct TargetTypeTraits<TNEW> {
    typedef __xxx_target target_category;//根据实际设备是host端还是device端进行选择
    typedef __tnew_device target_type;
};
```

  (2) 在`data_traits.h`中特化`DataTrait`模板类

    如果设备需要特殊的数据类型，则特化出设备的`DataTrait`类的实现，例如opencl数据类型的实现如下：

```c++
#ifdef USE_OPENCL
struct ClMem{
    ClMem(){
        dmem = nullptr;
        offset = 0;
    }

    ClMem(cl_mem* mem_in, int offset_in = 0) {
        dmem = mem_in;
        offset = offset_in;
    }

    ClMem(ClMem& right) {
        dmem = right.dmem;
        offset = right.offset;
    }

    ClMem& operator=(ClMem& right) {
        this->dmem = right.dmem;
        this->offset = right.offset;
        return *this;
    }

    ClMem& operator+(int offset_in) {
        this->offset += offset_in;
        return *this;
    }

    int offset{0};
    cl_mem* dmem;
};

template <>
struct DataTrait<AMD, AK_FLOAT> {
    typedef ClMem Dtype;
    typedef float dtype;
};

template <>
struct DataTrait<AMD, AK_DOUBLE> {
    typedef ClMem Dtype;
    typedef double dtype;
};

template <>
struct DataTrait<AMD, AK_INT8> {
    typedef ClMem Dtype;
    typedef char dtype;
};
#endif //use_opencl
```

  (3) 在`target_wrapper.h`中特化`TargetWrapper`模板类

    特化`TargetWrapper`模板类，在`target_wrapper.h`中声明函数，具体如下：

```c++
template <>
struct TargetWrapper<TNEW, __xxx_target> { //根据TNEW的具体类型修改__xxx_target，__host_target或者__device_target

    typedef xxx_event event_t;          //根据设备实现xxx_event
    typedef xxx_stream stream_t;        //根据设备实现xxx_stream

    static void get_device_count(int& count);

    static void set_device(int id);

    //We should add strategy to avoid malloc directly
    static void mem_alloc(void** ptr, size_t n);

    static void mem_free(void* ptr);

    static void mem_set(void* ptr, int value, size_t n);

    static void create_event(event_t& event, bool flag = false);

    static void create_stream(stream_t& stream);

    static void create_stream_with_flag(stream_t& stream, unsigned int flag);

    static void create_stream_with_priority(stream_t& stream, unsigned int flag, int priority);

    static void destroy_stream(stream_t& stream);

    static void destroy_event(event_t& event);

    static void record_event(event_t& event, stream_t stream);

    static void query_event(event_t& event);

    static void sync_event(event_t& event);

    static void sync_stream(event_t& event, stream_t& stream);

    static void sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                            size_t count, __DtoD);

    static void async_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                             size_t count, stream_t& stream, __DtoD);

    static void sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                            size_t count, __HtoD);

    static void async_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                             size_t count, stream_t& stream, __HtoD);

    static void sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                            size_t count, __DtoH);

    static void async_memcpy(void* dst, int dst_id, const void* src, int src_id, \
                             size_t count, stream_t& stream, __DtoH);

    static void sync_memcpy_p2p(void* dst, int dst_dev, const void* src, \
                                int src_dev, size_t count);

    static void async_memcpy_p2p(void* dst, int dst_dev, const void* src, \
                                 int src_dev, size_t count, stream_t& stream);

    static int get_device_id();
};

```

  (4) 在`impl/`目录下添加设备目录和实现

    在`saber/core/impl`目录下添加设备目录`tnew`
    * 实现`TargetWrapper<TNEW, __xxx_target>`结构体中各函数的定义    
      如果`TargetWrapper<TNEW, __xxx_target>`的实现与默认的模板类一致，则不用特化出该类。

```c++
typedef TargetWrapper<TNEW, __xxx_target> TNEW_API;
void TNEW_API::get_device_count(int &count) {
    // add implementation
}

void TNEW_API::set_device(int id){
    // add implementation
}
        
void TNEW_API::mem_alloc(void** ptr, size_t n){
    // add implementation
}
        
void TNEW_API::mem_free(void* ptr){
    if(ptr != nullptr){
        // add implementation
    }
}
...

```

    * 特化实现`device.h`中的`Device<TNEW>`

```c++
template <>
void Device<TNEW>::create_stream() {
    // add implementation
}

template <>
void Device<TNEW>::get_info() {

    // add implementation
}

```

  2.3. 在`saber/funcs`中实现设备相关的op

  参考[如何增加新的Operator](#0003)


3. <span id = '21003'> 在`framework`中添加设备的具体化或实例化 </span>  
  
  3.1. `framework/core`

  * `net.cpp`中添加实例化

```c++
#ifdef USE_TNEW_PLACE
  template class Net<TNEW, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
  template class Net<TNEW, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
#endif
```

  * `operator_func.cpp`中添加实例化

```c++
#ifdef USE_TNEW_PLACE
  template class OperatorFunc<TNEW, AK_FLOAT, Precision::FP32>;
#endif
```

  * `worker.cpp`中添加实例化

```c++
#ifdef USE_TNEW_PLACE
  template class Worker<TNEW, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
  template class Worker<TNEW, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
#endif
```

  * `operator_attr.cpp`中添加实例化

```c++
template
OpAttrWarpper& OpAttrWarpper::__alias__<TNEW, AK_FLOAT, Precision::FP32>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<TNEW, AK_FLOAT, Precision::FP16>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<TNEW, AK_FLOAT, Precision::INT8>(const std::string& op_name);
```

  * `parameter.h`中添加设备的实现

```c++
#ifdef USE_TNEW_PLACE
template<typename Dtype>
class PBlock<Dtype, TNEW> {
public:
  typedef Tensor4d<TNEW, DataTypeRecover<Dtype>::type> type;

  PBlock() {
    _inner_tensor = std::make_shared<type>(); 
  }
  ...
}
#endif //TNEW
```

  * `type_traits_extend.h`中添加设备的实现

```c++
template<>
struct target_host<saber::TNEW> {
    typedef saber::X86 type; //根据TNEW选择正确的host type
};
```

  3.2. `framework/graph`

  * `graph.cpp`中添加实例化
  
```c++
  #ifdef USE_TNEW_PLACE
  template class Graph<TNEW, AK_FLOAT, Precision::FP32>;
  template class Graph<TNEW, AK_FLOAT, Precision::FP16>;
  template class Graph<TNEW, AK_FLOAT, Precision::INT8>;
  #endif
```

  3.3. `framework/model_parser`

  * `parser.cpp`中添加实例化
  
```c++
  #ifdef USE_TNEW_PLACE
  template
  Status load<TNEW, AK_FLOAT, Precision::FP32>(graph::Graph<TNEW, AK_FLOAT, Precision::FP32>* graph,
          const char* model_path);
  template
  Status load<TNEW, AK_FLOAT, Precision::FP16>(graph::Graph<TNEW, AK_FLOAT, Precision::FP16>* graph,
          const char* model_path);
  template
  Status load<TNEW, AK_FLOAT, Precision::INT8>(graph::Graph<TNEW, AK_FLOAT, Precision::INT8>* graph,
          const char* model_path);
  
  template
  Status save<TNEW, AK_FLOAT, Precision::FP32>(graph::Graph<TNEW, AK_FLOAT, Precision::FP32>* graph,
          std::string& model_path);
  template
  Status save<TNEW, AK_FLOAT, Precision::FP16>(graph::Graph<TNEW, AK_FLOAT, Precision::FP16>* graph,
          std::string& model_path);
  template
  Status save<TNEW, AK_FLOAT, Precision::INT8>(graph::Graph<TNEW, AK_FLOAT, Precision::INT8>* graph,
          std::string& model_path);
  
  template
  Status load<TNEW, AK_FLOAT, Precision::FP32>(graph::Graph<TNEW, AK_FLOAT, Precision::FP32>* graph,
          std::string& model_path);
  template
  Status load<TNEW, AK_FLOAT, Precision::FP16>(graph::Graph<TNEW, AK_FLOAT, Precision::FP16>* graph,
          std::string& model_path);
  template
  Status load<TNEW, AK_FLOAT, Precision::INT8>(graph::Graph<TNEW, AK_FLOAT, Precision::INT8>* graph,
          std::string& model_path);
  
  template
  Status save<TNEW, AK_FLOAT, Precision::FP32>(graph::Graph<TNEW, AK_FLOAT, Precision::FP32>* graph,
          const char* model_path);
  template
  Status save<TNEW, AK_FLOAT, Precision::FP16>(graph::Graph<TNEW, AK_FLOAT, Precision::FP16>* graph,
          const char* model_path);
  template
  Status save<TNEW, AK_FLOAT, Precision::INT8>(graph::Graph<TNEW, AK_FLOAT, Precision::INT8>* graph,
          const char* model_path);
  #endif
```

  * `model_io.cpp`中添加实例化

```c++
#ifdef USE_TNEW_PLACE
template class NodeIO<TNEW, AK_FLOAT, Precision::FP32>;
template class NodeIO<TNEW, AK_FLOAT, Precision::FP16>;
template class NodeIO<TNEW, AK_FLOAT, Precision::INT8>;
#endif
```

  3.4. `framework/operators`

    为`framework/operators`目录下所有op添加实例化或具体化

    以`activation.cpp`为例，实例化如下：

```c++
#ifdef USE_TNEW_PLACE
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::FP32);
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::FP16);
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::INT8);
template class ActivationHelper<TNEW, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, TNEW, AK_FLOAT, Precision::FP32);
#endif
```

    如果TNEW设备函数的实现与现有模板实现不一致，可以特化实现如下（以init()为例）：

```c++
#ifdef USE_TNEW_PLACE
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::FP32);
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::FP16);
INSTANCE_ACTIVATION(TNEW, AK_FLOAT, Precision::INT8);
template <>
Status ActivationHelper<TNEW, AK_FLOAT, Precision::FP32>::Init(OpContext<TNEW> &ctx,\
        const std::vector<Tensor4dPtr<TNEW, AK_FLOAT> >& ins, \
                std::vector<Tensor4dPtr<TNEW, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_activation.init(ins, outs, _param_activation, SPECIFY, SABER_IMPL, ctx)); //在这里选择实现方式
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, TNEW, AK_FLOAT, Precision::FP32);
#endif
```

    在`ANAKIN_REGISTER_OP(Activation)`中添加TNEW的注册

```c++
#ifdef USE_TNEW_PLACE
.__alias__<TNEW, AK_FLOAT, Precision::FP32>("activation")
#endif
```

4. 注意事项
  
  不要修改`Tensor`/`Buffer`/`Env`/`Context`这些类函数的接口和实现




