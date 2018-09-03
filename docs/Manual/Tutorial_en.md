# Anakin Tutorial ##

This tutorial will briefly explain how Anakin works, some of the basic Anakin APIs, and how to call these APIs.

## Table of contents ###

- [How does the Anakin work?](#principle)
- [Anakin APIs](#api)
- [Code Example](#example)

## <span id = 'principle'> How does the Anakin work?</span> ###

![Anakin_principle](pics/anakin_fm_en.png)

The calcutional process of Anakin is mainly divided into the following tree steps:

- Convert other models to Anakin models through [Anakin Parser](Converter_en.md)  
  Before using Anakin, users must convert other models to Anakin ones. For any convenience, [Anakin Parser](Converter_en.md) will do all that for you.

- Generate Anakin computation graph  
  This step will generate a raw Anakin computation graph by loading Anakin model. And then, it is very necessary to optimize the raw graph when you first load model. You just need to call corresponding API to optimize the raw graph.
  
- Perform computation graph
  Anakin will choose different platforms to perform real calculations.


## <span id ='api'>Anakin APIs </span> ###
### Tensor ####
  
`Tensor` provides basic data operation and management, and provides unified data interface for ops. `Tensor` has the following attributes:    

- Buffer  
   Data storage area
- Shape  
   Dimention information of Tensor
- Event  
  Synchronization for asynchronous calculations

The `Tensor` class contains three `Shape` objects, which are `_shape`, `_valid_shape` and `offset`. `_shape` is the actual spatial information of `tensor`; `_valid_shape` indicates the spatial information used by the current `tensor`; `_offset` indicates where the current `tensor` data pointer is. The different dimensions of `Tensor` correspond to the vectors and matrices in mathematics, respectively, as shown in the following table.

Dimentions | Math entity |
 :----: | :----:
1 | vector
2 | matrix
3 | 3-tensor
n | n-tensor

#### Declaring a tensor object

`Tensor` receives three template parameters:

```c++
 template<typename TargetType, DataType datatype, typename LayOutType = NCHW>
 class Tensor .../* Inherit other class */{
  //some implements
  ...
 };
```

TargetType indicates the type of platfor such as X86, GPU and so on. There has a corresponding identifier within Anakin corresponding to it. Datatype is common data type, which has a corresponding identifier within Anakin. [LayOutType](#layout) is layout type of data like batch x channel x height x width [NxCxHxW], which is identified by using a struct within Anakin. The following tables show the correspondence between data type within Anakin and basic data type.

1. <span id='target'>TargetType</sapn>

 Anakin TargetType | platform
  :----: | :----:|
  NV | NVIDIA GPU
  ARM | ARM
  AMD | AMD GPU
  X86 | X86
  NVHX86 | NVIDIA GPU with Pinned Memory

2. <sapn id='datatype'>DataType</span>

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


3. <span id = 'layout'>LayOutType </span>

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

In theory, Anakin supports the declaration of more than one dimensions of tensor, but for Op in Anakin, only NW, NHW, NCHW, NCHW_C4 are supported, of which NCHW is the default LayOutType, and NCHW_C4 is a special LayOutType for int8.  

Example

> `The following examples will show you how to use a tensor . We recommend that you should read this quick start first.`

> `For more details about tensor, please see  ` *soure_path/saber/core/tensor.h*


> 1. using a shape object to initialize a tensor.
``` c++  
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

>`Note: Shape's dimention must be same as tensor's [LayoutType](#layout). If Shape has the layout (N, C, H, W), for exmaple, the Tensor's one must be NCHW, otherwise it will result an error. The example are shown below:`

```c++
   // A 4-D tensor.
   Tensor<X86, AK_FLOAT> mytensor2(shape2);  //right

   //A 4-D tensor which is resident at GPU and its datatype is AK_INT8
   Tensor<NV, AK_INT8> mytensor3(shape2);   //right
   
   Tensor<X86, AK_FLOAT, NHW> mytensor4(shape2); //wrong!! shape's dimetion must be equal to tensor's Layout.
   Tensor<NV, AK_FLOAT, NCHW_C4> mytensor5(shape2); //wrong!!!!

```

> 2. using existing data and shape to initialize a tensor.

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

> 3. using an existing tensor to initialize a tensor.

```c++
   Tensor<NV, AK_FLOAT> tensor(exist_tensor);
```

> Note : Typecally, you can use `typedef Tensor<X86, AK_FLOAT> Tensor4d_X86` for convenient.


#### Feeding tensor's buffer

Feeding tensor depends on the way you declare a tensor. Let's see how to feed a tensor.

```c++
Let's first look back to the way of tensor declarations.

1. Tensor<X86, AK_FLOAT> mytensor;
2. Tensor<X86, AK_FLOAT, W> mytensor1(shape1);
3. Tensor<X86, AK_FLOAT> mytensor(data_ptr, TargetType, device_id, shape);
4. Tensor<NV, AK_FLOAT> tensor(exist_tensor);

The corresponding method of feeding tensor are below:

1: Declare a empty tensor and no memory allocated. So, you need to allocate memory for that tensor.
            
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

2: It will automatically allocate memory for tensor in this way. 

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor1.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...

3: In this method, we still do not allocate memory for tensor. Although we do not allocate memory
manually,  the allocation of memory inside the constructor depends on the circumstances. If data_ptr 
and mytensor are both resident at the same platform, the tensor will share the buffer where data_ptr 
holds. However, if they are not(for exmaple, data_ptr is at X86, however tensor is at GPU), the constructor 
will allocate memory for mytensor and copy the data from data_ptr to mytensor's buffer. 

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...

4: In this method, we do not allocate meory for tensor. It will automatically allocate memory for tensor.

          //Get writable pointer to mytensor.
          //parama index (int): where you start to write.
          //Dtype is your data type such int, float or double.
          Dtype *p = mytensor.mutable_data(index/*=0*/);
          //write data to mytensor
          for(int i = 0; i < mytensor.size(); i++){
            p[i] = 1.0f;
          }
          //do something ...


In addition, there is a read-only pointer to tensor. You can use it as shown below:

        //Get read-only pointer to mytensor.
        //parama index (int): where you start to read.
        //Dtype is your data type such int, float or double.
         Dtype *p = mytensor.data(index/*=0*/);
        //do something ...
```

For more details about tensor' API , please refer to *soure_path/saber/core/tensor.h*

#### Getting a Tensor object's shape

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

#### Setting a tensor object's shape

We can exploit one of the tensor's member function set_shape to set a tensor's shape. Let's look at the defination of set_shape.

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

This member function only sets the shape of tensor. `All the` [LayOutType](#layout) `of these Shape object [valid_shape, shape, offset] must be the same as current tensor's.` If they are not, it will return SaberInvalidValue, Otherwise, set correspond shape.

```c++

// some declarations
// ...
//valid_shape, shape , offset are Shape object;
//All these Shape object's LayOutType must be equal to mytensor's.
mytensor.set_shape(valid_shape, shape, offset);

```

#### Reshaping tensor

```c++
//some declarations
Shape shape, valid_shape, offset;

//do some initializations
... 
mytensor.reshape(valid_shape, shape, offset);
```
`Note: Reshape also requres that shape's` [LayOutType](#layout) `must be same as tensor's`


### Graph ###

`Graph` class supports several operations such as generating a compute graph from loading Anakin models, graph optimization, saving models.

#### Graph declaration

Like `Tensor`, graph also accepts three tenplate parameters.
```c++

template<typename TargetType, DataType Dtype, Precision Ptype>
class Graph ... /* inherit other class*/{
  
  //some implements
  ...

};
```

As mentioned above, [TargetType](#target) and [DataType](#datatype) are data type of Anakin. [TargetType](#target) indicates platform type such as NV, X86. [DataType](#datatype) is Anakin's basic data type, which is corresponding to C++/C basic data type. [Precision](#precision) is kind of accuracy type, which will be introduced later. 


```c++

//Create a empty graph object.
Graph graph = Graph<NV, AK_FLOAT, Precision::FP32> tmp();

//Create a pointer to a empty graph.
Graph *graph = new Graph<NV, AK_FLOAT, Precision::FP32>();

//Create a pointer to a empty graph.
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();

```

#### Loading Anakin models

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

#### Optimizing graph

```c++
//some declarations
...
//Load graph.
...
//According to the ops of loaded graph, optimize compute graph.
graph->Optimize();

```
> `Note: It must be optimized when you load a graph for the first time.`

#### Saving models

You can save a model at any time. Typically, you can save a optimized model, thus, you can directly use that without optimizing again.

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

#### Reshaping tensor of graph

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

#### Resetting batch size

`Graph` class supports reseting a saved model.

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

###  Net ###

`Net` is a real executor of a computation graph. We can get input/output tensors of a graph through `Net` object.

#### Creating a graph executor
 
`Net` accepts four template parameters.

```c++
template<typename TargetType, DataType Dtype, Precision PType OpRunType RunType = OpRunType::ASYNC>
class Net{
  //some implements
  ...

};
```

Since some Ops may support many acurracy types, we can specify a special acurracy type through Precision. OpRunType indicates the type of Synchronization or Asynchronization, of which Asynchronization is default. OpRunType::SYNC means Synchronization. There is only single stream in GPU if you set OpRunType as OpRunType::SYNC. while if you set OpRunType as OpRunType::ASYNC, there are multi-stream in GPU And these streams are asynchronous. In fact, Precision and OpTunType are emum class, for more details, please referring to *source_root/framework/core/types.h*.

1. <span id = 'precision'> Precision </span>

Precision | Op support
:---: | :---:
Precision::INT4 | NO
Precision::INT8 | NO
Precision::FP16 | NO
Precision::FP32 | YES
Precision::FP64 | NO


Op only support FP32 for now, but other precision type will be supported in the future.

2. OpRunType

OpRunType | Sync/Aync |Description
:---: | :---: | :---:
OpRunType::SYNC | Synchronization | single-stream on GPU
OpRunType::ASYNC | Asynchronization | multi-stream on GPU

Create a executor using a graph object. 
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

#### Getting input/output tensors

`Get input/output tensors and feed the input tensors.` In order to get an input/output tensor, you must specify its name with a given string such as "input_0", "input_1", "input_2" and so on. If you want to know which input tensor the "input_i" are corresponding with, please check the dash board which can be found [Anakin Parser](Converter_en.md). The following code show you how to do this.
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

After getting the input tensors, we can feed data into them.

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

Analogously, we can use get_out to get output tensors. What different with getting input tensors is that we need to specify output node's name, rather than a special string. You need to check dash board again to find output node's name. You can find dash board usage in [Anakin Parser](Converter_en.md). If we has an output node named pred_out, for example, then we can get the output tensor through the following code.

```c++
//Note: this tensor are resident at GPU.
Tensor<NV, AK_FLOAT>* tensor_out_d = executor.get_out("pred_out");

```


#### Executing graph

When all the prepare things are finished, and then you just type the following code to do inference!
```c++
executor.prediction();
```
 
## <span id='example'> Code Example </span> ##

The following examples will show you how to call Anakin to do inference.

Before you start, please make sure that you have had Anakin models. If you don't have it, please use [Anakin Parser](Converter_en.md) to convert your models to Anakin ones.

### Single-thread

The single-thread example is at *source_root/test/framework/net/net_exec_test.cpp`*

```c++

std::string model_path = "your_Anakin_models/xxxxx.anakin.bin";
// Create an empty graph object.
auto graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
// Load Anakin model.
auto status = graph->load(model_path);
if(!status ) {
    LOG(FATAL) << " [ERROR] " << status.info();
}
// Reshape
graph->Reshape("input_0", {10, 384, 960, 10});
// You must optimize graph for the first time.
graph->Optimize();
// Create a executer.
Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph);

//Get your input tensors through some specific string such as "input_0", "input_1", and 
//so on. 
//And then, feed the input tensor.
//If you don't know Which input do these specific string ("input_0", "input_1") correspond with, you can launch dash board to find out.
auto d_tensor_in_p = net_executer.get_in("input_0");
Tensor4d<X86, AK_FLOAT> h_tensor_in;
auto valid_shape_in = d_tensor_in_p->valid_shape();
for (int i=0; i<valid_shape_in.size(); i++) {
    LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i]; //see tensor's dimentions
}
h_tensor_in.re_alloc(valid_shape_in);
float* h_data = h_tensor_in.mutable_data();
for (int i=0; i<h_tensor_in.size(); i++) {
    h_data[i] = 1.0f;
}
d_tensor_in_p->copy_from(h_tensor_in);

//Do inference.
net_executer.prediction();

//Get result tensor through the name of output node.
//And also, you need to see the dash board again to find out how many output nodes are and remember their name.

//For example, you've got a output node named obj_pre_out
//Then, you can get an output tensor.
auto d_tensor_out_0_p = net_executer.get_out("obj_pred_out"); //get_out returns a pointer to output tensor.
auto d_tensor_out_1_p = net_executer.get_out("lc_pred_out"); //get_out returns a pointer to output tensor.
//......
// do something else ...
//...
//save model.
//You might not optimize the graph when you load the saved model again.
std::string save_model_path = model_path + std::string(".saved");
auto status = graph->save(save_model_path);
if (!status ) {
    LOG(FATAL) << " [ERROR] " << status.info();
}

```

