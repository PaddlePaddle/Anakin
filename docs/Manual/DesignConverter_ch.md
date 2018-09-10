# Parser的编写指南
下文称Anakin为AK，运算操作为OP,本文参考Tensorflow的Parser编写,参考代码目录为tools/external_converter_v2/parser/tensorflow
## Parser的功能和执行流程
功能是将其他深度学习框架(如CAFFE，FLUID，TENSORFLOW，ONNEX)的模型转换为AK的模型.对AK的作用是屏蔽不同框架间的差异，这种差异包括模型存储、OP的定义、图差异
因此Parser的执行流程是：
1. 将源框架的模型载入Parser
2. 将原框架的图解析为AK中的OP节点和OP节点的连接关系
3. 进行OP定义的转换和图优化
4. 将符合AK标准的图写入protobuf
## Parser的目录结构
Parser工具在tools/external_converter_v2/parser目录下
Parser的目录主要包含3部分:
1. Parser的运行配置文件包括 config.py, config.yaml, converter.py, 用户只用执行converter.py，Parser就会按照config.yaml中的声明去解析模型
2. Parser的公共定义，包括operations,pbs,proto三个目录. Parser的公共工具函数 graph*.py logger.py utils.py
3. 各个框架对应的Parser，其目录的命名方式为框架名,如caffe, tensorflow
## Parser的编写流程
### 1、声明你的Parser
1. 在config.yaml中填写你的Parser运行的必要信息，包括ProtoPath和SavePath等.OPTIONS/Framework改为你的Parser的类型，TARGET下填写对应的参数列表
2. 添加你的Parser目录，如tensorflow，导出你的Parser符号.注意，Parser的框架默认调用你的Parser类中的__call__方法来执行解析，这个方法需要返回填写完毕的GraphProtoIO对象
3. 在config.py中Configuration下__init__函数中增加对你的Parser的调用，将yaml中读取的配置信息传给你的Parser，此处调用你的Parser中的__init__方法
### 2、添加你的Parser主体
可以参考parser_tf.py
1. 你需要在Parser主体构造时获取模型路径，input，ouput名字等解析必须的信息，
2. 在__call__中返回填写好的GraphProtoIO对象，该对象为填写protobuf的辅助工具
3. 建议Parser的解析过程分成三部分，先将原框架的模型载入并转换为一种便于修改的中间的图形式；对中间图修改使得图满足AK的要求；将满足要求的中间图利用NodeProtoIO和GraphProtoIO这两个辅助类填入protobuf.具体细节可以参考parser_tf
### 3、读取原始模型，并将模型转换为中间类型
可以参考parse_tf_2_med.py
1. 这一步与原始框架结合紧密，你可能需要import原始框架的工具函数来完成模型的裁剪、固定、加载等操作
2. 大部分的框架都是使用tensor来连接OP的，但AK中是OP直接相连，这点需要注意
3. AK的shape默认是4维的，有的参数的shape不足4维，需要Parser补全
### 4、对中间类型的图进行优化
可以参考med_graph.py
1. 由于AK不支持普通OP多输出的情况，需要在多输出的OP后面补上Splite类型的OP节点
2. 对于Convlution后接Batchnorm这种可以合并又不会导致OP定义改变的情况，需要Parser在这一步做掉
3. AK规定所有的输入类型OP的名字必须是input_x这种命名方式，其中x为从0开始的数字
### 5、将中间类型的图以GraphProtoIO的方式保存
可以参考parse_med_2_ak.py 和 parser_tf.py
1. 你首先需要构造Node节点，Node节点的名字是OP的名字(如conv2d_1_a_0)，Node节点中OP成员变量的名字是Node节点的类型(如Convlution)
2. Node节点需要按照输入的顺序用Node的add_in方法填写输入Node的名字，add_out方法按顺序填写输出Node的名字
3. 通过调用GraphProtoIO的add_node方法将构造好的Node的__call__方法的返回值作为参数，将Node节点加入AK的graph中
4. 调用GraphProtoIO的add_in_edge和add_out_edge完成AK图中OP间关系的构建. 如果Node中的in和out填写正确，你也可以通过调用GraphProtoIO的format_edge_from_nodes方法完成这个工作
5. AK的模型需要Parser给出输出Node的名字，使用GraphProtoIO的add_out方法填写输出Node的名字
### 6、检查模型解析的正确性
1. 默认的config.yaml配置会在解析结束后启动一个web服务器展示解析后的AK模型图，你需要对比原框架的模型图进行验证.这里最容易出现的错误是边关系的错误，表现为图非常乱，你需要逐条边地检查错误.第二个容易出错的地方是参数漏填，需要你检查OP中的属性
2. 将解析后的模型放入AK中执行，使用相同的输入，原框架与AK有相同的输出.若果输出不一致可以开启AK的DEBUG模式，在net.cpp中将没层的输出打印.如果AK在解析阶段陷入死循环，大概率是边的关系出错.
## 如何添加新OP
1. 需要在AK代码中加入该OP的实现，包括对应设备Saber的OP，Saber单测和Framework中的OP
2. 根据Framework的OP在ops.py中添加Parser公共的OP定义
3. 从原框架的模型中解析出该OP的节点，并在AK的graph中填入该OP节点
## AK模型与其他框架模型的不同之处
+ AK模型与CAFFE的模型相似，因此与其他模型有很多不同的地方，需要Parser在解析过程中处理掉.
+ 最大的不同是与FLUID或TENSORFLOW这种OP粒度很细的框架，AK的模型中OP的粒度很粗，这是为了节省访存开销.这会导致解析这些框架的模型时存在大量的合并操作.
+ 其次是OP的行为不同,如TENSORFLOW中Pooling默认都是exclusive的，而AK中是inclusive的.TENSORFLOW的Padding如果是奇数pad则在右方和下方多pad，AK是在左方和上方多Pad
+ AK默认的布局是NCHW，如果其他框架的OP是其他形式的，需要在Parser中做weights的布局转换，并处理reshape的问题.
+ AK中有的weights是需要预先做布局转换的(如GRU，LSTM).AK中也支持同一OP的不同算法，如(GRU，Pooling).

