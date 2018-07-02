# Anakin Unit Testing Framework 

为 anakin 设计的单元测试模型，可以移植到其他程序里作为单元测试模块框架。 和gtest类似的一个本地单元测试模块，实现了非常简单的基础功能，但是很实用。

推荐和我们的logger模块一起使用，这样在aktest模块就可以拥有几乎和gtest一样的功能。


1.**包含文件**
```bash
.
|-- aktest.h
|-- engine_test.h
`-- test_base.h
```

编译的时候引入这三个文件，并```#inlcude "aktest.h"```文件即可。

2.**Example**

```c++
#include <iostream>
#include "aktest.h"

using ::anakin::test::Test;

/** 
 * \brief 自定义的测试class，继承自Test类。
 *        这是单元测试的一个单元集合，包含一组
 *        测试函数test function。 
 */
class TestClass2:public Test{
public:
	TestClass2():name("second inner param."){}
	
	/** 
	 * \brief setup 函数用来对所有包含TestClass2的测试函数提供初始化，
	 *        其功能在所有函数开始运行的开始阶段自动调用，你可以实例化
     *        自己的setup函数内容。
	 */
    void SetUp(){}
	/** 
	 * \brief tear down 函数用来对所有本测试类的测试函数提供终止时操作，
	 *        典型的用法比如释放资源文件，进行垃圾回收等等，你可以自己
	 *		  实例化这个函数内容。
	 */
    void TearDown(){}

protected:
	/**
	 * \brief protected 里面可以包含你想要共享给所有测试函数的公共变量，
	 * 		  比如name变量，它在所有的TestClass2的TEST测试函数的实例中
	 *		  可见。
	 */
	std::string name;
};

/** 
 * \brief 添加TestClass1，与TestClass2功能类似，你可以添加多个testclass。
 */
class TestClass1:public Test{
public:

	TestClass1():name("test_inner_class_param"){}

	void SetUp(){}	

	void TearDown(){}

protected:
	std::string name;
};

/**
 * \brief 添加单元测试实例，TEST(className, functionName)用来声明一个测试案例，
 * 		  可以在其中添加任意测试内容。
 */
TEST(TestClass1,testfunc0){
	std::cout<<" FUNCTION_0 :test great suc! param=1? "<<name<<std::endl;
	for(int i=0;i<1000000;i++){}
}

TEST(TestClass1,testfunc1){
	std::cout<<" FUNCTION_1: test great suc! param=1? "<<name<<std::endl;
	for(int i=0;i<10000000;i++){}
}

TEST(TestClass1,testfunc2){
    std::cout<<" FUNCTION_2: test great suc! param=1? "<<name<<std::endl;
    for(int i=0;i<10000000;i++){}
}

// 这是 TestClass2 的测试实例。
TEST(TestClass2,seconde_test_func){
    std::cout<<" FUNCTION_second: test great suc! param=1? "<<name<<std::endl;
    for(int i=0;i<10000000;i++){}
}


int main(int argc, const char** argv){
	InitTest();		// 初始化 aktest.
	RUN_ALL_TESTS(argv[0]); // 运行所有的 test case.
	return 0;
}
```

4.**输出样例**
```bash

[***********] Running main() for ./test_example.
[    SUM    ] Running 4 test function from 2 test class.

[===========] Running 1 tests from TestClass2.
[ RUN       ] TestClass2.seconde_test_func
 FUNCTION_second: test great suc! param=1? second inner param.
[        OK ] TestClass2.seconde_test_func ( 25.00 ms )
[===========] 1 tests from TestClass2 class ran. ( 25.00 ms total )

[===========] Running 3 tests from TestClass1.
[ RUN       ] TestClass1.testfunc0
 FUNCTION_0 :test great suc! param=1? test_inner_class_param
[        OK ] TestClass1.testfunc0 ( 2.00 ms )
[ RUN       ] TestClass1.testfunc1
 FUNCTION_1: test great suc! param=1? test_inner_class_param
[        OK ] TestClass1.testfunc1 ( 25.00 ms )
[ RUN       ] TestClass1.testfunc2
 FUNCTION_2: test great suc! param=1? test_inner_class_param
[        OK ] TestClass1.testfunc2 ( 25.00 ms )
[===========] 3 tests from TestClass1 class ran. ( 52.00 ms total )

```
