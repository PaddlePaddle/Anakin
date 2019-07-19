#include "core_test.h"
#include "any.h"
#include "singleton.h"
#include "tls.h"
#include "parameter.h"
#include "thread_pool.h"

#ifdef USE_CUDA
#include "cuda_funcs.h"
#include "sass_funcs.h"
#endif

#include "tensor.h"

#if 0 //#ifdef USE_CUDA
TEST(CoreComponentsTest, sass_test) {
    LOG(INFO) << "test for cuda code function";
    //anakin::saber::Tensor<3, RTCUDA, float, NCHW> ts;
    //LOG(WARNING) << " tensor num " << ts.num();
    //ts.set_offset(8);
    //my_print();
    LOG(INFO) << "test for sass code function 1";
    invoke_test();
    LOG(INFO) << "test for sass code function 2";
    invoke_test_2();
}
#endif

TEST(CoreComponentsTest, core_base_types_any_test) {
    LOG(INFO) << "test for any class .";
    LOG(WARNING) << " level 1 : base type int (set 42 to any)";
    const int a = 42;
    any any_a(42);
    int result_a = any_cast<int>(any_a);

    LOG(INFO) << "casted result : " <<  result_a;
    LOG(WARNING) << " level 2 : base type float (set 42.8 to any)";
    float b = 42.8;
    any any_b = b;
    float result_b = any_cast<float>(any_b);
    LOG(INFO) << "casted result : " <<  result_b << " decide: ";

    LOG(WARNING) << " level 3 : ptuple type (set PTuple<float> to any)";
    PTuple<float> p_tuple_float(3.2f, 3.3f, 3.5f);
    p_tuple_float.push_back(4.3); // push_back

    any p_tuple_float_any = p_tuple_float;
    auto result_p_tuple_float_any = any_cast<PTuple<float>>(p_tuple_float_any);

    for (int i = 0; i < result_p_tuple_float_any.size(); i++) {
        LOG(INFO) << " any casted PTuple<float>[" << i << "]: " << result_p_tuple_float_any[i];
    }

    struct target {
        void print() {
            LOG(INFO) << " target struct Successfully recovered.";
        }
    };

    LOG(WARNING) << " level 5 : struct type";

    target tg;

    any any_tg = tg;

    target result_tg = any_cast<target>(any_tg);

    result_tg.print();

    LOG(WARNING) << " level other : struct type";

    any any_tg_copy = any_tg;

    target result_tg_copy = any_cast<target>(any_tg);

    result_tg_copy.print();
}

void at_exit_in_test() {
    LOG(WARNING) << "core_base_types_singleton_test exit successfully!";
}

TEST(CoreComponentsTest, core_base_types_singleton_test) {
    struct target {
        target() {
            LOG(INFO) << " singleton target constructed";
        }
    };
    typedef Singleton<target, at_exit_in_test> sg_target;
    sg_target::Global();
}

typedef AnakinThreadLocalVar<int> sg_tls;
void thread_func_0() {
    int* tmp = sg_tls::value();
    *tmp = 3;
    LOG(INFO) << " thread_func_0 value: " << *(sg_tls::value());
}
void thread_func_1() {
    int* tmp = sg_tls::value();
    *tmp = 4;

    LOG(INFO) << " thread_func_0 value: " << *(sg_tls::value());
}
TEST(CoreComponentsTest, core_base_types_tls_test) {
    LOG(INFO) << " Create tls var 0 , check in two thread.";
    std::thread first(thread_func_0);
    std::thread sec(thread_func_1);
    first.join();
    sec.join();
    LOG(INFO) << " main thread var: " << *(sg_tls::value());
}

int thread_pool_func(int i) {
    LOG(INFO) << " thread_pool_func input : " << i;
    //std::this_thread::sleep_for(std::chrono::seconds(0));
    return i;
}

TEST(CoreComponentsTest, core_base_types_thread_pool_test) {
    LOG(INFO) << " Create thread pool with thread num = 12 ";
    ThreadPool thread_pool_test(100);
    thread_pool_test.launch();
    std::function<int(int)> test = thread_pool_func;

    for (int i = 0; i < 50; i++) {
        // run async
        auto ret = thread_pool_test.RunAsync(test, i);
        LOG(INFO) << " return : " << ret.get();

        // run sync
        //auto sync_ret = thread_pool_test.RunSync(test, i);
    }
}


int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
