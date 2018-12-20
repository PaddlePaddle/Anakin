/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_THREAD_POOL_H
#define ANAKIN_THREAD_POOL_H 

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <future>
#include <mutex> 
#include <condition_variable>
#include "framework/core/thread_safe_macros.h"
#include "framework/core/type_traits_extend.h"
#include "utils/logger/logger.h"
#include "anakin_config.h"
#ifdef USE_SGX
#include <support/sgx/sgx_mutex>
#endif

namespace anakin {

class ThreadPool {
public:
    ThreadPool(int num_thread):_num_thread(num_thread) {}
    virtual ~ThreadPool();

    void launch();

    /** 
     *  \brief Lanuch the normal function task in sync.
     */
    template<typename functor, typename ...ParamTypes>
    typename function_traits<functor>::return_type RunSync(functor function, ParamTypes ...args);

    /**
     *  \brief Lanuch the normal function task in async.
     */
    template<typename functor, typename ...ParamTypes>
    typename std::future<typename function_traits<functor>::return_type> RunAsync(functor function, ParamTypes ...args);
    
    /// Stop the pool.
    void stop();

private:
    /// The initial function should be overrided by user who derive the ThreadPool class.
    virtual void init();

    /// Auxiliary function should be overrided when you want to do other things in the derived class.
    virtual void auxiliary_funcs();

private:
    int _num_thread;
    std::vector<std::thread> _workers;
    std::queue<std::function<void(void)> > _tasks GUARDED_BY(_mut);
    std::mutex _mut;
    std::condition_variable _cv;
    bool _stop{false};
};

} /* namespace anakin */

#include "thread_pool.inl"

#endif
