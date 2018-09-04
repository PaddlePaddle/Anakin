
namespace anakin {

void ThreadPool::launch() {
    for(size_t i = 0; i<_num_thread; ++i) {
        _workers.emplace_back(
            [i ,this]() {
                // initial 
                this->init();
                for(;;) {
                    std::function<void(void)> task;
                    {
                        std::unique_lock<std::mutex> lock(this->_mut);
                        while(!this->_stop && this->_tasks.empty()) { 
                            this->_cv.wait(lock); 
                        }
                        if(this->_stop) {
                            return ;
                        }
                        task = std::move(this->_tasks.front()); 
                        this->_tasks.pop();
                    }
                    DLOG(INFO) << " Thread (" << i <<") processing";
                    auxiliary_funcs();
                    task();
                }
            }
        );
    }
}

void ThreadPool::stop() {
    std::unique_lock<std::mutex> lock(this->_mut);
    _stop = true;
}

void ThreadPool::init() {}

void ThreadPool::auxiliary_funcs() {}

ThreadPool::~ThreadPool() {
    stop();
    this->_cv.notify_all();
    for(auto & worker: _workers){ 
        worker.join(); 
    }
}

template<typename functor, typename ...ParamTypes>
typename function_traits<functor>::return_type ThreadPool::RunSync(functor function, ParamTypes ...args) 
                    EXCLUSIVE_LOCKS_REQUIRED(_mut) { 
    auto task = std::make_shared<std::packaged_task<typename function_traits<functor>::return_type(void)> >( \
            std::bind(function, std::forward<ParamTypes>(args)...)
    );
    std::future<typename function_traits<functor>::return_type> result = task->get_future(); 
    { 
        std::unique_lock<std::mutex> lock(this->_mut);
        this->_tasks.emplace( [&]() { (*task)(); } );
    }
    this->_cv.notify_one();
    return result.get();
}

template<typename functor, typename ...ParamTypes>
std::future<typename function_traits<functor>::return_type> ThreadPool::RunAsync(functor function, ParamTypes ...args) 
                    EXCLUSIVE_LOCKS_REQUIRED(_mut) { 
    auto task = std::make_shared<std::packaged_task<typename function_traits<functor>::return_type(void)> >( \
            std::bind(function, std::forward<ParamTypes>(args)...)
    );
    std::future<typename function_traits<functor>::return_type> result = task->get_future(); 
    { 
        std::unique_lock<std::mutex> lock(this->_mut);
        this->_tasks.emplace( [=]() { (*task)(); } );
    }
    this->_cv.notify_one();
    return result;
}

} /* namespace anakin */
