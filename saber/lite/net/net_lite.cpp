#include "saber/lite/net/net_lite.h"

namespace anakin{}

namespace saber{}

namespace lite{}

using namespace anakin::saber;
using namespace anakin::saber::lite;

//template <typename dtype>
Net::Net(PowerMode mode, int threads) {
    //! init the env, only get device info
    Env::env_init();
    _mode = mode;
    _threads = threads;
    _ctx = new Context;
    _ctx->set_run_mode(_mode, _threads);
}

//template <typename dtype>
Net::~Net() {
    delete _ctx;
    _ctx = nullptr;
}

SaberStatus Net::set_run_mode(PowerMode mode, int threads) {
    _mode = mode;
    _threads = threads;
    _ctx->set_run_mode(_mode, _threads);
    return SaberSuccess;
}

SaberStatus Net::set_device_cache(size_t l1_cache, size_t l2_cache) {
    _ctx->set_cache(l1_cache, l2_cache, 0);
    return SaberSuccess;
}

//template <typename dtype>
SaberStatus Net::load_model(const char *opt_file, const char *model_file) {
    FILE* fp = fopen(opt_file, "rb");
    if (!fp) {
        printf("open %s failed\n", opt_file);
        return SaberInvalidValue;
    }
    int tensor_size = 0;
    fscanf(fp, "Tensor number %d\n", &tensor_size);
    printf("tensor size %d\n", tensor_size);
    //! gen tensors
    for (int i = 0; i < tensor_size; ++i) {
        char tensor_name[256];
        char tensor_shared_name[256];
        int real_shape[4];
        int valid_shape[4];
        int is_shared = 0;
        int nscan = fscanf(fp, "%256s %d,%d,%d,%d %d,%d,%d,%d %d %256s\n",
                           tensor_name,
                           &valid_shape[0],
                           &valid_shape[1],
                           &valid_shape[2],
                           &valid_shape[3],
                           &real_shape[0],
                           &real_shape[1],
                           &real_shape[2],
                           &real_shape[3],
                           tensor_shared_name);
        if (nscan != 11) {
            printf("load param error: %s\n", tensor_name);
            return SaberInvalidValue;
        }
        if(_tensors.find(tensor_name) != _tensors.end()) {
            printf("tensor is already registered: %s\n", tensor_name);
            return SaberInvalidValue;
        }
        Tensor<CPU, AK_FLOAT>* tensor = new Tensor<CPU, AK_FLOAT>;
        if (is_shared > 0) {
            tensor->set_shape(valid_shape);
            tensor->share_from(*_tensors[tensor_shared_name]);
        } else {
            tensor->re_alloc(Shape(real_shape));
            tensor->set_shape(Shape(valid_shape));
        }
        _tensors[tensor_name] = tensor;
        printf("%s vshape: %d,%d,%d,%d, rshape: %d,%d,%d,%d share:%s\n",
               tensor_name,
               valid_shape[0],
               valid_shape[1],
               valid_shape[2],
               valid_shape[3],
               real_shape[0],
               real_shape[1],
               real_shape[2],
               real_shape[3],
               tensor_shared_name);
    }


    return SaberSuccess;
}

//SaberStatus Net::load_model_from_memory(const void *memory) {
//    return SaberUnImplError;
//}

SaberStatus Net::prediction() {
    return SaberSuccess;
}

SaberStatus Net::init() {
    return SaberSuccess;
}

//} // namespace lite

//} // namespace saber

//} // namespace anakin
