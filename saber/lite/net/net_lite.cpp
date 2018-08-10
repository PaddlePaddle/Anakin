#include "saber/lite/net/net_lite.h"
#include "saber/lite/net/saber_factory_lite.h"
#include "saber/lite/core/tensor_op_lite.h"
namespace anakin{

namespace saber{

namespace lite{

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
    if (_weights) {
        fast_free(_weights);
        _weights = nullptr;
    }
    for (int i = 0; i < _ops.size(); ++i) {
        delete _ops[i];
    }
    for (auto it = _tensors.begin(); it != _tensors.end(); it++) {
        delete it->second;
        it->second = nullptr;
    }
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
SaberStatus Net::load_model(const char *lite_model) {
//    FILE *fp_w = fopen(model_file, "rb");
//    if(!fp_w) {
//        printf("load weights failed: %s\n", model_file);
//        return SaberInvalidValue;
//    }
//    fseek(fp_w, 0, SEEK_END);
//    long fsize = ftell(fp_w);
//    fseek(fp_w, 0, SEEK_SET);
//    if(_weights) {
//        delete [] _weights;
//        _weights = nullptr;
//    }
//    _weights = static_cast<float*>(fast_malloc(fsize + 1));//new float[fsize + 1];
//    fread(_weights, fsize, 1, fp_w);
//    fclose(fp_w);

    FILE* fp = fopen(lite_model, "rb");
    if (!fp) {
        printf("open %s failed\n", lite_model);
        return SaberInvalidValue;
    }
    long wsize;
    fscanf(fp, "Wsize %lu\n", &wsize);
    if (_weights) {
        delete [] _weights;
        _weights = nullptr;
    }
    _weights = static_cast<float*>(fast_malloc(wsize + 1));
    fread(_weights, wsize, 1, fp);

    int tensor_size = 0;
    int nscan = fscanf(fp, "Tensor number %d\n", &tensor_size);
    //printf("tensor size %d\n", tensor_size);
    //! gen tensors
    for (int i = 0; i < tensor_size; ++i) {
        char tensor_name[256];
        char tensor_shared_name[256];
        int real_shape[4];
        int valid_shape[4];
        int is_shared = 0;
        nscan = fscanf(fp, "%256s %d,%d,%d,%d %d,%d,%d,%d %d %256s\n",
                           tensor_name,
                           &valid_shape[0],
                           &valid_shape[1],
                           &valid_shape[2],
                           &valid_shape[3],
                           &real_shape[0],
                           &real_shape[1],
                           &real_shape[2],
                           &real_shape[3],
                           &is_shared,
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
        Shape vshape(valid_shape[0], valid_shape[1], valid_shape[2], valid_shape[3]);
        Shape rshape(real_shape[0], real_shape[1], real_shape[2], real_shape[3]);
        if (is_shared > 0) {
            tensor->set_shape(vshape);
            auto it = _tensors.find(tensor_shared_name);
            if (it == _tensors.end()) {
                printf("could not find tensor: %s\n", tensor_shared_name);
                return SaberInvalidValue;
            }
            tensor->share_from(*it->second);
        } else {
            tensor->re_alloc(rshape);
            tensor->set_shape(vshape);
        }
        _tensors[tensor_name] = tensor;
//        printf("%s vshape: %d,%d,%d,%d, rshape: %d,%d,%d,%d share:%s\n",
//               tensor_name,
//               valid_shape[0],
//               valid_shape[1],
//               valid_shape[2],
//               valid_shape[3],
//               real_shape[0],
//               real_shape[1],
//               real_shape[2],
//               real_shape[3],
//               tensor_shared_name);
    }

    //! get inputs and outputs name
    int in_num = 0;
    int out_num = 0;
    nscan = fscanf(fp, "inputs %d", &in_num);
    for (int i = 0; i < in_num; ++i) {
        char in_name[256];
        nscan = fscanf(fp, " %256s", in_name);
        _ins.push_back(in_name);
    }
    nscan = fscanf(fp, "\n");
    nscan = fscanf(fp, "outputs %d", &out_num);
    for (int i = 0; i < out_num; ++i) {
        char out_name[256];
        nscan = fscanf(fp, " %256s", out_name);
        _outs.push_back(out_name);
    }
    nscan = fscanf(fp, "\n");
//    printf("inputs: %d\n", _ins.size());
//    for (int i = 0; i < _ins.size(); ++i) {
//        printf("%s\n", _ins[i].c_str());
//    }
//    printf("outputs: %d\n", _outs.size());
//    for (int i = 0; i < _outs.size(); ++i) {
//        printf("%s\n", _outs[i].c_str());
//    }

    //! get ops and params
    _ops.clear();
    int ops_num = 0;
    nscan = fscanf(fp, "OPS %d\n", &ops_num);
    //printf("ops number: %d\n", ops_num);
    _tensor_ins.resize(ops_num);
    _tensor_outs.resize(ops_num);
    _ops.resize(ops_num);
    for (int i = 0; i < ops_num; ++i) {
        char op_type[256];
        char op_name[256];
        int in_num = 0;
        int out_num = 0;
        nscan = fscanf(fp, "%256s %256s %d %d ", op_type, op_name, &in_num, &out_num);
        if (nscan != 4) {
            printf("load ops: %s falied: %d\n", op_name, nscan);
            return SaberInvalidValue;
        }
        //printf("op type: %s, op_name: %s, ins: %d, outs: %d\n", op_type, op_name, in_num, out_num);
        std::vector<Tensor<CPU, AK_FLOAT>*> tensor_ins;
        for (int j = 0; j < in_num; ++j) {
            char in_name[256];
            nscan = fscanf(fp, "%256s ", in_name);
            auto it = _tensors.find(in_name);
            if (it == _tensors.end()) {
                printf("tensor name: %s not exits\n", in_name);
                return SaberInvalidValue;
            }
            //printf("find ins: %s\n", in_name);
            tensor_ins.push_back(it->second);
        }
        _tensor_ins[i] = tensor_ins;
        std::vector<Tensor<CPU, AK_FLOAT>*> tensor_outs;
        for (int j = 0; j < out_num; ++j) {
            char out_name[256];
            nscan = fscanf(fp, "%256s ", out_name);
            auto it = _tensors.find(out_name);
            if (it == _tensors.end()) {
                printf("tensor name: %s not exits\n", out_name);
                return SaberInvalidValue;
            }
            //printf("find outs: %s\n", out_name);
            tensor_outs.push_back(it->second);
        }
        _tensor_outs[i] = tensor_outs;

        //! create op and load param
        OpBase* op = OpRegistry::create_op(op_type);
        op->set_op_name(op_name);
        op->load_param(fp, _weights);
        _ops[i] = op;
    }
    //! get default input shape
    _last_input_shapes.resize(_ins.size());
    for (int i = 0; i < _ins.size(); ++i) {
        _last_input_shapes[i] = _tensors[_ins[i]]->valid_shape();
    }
    return init();
}

//SaberStatus Net::load_model_from_memory(const void *memory) {
//    return SaberUnImplError;
//}

SaberStatus Net::prediction() {
    if (_weights == nullptr) {
        printf("load model before prediction\n");
        return SaberNotInitialized;
    }
    _ctx->set_run_mode(_mode, _threads);
    for (int i = 0; i < _ins.size(); ++i) {
        if (_last_input_shapes[i] == _tensors[_ins[i]]->valid_shape()) {
            continue;
        } else {
            init();
            break;
        }
    }
    for (int i = 0; i < _ops.size(); ++i) {
        LCHECK_EQ(_ops[i]->dispatch(_tensor_ins[i], _tensor_outs[i]), SaberSuccess, "run op failed");
#if 1//def ENABLE_DEBUG
        for (int j = 0; j < _tensor_outs[i].size(); ++j) {
            double meanval = tensor_mean(*_tensor_outs[i][j]);
            printf("op: %s, mean: %.6f\n", _ops[i]->get_op_name(), meanval);
        }
#endif
    }
    return SaberSuccess;
}

SaberStatus Net::init() {
    for (int i = 0; i < _ops.size(); ++i) {
        LCHECK_EQ(_ops[i]->compute_output_shape(_tensor_ins[i], _tensor_outs[i]), SaberSuccess, "compute shape failed");
        LCHECK_EQ(_ops[i]->init(_tensor_ins[i], _tensor_outs[i], *_ctx), SaberSuccess, "init failed");
    }
    return SaberSuccess;
}

std::vector<Tensor<CPU, AK_FLOAT> *> Net::get_input() {
    std::vector<Tensor<CPU, AK_FLOAT>*> ins;
    for (int i = 0; i < _ins.size(); ++i) {
        ins.push_back(_tensors[_ins[i]]);
    }
    return ins;
}

Tensor<CPU, AK_FLOAT> * Net::get_input(std::string name) {
    if (_tensors.find(name) == _tensors.end()) {
        LCHECK_EQ(true, false, "input is not exits");
        return nullptr;
    }
    return _tensors[name];
}

std::vector<Tensor<CPU, AK_FLOAT>*> Net::get_output() {
    std::vector<Tensor<CPU, AK_FLOAT>*> out;
    for (int i = 0; i < _outs.size(); ++i) {
        out.push_back(_tensors[_outs[i]]);
    }
    return out;
}

Tensor<CPU, AK_FLOAT> * Net::get_output(std::string name) {
    if (_tensors.find(name) == _tensors.end()) {
        LCHECK_EQ(true, false, "output tensor is not exits");
        return nullptr;
    }
    return _tensors[name];
}

} // namespace lite

} // namespace saber

} // namespace anakin
