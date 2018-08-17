#include "saber/lite/net/net_lite.h"
#include "saber/lite/core/tensor_op_lite.h"
#include <fstream>
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

SaberStatus Net::load_model_weights(std::istream &stream, size_t size) {
    if (_weights) {
        fast_free(_weights);
        _weights = nullptr;
    }
    _weights = static_cast<float*>(fast_malloc(size));
    stream.read((char*)_weights, size);
    return SaberSuccess;
}

SaberStatus Net::load_model_info(std::istream &stream) {
    std::string strtmp;
    int tensor_size;
    stream >> strtmp >> tensor_size;
    //printf("tensor size %d\n", tensor_size);

    for (int i = 0; i < tensor_size; ++i) {
        std::vector<int> val_sh;
        std::vector<int> real_sh;
        int is_shared = 0;
        int val_size;
        int real_size;
        std::string tensor_name, shared_name;

        stream >> tensor_name >> val_size;
        val_sh.resize(val_size);
        for (int j = 0; j < val_size; ++j) {
            stream >> val_sh[j];
        }
        stream >> real_size;
        real_sh.resize(real_size);
        for (int j = 0; j < real_size; ++j) {
            stream >> real_sh[j];
        }
        stream >> is_shared >> shared_name;

        if(_tensors.find(tensor_name) != _tensors.end()) {
            printf("tensor is already registered: %s\n", tensor_name.c_str());
            return SaberInvalidValue;
        }
        Tensor<CPU, AK_FLOAT>* tensor = new Tensor<CPU, AK_FLOAT>;
        Shape vshape = val_sh;
        Shape rshape = real_sh;
        if (is_shared > 0) {
            tensor->set_shape(vshape);
            auto it = _tensors.find(shared_name);
            if (it == _tensors.end()) {
                printf("could not find tensor: %s\n", shared_name.c_str());
                return SaberInvalidValue;
            }
            tensor->share_from(*it->second);
        } else {
            tensor->re_alloc(rshape);
            tensor->set_shape(vshape);
        }
        _tensors[tensor_name] = tensor;
//        printf("%s vshape: %d,%d,%d,%d, rshape: %d,%d,%d,%d share:%s\n",
//               tensor_name.c_str(),
//               val_sh[0],
//               val_sh[1],
//               val_sh[2],
//               val_sh[3],
//               real_sh[0],
//               real_sh[1],
//               real_sh[2],
//               real_sh[3],
//               shared_name.c_str());
    }

    //! get inputs and outputs name
    int in_num = 0;
    int out_num = 0;
    stream >> strtmp >> in_num;
    for (int i = 0; i < in_num; ++i) {
        std::string in_name;
        stream >> in_name;
        _ins.push_back(in_name);
    }
    stream >> strtmp >> out_num;
    for (int i = 0; i < out_num; ++i) {
        std::string out_name;
        stream >> out_name;
        _outs.push_back(out_name);
    }
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
    stream >> strtmp >> ops_num;
    //printf("ops number: %d\n", ops_num);

    _tensor_ins.resize(ops_num);
    _tensor_outs.resize(ops_num);
    _ops.resize(ops_num);

    for (int i = 0; i < ops_num; ++i) {
        std::string op_type, op_name;
        int in_num = 0;
        int out_num = 0;
        stream >> op_type >> op_name >> in_num >> out_num;
        //printf("op type: %s, op_name: %s, ins: %d, outs: %d\n", op_type.c_str(), op_name.c_str(), in_num, out_num);
        std::vector<Tensor<CPU, AK_FLOAT>*> tensor_ins;
        for (int j = 0; j < in_num; ++j) {
            std::string in_tensor_name;
            stream >> in_tensor_name;
            auto it = _tensors.find(in_tensor_name);
            if (it == _tensors.end()) {
                printf("tensor name: %s not exits\n", in_tensor_name.c_str());
                return SaberInvalidValue;
            }
            //printf("find ins: %s\n", in_tensor_name.c_str());
            tensor_ins.push_back(it->second);
        }
        _tensor_ins[i] = tensor_ins;
        std::vector<Tensor<CPU, AK_FLOAT>*> tensor_outs;
        for (int j = 0; j < out_num; ++j) {
            std::string out_tensor_name;
            stream >> out_tensor_name;
            auto it = _tensors.find(out_tensor_name);
            if (it == _tensors.end()) {
                printf("tensor name: %s not exits\n", out_tensor_name.c_str());
                return SaberInvalidValue;
            }
            //printf("find outs: %s\n", out_tensor_name.c_str());
            tensor_outs.push_back(it->second);
        }
        _tensor_outs[i] = tensor_outs;

        //! create op and load param
        OpBase* op = OpRegistry::create_op(op_type);
#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
        op->set_op_name(op_name.c_str());
#endif
        op->load_param(stream, _weights);
        _ops[i] = op;
    }

    return SaberSuccess;
}

SaberStatus Net::load_model(const char *info_path, const char *weights_path) {
    std::fstream fp_w(weights_path, std::ios::in | std::ios::binary);
    // get length of weights:
    fp_w.seekg (0, std::ios::end);
    long long length = fp_w.tellg();
    fp_w.seekg (0, std::ios::beg);
    SaberStatus flag = load_model_weights(fp_w, length);
    if (!flag) {
        printf("load weights faild: %s\n", weights_path);
        return SaberInvalidValue;
    }
    fp_w.close();

    std::fstream fp_info(info_path, std::ios::in | std::ios::binary);

    flag = load_model_info(fp_info);
    if (!flag) {
        printf("load info faild: %s\n", info_path);
        return SaberInvalidValue;
    }
    fp_info.close();
    //! get default input shape
    _last_input_shapes.resize(_ins.size());
    for (int i = 0; i < _ins.size(); ++i) {
        _last_input_shapes[i] = _tensors[_ins[i]]->valid_shape();
    }
    return init();
}

SaberStatus Net::load_model(const char *lite_model_path) {

    std::fstream fp_merge(lite_model_path, std::ios::in | std::ios::binary);
    std::string strtmp;
    size_t weights_size;
    fp_merge >> strtmp >> weights_size;
    //printf("weights: %s, size : %lu\n", strtmp.c_str(), weights_size);
    //! get rid of \n
    fp_merge.seekg(1, std::ios::cur);

    //! load weights
    SaberStatus flag = load_model_weights(fp_merge, weights_size);
    if (!flag) {
        printf("load weights faild: %s\n", lite_model_path);
        return SaberInvalidValue;
    }
    //! load model
    flag = load_model_info(fp_merge);
    if (!flag) {
        printf("load info faild: %s\n", lite_model_path);
        return SaberInvalidValue;
    }
    fp_merge.close();
    //! get default input shape
    _last_input_shapes.resize(_ins.size());
    for (int i = 0; i < _ins.size(); ++i) {
        _last_input_shapes[i] = _tensors[_ins[i]]->valid_shape();
    }
    return init();
}

SaberStatus Net::load_model(const void *info_memory, size_t info_size,
                            const void *weights_memory, size_t weights_size) {
    std::string str_w(static_cast<const char*>(weights_memory), weights_size);
    std::istringstream w_stream(str_w);
    SaberStatus flag = load_model_weights(w_stream, weights_size);
    if (!flag) {
        printf("load model weights faild\n");
        return SaberInvalidValue;
    }

    std::string str_info(static_cast<const char*>(info_memory), info_size);
    std::istringstream info_stream(str_info);
    flag = load_model_info(info_stream);
    if (!flag) {
        printf("load model info faild\n");
        return SaberInvalidValue;
    }
    //! get default input shape
    _last_input_shapes.resize(_ins.size());
    for (int i = 0; i < _ins.size(); ++i) {
        _last_input_shapes[i] = _tensors[_ins[i]]->valid_shape();
    }
    return init();
}

SaberStatus Net::load_model(const void *merged_memory, size_t mem_size) {
    std::string casted_memory(static_cast<const char*>(merged_memory), mem_size);
    std::istringstream stream(casted_memory);
    std::string strtmp;
    size_t weights_size;
    stream >> strtmp >> weights_size;
    //printf("weights: %s, size : %lu\n", strtmp.c_str(), weights_size);
    //! get rid of \n
    stream.seekg(1, std::ios::cur);

    //! load weights
    SaberStatus flag = load_model_weights(stream, weights_size);
    if (!flag) {
        printf("load merged model weights faild\n");
        return SaberInvalidValue;
    }
    //! load model
    flag = load_model_info(stream);
    if (!flag) {
        printf("load merged model info faild\n");
        return SaberInvalidValue;
    }

    //! get default input shape
    _last_input_shapes.resize(_ins.size());
    for (int i = 0; i < _ins.size(); ++i) {
        _last_input_shapes[i] = _tensors[_ins[i]]->valid_shape();
    }
    return init();
}

#if 0
SaberStatus Net::load_model(const char* model_path) {
//    FILE* fp = fopen(lite_model_path, "rb");
//    if (!fp) {
//        printf("open %s failed\n", lite_model_path);
//        return SaberInvalidValue;
//    }
//    return load_model(fp);
}

//template <typename dtype>
SaberStatus Net::load_model(FILE* fp) {

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
#endif

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
            for (int j = 0; j < _ins.size(); ++j) {
                _last_input_shapes[j] = _tensors[_ins[j]]->valid_shape();
            }
            break;
        }
    }
    for (int i = 0; i < _ops.size(); ++i) {
        LCHECK_EQ(_ops[i]->dispatch(_tensor_ins[i], _tensor_outs[i]), SaberSuccess, "run op failed");
#ifdef ENABLE_DEBUG
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
