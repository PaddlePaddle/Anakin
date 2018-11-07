

#include "anakin_runner.h"
#include "anakin_config.h"
#include "saber/saber_types.h"
#include "framework/graph/graph.h"
#include "framework/core/net/net.h"

using namespace anakin;
using namespace anakin::graph;

#ifdef USE_X86_PLACE
class AnakinRunerTensorX86: public AnakinRunerTensorInterface {
public:

    AnakinRunerTensorX86(): tensor_ptr(nullptr) {};
    AnakinRunerTensorX86(Tensor<X86>* inner): tensor_ptr(static_cast<void*>(inner)) {};
    void reset(Tensor<X86>* inner) {
        tensor_ptr = static_cast<void*>(inner);
    }

    void set_dev_shape(std::vector<int>& new_shape) override {
        static_cast<Tensor<X86>*>(tensor_ptr)->reshape(static_cast<Shape>(new_shape));
    }
    void set_host_shape(std::vector<int>& new_shape) override {
        static_cast<Tensor<X86>*>(tensor_ptr)->reshape(static_cast<Shape>(new_shape));
    };

    void set_dev_lod_offset(std::vector<std::vector<int>>& new_seq_shape) override {
        static_cast<Tensor<X86>*>(tensor_ptr)->set_seq_offset(new_seq_shape);
    };
    void set_host_lod_offset(std::vector<std::vector<int>>& new_seq_shape) override {
        static_cast<Tensor<X86>*>(tensor_ptr)->set_seq_offset(new_seq_shape);
    };

    std::vector<int> get_dev_shape() override {
        return static_cast<Tensor<X86>*>(tensor_ptr)->valid_shape();
    };
    std::vector<int> get_host_shape() override {
        return static_cast<Tensor<X86>*>(tensor_ptr)->valid_shape();
    };

    int get_dev_size() override {
        return static_cast<Tensor<X86>*>(tensor_ptr)->valid_size();
    };
    int get_host_size() override {
        return static_cast<Tensor<X86>*>(tensor_ptr)->valid_size();
    };

    std::vector<std::vector<int>> get_dev_lod_offset() override {
        return static_cast<Tensor<X86>*>(tensor_ptr)->get_seq_offset();
    };
    std::vector<std::vector<int>> get_host_lod_offset() override {
        return static_cast<Tensor<X86>*>(tensor_ptr)->get_seq_offset();
    };

    void* get_dev_data() override {
        return static_cast<void*>(static_cast<Tensor<X86>*>(tensor_ptr)->mutable_data());
    };
    void* get_host_data() override {
        return static_cast<void*>(static_cast<Tensor<X86>*>(tensor_ptr)->mutable_data());
    };

private:
    void* tensor_ptr;

};

#endif

#ifdef NVIDIA_GPU
class AnakinRunerTensorNV: public AnakinRunerTensorInterface {
public:

    AnakinRunerTensorNV(): tensor_ptr_host(nullptr), tensor_ptr_dev(nullptr) {};

    AnakinRunerTensorNV(Tensor<NV>* tensor_ptr_dev_in, Tensor<NVHX86>* tensor_ptr_host_in):
        tensor_ptr_host(static_cast<void*>(tensor_ptr_host_in)),
        tensor_ptr_dev(static_cast<void*>(tensor_ptr_dev_in)) {}

    void reset(Tensor<NV>* tensor_ptr_dev_in, Tensor<NVHX86>* tensor_ptr_host_in) {
        tensor_ptr_host = static_cast<void*>(tensor_ptr_host_in);
        tensor_ptr_dev = static_cast<void*>(tensor_ptr_dev_in);
    }

    void set_dev_shape(std::vector<int>& new_shape) override {
        static_cast<Tensor<NV>*>(tensor_ptr_dev)->reshape(static_cast<Shape>(new_shape));
    }
    void set_host_shape(std::vector<int>& new_shape) override {
        static_cast<Tensor<NVHX86>*>(tensor_ptr_host)->reshape(static_cast<Shape>(new_shape));
    };

    void set_dev_lod_offset(std::vector<std::vector<int>>& new_seq_shape) override {
        static_cast<Tensor<NV>*>(tensor_ptr_dev)->set_seq_offset(new_seq_shape);
    };
    void set_host_lod_offset(std::vector<std::vector<int>>& new_seq_shape) override {
        static_cast<Tensor<NVHX86>*>(tensor_ptr_host)->set_seq_offset(new_seq_shape);
    };

    std::vector<int> get_dev_shape() override {
        return static_cast<Tensor<NV>*>(tensor_ptr_dev)->valid_shape();
    };
    std::vector<int> get_host_shape() override {
        return static_cast<Tensor<NVHX86>*>(tensor_ptr_host)->valid_shape();
    };

    int get_dev_size() override {
        return static_cast<Tensor<NV>*>(tensor_ptr_dev)->valid_size();
    };
    int get_host_size() override {
        return static_cast<Tensor<NVHX86>*>(tensor_ptr_host)->valid_size();
    };

    std::vector<std::vector<int>> get_host_lod_offset() override {
        return static_cast<Tensor<NV>*>(tensor_ptr_dev)->get_seq_offset();
    };
    std::vector<std::vector<int>> get_dev_lod_offset() override {
        return static_cast<Tensor<NVHX86>*>(tensor_ptr_host)->get_seq_offset();
    };

    void* get_dev_data() override {
        return static_cast<void*>(static_cast<Tensor<NV>*>(tensor_ptr_dev)->mutable_data());
    };
    void* get_host_data() override {
        return static_cast<void*>(static_cast<Tensor<NVHX86>*>(tensor_ptr_host)->mutable_data());
    };

    void copy_data_host_2_dev() {
        Tensor<NVHX86>* host_tensor_ptr = static_cast<Tensor<NVHX86>*>(tensor_ptr_host);
        Tensor<NV>* dev_tensor_ptr = static_cast<Tensor<NV>*>(tensor_ptr_dev);
        dev_tensor_ptr->copy_from(*host_tensor_ptr);
    };

    void copy_data_dev_2_host() {
        Tensor<NVHX86>* host_tensor_ptr = static_cast<Tensor<NVHX86>*>(tensor_ptr_host);
        Tensor<NV>* dev_tensor_ptr = static_cast<Tensor<NV>*>(tensor_ptr_dev);
        host_tensor_ptr->copy_from(*dev_tensor_ptr);
    };

private:
    void* tensor_ptr_host;
    void* tensor_ptr_dev;
};

#endif
template <typename TarType>
class AnakinRuner : public AnakinRunerInterface {};

#ifdef USE_X86_PLACE
template <>
class AnakinRuner<X86> : public AnakinRunerInterface {
public:
    bool load_model(std::string model_path, int max_batch_size) {
        delete _net;
        delete _graph;

        _graph = new Graph<X86, Precision::FP32>();
        _graph->load(model_path);
        _vin_name = _graph->get_ins();
        LOG(INFO) << "input size: " << _vin_name.size();
        LOG(INFO) << "set batchsize to " << max_batch_size;
        _vin_ak_tensor_ptr.resize(_vin_name.size());

        for (int j = 0; j < _vin_name.size(); ++j) {
            LOG(INFO) << "input name: " << _vin_name[j];
            _graph->ResetBatchSize(_vin_name[j], max_batch_size);
            _vin_ak_tensor_ptr[j] = new AnakinRunerTensorX86();
        }

        _vout_name = _graph->get_outs();
        LOG(INFO) << "output size: " << _vout_name.size();
        _vout_ak_tensor_ptr.resize(_vout_name.size());

        for (int j = 0; j < _vout_name.size(); ++j) {
            LOG(INFO) << "output name: " << _vout_name[j];
            _vout_ak_tensor_ptr[j] = new AnakinRunerTensorX86();
        }

        _graph->Optimize();
        _net = new Net<X86, Precision::FP32>(*_graph, true);
        return true;
    }

    AnakinRunerTensorInterface* get_input_tensor(int index) {
        _vin_ak_tensor_ptr[index]->reset(_net->get_in(_vin_name[index]));
        return _vin_ak_tensor_ptr[index];
    }

    AnakinRunerTensorInterface* get_output_tensor(int index) {
        _vout_ak_tensor_ptr[index]->reset(_net->get_out(_vout_name[index]));
        return _vout_ak_tensor_ptr[index];
    }

    int get_input_number() {
        return _vin_name.size();
    }
    int get_output_number() {
        return _vout_name.size();
    }
    void prediction() {
        auto in = _net->get_in(_vin_name[0]);
        _net->prediction();
    }

    ~AnakinRuner() {
        delete _net;
        delete _graph;
    }

private:
    Graph <X86, Precision::FP32>* _graph;
    Net<X86, Precision::FP32>* _net;
    std::vector<std::string>_vin_name;
    std::vector<std::string> _vout_name;
    std::vector<AnakinRunerTensorX86*> _vin_ak_tensor_ptr;
    std::vector<AnakinRunerTensorX86*> _vout_ak_tensor_ptr;
};

#endif

#ifdef NVIDIA_GPU
template <>
class AnakinRuner<NV> : public AnakinRunerInterface {
public:
    bool load_model(std::string model_path, int max_batch_size) {
        delete _net;
        delete _graph;

        _graph = new Graph<NV, Precision::FP32>();
        _graph->load(model_path);
        _vin_name = _graph->get_ins();
        _vin_host.resize(_vin_name.size());
        _vin_ak_tensor_ptr.resize(_vin_name.size());
        LOG(INFO) << "input size: " << _vin_name.size();
        LOG(INFO) << "set batchsize to " << max_batch_size;

        for (int j = 0; j < _vin_name.size(); ++j) {
            LOG(INFO) << "input name: " << _vin_name[j];
            _graph->ResetBatchSize(_vin_name[j], max_batch_size);
            _vin_host[j] = new Tensor<NVHX86>();
            _vin_ak_tensor_ptr[j] = new AnakinRunerTensorNV();
        }

        _vout_name = _graph->get_outs();
        _vout_host.resize(_vout_name.size());
        _vout_ak_tensor_ptr.resize(_vout_name.size());
        LOG(INFO) << "output size: " << _vout_name.size();

        for (int j = 0; j < _vout_name.size(); ++j) {
            LOG(INFO) << "output name: " << _vout_name[j];
            _vout_host[j] = new Tensor<NVHX86>();
            _vout_ak_tensor_ptr[j] = new AnakinRunerTensorNV();
        }

        _graph->Optimize();
        _net = new Net<NV, Precision::FP32>(*_graph, true);
        return true;

    }

    AnakinRunerTensorInterface* get_input_tensor(int index) {
        _vin_ak_tensor_ptr[index]->reset(_net->get_in(_vin_name[index]), _vin_host[index]);
        return _vin_ak_tensor_ptr[index];
    }

    AnakinRunerTensorInterface* get_output_tensor(int index) {
        _vout_ak_tensor_ptr[index]->reset(_net->get_out(_vout_name[index]), _vout_host[index]);
        return _vout_ak_tensor_ptr[index];;
    }
    int get_input_number() {
        return _vin_name.size();
    }
    int get_output_number() {
        return _vout_name.size();
    }
    void prediction() {
        auto in = _net->get_in(_vin_name[0]);
        cudaDeviceSynchronize();
        _net->prediction();
        cudaDeviceSynchronize();
    }

    ~AnakinRuner() {
        delete _net;
        delete _graph;
    }

private:
    Graph <NV, Precision::FP32>* _graph;
    Net<NV, Precision::FP32>* _net;
    std::vector<std::string>_vin_name;
    std::vector<std::string> _vout_name;
    std::vector<Tensor<NVHX86>*> _vin_host;
    std::vector<Tensor<NVHX86>*> _vout_host;
    std::vector<AnakinRunerTensorNV*> _vin_ak_tensor_ptr;
    std::vector<AnakinRunerTensorNV*> _vout_ak_tensor_ptr;
};

#endif

AnakinRunerInterface* get_anakinrun_instance(const char* device_type, int device_number) {
    printf("device type = %s , number = %d \n", device_type, device_number);
    std::string device_type_string(device_type);

    if (device_type_string == "NV") {
#if defined(NVIDIA_GPU)

        return new AnakinRuner<NV>();
#else
        LOG(FATAL) << "the so not support type " << device_type;
#endif
    } else if (device_type_string == "X86") {
#if defined(USE_X86_PLACE)
        set_ak_cpu_parallel();
        return new AnakinRuner<X86>();
#else
        LOG(FATAL) << "the so not support type " << device_type;
#endif

    } else {
        LOG(FATAL) << "the so not support type :" << device_type_string;
    }

    return nullptr;
}

const char* get_ak_cpu_arch_string() {
#ifdef USE_X86_PLACE
    return BUILD_X86_ARCH;
#else
    return "NO_X86";
#endif
}

#ifdef USE_X86_PLACE

#include "omp.h"
#include "mkl_service.h"
void set_ak_cpu_parallel() {
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    mkl_set_num_threads(1);
}
#endif





