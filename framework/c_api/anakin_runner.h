
#ifndef ANAKIN_ANAKIN_RUNNER_H
#define ANAKIN_ANAKIN_RUNNER_H


#include <vector>
#include <string>

extern "C" {

class AnakinRunerTensorInterface {
public:

    virtual void set_dev_shape(std::vector<int>& new_shape) {};
    virtual void set_host_shape(std::vector<int>& new_shape) {};

    virtual void set_dev_lod_offset(std::vector<std::vector<int>>& new_seq_shape) {};
    virtual void set_host_lod_offset(std::vector<std::vector<int>>& new_seq_shape) {};

    virtual std::vector<int> get_dev_shape() { return std::vector<int>();};
    virtual std::vector<int> get_host_shape() { return std::vector<int>();};

    virtual int get_dev_size() { return 0;};
    virtual int get_host_size() { return 0;};

    virtual std::vector<std::vector<int>> get_dev_lod_offset() { return std::vector<std::vector<int>>();};
    virtual std::vector<std::vector<int>> get_host_lod_offset() { return std::vector<std::vector<int>>();};

    virtual void* get_dev_data() { return nullptr;};
    virtual void* get_host_data() { return nullptr;};

    virtual void copy_data_dev_2_host() {};
    virtual void copy_data_host_2_dev() {};

};

class AnakinRunerInterface {
public:
    virtual bool load_model(std::string model_path, int max_batch_size) { return false;};

    virtual int get_input_number() { return 0;};

    virtual int get_output_number() { return 0;};

    virtual AnakinRunerTensorInterface* get_input_tensor(int index) { return nullptr;};

    virtual AnakinRunerTensorInterface* get_output_tensor(int index) { return nullptr;};

    virtual void prediction() {};

};


AnakinRunerInterface* get_anakinrun_instance(const char* device_type, int device_number);
char* get_ak_cpu_arch_string();
void set_ak_cpu_parallel();

}
#endif //ANAKIN_ANAKIN_RUNNER_H
