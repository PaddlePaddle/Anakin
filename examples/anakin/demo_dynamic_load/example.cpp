#include "anakin_helper.h"

class AKRNNExampleX86 {
public:
    AKRNNExampleX86(std::string ak_so_dir, std::string model_path): _model_path(model_path) {
        AKAutoChoose auto_choose(ak_so_dir);
        _anakin_obj = auto_choose.get_ak_instance("X86", 0);
        _anakin_obj->load_model(model_path, 100);
    }
    void run(std::string data_path) {
        std::vector<std::vector<float> > input_data = RNNDataHelper::get_input_data(data_path, ";", 0);
        int count = 0;

        for (auto input : input_data) {
            AnakinRunerTensorInterface* input_0 = _anakin_obj->get_input_tensor(0);
            std::vector<int> in_shape = {input.size(), 1, 1, 1};
            std::vector<std::vector<int>> in_seq = {{0, input.size()}};
            input_0->set_dev_shape(in_shape);
            input_0->set_dev_lod_offset(in_seq);
            float* in_ptr = static_cast<float*>(input_0->get_dev_data());

            for (int i = 0; i < input.size(); i++) {
                in_ptr[i] = input[i];
            }

            _anakin_obj->prediction();
            AnakinRunerTensorInterface* output_0 = _anakin_obj->get_output_tensor(0);
            float* out_ptr = static_cast<float*>(output_0->get_dev_data());
            SaveDataHelper::record_2_file(out_ptr, output_0->get_dev_size(),
                                          ("record_" + std::to_string(count)).c_str());
            count++;
        }
    }
private:
    AnakinRunerInterface* _anakin_obj;
    std::string _model_path;
};

class AKRNNExampleNV {
public:
    AKRNNExampleNV(std::string ak_so_dir, std::string model_path): _model_path(model_path) {
        AKAutoChoose auto_choose(ak_so_dir);
        _anakin_obj = auto_choose.get_ak_instance("NV", 0);
        _anakin_obj->load_model(model_path, 100);
        //        exit(0);
    }
    void run(std::string data_path) {
        std::vector<std::vector<float> > input_data = RNNDataHelper::get_input_data(data_path, ";", 0);
        int count = 0;

        for (auto input : input_data) {
            AnakinRunerTensorInterface* input_0 = _anakin_obj->get_input_tensor(0);
            std::vector<int> in_shape = {input.size(), 1, 1, 1};
            std::vector<std::vector<int>> in_seq = {{0, input.size()}};
            input_0->set_host_shape(in_shape);
            input_0->set_dev_shape(in_shape);
            input_0->set_dev_lod_offset(in_seq);
            float* in_ptr = static_cast<float*>(input_0->get_host_data());

            for (int i = 0; i < input.size(); i++) {
                in_ptr[i] = input[i];
            }

            input_0->copy_data_host_2_dev();
            _anakin_obj->prediction();
            AnakinRunerTensorInterface* output_0 = _anakin_obj->get_output_tensor(0);
            std::vector<int> out_shape = output_0->get_dev_shape();
            output_0->set_host_shape(out_shape);
            ShowDataHelper::show_vector(output_0->get_host_shape());
            output_0->copy_data_dev_2_host();
            float* out_ptr = static_cast<float*>(output_0->get_host_data());
            SaveDataHelper::record_2_file(out_ptr, output_0->get_host_size(),
                                          ("record_" + std::to_string(count)).c_str());
            count++;
        }
    }
private:
    AnakinRunerInterface* _anakin_obj;
    std::string _model_path;
};

class AKExampleAllOne {
public:
    AKExampleAllOne(std::string ak_so_dir, std::string model_path,
                    std::string dev_type): _model_path(model_path) {
        AKAutoChoose auto_choose(ak_so_dir);
        _anakin_obj = auto_choose.get_ak_instance(dev_type, 0);
        _anakin_obj->load_model(model_path, 1);
    }
    void run(std::string data_path) {
        for (int i = 0; i < _anakin_obj->get_input_number(); i++) {
            AnakinRunerTensorInterface* input_0 = _anakin_obj->get_input_tensor(i);
            std::vector<int> shape = input_0->get_dev_shape();
            input_0->set_host_shape(shape);
            float* in_ptr = static_cast<float*>(input_0->get_host_data());

            for (int i = 0; i < input_0->get_host_size(); i++) {
                in_ptr[i] = 1.f;
            }

            input_0->copy_data_host_2_dev();
        }

        _anakin_obj->prediction();

        for (int i = 0; i < _anakin_obj->get_output_number(); i++) {
            AnakinRunerTensorInterface* output_0 = _anakin_obj->get_output_tensor(i);
            std::vector<int> shape = output_0->get_dev_shape();
            output_0->set_host_shape(shape);
            ShowDataHelper::show_vector(shape);
            output_0->copy_data_dev_2_host();
            float* out_ptr = static_cast<float*>(output_0->get_host_data());
            SaveDataHelper::record_2_file(out_ptr, output_0->get_host_size(),
                                          ("record_output_" + std::to_string(i)).c_str());
        }
    }
private:
    AnakinRunerInterface* _anakin_obj;
    std::string _model_path;
};


int main() {
    std::string so_path = "../../x86_output";
    //    std::string model_path="/home/workspace/x86_model/language_model.anakin2.bin";
    //    std::string model_path="/home/workspace/x86_model/route-dnn.anakin2.bin";
    std::string model_path = "/home/liujunjie03/model/language_model.anakin2.bin";
    std::string data_path = "/home/liujunjie03/model/fake_realtitle.test_1000";

    //    AKRNNExampleNV ak_run(so_path,model_path);
    AKRNNExampleX86 ak_run(so_path, model_path);
    //    AKExampleAllOne ak_run(so_path,model_path,"X86");
    ak_run.run(data_path);
}