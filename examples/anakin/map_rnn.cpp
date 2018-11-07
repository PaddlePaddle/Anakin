#include "anakin_helper.h"
#include <string.h>
class Data {
public:
    Data(std::string file_name, int batch_size) :
        _batch_size(batch_size),
        _total_length(0) {
        _file.open(file_name);

        if (_file.is_open() != true) {
            printf("file open failed\n");
            exit(-1);
        };

        _file.seekg(_file.end);

        _total_length = _file.tellg();

        _file.seekg(_file.beg);
    }
    void get_batch_data(std::vector<std::vector<float>>& fea,
                        std::vector<std::vector<float>>& week_fea,
                        std::vector<std::vector<float>>& time_fea,
                        std::vector<int>& seq_offset);
private:
    std::fstream _file;
    int _total_length;
    int _batch_size;
};

void Data::get_batch_data(std::vector<std::vector<float>>& fea,
                          std::vector<std::vector<float>>& week_fea,
                          std::vector<std::vector<float>>& time_fea,
                          std::vector<int>& seq_offset) {
    if (_file.is_open() != true) {
        printf("file open failed\n");
        exit(-1);
    };

    int seq_num = 0;

    int cum = 0;

    char buf[10000];

    seq_offset.clear();

    seq_offset.push_back(0);

    fea.clear();

    week_fea.clear();

    time_fea.clear();

    while (_file.getline(buf, 10000)) {
        std::string s = buf;
        std::vector<std::string> deli_vec = {":"};
        std::vector<std::string> data_vec = ReadDataHelper::string_split(s, deli_vec);

        std::vector<std::string> seq;
        seq = ReadDataHelper::string_split(data_vec[0], {"|"});

        for (auto link : seq) {
            std::vector<std::string> data = ReadDataHelper::string_split(link, ",");
            std::vector<float> vec;

            for (int i = 0; i < data.size(); i++) {
                vec.push_back(atof(data[i].c_str()));
            }

            fea.push_back(vec);
        }

        std::vector<std::string> week_data;
        std::vector<std::string> time_data;

        week_data = ReadDataHelper::string_split(data_vec[2], ",");
        std::vector<float> vec_w;

        for (int i = 0; i < week_data.size(); i++) {
            vec_w.push_back(atof(week_data[i].c_str()));
        }

        week_fea.push_back(vec_w);

        time_data = ReadDataHelper::string_split(data_vec[1], ",");
        std::vector<float> vec_t;

        for (int i = 0; i < time_data.size(); i++) {
            vec_t.push_back(atof(time_data[i].c_str()));
        }

        time_fea.push_back(vec_t);

        cum += seq.size();
        seq_offset.push_back(cum);

        seq_num++;

        if (seq_num >= _batch_size) {
            break;
        }

    }
}


class AKRNNExampleX86 {
public:
    AKRNNExampleX86(std::string ak_so_dir, std::string model_path): _model_path(model_path) {
        AKAutoChoose auto_choose(ak_so_dir);
        _anakin_obj = auto_choose.get_ak_instance("X86", 0);
        _anakin_obj->load_model(model_path, 100);
    }
    void run(std::string data_path) {
        Data map_data(data_path, 1);
        std::vector<std::vector<float>> fea;
        std::vector<std::vector<float>> week_fea;
        std::vector<std::vector<float>> time_fea;
        std::vector<int> seq_offset;
        int count = 0;

        while (true) {
            seq_offset.clear();
            map_data.get_batch_data(fea, week_fea, time_fea, seq_offset);

            if (seq_offset.size() <= 1) {
                break;
            }

            std::vector<int> fea_shape = {fea.size(), 38, 1, 1};
            std::vector<int> week_fea_shape = {week_fea.size(), 10, 1, 1};
            std::vector<int> time_fea_shape = {time_fea.size(), 10, 1, 1};
            std::vector<std::vector<int>> lod = {seq_offset};

            AnakinRunerTensorInterface* input_fea = _anakin_obj->get_input_tensor(0);
            AnakinRunerTensorInterface* input_week_fea = _anakin_obj->get_input_tensor(0);
            AnakinRunerTensorInterface* input_time_fea = _anakin_obj->get_input_tensor(0);
            input_fea->set_host_shape(fea_shape);
            input_fea->set_dev_shape(fea_shape);
            input_week_fea->set_host_shape(week_fea_shape);
            input_week_fea->set_dev_shape(week_fea_shape);
            input_time_fea->set_host_shape(time_fea_shape);
            input_time_fea->set_dev_shape(time_fea_shape);

            float* input = static_cast<float*>(input_fea->get_host_data());

            for (int i = 0; i < fea.size(); i++) {
                memcpy(input + i * 38, &fea[i][0], sizeof(float) * 38);
            }

            input = static_cast<float*>(input_week_fea->get_host_data());

            for (int i = 0; i < week_fea.size(); i++) {
                memcpy(input + i * 10, &week_fea[i][0], sizeof(float) * 10);
            }

            input = static_cast<float*>(input_time_fea->get_host_data());

            for (int i = 0; i < time_fea.size(); i++) {
                memcpy(input + i * 10, &time_fea[i][0], sizeof(float) * 10);
            }

            input_fea->copy_data_host_2_dev();
            input_week_fea->copy_data_host_2_dev();
            input_time_fea->copy_data_host_2_dev();
            input_fea->set_dev_lod_offset(lod);
            _anakin_obj->prediction();

            AnakinRunerTensorInterface* output_0 = _anakin_obj->get_output_tensor(0);
            output_0->copy_data_dev_2_host();
            float* out_ptr = static_cast<float*>(output_0->get_host_data());
            SaveDataHelper::record_2_file(out_ptr, output_0->get_dev_size(),
                                          ("record_" + std::to_string(count)).c_str());
            count++;
        }
    }
private:
    AnakinRunerInterface* _anakin_obj;
    std::string _model_path;
};

int main() {
    std::string so_path = "../../output/libanakin.so";
    std::string model_path = "/home/mount/x86_model/gis_rnn.anakin2.bin";
    //    std::string model_path="/home/workspace/x86_model/route-dnn.anakin2.bin";
    //    std::string model_path="/home/workspace/x86_model/language_model.anakin2.bin";
    std::string data_path = "/home/mount/x86_model/test_features_sys";

    //    AKRNNExampleNV ak_run(so_path,model_path);
    //    AKRNNExampleX86 ak_run(so_path,model_path);
    AKRNNExampleX86 ak_run(so_path, model_path);
    ak_run.run(data_path);
}