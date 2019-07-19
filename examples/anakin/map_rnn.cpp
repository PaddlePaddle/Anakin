#include "anakin_helper.h"
#include <string.h>
bool g_print_data=false;
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
    AKRNNExampleX86(std::string ak_so_dir, std::string model_path,int max_batch): _model_path(model_path) {
        AKAutoChoose auto_choose(ak_so_dir);
        _anakin_obj = auto_choose.get_ak_instance("X86", 0);
        _anakin_obj->load_model(model_path, max_batch);
    }
    AKRNNExampleX86(std::string ak_so_dir,std::string ak_so_path, std::string model_path,int max_batch): _model_path(model_path) {
        AKAutoChoose auto_choose(ak_so_dir,ak_so_path);
        _anakin_obj = auto_choose.get_ak_instance("X86", 0);
        _anakin_obj->load_model(model_path, max_batch);
    }
    void run(std::string data_path,int bach_size=1) {
        Data map_data(data_path, bach_size);

        std::vector<std::vector<std::vector<float>>> fea_vec;
        std::vector<std::vector<std::vector<float>>> week_fea_vec;
        std::vector<std::vector<std::vector<float>>> time_fea_vec;
        std::vector<std::vector<int>> seq_offset_vec;
        MiniTimer load_time;
        load_time.start();
        int temp_conter = 0;

        while (true) {
            std::vector<int>* seq_offset = new std::vector<int>();
            std::vector<std::vector<float>>* fea = new std::vector<std::vector<float>>();
            std::vector<std::vector<float>>* week_fea = new std::vector<std::vector<float>>();
            std::vector<std::vector<float>>* time_fea = new std::vector<std::vector<float>>();
            map_data.get_batch_data(*fea, *week_fea, *time_fea, *seq_offset);

            if (seq_offset->size() <= 1) {
                break;
            }

            fea_vec.push_back(*fea);
            week_fea_vec.push_back(*week_fea);
            time_fea_vec.push_back(*time_fea);
            seq_offset_vec.push_back(*seq_offset);
            temp_conter++;
            //        LOG(INFO)<<seq_offset->size()<<","<<fea->size()<<","<<fea_vec.size();
        }
        double load_time_ms =load_time.end();
        printf("load time = %f , for %d batch data ,batch size = %d\n",load_time_ms,temp_conter,bach_size);

        MiniTimer timer;
        timer.start();
        for (int batch_id = 0; batch_id < temp_conter; batch_id++) {
            std::vector<int>& seq_offset = seq_offset_vec[batch_id];
            std::vector<std::vector<float>>& fea = fea_vec[batch_id];
            std::vector<std::vector<float>>& week_fea = week_fea_vec[batch_id];
            std::vector<std::vector<float>>& time_fea = time_fea_vec[batch_id];

            std::vector<int> fea_shape = {fea.size(), 38, 1, 1};
            std::vector<int> week_fea_shape = {week_fea.size(), 10, 1, 1};
            std::vector<int> time_fea_shape = {time_fea.size(), 10, 1, 1};
            std::vector<std::vector<int>> lod = {seq_offset};

            AnakinRunerTensorInterface* input_fea = _anakin_obj->get_input_tensor(0);
            AnakinRunerTensorInterface* input_week_fea = _anakin_obj->get_input_tensor(1);
            AnakinRunerTensorInterface* input_time_fea = _anakin_obj->get_input_tensor(2);
            input_fea->set_host_shape(fea_shape);
//            input_fea->set_dev_shape(fea_shape);
            input_week_fea->set_host_shape(week_fea_shape);
//            input_week_fea->set_dev_shape(week_fea_shape);
            input_time_fea->set_host_shape(time_fea_shape);
//            input_time_fea->set_dev_shape(time_fea_shape);

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

//            input_fea->copy_data_host_2_dev();
//            input_week_fea->copy_data_host_2_dev();
//            input_time_fea->copy_data_host_2_dev();
            input_fea->set_dev_lod_offset(lod);
            _anakin_obj->prediction();


            if(g_print_data){
                AnakinRunerTensorInterface* output_0 = _anakin_obj->get_output_tensor(0);
                for (int seq_id = 0; seq_id < seq_offset.size() - 1; seq_id++) {
                    int seq_len = seq_offset[seq_id + 1] - seq_offset[seq_id];
                    int seq_start = seq_offset[seq_id];

                    for (int i = 0; i < seq_len - 1; i++) {
                        printf("%f|", static_cast<float*>(output_0->get_host_data())[seq_start + i]);
                    }

                    printf("%f\n", static_cast<float*>(output_0->get_host_data())[seq_start + seq_len - 1]);
                }
            }


//            output_0->copy_data_dev_2_host();
//            float* out_ptr = static_cast<float*>(output_0->get_host_data());
//            SaveDataHelper::record_2_file(out_ptr, output_0->get_dev_size(),
//                                          ("record_" + std::to_string(count)).c_str());

        }
        double use_time=timer.end();
        std::cout<<"time = "<<use_time<<" ms, avg = "<<use_time/temp_conter<<" ms"<<std::endl;
    }
private:
    AnakinRunerInterface* _anakin_obj;
    std::string _model_path;
};

int main(int argc, const char** argv) {
    std::string so_dir = "../../x86_output";
    std::string so_path="../../output/libanakin.so";
    std::string model_path = "/home/liujunjie03/model/gis_rnn.anakin2.bin";
    std::string data_path = "/home/liujunjie03/model/test_features_sys";
    int max_batch=10000;
    int batch_size=1;
    if(argc>1) {
        data_path = argv[1];
    }

    if (argc > 2) {
        model_path = argv[2];
    }

    if (argc > 3) {
        max_batch = atoi(argv[3]);
    }

    if (argc > 4) {
        batch_size = atoi(argv[4]);
    }

    if (argc > 5) {
        g_print_data=atoi(argv[5]);
    }

    if (argc > 6) {
        so_dir=argv[6];
    }

    if(argc > 7){
        so_path = argv[7];
    }

    if(argc<=7){
        AKRNNExampleX86 ak_run(so_dir, model_path, max_batch);
        ak_run.run(data_path,batch_size);
    }else {
        AKRNNExampleX86 ak_run(so_dir, so_path, model_path, max_batch);
        ak_run.run(data_path,batch_size);
    }

}