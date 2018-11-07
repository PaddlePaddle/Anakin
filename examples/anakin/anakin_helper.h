#ifndef ANAKIN_ANAKIN_HELPER_H
#define ANAKIN_ANAKIN_HELPER_H

#include <string>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <map>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <sys/time.h>
#include <cpuid.h>
#include <dlfcn.h>
#include "anakin_runner.h"

typedef std::string* (*cpu_arch_func)();
class CPUID_Helper {
    typedef union {
        struct {
            unsigned stepping   : 4;
            unsigned model      : 4;
            unsigned family     : 4;
            unsigned type       : 2;
            unsigned _reserved0 : 2;
            unsigned model_ext  : 4;
            unsigned family_ext : 8;
            unsigned _reserved1 : 4;
        };

        uint32_t value;

    } cpu_signature_t;
public:
    CPUID_Helper() {
        uint32_t ebx, ecx, edx;
        get_max_ext_leaf();
        __get_cpuid(0x1, &_cpu_signature.value, &ebx, &ecx, &edx);
    }
    std::string get_cpu_arch() {


        // print the stepping id, model and family of the cpu
        printf("Stepping ID: %X\n", _cpu_signature.stepping);
        printf("Model: %X\n", _cpu_signature.model);
        printf("Family: %X\n", _cpu_signature.family);

        uint32_t model_num = _cpu_signature.model;

        // check the extended model number
        if (_cpu_signature.family == 0x6 || _cpu_signature.family == 0xF) {
            model_num |= (_cpu_signature.model_ext << 4);

            printf("Extended Model: %X\n", model_num);
        }

        return microarch_info(model_num);
    }

private:

    void get_max_ext_leaf() {
        uint32_t ebx, ecx, edx;

        // get the maximum cpuid extended leaf
        __get_cpuid(0x80000000, &_max_ext_leaf, &ebx, &ecx, &edx);
    }

    std::string microarch_info(uint32_t model_num) {
        // https://software.intel.com/en-us/articles/intel-architecture-and-processor-identification-with-cpuid-model-and-family-numbers
        // https://en.wikipedia.org/wiki/List_of_Intel_CPU_microarchitectures
        // http://instlatx64.atw.hu/
        //
        // TODO: Older microarchitectures identification

        switch (model_num) {
        // atom microarchitectures
        case 0x1C:
        case 0x26:
            return "atom";//"Atom - 45 nm";

        case 0x36:
            return "atom";//"Atom - 32 nm";

        // mainline microarchitectures
        case 0x03:
        case 0x04:
            return "prescott";//"Prescott - 90 nm";

        case 0x06:
            return "presler";//"Presler - 65 nm";

        case 0x0D:
            return "dothan";//"Dothan - 90 nm";

        case 0x0F:
        case 0x16:
            return "merom";//"Merom - 65 nm";

        case 0x17:
        case 0x1D:
            return "penryn";//"Penryn - 45 nm";

        case 0x1A:
        case 0x1E:
        case 0x2E:
            return "nehalem";//"Nehalem - 45 nm";

        case 0x25:
        case 0x2C:
        case 0x2F:
            return "westmere";//"Westmere - 32 nm";

        case 0x2A:
        case 0x2D:
            return "sandybridge";//"SandyBridge - 32 nm";

        case 0x3A:
        case 0x3E:
            return "ivybridge";//"IvyBridge - 22 nm";

        case 0x3C:
        case 0x3F:
            return "haswell";//"Haswell - 22 nm";

        case 0x3D:
        case 0x4F:
            return "broadwell";//"Broadwell - 14 nm";

        case 0x55:
        case 0x5E:
            return "knl";//"Skylake - 14 nm";

        case 0x8E:
        case 0x9E:
            return "kabylake";//KabyLake - 14 nm";

        default:
            return "<Unknow>";
        }
    }
private:
    unsigned int _max_ext_leaf = 0;
    cpu_signature_t _cpu_signature;
};

class AKAutoChoose {
    typedef AnakinRunerInterface* (*create_func)(const char* device_type, int device_number);
    typedef std::string(*get_string_func)();
public:
    AKAutoChoose(std::string ak_so_dir): _ak_so_dir(ak_so_dir) {

    }
    AnakinRunerInterface* get_ak_instance(std::string device_type, int device_num) {
        if (device_type == "X86") {
            std::string this_cpu_arch = _cpu_helper.get_cpu_arch();
            //FIXME:choose real path
            std::string so_path = _ak_so_dir;
            void* so_handle = dlopen(so_path.c_str(), RTLD_LAZY);

            if (!so_handle) {
                fprintf(stderr, "Error: load so `%s' failed.\n", so_path.c_str());
                exit(-1);
            }

            create_func create_anakin = (create_func) get_symble(so_handle, "get_anakinrun_instance");
            get_string_func get_cpu_arch_string = (get_string_func) get_symble(so_handle,
                                                  "get_ak_cpu_arch_string");
            std::string so_cpu_arch = get_cpu_arch_string();

            if (this_cpu_arch != so_cpu_arch) {
                fprintf(stderr, "Error: load so not equal %s != %s .\n", this_cpu_arch.c_str(),
                        so_cpu_arch.c_str());
                exit(-1);
            } else {
                printf("choose x86 arch %s \n", so_cpu_arch.c_str());
            }

            AnakinRunerInterface* anakin_obj = create_anakin("X86", device_num);
            return anakin_obj;
        } else if (device_type == "NV") {
            //FIXME:choose real path
            std::string so_path = _ak_so_dir;
            void* so_handle = dlopen(so_path.c_str(), RTLD_LAZY);

            if (!so_handle) {
                fprintf(stderr, "Error: load so `%s' failed.\n", so_path.c_str());
                exit(-1);
            }

            create_func create_anakin = (create_func) get_symble(so_handle, "get_anakinrun_instance");
            AnakinRunerInterface* anakin_obj = create_anakin("NV", device_num);
            return anakin_obj;
        } else {
            fprintf(stderr, "Error: not support device type %s /n", device_type.c_str());
            exit(-1);
        }
    }
private:
    void* get_symble(void* so_handle, const char* symbol) {
        void* func = dlsym(so_handle, symbol);

        if (func == nullptr) {
            fprintf(stderr, "Error: interface changed! can`t find  %s .\n", symbol);
            exit(-1);
        }

        return func;
    }
    CPUID_Helper _cpu_helper;
    std::string _ak_so_dir;
};

class MiniTimer {
public:
    MiniTimer() {};

    void start() {
        gettimeofday(&start_time, NULL);
    }

    double end() {
        gettimeofday(&end_time, NULL);
        double ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec)
                    / 1000.0;
        return ms;
    }

    double get_msecond() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
    }

    struct timeval start_time;
    struct timeval end_time;
};


class ReadDataHelper {
public:
    static std::vector<std::string> string_split(std::string in_str, std::string delimiter) {
        std::vector<std::string> seq;
        int found = in_str.find(delimiter);
        int pre_found = -1;

        while (found != std::string::npos) {
            if (pre_found == -1) {
                seq.push_back(in_str.substr(0, found));
            } else {
                seq.push_back(in_str.substr(pre_found + delimiter.length(),
                                            found - delimiter.length() - pre_found));
            }

            pre_found = found;
            found = in_str.find(delimiter, pre_found + delimiter.length());
        }

        seq.push_back(in_str.substr(pre_found + 1, in_str.length() - (pre_found + 1)));
        return seq;
    }

    static std::vector<std::string> string_split(std::string in_str,
            std::vector<std::string>& delimiter) {
        std::vector<std::string> in;
        std::vector<std::string> out;
        out.push_back(in_str);

        for (auto del : delimiter) {
            in = out;
            out.clear();

            for (auto s : in) {
                auto out_s = string_split(s, del);

                for (auto o : out_s) {
                    out.push_back(o);
                }
            }
        }

        return out;
    }
};

class SaveDataHelper {
public:
    static void record_2_file(const float* data_ptr, int size, const char* locate) {
        FILE* fp = fopen(locate, "w+");

        if (fp == nullptr) {
            std::cout << "file open field " << locate << std::endl;
        } else {
            for (int i = 0; i < size; ++i) {
                fprintf(fp, "%f \n", i, (data_ptr[i]));
            }

            fclose(fp);
        }

        std::cout << "!!! write success: " << locate << std::endl;
    }
};

class ShowDataHelper {
public:
    template <typename Dtype>
    static void show_vector(std::vector<Dtype> in) {
        std::cout << "show_vector :";

        for (int i = 0; i < in.size(); i++) {
            std::cout << std::to_string(in[i]) << ",";
        }

        std::cout << std::endl;

    }
};

class RNNDataHelper {
public:
    static void split_string(const std::string& s,
                             std::vector<std::string>& v, const std::string& c) {
        std::string::size_type pos1, pos2;
        pos2 = s.find(c);
        pos1 = 0;

        while (std::string::npos != pos2) {
            v.push_back(s.substr(pos1, pos2 - pos1));

            pos1 = pos2 + c.size();
            pos2 = s.find(c, pos1);
        }

        if (pos1 != s.length()) {
            v.push_back(s.substr(pos1));
        }
    }

    static int split_word_from_file(
        std::vector<std::vector<float> >& word_idx,
        const std::string input_file_path,
        const std::string split_token,
        const std::string inner_split_token,
        const int col_select) {

        std::ifstream infile(input_file_path.c_str());

        if (!infile.good()) {
            std::cout << "Cannot open " << std::endl;
            return -1;
        }

        std::cout << "found filename: " << input_file_path << std::endl;
        std::string line;
        std::vector<std::string> split_v;
        std::vector<std::string> split_w;
        int word_count = 0;

        while (std::getline(infile, line)) {
            split_v.clear();
            split_string(line, split_v, split_token);
            assert(split_v.size() >= col_select + 1);
            std::vector<float> word;
            std::vector<float> mention;
            split_w.clear();
            split_string(split_v[col_select], split_w, inner_split_token);

            for (auto w : split_w) {
                word.push_back(atof(w.c_str()));
                word_count++;
            }

            word_idx.push_back(word);
        }

        return word_count;
    }

    static std::vector<std::vector<float> > get_input_data(std::string data_file_path,
            std::string data_split_word, int split_index) {
        std::vector<std::vector<float> > word_idx;

        if (split_word_from_file(word_idx, data_file_path, data_split_word, " ", split_index) == -1) {
            std::cout << " NOT FOUND " << data_file_path << std::endl;
            exit(-1);
        }

        return word_idx;
    };
};



#endif //ANAKIN_ANAKIN_HELPER_H
