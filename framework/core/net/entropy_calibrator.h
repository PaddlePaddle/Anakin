
/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_ENTROPY_CALIBRATOR_H
#define ANAKIN_ENTROPY_CALIBRATOR_H

#include "anakin_config.h"

#ifndef USE_SGX

#include "framework/core/net/calibrator.h"

namespace anakin {

/** 
 *  \brief Net class used for execution of graph and it is thread safety.
 */
template<typename Ttype>
class EntropyCalibrator: public Calibrator<Ttype> {
public:
    EntropyCalibrator(BatchStream<Ttype>* stream,
              int batch_size,
              std::string calibrator_file, 
              Net<Ttype, Precision::FP32, OpRunType::SYNC>* net,
              int bin_num):
         Calibrator<Ttype>(stream, batch_size, calibrator_file, net), 
         _bin_num(bin_num) {}

    ~EntropyCalibrator() { 
        for (auto tensor : _in_vec) {
             delete tensor;
             tensor = nullptr;
        }
    }

    virtual void init_statistics(int tensor_num);

    virtual int get_batch_data(std::vector<Tensor4dPtr<Ttype>> inputs);

    virtual void reset_data_stream();

    //virtual int get_batch_size() {return _batch_size;}

    virtual void read_calibrator();

    virtual void write_calibrator();

    virtual void generate_calibrator_table();

    virtual CalibrationAlgoType get_algorithm() {return ENTROPY;}

private:
    void get_ref_q(std::vector<int>& ref_p, std::vector<float>& ref_q);

    void expand_to_q(std::vector<int>& ref_p, std::vector<float>& ref_q, std::vector<float>& q);

    float get_kl_divergence(std::vector<int>&ref_p, std::vector<float>& q);

    void get_histgrams(std::vector<Tensor4dPtr<Ttype>> in_vec,
            std::vector<OperatorFunc<Ttype, Precision::FP32 >> exec_funcs);

    void get_max_values(std::vector<Tensor4dPtr<Ttype>> in_vec,
            std::vector<OperatorFunc<Ttype, Precision::FP32 >> exec_funcs);

    void get_kl_threshold(std::vector<std::string>& tensor_name_list);

    float max_data(Tensor4dPtr<Ttype> tensor, int tensor_id);

    void histgram(Tensor4dPtr<Ttype> tensor, int tensor_id);

    int get_bin_num() {return _bin_num;}

    std::vector<float>& max_vec() {return _max_vec;}

    std::vector<std::vector<int>>& hist_vecs() {return _hist_vecs;}
    
protected:
    std::vector<Tensor4dPtr<X86>> _in_vec;

    std::map<std::string, float> _scale_map;

    std::vector<float> _max_vec;

    std::vector<std::vector<int>> _hist_vecs;

    int _bin_num;
};
}
#endif // USE_SGX

#endif
