
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

#ifndef ANAKIN_CALIBRATOR_H
#define ANAKIN_CALIBRATOR_H

#include "anakin_config.h"

#ifndef USE_SGX

#include "framework/core/net/batch_stream.h"
#include "framework/core/base.h"
#include "framework/core/operator/operator.h"
#include <map>
#include "framework/core/net/net.h"

namespace anakin {

/** 
 *  \brief Net class used for execution of graph and it is thread safety.
 */
template<typename Ttype>
class Calibrator {
public:
    Calibrator(BatchStream<Ttype>* batch_stream,
              int batch_size,
              std::string calibrator_file,
              Net<Ttype, Precision::FP32, OpRunType::SYNC>* net) :
            _batch_stream(batch_stream),
            _batch_size(batch_size),
            _calibrator_file(calibrator_file),
            _net(net){}

    virtual ~Calibrator() {}

    virtual int get_batch_data(std::vector<Tensor4dPtr<Ttype>> inputs) = 0;

    virtual int get_batch_size() {return _batch_size;}

    virtual void reset_data_stream() = 0;

    virtual void read_calibrator() = 0;

    virtual void write_calibrator() = 0;

    virtual void generate_calibrator_table() = 0;

    virtual CalibrationAlgoType get_algorithm() = 0;

    std::vector<Tensor4dPtr<Ttype>> get_in_vec() {
        return _net->get_in_list();
    }

    std::vector<OperatorFunc<Ttype, Precision::FP32>> get_exec_funcs() {
        return _net->_exec_funcs;
    }

    std::vector<std::string> get_tensor_name_list() {
        return _net->_tensor_name_list;
    }
    
protected:

    BatchStream<Ttype>* _batch_stream;
    std::string _calibrator_file;
    int _batch_size;
    Net<Ttype, Precision::FP32, OpRunType::SYNC>* _net;

};
}
#endif // USE_SGX

#endif
