

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

#ifndef ANAKIN_BATCH_STREAM_H
#define ANAKIN_BATCH_STREAM_H

#include "anakin_config.h"

#ifndef USE_SGX
#include "framework/core/parameter.h"
#include "framework/core/data_types.h"
#include "saber/saber_types.h"
#include <fstream>
#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif
namespace anakin {

/** 
 *  \brief BatchStream  used for get batch data from some file list.
 */
template<typename Ttype>
class BatchStream {
public:
    BatchStream(std::string file, int batch_size);

#ifdef USE_OPENCV
    BatchStream(std::string image_list, int channel, int height, int width, \
        std::vector<float> mean = {1.f, 1.f, 1.f}, std::vector<float> scale = {1.f, 1.f, 1.f});
#endif
    ~BatchStream();

    void reset();

    int get_batch_data(std::vector<Tensor4dPtr<Ttype>> outs);
private:
    int _batch_size;
    std::vector<std::string> _file_list;
    Tensor<X86> _host_tensor;
    std::ifstream _ifs;
    int _num;
    int _channel;
    int _height;
    int _width;
    int _file_id;
    std::vector<float> _mean;
    std::vector<float> _scale;
    bool _flag_from_image{false};
};

}

#endif // USE_SGX
#endif
