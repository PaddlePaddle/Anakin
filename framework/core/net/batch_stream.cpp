

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

#include "framework/core/net/batch_stream.h"
namespace anakin {

/**
 *  \brief Net class used for execution of graph and it is thread safety.
 */
template<typename Ttype>
BatchStream<Ttype>::BatchStream(std::string file, int batch_size): _batch_size(batch_size) {
    std::ifstream ifs(file, std::ofstream::out | std::ofstream::binary);
    CHECK(ifs.is_open()) << file << "can not be opened";

    while (ifs.good()) {
        std::string new_file;
        std::getline(ifs, new_file);
        _file_list.push_back(new_file);
    }

    _file_id = 0;
    _ifs.open(_file_list[_file_id++]);
    CHECK(_ifs.is_open()) << _file_list[_file_id - 1] << "can not be opened";
    _ifs.read((char*)(&_num), 4);
    _ifs.read((char*)(&_channel), 4);
    _ifs.read((char*)(&_height), 4);
    _ifs.read((char*)(&_width), 4);
    Shape shape = std::vector<int> {batch_size, _channel, _height, _width};
    auto tensor = new Tensor<X86>(shape);
    _cpu_tensors.push_back(tensor);
}

template<typename Ttype>
void BatchStream<Ttype>::reset() {
    _file_id = 0;
    _ifs.open(_file_list[_file_id++]);
    CHECK(_ifs.is_open()) << _file_list[_file_id - 1] << "can not be opened";
    _ifs.read((char*)(&_num), 4);
    _ifs.read((char*)(&_channel), 4);
    _ifs.read((char*)(&_height), 4);
    _ifs.read((char*)(&_width), 4);

}

template<typename Ttype>
int BatchStream<Ttype>::get_batch_data(std::vector<Tensor4dPtr<Ttype>> outs) {
    Shape shape = std::vector<int> {_batch_size, _height, _width, _channel};
    _cpu_tensors[0]->reshape(shape);

    int num = std::min(_num, _batch_size);
    int image_size = _channel * _height * _width;
#ifdef USE_CUDA
    auto data = _cpu_tensors[0]->mutable_data();
#else
    auto data = outs[0]->mutable_data();
#endif

    _ifs.read((char*)(data), sizeof(float) *  image_size * num);
    data += image_size * num;
    _num -= num;

    while (num < _batch_size) {
        if (_file_id >= _file_list.size()) {
            _ifs.close();
            break;
        }

        _ifs.close();
        _ifs.open(_file_list[_file_id++]);
        _ifs.read((char*)(&_num), 4);
        _ifs.read((char*)(&_channel), 4);
        _ifs.read((char*)(&_height), 4);
        _ifs.read((char*)(&_width), 4);
        int cur_num = std::min(_num, _batch_size - num);
        _num -= cur_num;
        _ifs.read((char*)(data), sizeof(float) *  image_size * cur_num);
        num += cur_num;
        data += image_size * cur_num;
    }

    if (num != 0) {
        //outs[0]->reshape(Shape{num, _channel, _height,_width});
        Shape shape = std::vector<int> {num, _height, _width, _channel};
        _cpu_tensors[0]->reshape(shape);
        outs[0]->reshape(shape);
#ifdef USE_CUDA
        outs[0]->copy_from(*_cpu_tensors[0]);
#endif
    }

    return num;
}

#ifdef USE_CUDA
template class BatchStream<NV>;
#endif
#ifdef USE_X86_PLACE
template class BatchStream<X86>;
#endif
}
