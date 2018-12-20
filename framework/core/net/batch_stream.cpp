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

#ifndef USE_SGX

#include "saber/core/tensor_op.h"
namespace anakin {
using namespace anakin::saber;
#ifdef USE_OPENCV
using namespace cv;
void fill_tensor_with_cvmat(const Mat& img_in, Tensor<X86>& tout, int num, \
    const int width, const int height, const float* mean, const float* scale, float& max_val) {
    cv::Mat im;
    max_val = 0.f;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    float* ptr_data_in = tout.mutable_data();
    int stride = width * height;
    for (int i = 0; i < num; i++) {
        float* ptr_in = ptr_data_in + i * tout.channel() * tout.height() * tout.width();
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                ptr_in[r * width + c] = (im.at<cv::Vec3b>(r, c)[0] - mean[0]) * scale[0];
                ptr_in[stride + r * width + c] = (im.at<cv::Vec3b>(r, c)[1] - mean[1]) * scale[1];
                ptr_in[2 * stride + r * width + c] = (im.at<cv::Vec3b>(r, c)[2] - mean[2]) * scale[2];
                if (fabsf(ptr_in[r * width + c]) > fabsf(max_val)) {
                    max_val = ptr_in[r * width + c];
                }
                if (fabsf(ptr_in[stride + r * width + c]) > fabsf(max_val)) {
                    max_val = ptr_in[stride + r * width + c];
                }
                if (fabsf(ptr_in[2 * stride + r * width + c]) > fabsf(max_val)) {
                    max_val = ptr_in[2 * stride + r * width + c];
                }
            }
        }
    }
}
#endif
/** 
 *  \brief Net class used for execution of graph and it is thread safety.
 */
template<typename Ttype>
BatchStream<Ttype>::BatchStream(std::string file, int batch_size):_batch_size(batch_size) {
    std::ifstream ifs(file, std::ifstream::in);
    CHECK(ifs.is_open()) << file << " can not be opened";
    while (ifs.good()) {
        std::string new_file;
        std::getline(ifs, new_file);
        _file_list.push_back(new_file);
    }
    _file_id = 0;
    _ifs.open(_file_list[_file_id++]);
    CHECK(_ifs.is_open()) << _file_list[_file_id -1] << "can not be opened";
    _ifs.read((char*)(&_num), 4);
    _ifs.read((char*)(&_channel), 4);
    _ifs.read((char*)(&_height), 4);
    _ifs.read((char*)(&_width), 4);
    Shape shape = std::vector<int> {batch_size, _channel, _height, _width};
    _host_tensor.reshape(shape);
    _flag_from_image = false;
}

template<typename Ttype>
void BatchStream<Ttype>::reset() {
    _file_id = 0;
    _ifs.open(_file_list[_file_id++]);
    CHECK(_ifs.is_open()) << _file_list[_file_id -1] << "can not be opened";
    _ifs.read((char*)(&_num), 4);
    _ifs.read((char*)(&_channel), 4);
    _ifs.read((char*)(&_height), 4);
    _ifs.read((char*)(&_width), 4);
}

template<typename Ttype>
BatchStream<Ttype>::~BatchStream() {}

#ifdef USE_OPENCV
template<typename Ttype>
BatchStream<Ttype>::BatchStream(std::string image_list, int channel, int height, int width, \
        std::vector<float> mean, std::vector<float> scale) {

    if (channel != mean.size() || channel != scale.size()) {
        LOG(FATAL) << "channel size must = mean size && scale size";
    }
    _num = 1;
    _batch_size = 1;
    _channel = std::max(1, channel);
    _height = std::max(1, height);
    _width = std::max(1, width);
    _mean = mean;
    _scale = scale;
    _file_list.clear();
    std::fstream fp(image_list);
    std::string line;
    while (getline(fp, line)) {
        _file_list.push_back(line);
    }
    LOG(INFO) << "image size: " << _num << ", " << _channel << ", " << _height << ", " << _width;
    LOG(INFO) << "total test image number: " << _file_list.size();
    for (int i = 0; i < _file_list.size(); ++i) {
        LOG(INFO) << "image path: " << _file_list[i];
    }
    _flag_from_image = true;
    Shape shape = std::vector<int> {_num, _channel, _height, _width};
    _host_tensor.reshape(shape);
}
#endif
template<typename Ttype>
int BatchStream<Ttype>::get_batch_data(std::vector<Tensor4dPtr<Ttype>> outs) {     
     Shape shape = std::vector<int>{_batch_size, _height, _width, _channel};
     int num = std::min(_num, _batch_size);
     int image_size = _channel * _height * _width;
#ifdef USE_CUDA
     auto data = static_cast<float*>(_host_tensor.mutable_data());
#else
     auto data = static_cast<float*>(outs[0]->mutable_data());
#endif
    if (!_flag_from_image) {
        _ifs.read((char* )(data), sizeof(float) *  image_size * num);
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
            _ifs.read((char* )(data), sizeof(float) *  image_size * cur_num);
            num += cur_num;
            data += image_size * cur_num;
        }
        if (num != 0) {
            //outs[0]->reshape(Shape{num, _channel, _height,_width});
            Shape shape = std::vector<int>{num, _height,_width, _channel};
            _host_tensor.reshape(shape);
            outs[0]->reshape(shape);
#ifdef USE_CUDA
            outs[0]->copy_from(_host_tensor);
#endif
        }
        return num;
    }
#ifdef USE_OPENCV
    if (_flag_from_image) {
        if (_file_list.size() < 2) {
            return 0;
        }
        cv::Mat img = cv::imread(_file_list.back(), cv::IMREAD_COLOR);
        if (img.empty()) {
            LOG(FATAL) << "load image " << _file_list.back() << " failed";
        }
        float max_val = 0.f;
        fill_tensor_with_cvmat(img, _host_tensor, _num, _width, _height, _mean.data(), _scale.data(), max_val);
        double mean_val = tensor_mean_value_valid(_host_tensor);
        LOG(INFO) << "load image " << _file_list.back() << " successed, with mean value: " << mean_val << ", max_val: " << max_val;
        _file_list.pop_back();
        Shape shape = std::vector<int>{_num, _channel, _height,_width};
	    outs[0]->reshape(shape);
        outs[0]->copy_from(_host_tensor);
        return 1;   
    }
#endif
    return 0;
}

#ifdef USE_CUDA
template class BatchStream<NV>;
#endif
#ifdef USE_X86_PLACE
template class BatchStream<X86>;
#endif
}

#endif // USE_SGX
