/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_FRAMEWORK_LITE_CODE_GENERATE_CPP_H
#define ANAKIN_FRAMEWORK_LITE_CODE_GENERATE_CPP_H

#include "framework/lite/op_map.h"
#include "framework/lite/code_gen_base.h"

namespace anakin {

namespace lite {

/**  
 *  \brief class to generate cpp files.
 *
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class GenCPP : public CodeGenBase<Ttype, Dtype, Ptype> {
public:
	explicit GenCPP(std::string model_name, std::string model_dir = ".") {
		_cpp_file_name = model_dir + '/' + model_name + ".cpp";
		_h_file_name = model_dir + '/' + model_name + ".h";
		_model_file_name = model_dir + '/' + model_name + ".bin";
		_weights.open(_model_file_name);
		_code_name = model_name;
	}
	~GenCPP()=default;

	/// generate all cpp files
	virtual void gen_files() {
		this->gen_header();
		this->gen_source();
	}

private:
	void gen_header_start();
	void gen_header_end();
	void gen_source_start();
	void gen_source_end();

	/**
	 * \brief generate tensors for edges
	 */
	void gen_tensors();

	/**
	 * \brief generate model's inputs and outputs
	 */
	void gen_model_ios();

	/**
	 * \brief generate and parsing operations for model
	 */
	virtual void gen_and_parse_ops() final;

	/**
	 * \brief generate initial impl api for model
	 */
	void gen_init_impl();

	/**
	 * \brief generate running api impl for model
	 */
	void gen_api_impl();	

	/**
	 * \biref generata header file
	 */
	void gen_header();

	/**
	 * \biref generata source file
	 */
	void gen_source();

private:
	std::string _cpp_file_name;
	std::string _h_file_name;
	std::string _model_file_name;
	std::string _code_name;
	CodeWritter _code;
	WeightsWritter<Ttype, Dtype> _weights;
};

} /* namespace lite */

} /* namespace anakin */

#endif
