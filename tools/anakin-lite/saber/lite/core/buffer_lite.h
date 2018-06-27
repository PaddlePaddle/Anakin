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

#ifndef ANAKIN_SABER_LITE_CORE_BUFFER_LITE_H
#define ANAKIN_SABER_LITE_CORE_BUFFER_LITE_H

#include "saber/lite/core/common_lite.h"

namespace anakin{

namespace saber{

namespace lite{

template <ARMType ttype>
class Buffer{
public:
    typedef typename TargetTrait<ttype>::bdtype dtype;
    /**
     * \brief constructor
     */
    Buffer();
    /**
     * \brief constructor, allocate data
     */
    explicit Buffer(size_t size);

    /**
     * \brief construct from existence data
     * @param data
     * @param size
     */
    Buffer(dtype* data, size_t size);

    /**
     * \brief assigned function
     */
    Buffer& operator = (Buffer& buf);
	
    /**
     * \brief destructor
     */
    ~Buffer();

	/**
	* \brief deep copy function
	*/
	void copy_from(Buffer<ttype>& buf);

    /**
     * \brief set _data to (c) with length of (size)
     */
    void mem_set(int c, size_t size);

    /**
     * \brief re-alloc memory
     */
    void re_alloc(size_t size);

    /**
     * \brief alloc memory
     */
    void alloc(size_t size);

    /**
     * \brief free memory
     */
    void clean();

    /**
     * \brief return const data pointer
     */
    const dtype* get_data();

    /**
     * \brief return mutable data pointer
     */
    dtype* get_data_mutable();

    /**
     * \brief return total size of memory, in bytes
     */
    size_t get_capacity();

protected:
    dtype* _data;
    bool _own_data;
    size_t _capacity;

};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_LITE_CORE_BUFFER_LITE_H
