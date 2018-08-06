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

#ifndef ANAKIN_FUNCTOR_H
#define ANAKIN_FUNCTOR_H 

#include <functional>

namespace anakin {

namespace core {

/** 
 *  \brief Functor class for algorithm.
 *  Template parameter:
 *	  - ParamTypes	Functor parameter pack.
 *    - RetType		Functor return type.	
 */
template<typename RetType = void, typename ...ParamTypes>
class Functor {
	typedef std::function<RetType(ParamTypes...)> FuncType;
public:
	Functor(){}
    Functor(FuncType& func):_func(func){};

    Functor<RetType, ParamTypes...>& operator=(FuncType& func) { _func=func; return ;}
	
	  /// must be overwritten
    virtual RetType operator()(ParamTypes ...parameters) = 0;

private:
    FuncType _func;
};


} /* namespace core */

} /* namespace anakin */

#endif /* ANAKIN_FUNCTOR_H */
