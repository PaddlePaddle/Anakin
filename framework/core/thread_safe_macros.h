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

#ifndef ANAKIN_THREAD_SAFE_MACROS_H
#define ANAKIN_THREAD_SAFE_MACROS_H 

/// Anakin uses the following  annotations to detect and warn about potential issues 
/// that could result in data races and deadlocks. 
/// The thread safety issues detected by the gcc analysis pass include:
///     1. Accesses to shared variables and function calls are not guarded by proper (read or write) locks
///     2. Locks are not acquired in the specified order.
///     3. A cycle in the locking order.
///     4. Try to acquire a lock that is already held by the same thread.
///     5. Useful when mutex locks are non-reentrant.
///     6. Locks are not acquired and released in the same routine (or in the same block scope).
///     7. Having critical sections starting and ending in the same routine is a good practice.
#if defined(__GNUC__) && defined(__SUPPORT_TS_ANNOTATION__)
/// The GUARDED_BY and GUARDED_VAR document a shared variable/field that needs to be protected by a / any lock.
/// GUARDED_BY specifies a paritcular lock should be held when thread accesses the annotated variable.
/// GUARDED_VAR should be used when you can't express the name of lock.
#define GUARDED_BY(x)          __attribute__ ((guarded_by(x)))
#define GUARDED_VAR            __attribute__ ((guarded))
/// these two annotations document a memory location pointer that should
/// be guarded by a lock when dereferencing the pointer.
/// note:
///      they don't support smart pointer.
///      the pointer itself can be guarded by mutex. and they don't support smart pointer.
///       e.g int *a GUARDED_BY(mutex_1) PT_GUARDED_BY(mutex_2);
#define PT_GUARDED_BY(x)       __attribute__ ((point_to_guarded_by(x)))
#define PT_GUARDED_VAR         __attribute__ ((point_to_guarded))
/// These two annotations document the acquisition order between locks that can be held by a thread simultaneously.
#define ACQUIRED_AFTER(...)    __attribute__ ((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...)   __attribute__ ((acquired_before(__VA_ARGS__)))
/// This annotation documents if a class/type is a lockable type (e.g. a mutex lock class).
#define LOCKABLE               __attribute__ ((lockable))
/// This annotation documents if a class is a scoped lock type. 
/// A scoped lock object acquires a lock at construction and releases it when the object goes out of scope.
#define SCOPED_LOCKABLE        __attribute__ ((scoped_lockable))
/// function annotations: the following annotations specify lock and unlock primitives
/// note: for the lock and unlock methods of a lockable class, 
///       the annotations don't need to take any argument.
///       they can also be formal parameters or parameter positions (1-based) of the lock/unlock primitives
#define EXCLUSIVE_LOCK_FUNCTION(...)    __attribute__ ((exclusive_lock(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)       __attribute__ ((shared_lock(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...)            __attribute__ ((unlock(__VA_ARGS__)))
/// the following two trylock annotations take optional arguments that specify the locks the primitives try to acquire.
/// they take a integer or boolean argument (succ_return value maybe true or 0)
/// that specifies the return value of a successful lock acquisition
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__ ((exclusive_trylock(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__ ((shared_trylock(__VA_ARGS__)))
/// These following three annotations document lock requirements of functions/methods. 
/// If a function expects certain locks to be held before it is called, 
/// it needs to be annotated with EXCLUSIVE_LOCKS_REQUIRED and/or SHARED_LOCKS_REQUIRED. 
/// Note:
///     they are intended to be applied to internal/private functions/methods, not to public APIs
#define EXCLUSIVE_LOCKS_REQUIRED(...)   __attribute__ ((exclusive_locks_required(__VA_ARGS__)))
#define SHARED_LOCKS_REQUIRED(...)      __attribute__ ((shared_locks_required(__VA_ARGS__)))
#define LOCKS_EXCLUDED(...)             __attribute__ ((locks_excluded(__VA_ARGS__)))
/// If a function/method returns a lock without acquiring it, the returned lock needs to be documented using LOCK_RETURNED annotation
#define LOCK_RETURNED(x)                __attribute__ ((lock_returned(x)))
/// we can prevent the thread safety analysis on function by NO_THREAD_SAFETY_ANALYSIS
#define NO_THREAD_SAFETY_ANALYSIS       __attribute__ ((no_thread_safety_analysis))
#else 
/// Note: 
///  When the compiler is not GCC, these annotations are define null.
#define GUARDED_BY(x)
#define GUARDED_VAR
#define PT_GUARDED_BY(x)
#define PT_GUARDED_VAR
#define ACQUIRED_AFTER(arg)
#define ACQUIRED_BEFORE(arg)
#define EXCLUSIVE_LOCKS_REQUIRED(arg)
#define SHARED_LOCKS_REQUIRED(arg)
#define LOCKS_EXCLUDED(arg)
#define LOCK_RETURNED(x)
#define LOCKABLE
#define SCOPED_LOCKABLE
#define EXCLUSIVE_LOCK_FUNCTION(arg)
#define SHARED_LOCK_FUNCTION(arg)
#define EXCLUSIVE_TRYLOCK_FUNCTION(arg)
#define SHARED_TRYLOCK_FUNCTION(arg)
#define UNLOCK_FUNCTION(arg)
#define NO_THREAD_SAFETY_ANALYSIS
#endif /// defined(__GNUC__) && defined(__SUPPORT_TS_ANNOTATION__)

#endif
