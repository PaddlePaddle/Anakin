set(SGX_ENCLAVE_SIGNER ${SGX_SDK}/bin/x64/sgx_sign)
set(SGX_EDGER8R ${SGX_SDK}/bin/x64/sgx_edger8r)

set(TRUSTED_DIR ${CMAKE_CURRENT_BINARY_DIR}/trusted)
set(UNTRUSTED_DIR ${CMAKE_CURRENT_BINARY_DIR}/untrusted)

macro(anakin_sgx_copy_example part file)
  add_custom_command(
    OUTPUT ${ANAKIN_SGX}/${part}/${file}
    COMMAND ${CMAKE_COMMAND}
       ARGS -E copy
            ${ANAKIN_SGX}/${part}/example/${file}
            ${ANAKIN_SGX}/${part}/${file}
    DEPENDS ${ANAKIN_SGX}/${part}/example/${file}
    COMMENT "Using the example ${file} for SGX ${part}"
  )
endmacro()

file(GLOB examples "enclave/example/*")
foreach(example ${examples})
  get_filename_component(file ${example} NAME)
  anakin_sgx_copy_example("enclave" ${file})
endforeach()

file(GLOB examples "app/example/*")
foreach(example ${examples})
  get_filename_component(file ${example} NAME)
  anakin_sgx_copy_example("app" ${file})
endforeach()

set(ENCLAVE_EDL ${ANAKIN_SGX}/enclave/enclave.edl)
set(ENCLAVE_LDS ${ANAKIN_SGX}/enclave/version.lds)
set(ENCLAVE_KEY ${ANAKIN_SGX}/enclave/sign_enclave.pem)
set(ENCLAVE_CONFIG ${ANAKIN_SGX}/enclave/config.xml)
set(ECALL_EDL ${ANAKIN_SGX}/enclave/ecall.edl)
set(ECALL_SRC ${ANAKIN_SGX}/enclave/ecall.cpp)
set(APP_SRC ${ANAKIN_SGX}/app/app.cpp)
set(INIT_SRC ${ANAKIN_SGX}/app/init.c)
set(OCALL_SRC ${ANAKIN_SGX}/app/ocall.c)

add_custom_command(
  OUTPUT ${TRUSTED_DIR}/enclave_t.c ${TRUSTED_DIR}/enclave_t.h
         ${UNTRUSTED_DIR}/enclave_u.c ${UNTRUSTED_DIR}/enclave_u.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TRUSTED_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNTRUSTED_DIR}
  COMMAND ${SGX_EDGER8R}
     ARGS ${ENCLAVE_EDL}
          --trusted --trusted-dir ${TRUSTED_DIR}
          --untrusted --untrusted-dir ${UNTRUSTED_DIR}
          --search-path ${SGX_SDK}/include
          --search-path ${ANAKIN_SGX}/enclave
  DEPENDS ${SGX_EDGER8R} ${ENCLAVE_EDL} ${ECALL_EDL}
  COMMENT "Generatring enclave bridge for ${ENCLAVE_EDL}..."
)

set(TRUSTED_SRC ${TRUSTED_DIR}/enclave_t.c)
anakin_fetch_files_with_suffix("${ANAKIN_SGX}/enclave/src" "c" TRUSTED_SRC)
anakin_fetch_files_with_suffix("${ANAKIN_SGX}/enclave/src" "cpp" TRUSTED_SRC)
add_library(anakin_trusted STATIC ${TRUSTED_SRC})
target_link_libraries(anakin_trusted ${SGX_CONFIG_INTERFACE})

set(UNTRUSTED_SRC ${UNTRUSTED_DIR}/enclave_u.c)
add_library(anakin_untrusted STATIC ${UNTRUSTED_SRC})

target_include_directories(anakin_trusted PUBLIC
  ${TRUSTED_DIR}
  ${ANAKIN_FRAMEWORK}/graph
  ${ANAKIN_FRAMEWORK}/core/net
  ${ANAKIN_SABER}
)

target_include_directories(anakin_untrusted PUBLIC ${UNTRUSTED_DIR})
target_include_directories(anakin_untrusted PUBLIC ${SGX_SDK}/include)

set(MKL_PATCHED_DIR ${ANAKIN_ROOT}/third-party/mkl-patched)
set(MKL_PATCHED_LIB ${MKL_PATCHED_DIR}/libmkl_patch.a)
set(MKL_PATCHED_URL "https://raw.githubusercontent.com/rdzhou/mkl_patch/master/libmkl_patch.a")

add_executable(anakin_enclave ${ECALL_SRC} ${ECLAVE_LDS} ${MKL_PATCHED_LIB})

add_custom_command(
  OUTPUT ${MKL_PATCHED_LIB}
  COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${MKL_PATCHED_DIR}
  COMMAND wget ARGS -O ${MKL_PATCHED_LIB} ${MKL_PATCHED_URL}
  COMMENT "Downloading mkl patch for sgx build from ${MKL_PATCHED_URL}..."
  VERBATIM
)

add_dependencies(anakin_enclave
   ${anakin_lib_static}
   ${ANAKIN_SABER_LIB_TARGET}
   anakin_trusted
)

add_custom_target(enclave_assets DEPENDS
  ${ENCLAVE_LDS} ${ENCLAVE_KEY} ${ENCLAVE_CONFIG}
)

add_dependencies(anakin_enclave enclave_assets)

set(SGX_JIT_LDS ${ANAKIN_SGX}/enclave/enclave.lds)

set_target_properties(anakin_enclave PROPERTIES LINK_DEPENDS ${SGX_JIT_LDS})

target_link_libraries(anakin_enclave
  -Wl,-T,${SGX_JIT_LDS}
  -L${SGX_SDK}/lib64
  -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles
  -Wl,-Bstatic -Wl,-Bsymbolic -Wl,-pie,-eenclave_entry
  -Wl,--undefined,__anakin_enclave_init_status
  -Wl,--defsym,__ImageBase=0 -Wl,--gc-sections
  -Wl,--version-script=${ENCLAVE_LDS}
)

if(SGX_SIM_MODE)
  set(SGX_LIB_TYPE "_sim")
endif()

# link anakin components
target_link_libraries(anakin_enclave
  -Wl,--whole-archive
  -lsgx_trts${SGX_LIB_TYPE} anakin_trusted ${anakin_lib_static} ${ANAKIN_SABER_LIB_TARGET}
  -Wl,--no-whole-archive
)

# link mkl
target_link_libraries(anakin_enclave
  -Wl,--start-group
  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
  -Wl,--end-group
  ${MKL_PATCHED_LIB}
)

# link standard sgx libraries
target_link_libraries(anakin_enclave
  -Wl,--start-group
  -lsgx_tcxx -lsgx_tstdc -lsgx_tcrypto -lsgx_tservice${SGX_LIB_TYPE}
  -Wl,--end-group
)

# sign the enclave
add_custom_command(
  OUTPUT anakin_enclave.signed
  DEPENDS anakin_enclave ${ENCLAVE_KEY} ${ENCLAVE_CONFIG}
  COMMAND sgx_sign
     ARGS sign -key ${ENCLAVE_KEY} -enclave anakin_enclave
     -out anakin_enclave.signed -config ${ENCLAVE_CONFIG}
  COMMENT "Signing the enclave using\n  key: ${ENCLAVE_KEY}\n  config: ${ENCLAVE_CONFIG}"
)

add_custom_target(anakin_enclave_signed ALL
  DEPENDS anakin_enclave.signed
)

add_executable(anakin_app ${APP_SRC} ${OCALL_SRC} ${INIT_SRC})
target_compile_options(anakin_app PUBLIC -UNDEBUG)
target_link_libraries(anakin_app
  anakin_untrusted
  -L${SGX_SDK}/sdk_libs
  -lsgx_urts${SGX_LIB_TYPE}
  -lsgx_uae_service${SGX_LIB_TYPE}
  -lpthread
)

set(CLI11_SRC ${UNTRUSTED_DIR}/CLI11.hpp)
add_custom_command(
  OUTPUT ${CLI11_SRC}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNTRUSTED_DIR}
  COMMAND wget ARGS
     -q https://github.com/CLIUtils/CLI11/releases/download/v1.7.1/CLI11.hpp
     -O ${UNTRUSTED_DIR}/CLI11.hpp
  COMMENT "Downloading CLI11.hpp..."
)
add_custom_target(cli11 ALL
  DEPENDS ${CLI11_SRC}
)
add_dependencies(anakin_app cli11)
