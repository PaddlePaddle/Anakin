function(anakin_fetch_files_with_suffix search_dir suffix outputs)
    exec_program(ls ${search_dir}                
             ARGS "*.${suffix}"                  
             OUTPUT_VARIABLE OUTPUT              
             RETURN_VALUE VALUE)                 
    if(NOT VALUE)                                
        string(REPLACE "\n" ";" OUTPUT_LIST "${OUTPUT}")                                                                                                             
        set(abs_dir "")                          
        foreach(var ${OUTPUT_LIST})              
            set(abs_dir ${abs_dir} ${search_dir}/${var})
            #message(STATUS "fetch_result: ${abs_dir}")
        endforeach()                             
        set(${outputs} ${${outputs}} ${abs_dir} PARENT_SCOPE)
    else()                                       
        #message(WARNING "anakin_fetch_files_recursively ${BoldRed}failed${ColourReset}:\n"
        #                "real_dir:${BoldYellow}${search_dir}${ColourReset}\n"
        #                "suffix:*.${BoldYellow}${suffix}${ColourReset} \n")
    endif()                                      
endfunction()



set(BIN_NAME bmkernel_bin)
set(LINK_CONFIG link/bm1682_ddr.lds)
set(BM_ROOT /usr/local/include/bm)
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/bm/device "c" DEVICE_KERNEL_SRC)
message(${DEVICE_KERNEL_SRC})
#string(REPLACE ".c" ".o" DEVICE_KERNEL_OBJ ${DEVICE_KERNEL_SRC})
#message(${DEVICE_KERNEL_OBJ})

set(OBJ_PATH "")
FOREACH(FILE_PATH ${DEVICE_KERNEL_SRC})
  STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${FILE_PATH})
  set(OBJ_PATH ${OBJ_PATH} ${FILE_NAME}.o)
  message(${FILE_NAME}.o)
ENDFOREACH(FILE_PATH)
message(${OBJ_PATH})

add_custom_command(OUTPUT bm_kernel_tmp
    COMMAND arm-none-eabi-gcc ${DEVICE_KERNEL_SRC} -c -mcpu=arm926ej-s -mfpu=vfp -fno-short-enums -std=gnu99 -O2 -Wall -Werror -ffunction-sections -fdata-sections -nostdlib -DENABLE_PRINT -I${BM_ROOT}/include/config -I${BM_ROOT}/include/common -I${BM_ROOT}/include/c_model -I${BM_ROOT}/include/firmware_core -I${BM_ROOT}/include/bmlib
    COMMAND arm-none-eabi-gcc -T ${BM_ROOT}/${LINK_CONFIG} -mcpu=arm926ej-s -mfpu=vfp -fno-short-enums -Wl,--check-sections -Wl,--gc-sections -Wl,--unresolved-symbols=report-all -Wl,--no-enum-size-warning -o ${BIN_NAME}.elf -Wl,--start-group -lc -lm ${OBJ_PATH}  ${BM_ROOT}/lib/device/fw-top.a ${BM_ROOT}/lib/device/fw-arm.a -Wl,--end-group
    COMMAND arm-none-eabi-objcopy -O binary -R *.slow* ${BIN_NAME}.elf ${BIN_NAME}_itcm.bin
    COMMAND hexdump -v -e '1/4 \"%08x\\n\"' ${BIN_NAME}_itcm.bin > ${BIN_NAME}_itcm.hex.sim
    COMMAND arm-none-eabi-objcopy -O binary -j *.slow* ${BIN_NAME}.elf ${BIN_NAME}_ddr.bin
    COMMAND hexdump -v -e '1/4 \"%08x\\n\"' ${BIN_NAME}_ddr.bin > ${BIN_NAME}_ddr.hex.sim
    COMMAND printf "%x" 0xAABBCCDD > ${BIN_NAME}.bin
    COMMAND printf "%x" 0x0 >> ${BIN_NAME}.bin
    COMMAND printf "%x" 0x0 >> ${BIN_NAME}.bin
    COMMAND printf "%x" 0x0 >> ${BIN_NAME}.bin

    COMMAND printf \"%x\" `wc -c < ${BIN_NAME}_itcm.hex.sim` >> ${BIN_NAME}.bin

    COMMAND cat ${BIN_NAME}_itcm.hex.sim >> ${BIN_NAME}.bin
    COMMAND cat ${BIN_NAME}_ddr.hex.sim >> ${BIN_NAME}.bin
    COMMAND mkdir -p /usr/local/include/bm
    COMMAND cp ${BIN_NAME}.bin /usr/local/include/bm/
    COMMENT "BM Kernel compilation..."
)
add_custom_target(ANAKIN ALL DEPENDS bm_kernel_tmp)
