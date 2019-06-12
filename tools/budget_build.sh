#!/bin/bash
set -e
# This script shows how one can build a anakin for the <NVIDIA> gpu platform
ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"
BUDGET_RELEASE_ROOT=$ANAKIN_ROOT/budget_release
# build the target into gpu_build.
BUILD_ROOT=$ANAKIN_ROOT/budget_build

mkdir -p $BUILD_ROOT
echo "-- Build anakin budget(NVIDIA/X86) into: $BUILD_ROOT"

# create env script
create_env_scripts() {
    if [ ! $# -eq 1 ]; then
        exit 1
    fi
    CURRENT_PATH=$1
    if [ ! -d "$CURRENT_PATH" ]; then
        mkdir -p $CURRENT_PATH
    fi
    if [ -f  $CURRENT_PATH/env.sh ]; then
        rm -f $CURRENT_PATH/env.sh
    fi
    echo "#!/bin/bash" >> $CURRENT_PATH/env.sh
    echo "echo \"LD_LIBRARY_PATH=$CURRENT_PATH\"" >> $CURRENT_PATH/env.sh
}

# build anakin for budget term
build_for_budget() {
    if [ ! $# -eq 1 ]; then
		exit 1
	fi
    # Now, actually build the gpu target.
    IF_OPEN_TIMER=$1
    if [ "$IF_OPEN_TIMER" = "TRUE" ]; then
        echo "-- Building anakin with timer..."
        option='YES'
        output=$BUDGET_RELEASE_ROOT/with_timer
        create_env_scripts $output
    else
        echo "-- Building anakin without timer..."
        option='NO'
        output=$BUDGET_RELEASE_ROOT/without_timer
        create_env_scripts $output
    fi
    cd $BUILD_ROOT
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
	    -DUSE_ARM_PLACE=NO \
	    -DUSE_GPU_PLACE=YES \
        -DNVIDIA_GPU=YES \
        -DAMD_GPU=NO \
	    -DUSE_X86_PLACE=YES \
	    -DUSE_BM_PLACE=NO \
	    -DBUILD_FAT_BIN=YES \
	    -DBUILD_WITH_UNIT_TEST=YES \
        -DBUILD_RPC=OFF \
   	    -DUSE_PYTHON=OFF \
        -DUSE_GFLAGS=OFF \
	    -DENABLE_DEBUG=OFF \
	    -DENABLE_VERBOSE_MSG=NO \
	    -DDISABLE_ALL_WARNINGS=YES \
	    -DENABLE_NOISY_WARNINGS=NO \
        -DUSE_OPENMP=YES\
	    -DBUILD_SHARED=YES\
	    -DBUILD_WITH_FRAMEWORK=YES\
        -DENABLE_OP_TIMER=$option
        
    # build target lib or unit test.
    if [ "$(uname)" = 'Darwin' ]; then
        make "-j$(sysctl -n hw.ncpu)" && make install
    else
        make "-j$(nproc)"   && make install
    fi
    # clean and move build results
    cp -r $ANAKIN_ROOT/output/lib* $output
    cp -r $ANAKIN_ROOT/output/unit_test/net_audit_exec $output
}
 
build_for_budget "TRUE"
build_for_budget "FALSE"
