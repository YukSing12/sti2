CUR_DIR=`pwd`

rm -rf build
mkdir -p build

STI2_ROOT=${CUR_DIR}/../..
INSTALL_PREFIX=${STI2_ROOT}/install

cd ${CUR_DIR}/build
cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE=OFF \
	-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-DBUILD_TF=OFF \
	-DBUILD_PYT=OFF \
	-DBUILD_TRT=ON \
	-DBUILD_MULTI_GPU=OFF \
	-DUSE_NVTX=OFF \
	..


make -j$(nproc) 
cp ${CUR_DIR}/build/lib/libErniePlugin.so ${STI2_ROOT}/so/plugins/libErniePlugin.so
cp ${CUR_DIR}/build/bin/ernie_infer ${STI2_ROOT}/bin
