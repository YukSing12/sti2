CUR_DIR=`pwd`

rm -rf build
mkdir -p build

STI2_ROOT=${CUR_DIR}/../..
INSTALL_PREFIX=${STI2_ROOT}

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
make install